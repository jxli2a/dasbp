import psutil
import pyDAS
import sys
from numba import jit, prange, njit
import numpy as np
import h5py
import tqdm  # Progress bar
import dateutil.parser
import datetime
from scipy.signal import tukey
from copy import deepcopy
import os
# Time zone variables
import pytz
UTC = pytz.timezone("UTC")
PST = pytz.timezone("US/Pacific")

# Parallelization
Ncores = psutil.cpu_count(logical=False)

# C++/pybind11 functions
path_dasutil = os.path.split(os.path.abspath(__file__))[0]
pyDAS_path = os.path.join(path_dasutil, "build")
pyDAS_python = os.path.join(path_dasutil, "python")
os.environ['LD_LIBRARY_PATH'] = pyDAS_path
sys.path.insert(0, pyDAS_path)
sys.path.insert(0, pyDAS_python)


@jit(nopython=True, parallel=True)
def detrend(y):
    x = np.arange(len(y))
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
    b = y_mean - m * x_mean
    return y - (m*x + b)


@njit(parallel=True)
def detrend_2D(data):
    nCh = data.shape[0]
    x = np.arange(data.shape[1])
    x_mean = np.mean(x)
    for idx in prange(nCh):
        y = data[idx, :]
        y_mean = np.mean(y)
        m = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
        b = y_mean - m * x_mean
        data[idx, :] -= (m*x + b)
    return data


@njit(parallel=True)
def preprocess_medfilt(data):
    nt = data.shape[1]
    for idx in prange(nt):
        data[:, idx] -= np.median(data[:, idx])
    return data


def bandpass2D_c(data, freqmin, freqmax, dt, order=6, zerophase=False, nThreads=Ncores):
    phase = 0 if zerophase else 1
    data_bp = pyDAS.lfilter(data, freqmin*dt, order,
                            freqmax*dt, order, phase, nThreads)
    return data_bp


def readFile_HDF(filelist, fmin, fmax, desampling=True, taper=0.4, nChbuffer=1000, verbose=False, system=None, **kwargs):
    """
    read and pre-process list of HDF5 files
    """
    nFiles = len(filelist)
    fsRaw = np.zeros(nFiles, dtype=float)
    ntRaw = np.zeros(nFiles, dtype=int)
    nChRaw = np.zeros(nFiles, dtype=int)
    nChSamp = np.zeros(nFiles, dtype=float)

    order = kwargs.get("order", 4)
    zerophase = kwargs.get("zerophase", True)

    # First reading sampling parameters
    for idx, ifile in enumerate(filelist):
        with h5py.File(ifile, 'r') as fid:
            preproc = True if 'Data' in fid else False
            if preproc:
                # Converting ping period from nanoseconds to seconds and compute sampling rate
                fsRaw[idx] = fid['Data'].attrs["fs"]
                ntRaw[idx] = fid['Data'].attrs["nt"]
                nChRaw[idx] = fid['Data'].attrs["nCh"]
                attrs_names = fid['Data'].attrs.keys()
                # Possible key name for channel sampling interval
                if 'dCh' in attrs_names:
                    nChSamp[idx] = fid['Data'].attrs["dCh"]
                elif 'ChSamp' in attrs_names:
                    nChSamp[idx] = fid['Data'].attrs["ChSamp"]
            else:
                # Converting ping period from nanoseconds to seconds and compute sampling rate
                fsRaw[idx] = fid['Acquisition'].attrs.get(
                    "PulseRate")/fid['Acquisition']['Custom'].attrs.get("Decimation Factor")
                ntRaw[idx] = len(fid['Acquisition']['Raw[0]']
                                 ['Custom']['SampleCount'][:])
                nChRaw[idx] = fid['Acquisition']["Custom"].attrs.get(
                    "Num Output Channels")
                nChSamp[idx] = fid['Acquisition'].attrs.get(
                    "SpatialSamplingInterval")
    # Checking if all files have same number of channels and sampling rate
    if not np.all(fsRaw == fsRaw[0]):
        raise ValueError("Data do not have same sampling rate!")
    if not np.any(nChRaw == nChRaw[0]):
        raise ValueError("Data do not have same number of channels!")
    if not np.any(nChSamp == nChSamp[0]):
        raise ValueError("Data do not have same channel sampling!")
    fsRaw = fsRaw[0]
    nCh = nChRaw[0]
    nChSamp = nChSamp[0]
    # For reading fewer channels from original data
    min_ch = int(kwargs.get("min_ch", 0))
    max_ch = int(kwargs.get("max_ch", nCh))
    if max_ch > nCh:
        raise ValueError(
            "Maximum channel number (max_ch=%s) greater than total number of channels (nCh=%s)!" % (max_ch, nCh))
    nCh = max_ch - min_ch
    # Setting size of channel buffer based on number of channels to read
    nChbuffer = min(nChbuffer, nCh)
    # Total number of time samples
    ntRawTot = np.sum(ntRaw)
    # Getting additional parameters
    # Datetime value of the minimum request time
    minTime = kwargs.get("minTime", None)
    # Datetime value of the maximum request time
    maxTime = kwargs.get("maxTime", None)
    if minTime is not None and maxTime is not None:
        if minTime >= maxTime:
            raise ValueError(
                "minTime smaller or equal than maxTime! Change parameter values!")
        intervalSec = maxTime.timestamp() - minTime.timestamp()
        ntRawTot = np.ceil(intervalSec*fsRaw).astype(int)
    nt = ntRawTot
    fs = fsRaw
    fsRatio = 1
    # Allocating memory
    DAS_data = np.zeros((nCh, nt), dtype=np.float32)
    trace_buffer = np.zeros((nChbuffer, ntRawTot), dtype=np.float32)
    w_taper = tukey(ntRawTot, alpha=taper)
    nChunks = np.ceil(nCh/nChbuffer).astype(int)
    itraces = 0
    for ichnk in tqdm.tqdm(range(nChunks), desc="Processing data..."):
        # Pointer to initial time sample
        itime = 0
        # Getting number of traces to process
        ntraces = min(nChbuffer, nCh-itraces)
        ntRawLeft = np.copy(ntRawTot)
        # Reading raw data from all files
        for idx, ifile in enumerate(filelist):
            with h5py.File(ifile, 'r') as fid:
                preproc = True if 'Data' in fid else False
                if preproc:
                    startTime = dateutil.parser.parse(
                        fid["Data"].attrs["startTime"])
                else:
                    RawDataTime = (fid["Acquisition"]["Raw[0]"]
                                   ["RawDataTime"][0].astype(np.float)*1e-6)
                    startTime = datetime.datetime.fromtimestamp(
                        RawDataTime, tz=PST).astimezone(UTC)
                if idx == 0:
                    begTime = deepcopy(startTime)
                # Reading time interval
                it_min = 0
                it_max = ntRaw[idx]
                # Reading only data interval
                if minTime is not None and maxTime is not None:
                    it_min = max(
                        0, int((minTime.timestamp() - startTime.timestamp())*fsRaw))
                    it_max = min(ntRaw[idx], it_min+ntRawLeft)
                    ntRawLeft = ntRawLeft - it_max + it_min
                    if idx == 0:
                        if it_min > 0:
                            begTime = deepcopy(minTime)
                        else:
                            begTime = deepcopy(startTime)
                ntSamp = it_max - it_min
                if preproc:
                    trace_buffer[:ntraces, itime:itime+ntSamp] = fid['Data'][min_ch +
                                                                             itraces:min_ch+itraces+ntraces, it_min:it_max]
                else:
                    trace_buffer[:ntraces, itime:itime+ntSamp] = fid['Acquisition']['Raw[0]']['RawData'][min_ch +
                                                                                                         itraces:min_ch+itraces+ntraces, it_min:it_max]

                itime += ntSamp
        # Detrending data
        if kwargs.get("detrend", True):
            trace_buffer = detrend_2D(trace_buffer)
        # Tapering
        if kwargs.get("tapering", True):
            trace_buffer = trace_buffer*w_taper
        # Bandpassing the data
        if kwargs.get("filter", True):
            trace_buffer = bandpass2D_c(
                trace_buffer, fmin, fmax, 1.0/fsRaw, order=order, zerophase=zerophase)
        DAS_data[itraces:itraces+ntraces,
                 :] = (trace_buffer[:ntraces, ::fsRatio])[:, :nt]
        itraces += ntraces
        # time_done_process = time.time()
    if kwargs.get("median", True):
        DAS_data = preprocess_medfilt(DAS_data)
    # Converting raw amplitude to micro-strain
    if system is not None:
        if system == "OptaSense":
            with h5py.File(filelist[0], 'r') as fid:
                if "Acquisition_origin" in fid or "Acquisition" in fid:
                    if preproc:
                        G = fid["Acquisition_origin"].attrs.get("GaugeLength")
                        n = fid["Acquisition_origin"].attrs.get(
                            "Fibre Refractive Index")
                        lamd = fid["Acquisition_origin"].attrs.get(
                            "Laser Wavelength (nm)")*1e-9
                    else:
                        G = fid['Acquisition'].attrs.get("GaugeLength")
                        n = fid['Acquisition']['Custom'].attrs.get(
                            "Fibre Refractive Index")
                        lamd = fid['Acquisition']['Custom'].attrs.get(
                            "Laser Wavelength (nm)")*1e-9
                    eta = 0.78  # photo-elastic scaling factor for longitudinal strain in isotropic material
                    factor = 4.0*np.pi*eta*n*G/lamd
                    # Conversion factor from raw to delta phase
                    radconv = 10430.378850470453
                    DAS_data = DAS_data/factor/radconv*1e6
                else:
                    print("WARNING! Missing GaugeLength, Fibre Refractive Index, and Laser Wavelength info from metadata. Skipping data strain conversion!")
        else:
            print("WARNING! %s is not a known system")
    # header info
    info = {}
    info['dx'] = nChSamp
    info['dt'] = 1/fs
    info['fs'] = fs
    info['nt'] = nt
    info['nx'] = nCh
    info['begTime'] = begTime
    info['endTime'] = begTime + datetime.timedelta(seconds=nt/fs)
    return DAS_data, info
