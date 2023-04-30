import numpy as np
from .util import init_phase_window_para

# CUDA
import pycuda.driver as cuda
from pycuda import gpuarray

# import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import clear_context_caches
import threading


def ccMultiShiftKernel():
    mod = SourceModule(
        """
    #include<stdio.h>
    __global__  void 
    ccMultiShiftKernel(float *d_continuous, float *d_template, float *d_cc, int *d_index_shift, int npts_continuous,
                int npts_template, int ntrace, int nshift, int cclen) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = x + y * blockDim.x * gridDim.x;
        int j, index1, index2, ishift;
        float cc_temp;
        float temp1 = 0.0;
        float temp2 = 0.0;
        float temp3 = 0.0;
        index1 = floor(i * 1.0 / npts_continuous);
        if (index1 < ntrace) {
            for (j = 0; j < npts_template; j++) {
                temp1 = temp1 + d_continuous[i + j] * d_template[j + index1 * npts_template];
                temp2 = temp2 + d_continuous[i + j] * d_continuous[i + j];
                temp3 = temp3 + d_template[j + index1 * npts_template] * d_template[j + index1 * npts_template];
            }
            cc_temp = temp1 / sqrt(temp2 * temp3);
            if (temp2 == 0 || temp3 == 0) cc_temp = 0.0;
            __syncthreads();
            for (ishift=0; ishift < nshift; ishift++) {
                index2 = i-index1 * npts_continuous - d_index_shift[index1 * nshift + ishift];
                if (index2 > 0 && index2 < cclen) {
                    atomicAdd(&d_cc[ishift*cclen+index2], cc_temp);
                }
            }
        }
    }
    """
    )
    return mod.get_function("ccMultiShiftKernel")


def ccMultiShiftWeightedKernel():
    mod = SourceModule(
        """
    #include<stdio.h>
    __global__  void 
    ccMultiShiftWeightedKernel(float *d_continuous, float *d_template, float *d_cc, float *d_weight, int *d_index_shift, int npts_continuous,
                int npts_template, int ntrace, int nshift, int cclen) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = x + y * blockDim.x * gridDim.x;
        int j, index1, index2, ishift;
        float cc_temp;
        float temp1 = 0.0;
        float temp2 = 0.0;
        float temp3 = 0.0;
        index1 = floor(i * 1.0 / npts_continuous);
        if (index1 < ntrace) {
            for (j = 0; j < npts_template; j++) {
                temp1 = temp1 + d_continuous[i + j] * d_template[j + index1 * npts_template];
                temp2 = temp2 + d_continuous[i + j] * d_continuous[i + j];
                temp3 = temp3 + d_template[j + index1 * npts_template] * d_template[j + index1 * npts_template];
            }
            cc_temp = temp1 / sqrt(temp2 * temp3);
            if (temp2 == 0 || temp3 == 0) cc_temp = 0.0;
            __syncthreads();
            for (ishift=0; ishift < nshift; ishift++) {
                index2 = i-index1 * npts_continuous - d_index_shift[index1 * nshift + ishift];
                if (index2 > 0 && index2 < cclen) {
                    atomicAdd(&d_cc[ishift*cclen+index2], cc_temp * d_weight[index1]);
                }
            }
        }
    }
    """
    )
    return mod.get_function("ccMultiShiftWeightedKernel")


def batch_process_cc_shift_multi(
    data_continuous,
    template,
    snr,
    shift_index,
    p_or_s="s",
    padding="valid",
    deviceid=0,
    para=init_phase_window_para(),
):
    """
    Multi-channel cc using gpu.
    *** Cross corrlate high-snr template with continuous data channel by channel,
        and stack the correlation by traveltime curves given in shift_index.
    *** High-snr: (snr > para['snr_threshold'])
    Input:
        data_continuous: [nx, nt_cont]
        template:        [nx, nt_temp]
        snr:             [nx, 1]
        shift_index      [nx, nshift]
        mode:
    Output:
        cc_multi:        [nshift, ncc]

    """
    # initialize device
    cuda.init()
    device = cuda.Device(deviceid)
    context = device.make_context()
    fun_cc_kernel = ccMultiShiftKernel()
    # select high-SNR channel
    if p_or_s == "s":
        isnr = np.squeeze(snr > para["SNR_threshold_S"])
    elif p_or_s == "p":
        isnr = np.squeeze(snr > para["SNR_threshold_P"])
    data_good = data_continuous[isnr, :]
    temp_good = template[isnr, :]
    shft_good = shift_index[isnr, :]
    # size for 2D grids, 1D blocks
    shift_max = np.max(shft_good)
    nt_cont = data_good.shape[1]
    nx_good = data_good.shape[0]
    nt_temp = temp_good.shape[1]
    ncc = int(nt_cont - nt_temp - shift_max)
    nshift = shift_index.shape[1]
    cc_multi = np.zeros((nshift, ncc)).astype(np.float32)
    # processing every nbuffer = 2000 channels
    # nbuffer = 2000
    # nchunk = int(np.ceil(nx_good/nbuffer))
    # nblock_x = 16384
    nchunk = 1
    nbuffer = int(np.ceil(nx_good / nchunk))
    nthread = 1024
    nblock_x = 65535
    ntrace = min(nbuffer, nx_good)
    # convert to 1D array on device
    d_cont_1d = cuda.mem_alloc(ntrace * nt_cont * 4)
    d_temp_1d = cuda.mem_alloc(ntrace * nt_temp * 4)
    d_shft_1d = cuda.mem_alloc(ntrace * nshift * 4)
    d_xcor_1d = gpuarray.to_gpu(np.zeros(ncc * nshift).astype(np.float32))
    itrace = 0
    try:
        # for i in tqdm(range(nchunk), desc="On device: {0} crosss Correlation ...".format(deviceid)):
        for i in range(nchunk):
            ntrace = min(nbuffer, nx_good - itrace)
            nelement = ntrace * nt_cont
            nblock_y = int(np.ceil(nelement / nthread / nblock_x))
            cont_1d = (
                data_good[itrace : itrace + ntrace, :].flatten().astype(np.float32)
            )
            temp_1d = (
                temp_good[itrace : itrace + ntrace, :].flatten().astype(np.float32)
            )
            shft_1d = shft_good[itrace : itrace + ntrace, :].flatten().astype(np.int32)
            cuda.memcpy_htod(d_cont_1d, cont_1d)
            cuda.memcpy_htod(d_temp_1d, temp_1d)
            cuda.memcpy_htod(d_shft_1d, shft_1d)
            fun_cc_kernel(
                d_cont_1d,
                d_temp_1d,
                d_xcor_1d,
                d_shft_1d,
                np.int32(nt_cont),
                np.int32(nt_temp),
                np.int32(ntrace),
                np.int32(nshift),
                np.int32(ncc),
                block=(nthread, 1, 1),
                grid=(nblock_x, nblock_y, 1),
            )
            cc_multi += d_xcor_1d.get().reshape((nshift, ncc))
            d_xcor_1d.fill(0)
            itrace += ntrace
    except:
        d_cont_1d.free()
        d_temp_1d.free()
        d_shft_1d.free()
        d_xcor_1d.gpudata.free()
        context.pop()
        context.detach()
        context = None
        return [], [], [], []
    # finalize device
    d_cont_1d.free()
    d_temp_1d.free()
    d_shft_1d.free()
    d_xcor_1d.gpudata.free()
    context.pop()
    context.detach()
    context = None
    clear_context_caches()
    # return cc_multi/nx_good, 0, nt_cont-nt_temp-shift_max, nx_good
    if padding == "valid":
        icc_beg = int(np.floor(nt_temp / 2))
        icc_end = icc_beg + ncc
    elif padding == "same":
        # padding zeros on the left s.t. the ccmulti start from 0 and end at nt_cont-1
        icc_beg = 0
        icc_end = nt_cont
        ibeg = int(np.floor(nt_temp / 2))
        cc_multi = np.concatenate(
            [
                np.zeros((nshift, ibeg)),
                cc_multi,
                np.zeros((nshift, nt_cont - ibeg - ncc)),
            ],
            axis=1,
        )
    return cc_multi / nx_good, icc_beg, icc_end, nx_good


def batch_process_cc_shift_weighted_multi(
    data_continuous,
    template,
    snr,
    shift_index,
    weight,
    p_or_s="s",
    padding="valid",
    deviceid=0,
    para=init_phase_window_para(),
):
    """
    Weighted Multi-channel cc using gpu.
    *** Cross corrlate high-snr template with continuous data channel by channel,
        and stack the correlation by traveltime curves given in shift_index.
    *** High-snr: (snr > para['snr_threshold'])
    Input:
        data_continuous: [nx, nt_cont]
        template:        [nx, nt_temp]
        snr:             [nx, 1]
        shift_index      [nx, nshift]
        mode:
    Output:
        cc_multi:        [nshift, ncc]

    """
    # initialize device
    cuda.init()
    device = cuda.Device(deviceid)
    context = device.make_context()
    fun_cc_kernel = ccMultiShiftWeightedKernel()
    # select high-SNR channel
    if p_or_s == "s":
        isnr = np.squeeze(snr > para["SNR_threshold_S"])
    elif p_or_s == "p":
        isnr = np.squeeze(snr > para["SNR_threshold_P"])
    data_good = data_continuous[isnr, :]
    temp_good = template[isnr, :]
    shft_good = shift_index[isnr, :]
    weit_good = weight[isnr]
    # size for 2D grids, 1D blocks
    shift_max = np.max(shft_good)
    nt_cont = data_good.shape[1]
    nx_good = data_good.shape[0]
    nt_temp = temp_good.shape[1]
    ncc = int(nt_cont - nt_temp - shift_max)
    nshift = shift_index.shape[1]
    cc_multi = np.zeros((nshift, ncc)).astype(np.float32)
    # processing every nbuffer = 2000 channels
    # nbuffer = 2000
    # nchunk = int(np.ceil(nx_good/nbuffer))
    # nblock_x = 16384
    nchunk = 1
    nbuffer = int(np.ceil(nx_good / nchunk))
    nthread = 1024
    nblock_x = 65535
    ntrace = min(nbuffer, nx_good)
    # convert to 1D array on device
    d_cont_1d = cuda.mem_alloc(ntrace * nt_cont * 4)
    d_temp_1d = cuda.mem_alloc(ntrace * nt_temp * 4)
    d_weit_1d = cuda.mem_alloc(ntrace * 4)
    d_shft_1d = cuda.mem_alloc(ntrace * nshift * 4)
    d_xcor_1d = gpuarray.to_gpu(np.zeros(ncc * nshift).astype(np.float32))
    itrace = 0
    try:
        # for i in tqdm(range(nchunk), desc="On device: {0} crosss Correlation ...".format(deviceid)):
        for i in range(nchunk):
            ntrace = min(nbuffer, nx_good - itrace)
            nelement = ntrace * nt_cont
            nblock_y = int(np.ceil(nelement / nthread / nblock_x))
            cont_1d = (
                data_good[itrace : itrace + ntrace, :].flatten().astype(np.float32)
            )
            temp_1d = (
                temp_good[itrace : itrace + ntrace, :].flatten().astype(np.float32)
            )
            weit_1d = weit_good[itrace : itrace + ntrace].flatten().astype(np.float32)
            shft_1d = shft_good[itrace : itrace + ntrace, :].flatten().astype(np.int32)
            cuda.memcpy_htod(d_cont_1d, cont_1d)
            cuda.memcpy_htod(d_temp_1d, temp_1d)
            cuda.memcpy_htod(d_weit_1d, weit_1d)
            cuda.memcpy_htod(d_shft_1d, shft_1d)
            fun_cc_kernel(
                d_cont_1d,
                d_temp_1d,
                d_xcor_1d,
                d_weit_1d,
                d_shft_1d,
                np.int32(nt_cont),
                np.int32(nt_temp),
                np.int32(ntrace),
                np.int32(nshift),
                np.int32(ncc),
                block=(nthread, 1, 1),
                grid=(nblock_x, nblock_y, 1),
            )
            cc_multi += d_xcor_1d.get().reshape((nshift, ncc))
            d_xcor_1d.fill(0)
            itrace += ntrace
    except:
        d_cont_1d.free()
        d_temp_1d.free()
        d_shft_1d.free()
        d_xcor_1d.gpudata.free()
        context.pop()
        context.detach()
        context = None
        return [], [], [], []
    # finalize device
    d_cont_1d.free()
    d_temp_1d.free()
    d_weit_1d.free()
    d_shft_1d.free()
    d_xcor_1d.gpudata.free()
    context.pop()
    context.detach()
    context = None
    clear_context_caches()
    if padding == "valid":
        icc_beg = int(np.floor(nt_temp / 2))
        icc_end = icc_beg + ncc
    elif padding == "same":
        # padding zeros on the left s.t. the ccmulti start from 0 and end at nt_cont-1
        icc_beg = 0
        icc_end = nt_cont
        ibeg = int(np.floor(nt_temp / 2))
        cc_multi = np.concatenate(
            [
                np.zeros((nshift, ibeg)),
                cc_multi,
                np.zeros((nshift, nt_cont - ibeg - ncc)),
            ],
            axis=1,
        )
    return cc_multi / nx_good, icc_beg, icc_end, nx_good


class ccMultiThread(threading.Thread):
    """
    Thread for calling multi shift kernel
    """

    def __init__(
        self,
        data_cont,
        data_temp,
        snr,
        shift,
        p_or_s="s",
        padding="valid",
        para=init_phase_window_para(),
        deviceID=0,
    ):
        super().__init__()
        self.data_cont = data_cont
        self.data_temp = data_temp
        self.snr = snr
        self.shift = shift
        self.padding = padding
        self.para = para
        self.phase = p_or_s
        self.deviceID = deviceID

    def run(self):
        self.cc_multi, ibeg, iend, ngood = batch_process_cc_shift_multi(
            self.data_cont,
            self.data_temp,
            self.snr,
            self.shift,
            p_or_s=self.phase,
            padding=self.padding,
            deviceid=self.deviceID,
            para=self.para,
        )
        self.cc_info = {}
        self.cc_info["ibeg"] = ibeg
        self.cc_info["iend"] = iend
        self.cc_info["ngood"] = ngood

    def join(self):
        super().join()
        return self.cc_multi, self.cc_info


class ccMultiWeightedThread(threading.Thread):
    """
    Thread for calling multi shift kernel
    """

    def __init__(
        self,
        data_cont,
        data_temp,
        snr,
        shift,
        weight,
        p_or_s="s",
        padding="valid",
        para=init_phase_window_para(),
        deviceID=0,
    ):
        super().__init__()
        self.data_cont = data_cont
        self.data_temp = data_temp
        self.snr = snr
        self.shift = shift
        self.weight = weight
        self.padding = padding
        self.para = para
        self.phase = p_or_s
        self.deviceID = deviceID

    def run(self):
        self.cc_multi, ibeg, iend, ngood = batch_process_cc_shift_weighted_multi(
            self.data_cont,
            self.data_temp,
            self.snr,
            self.shift,
            self.weight,
            p_or_s=self.phase,
            padding=self.padding,
            deviceid=self.deviceID,
            para=self.para,
        )
        self.cc_info = {}
        self.cc_info["ibeg"] = ibeg
        self.cc_info["iend"] = iend
        self.cc_info["ngood"] = ngood

    def join(self):
        super().join()
        return self.cc_multi, self.cc_info
