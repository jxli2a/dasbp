import pickle
import numpy as np
import pandas as pd
from .util import *
import h5py
import os
import dateutil.parser as dateparse
import pytz
from scipy.interpolate import interp1d


def read_das_config(fn_yaml):
    import yaml

    with open(fn_yaml, "r") as fid:
        config = yaml.safe_load(fid)
    for key in config.keys():
        if "fn_dasinfo" in config[key].keys():
            config[key]["dasinfo"] = read_das_info(config[key]["fn_dasinfo"])
    return config


def read_dasdb_filelist(fn_filelist, tbeg=None, tend=None):
    """
    read das database filelist created by 'create_das_filelist'
    """
    # read continuous recording timelist
    df = pd.read_csv(fn_filelist)
    df["begTime"] = pd.to_datetime(df["begTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])
    # default timezone: UTC
    for i in range(len(df)):
        if df.loc[i, "begTime"].tzname() is None:
            df.at[i, "begTime"] = (
                df.loc[i, "begTime"].tz_localize(pytz.UTC).tz_convert(pytz.UTC)
            )
        if df.loc[i, "endTime"].tzname() is None:
            df.at[i, "endTime"] = (
                df.loc[i, "endTime"].tz_localize(pytz.UTC).tz_convert(pytz.UTC)
            )
    if tbeg is None and tend is None:
        return df
    elif tbeg is None:
        tbeg = datetime.min
    elif tend is None:
        tend = datetime.max
    # select df within the [tbeg, tend] range
    df["endTime"] = df["endTime"] - pd.Timedelta(seconds=0.001)
    df["select"] = False
    # for i, row in tqdm(df.iterrows(), desc='Select filelist ...'):
    for i, row in df.iterrows():
        if not (row["begTime"] > tend or row["endTime"] < tbeg):
            df.at[i, "select"] = True
    df = df[df["select"]]
    df.reset_index(inplace=True, drop=True)
    df["endTime"] = df["endTime"] + pd.Timedelta(seconds=0.001)
    return df[["filename", "begTime", "endTime", "fs"]]


def read_catalog(fn, name="short"):
    """
    read catalog csv file
    convert time string to datetime
    add 'dir' field
    """
    df = pd.read_csv(fn)
    map_name_helper(df, map_name_event, name)
    if name == "long":
        time_key = ["event_time", "begin_time", "end_time"]
    elif name == "short":
        time_key = ["time", "tbeg", "tend"]
    for key in time_key:
        if key in df.keys():
            df[key] = pd.to_datetime(df[key])
    df["dir"] = format_catalog_dir(df)
    return df


def read_USGS_FF_PARAM(fn_param):
    df = pd.read_csv(fn_param, header=None, skiprows=10, sep="\s+")
    df.columns = [
        "lat",
        "lon",
        "dep",
        "slip",
        "rak",
        "stk",
        "dip",
        "t_rup",
        "t_ris",
        "t_fal",
        "mo",
    ]
    lonq = df["lon"].values
    latq = df["lat"].values
    depq = df["dep"].values
    slipq = df["slip"].values
    rupq = df["t_rup"].values
    LON = lonq.reshape((20, 15), order="F")
    LAT = latq.reshape((20, 15), order="F")
    SLIP = slipq.reshape((20, 15), order="F") / 100
    DEP = depq.reshape((20, 15), order="F")
    RUP = rupq.reshape((20, 15), order="F")
    return df, LON, LAT, DEP, SLIP, RUP


def read_USGS_MOMENT_RATE(fn_moment):
    df = pd.read_csv(fn_moment, header=None, skiprows=2, sep="\s+")
    df.columns = ["time", "moment_rate"]
    moment_time = df["time"].values
    moment_rate = df["moment_rate"].values
    return moment_time, moment_rate


def read_scardec_stf(fn, timeq=np.arange(-3, 15, 0.01)):
    """"""
    stf = np.loadtxt(fn, dtype="float")
    func = interp1d(stf[:, 0], stf[:, 1], bounds_error=False, fill_value=0)
    stf_average_q = func(timeq)
    # lag from cc
    return timeq - 1.09, stf_average_q


def get_finite_fault_info():
    ff_dict = {
        "ID": "nc73584926",
        "strike": 358,
        "dip": 49,
        "lon": -119.500000,
        "lat": 38.507999,
        "dep": 7.500000,
    }
    return ff_dict


def get_finite_fault_query(fn_param):
    """"""
    _, LON_F, LAT_F, DEP_F, SLIP, _ = read_USGS_FF_PARAM(fn_param)
    # finite fault interpolation dimension
    corner_lon = np.array(
        [LON_F[0, 0], LON_F[0, -1], LON_F[-1, -1], LON_F[-1, 0]])
    corner_lat = np.array(
        [LAT_F[0, 0], LAT_F[0, -1], LAT_F[-1, -1], LAT_F[-1, 0]])
    LON_FQ, LAT_FQ = meshgrid_from_corners(corner_lon, corner_lat, 60, 80)
    LON_FQ = LON_FQ[:, :40][20:, :]
    LAT_FQ = LAT_FQ[:, :40][20:, :]
    DEP_FQ = interp2_griddata_latlon(DEP_F, LON_F, LAT_F, LON_FQ, LAT_FQ)
    SLIP_FQ = interp2_griddata_latlon(SLIP, LON_F, LAT_F, LON_FQ, LAT_FQ)
    FQ_dict = {
        "lon": LON_FQ,
        "lat": LAT_FQ,
        "dep": DEP_FQ,
        "slip": SLIP_FQ,
    }
    return FQ_dict


def read_das_info(fn_das_info, all=False, name="short"):
    # read DAS quality and location info
    df = pd.read_csv(fn_das_info)
    map_name_helper(df, map_name_dasinfo, name)
    if all:
        return df
    else:
        if "status" in df.keys():
            df = df[df["status"] == "good"]
            df = df[~np.isnan(df.lat)]
            df.reset_index(drop=True, inplace=True)
        else:
            df = df[~np.isnan(df.lat)]
        return df


def read_traveltime_table(fn_table):
    # read traveltime table
    df = pd.read_csv(fn_table, header=None, sep="\s+")
    if len(df.columns) == 1:
        df = pd.read_csv(fn_table)
        df.rename(
            columns={"channel_index": "ichan", "P": "tp", "S": "ts"}, inplace=True
        )
    else:
        df.columns = ["ichan", "lon", "lat", "dist", "tp", "ts"]
    return df


def read_traveltime_search_grids(fn_grids):
    """"""
    grid_info = {}
    with h5py.File(fn_grids, "r") as h5:
        TT = h5["tt"][:]
        LON = h5["lon"][:]
        LAT = h5["lat"][:]
        DEP = h5["dep"][:]
        grid_info["grid_lon"] = h5["dim"].attrs["grid_lon"]
        grid_info["grid_lat"] = h5["dim"].attrs["grid_lat"]
        grid_info["grid_dep"] = h5["dim"].attrs["grid_dep"]
        shp = h5["dim"].attrs["shape"]
    return TT, LON, LAT, DEP, shp, grid_info


def write_h5(fn, key_name, data, info):
    """
    write general data to hdf5 file with key name
    """
    with h5py.File(fn, "a") as fid:
        if key_name in fid.keys():
            del fid[key_name]
        fid.create_dataset(key_name, data=data)
        for key, val in info.items():
            fid[key_name].attrs.modify(key, val)


def write_das_window_data_h5(fn, data, info, attrs=None):
    """ """
    # required_keys = ["dx", "dt", "begTime", "endTime", "unit"]
    attrs_dict = {}
    attrs_dict["dx_m"] = info["dx"]
    attrs_dict["dt_s"] = info["dt"]
    attrs_dict["begin_time"] = info["begTime"]
    attrs_dict["end_time"] = info["endTime"]
    attrs_dict["unit"] = info["unit"]
    # optinal_keys: event attributes
    if attrs is not None:
        attrs_event = change_event_keyname(attrs, method="s2l")
        for key in attrs_dict.keys():
            if key in attrs_event.keys():
                # assert attrs_dict[key] == attrs_event[key]
                continue
        # attrs_dict = {**attrs_dict, **attrs_event}
        attrs_dict = {**attrs_event, **attrs_dict}
    # datetime values -> isoformat
    for key, val in attrs_dict.items():
        try:
            valstr = val.isoformat()
            attrs_dict[key] = valstr
        except:
            continue
    # write to hdf5 file
    write_h5(fn, "data", data, attrs_dict)


def read_das_window_data_h5(fn):
    """"""
    if not os.path.exists(fn):
        raise FileExistsError(f"file {fn} does not exist")
    with h5py.File(fn, "r") as fid:
        data = fid["data"][:]
        info = {}
        for key in fid["data"].attrs.keys():
            info[key] = fid["data"].attrs[key]
            try:
                valstr = dateparse.parse(info[key])
                info[key] = valstr
            except:
                continue
    # required_keys = ["dx", "dt", "begTime", "endTime", "unit"]
    info_required = {}
    info_required["dx"] = info["dx_m"]
    info_required["dt"] = info["dt_s"]
    info_required["nt"] = data.shape[1]
    info_required["nx"] = data.shape[0]
    info_required["begTime"] = info["begin_time"]
    info_required["endTime"] = info["end_time"]
    info_required["unit"] = info["unit"]
    # optinal event keys
    info_event = change_event_keyname(info, method="l2s")
    info = {**info_required, **info_event}
    # time_axis
    if "time" in info.keys():
        info["time_axis"] = (
            np.arange(info["nt"]) * info["dt"]
            + (info["begTime"] - info["time"]).total_seconds()
        )
    return data, info


def write_das_eventphase_data_h5(fn, event_dict, data_list, info_list):
    """
    write DAS event phase data to hdf5 file
    """
    # keys for event attributes
    event_attrs_keys = [
        "event_id",
        "event_time",
        "longitude",
        "latitude",
        "depth_km",
        "magnitude",
    ]
    # keys for event phase data attributes
    phase_data_attrs_keys = ["nx", "nt", "dt"]
    # optional keys for event phase datasets
    phase_optional_dataset_keys = ["snr", "shift_index", "traveltime"]
    # change event dict keys if it is in short format
    if "ID" in event_dict.keys():
        event_dict_long = change_event_keyname(event_dict, method="s2l")
    elif "event_id" in event_dict.keys():
        event_dict_long = event_dict
    if not isinstance(event_dict_long["event_time"], str):
        event_dict_long["event_time"] = event_dict_long["event_time"].isoformat()
    with h5py.File(fn, "a") as fid:
        # group: phase data
        if "data" in fid.keys():
            g_phase_data = fid["data"]
        else:
            g_phase_data = fid.create_group("data")
        # group.attrs: event dict
        for key in event_attrs_keys:
            g_phase_data.attrs.modify(key, event_dict_long[key])
        # group.group: phase group
        if isinstance(data_list, np.ndarray):
            data_list2 = [data_list]
            info_list2 = [info_list]
        elif isinstance(data_list, list):
            data_list2 = data_list
            info_list2 = info_list
        for data, info in zip(data_list2, info_list2):
            # this phase group:
            phase_name = info["phase_name"]
            if phase_name in g_phase_data.keys():
                g_phase = g_phase_data[phase_name]
            else:
                g_phase = g_phase_data.create_group(phase_name)
            # group.group.dataset: phase datasets
            # dataset: phase window
            if "data" in g_phase.keys() and g_phase["data"].shape == data.shape:
                d_phase = g_phase["data"]
                d_phase[:] = data.astype(np.float32)
            else:
                if "data" in g_phase.keys():
                    del g_phase["data"]
                d_phase = g_phase.create_dataset(
                    "data", data=data.astype(np.float32))
            for key in phase_data_attrs_keys:
                d_phase.attrs.modify(key, info[key])
            # dataset: phase optional dataset
            for key in phase_optional_dataset_keys:
                if not key in info.keys():
                    continue
                if key in g_phase.keys() and g_phase[key].shape == info[key].shape:
                    d_phase = g_phase[key]
                    d_phase[:] = info[key]
                else:
                    if key in g_phase.keys():
                        del g_phase[key]
                    d_phase = g_phase.create_dataset(key, data=info[key])
                if key == "traveltime":
                    if "tref" in info.keys():
                        d_phase.attrs.modify("tref", info["tref"])


def read_das_eventphase_data_h5(
    fn, phase=None, event=False, dataset_keys=None, attrs_only=False
):
    """
    read event phase data from hdf5 file
    """
    if isinstance(phase, str):
        phase = [phase]
    data_list = []
    info_list = []
    with h5py.File(fn, "r") as fid:
        g_phases = fid["data"]
        phase_avail = g_phases.keys()
        if phase is None:
            phase = list(phase_avail)
        for phase_name in phase:
            if not phase_name in g_phases.keys():
                raise (f"{fn} does not have phase: {phase_name}")
            g_phase = g_phases[phase_name]
            if attrs_only:
                data = []
            else:
                data = g_phase["data"][:]
            info = {}
            for key in g_phase["data"].attrs.keys():
                info[key] = g_phases[phase_name]["data"].attrs[key]
            if dataset_keys is not None:
                for key in dataset_keys:
                    if key in g_phase.keys():
                        info[key] = g_phase[key][:]
                        for kk in g_phase[key].attrs.keys():
                            info[kk] = g_phase[key].attrs[kk]
            data_list.append(data)
            info_list.append(info)
        if event:
            event_dict = dict(
                (key, fid["data"].attrs[key]) for key in fid["data"].attrs.keys()
            )
            event_dict = change_event_keyname(event_dict, method="l2s")
            info_list[0]["event"] = event_dict
    return data_list, info_list


def save_bp_location_result(
    cc_P, cc_P_info, cc_S, cc_S_info, grid_info, fn_save_info, fn_save_data
):
    # save backprojection result
    info = {"cc_P_info": cc_P_info,
            "cc_S_info": cc_S_info, "grid_info": grid_info}
    data = {"cc_P": cc_P, "cc_S": cc_S}
    with open(fn_save_info, "wb") as fid:
        pickle.dump(info, fid)
    with open(fn_save_data, "wb") as fid:
        pickle.dump(data, fid)


def load_bp_location_result(fn_save_data, fn_save_info):
    # load cc_PS and info
    with open(fn_save_info, "rb") as fid:
        info_load = pickle.load(fid)
    with open(fn_save_data, "rb") as fid:
        dat_load = pickle.load(fid)
    return (
        dat_load["cc_P"],
        info_load["cc_P_info"],
        dat_load["cc_S"],
        info_load["cc_S_info"],
        info_load["grid_info"],
    )
