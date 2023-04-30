import numpy as np
import pandas as pd

import sys
import os

path_util = os.path.split(os.path.abspath(__file__))[0]
path_dasutil = os.path.join(path_util, "DAS-utilities")
pyDAS_path = os.path.join(path_dasutil, "build")
pyDAS_python = os.path.join(path_dasutil, "python")

os.environ["LD_LIBRARY_PATH"] += ":" + pyDAS_path
sys.path.insert(0, pyDAS_path)
sys.path.insert(0, pyDAS_python)
import DASutils


from .reader import *
from .util import *
from .bp_util import backproject_cc_PS, merge_bp_image_dict_two_system, get_egf_window


def run_preprocess_workflow(
    event_dict, egf_dict, freqmin, freqmax, path_processed_bp, config
):
    """"""
    mkdir(path_processed_bp)
    systems = config.keys()
    catalog = pd.DataFrame([event_dict, egf_dict])
    event_id = event_dict["ID"]
    egf_id = egf_dict["ID"]

    para = init_phase_window_para()
    # preprocess event data from raw data
    for system in systems:
        for i, event in catalog.iterrows():
            event_dict = event.to_dict()
            event_id = event_dict["ID"]
            filelist = read_dasdb_filelist(
                config[system]["fn_timelist"], event_dict["tbeg"], event_dict["tend"]
            )
            filelist = [
                os.path.join(config[system]["path_continuous_data"], f)
                for f in list(filelist["filename"])
            ]
            data, info = DASutils.readFile_HDF(
                filelist,
                freqmin,
                freqmax,
                desampling=False,
                taper=0.1,
                nChbuffer=5000,
                minTime=event_dict["tbeg"],
                maxTime=event_dict["tend"],
                system="OptaSense",
                zerophase=True,
                order=14,
            )
            info["unit"] = "microstrain"
            # event data
            write_das_window_data_h5(
                f"{path_processed_bp}/{event_id}_{system}.h5",
                data,
                info,
                attrs=event_dict,
            )

    # EGF from aftershock
    table = read_traveltime_table(
        f"{config['north']['path_traveltime']}/{egf_id}_north.table"
    )
    tref = np.min(table["tp"])

    for system in systems:
        data, info = read_das_window_data_h5(
            f"{path_processed_bp}/{egf_id}_{system}.h5"
        )
        table = read_traveltime_table(
            f"{config[system]['path_traveltime']}/{egf_id}_{system}.table"
        )
        tref = np.min(table["tp"])
        egf_data_P, snr, shift = get_egf_window(
            data * 1e6,
            info,
            table,
            tref=tref,
            p_or_s="p",
            para=para,
        )
        egf_data_P /= 1e6
        egf_info_P = {
            "phase_name": "P",
            "snr": snr,
            "shift_index": shift,
            "traveltime": table["tp"],
            "tref": tref,
            "nx": egf_data_P.shape[0],
            "nt": egf_data_P.shape[1],
            "dt": info["dt"],
            "fn_grids": config[system]["fn_P_grids"],
        }
        egf_data_S, snr, shift = get_egf_window(
            data * 1e6,
            info,
            table,
            tref=tref,
            p_or_s="s",
            para=para,
        )
        egf_data_S /= 1e6
        egf_info_S = {
            "phase_name": "S",
            "snr": snr,
            "shift_index": shift,
            "traveltime": table["ts"],
            "tref": tref,
            "nx": egf_data_S.shape[0],
            "nt": egf_data_S.shape[1],
            "dt": info["dt"],
            "fn_grids": config[system]["fn_S_grids"],
        }
        fn_egf = f"{path_processed_bp}/{egf_id}_{system}_egf.h5"
        write_das_eventphase_data_h5(
            fn_egf, event_dict, [egf_data_P, egf_data_S], [egf_info_P, egf_info_S]
        )


def run_bp_workflow(event_id, egf_id, path_processed_bp, config):
    systems = config.keys()
    bp_data_dict = {"north": {}, "south": {}}
    # -- load data and egf
    for system in systems:
        data, info = read_das_window_data_h5(
            f"{path_processed_bp}/{event_id}_{system}.h5"
        )
        egf_data, egf_info = read_das_eventphase_data_h5(
            f"{path_processed_bp}/{egf_id}_{system}_egf.h5",
            dataset_keys=["snr", "traveltime", "shift_index"],
        )
        # bp grids
        egf_info[0]["fn_grids"] = config[system]["fn_P_grids"]
        egf_info[1]["fn_grids"] = config[system]["fn_S_grids"]
        # taptest channels
        ichan_taptest = config[system]["dasinfo"].ichan.values
        data = data[ichan_taptest, :]
        info["nx"] = data.shape[0]
        bp_data_dict[system]["data"] = data
        bp_data_dict[system]["info"] = info
        bp_data_dict[system]["egf_data"] = egf_data
        bp_data_dict[system]["egf_info"] = egf_info

    # bp grids:
    TP, LON, LAT, DEP, shp, center = read_traveltime_search_grids(
        config["south"]["fn_P_grids"]
    )
    tref_array = TP[:, 0]
    tref = np.min(tref_array)
    ibeg = 4000
    iend = 6000
    cc_time_axis = bp_data_dict["south"]["info"]["time_axis"][ibeg:iend]
    para = init_phase_window_para()

    # bp
    bp_image_dict = {"north": {}, "south": {}}
    for i, system in enumerate(systems):
        (
            cc_P,
            cc_P_info,
            cc_S,
            cc_S_info,
            cc_PS,
            cc_PS_info,
            _,
        ) = backproject_cc_PS(
            bp_data_dict[system]["data"],
            bp_data_dict[system]["info"],
            bp_data_dict[system]["egf_data"][0],
            bp_data_dict[system]["egf_data"][1],
            bp_data_dict[system]["egf_info"][0],
            bp_data_dict[system]["egf_info"][1],
            para=para,
            tref=tref,
            tref_array=tref_array,
        )
        bp_image_dict[system]["ccP"] = cc_P[:, ibeg:iend]
        bp_image_dict[system]["ccP_info"] = cc_P_info
        bp_image_dict[system]["ccS"] = cc_S[:, ibeg:iend]
        bp_image_dict[system]["ccS_info"] = cc_S_info
        bp_image_dict[system]["ccPS"] = cc_PS[:, ibeg:iend]
        bp_image_dict[system]["ccPS_info"] = cc_PS_info

    # merge two system
    bp_image = merge_bp_image_dict_two_system(bp_image_dict)
    bp_image["grid_info"] = {
        "xx": center["grid_lon"],
        "yy": center["grid_lat"],
        "zz": center["grid_dep"],
        "tt": cc_time_axis,
        "tref": tref_array,
    }
    return bp_data_dict, bp_image_dict, bp_image
