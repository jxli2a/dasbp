import numpy as np
from .util import *
from .kernel import ccMultiWeightedThread, ccMultiThread
from .reader import read_traveltime_search_grids


def get_weights(cc_south_P_info, cc_south_S_info, cc_north_P_info, cc_north_S_info):
    ngood_south_P = cc_south_P_info["ngood"]
    ngood_south_S = cc_south_S_info["ngood"]
    ngood_north_P = cc_north_P_info["ngood"]
    ngood_north_S = cc_north_S_info["ngood"]
    ngood_total = ngood_south_P + ngood_south_S + ngood_north_P + ngood_north_S
    weight_south_P = ngood_south_P / ngood_total
    weight_south_S = ngood_south_S / ngood_total
    weight_north_P = ngood_north_P / ngood_total
    weight_north_S = ngood_north_S / ngood_total
    return weight_south_P, weight_south_S, weight_north_P, weight_north_S


def merge_bp_image_dict_two_system(bp_image_dict):
    """"""
    w_south_P, w_south_S, w_north_P, w_north_S = get_weights(
        bp_image_dict["south"]["ccP_info"],
        bp_image_dict["south"]["ccS_info"],
        bp_image_dict["north"]["ccP_info"],
        bp_image_dict["north"]["ccS_info"],
    )

    bp_image_merged = {
        "ccPS": bp_image_dict["south"]["ccP"] * w_south_P
        + bp_image_dict["south"]["ccS"] * w_south_S
        + bp_image_dict["north"]["ccP"] * w_north_P
        + bp_image_dict["north"]["ccS"] * w_north_S,
        "ccP": (
            bp_image_dict["south"]["ccP"] * w_south_P
            + bp_image_dict["north"]["ccP"] * w_north_P
        )
        / (w_south_P + w_north_P),
        "ccS": (
            bp_image_dict["south"]["ccS"] * w_south_S
            + bp_image_dict["north"]["ccS"] * w_north_S
        )
        / (w_south_S + w_north_S),
    }

    return bp_image_merged


def get_egf_window(
    event_data_raw,
    event_data_info,
    tt_table,
    p_or_s="s",
    tref=None,
    para=init_phase_window_para(),
):
    """
    Get the phase window and calculate the SNR value
    time_axis must have 0 anchored @ event origin time
    event_data_raw:  raw data extracted from h5 files, include all channels (good+bad)
    event_data_info: raw data header info
    tt_table:        channel info table with traveltimes
    tref:            shift index reference time, default=min(TS)
    """
    # parameters
    dt = event_data_info["dt"]
    t0 = event_data_info["time_axis"][0]
    snr_signal_beg = para["SNR_signal_beg"]
    snr_noise_win = para["SNR_noise_win"]
    if p_or_s == "s":
        travel_time_phase = tt_table["ts"].values
        phase_win = para["winS"]
    elif p_or_s == "p":
        travel_time_phase = tt_table["tp"].values
        phase_win = para["winP"]
    # processing
    # event_data = np.copy(event_data_raw[tt_table['ichan'], :])
    event_data = np.copy(event_data_raw)
    event_data = event_data[tt_table["ichan"], :]
    # select phase window & calculate SNR
    nwin = int(np.floor(phase_win / dt))
    nnos = int(np.floor(snr_noise_win / dt))
    # use only good channels
    nchan = event_data.shape[0]
    egf = np.zeros((nchan, nwin))
    snr = np.zeros((nchan, 1))
    if tref is None:
        shift_index = np.round(
            (travel_time_phase - np.min(travel_time_phase)) / dt
        ).astype(int)
    else:
        shift_index = np.round((travel_time_phase - tref) / dt).astype(int)
    for ichan in range(nchan):
        tphase = travel_time_phase[ichan]
        isig_beg = int(np.floor((tphase + snr_signal_beg - t0) / dt))
        inos_beg = int(np.floor((-snr_noise_win - t0) / dt))
        iph_beg = int(np.floor((tphase - phase_win / 2 - t0) / dt))
        esig = np.mean(np.square(event_data[ichan][isig_beg : isig_beg + nwin]))
        enos = np.mean(np.square(event_data[ichan][inos_beg : inos_beg + nnos]))
        if np.isclose(enos, 0):
            snr[ichan] = -1000
        else:
            snr[ichan] = 10 * np.log10(esig / enos)
        try:
            egf[ichan][:] = event_data[ichan][iph_beg : iph_beg + nwin]
        except:
            print(
                "Warning: ichan={0}, iph_beg={1}, nwin={2}".format(
                    ichan, iph_beg, nwin
                )
            )
    return egf, snr, shift_index


def backproject_cc_PS(
    data_cont,
    data_cont_info,
    egf_P,
    egf_S,
    egf_P_info,
    egf_S_info,
    padding="same",
    para=init_phase_window_para(),
    tref=None,
    tref_array=None,
):
    """
    Back-project correlogram for both P&S phase to BP grids
    s.t. the CC's are coherently stacked across channels
    Input:
        data_cont: continuous data, shape=[nchan, ntime];
                   should only include good channels
        data_cont_info: header info for continuous data
        egf_P:    egf of P phase, shape=[nchan, ntime]
        egf_S:    egf of S phase, shape=[nchan, ntime]
        egf_P_info:     P phase info: {'snr', 'fn_grids'}
        egf_S_info:     S phase info: {'snr', 'fn_grids'}
        padding:   'same', the P and S CC are padded with zeros s.t. they have the same # of time samples as data_cont
    Output:
        [P, S, PS]
        cc:        stacked CC time series for different search grids, shape=[nchan, ncc]
        cc_info:   stacked CC header info; include start and end index
        grid_info: grid search header info
    """
    fn_grids_P = egf_P_info["fn_grids"]
    fn_grids_S = egf_S_info["fn_grids"]
    dt = data_cont_info["dt"]
    # read traveltime search grids and convert to shift index
    TP, LON, LAT, DEP, shp, center = read_traveltime_search_grids(fn_grids_P)
    TS, _, _, _, _, _ = read_traveltime_search_grids(fn_grids_S)
    # select part of the stations/channels
    if "ista" in egf_P_info.keys():
        TP = TP[:, egf_P_info["ista"]]
    if "ista" in egf_S_info.keys():
        TS = TS[:, egf_S_info["ista"]]
    grid_info = {}
    grid_info["LON"] = LON
    grid_info["LAT"] = LAT
    grid_info["DEP"] = DEP
    grid_info["shp"] = shp
    grid_info["center"] = center
    # ngrid = TP.shape[0]
    if tref is None:
        tref = np.min(TP)
    shift_multi_P = get_shift_index(TP, tref, dt)
    shift_multi_S = get_shift_index(TS, tref, dt)
    # grid search travel times for stacking maximum CC
    list_threads = []
    if "weight" in egf_P_info.keys():
        list_threads.append(
            ccMultiWeightedThread(
                data_cont,
                egf_P,
                egf_P_info["snr"],
                shift_multi_P.T,
                egf_P_info["weight"],
                p_or_s="p",
                padding=padding,
                para=para,
                deviceID=0,
            )
        )
    else:
        list_threads.append(
            ccMultiThread(
                data_cont,
                egf_P,
                egf_P_info["snr"],
                shift_multi_P.T,
                p_or_s="p",
                padding=padding,
                para=para,
                deviceID=0,
            )
        )
    if "weight" in egf_S_info.keys():
        list_threads.append(
            ccMultiWeightedThread(
                data_cont,
                egf_S,
                egf_S_info["snr"],
                shift_multi_S.T,
                egf_S_info["weight"],
                p_or_s="s",
                padding=padding,
                para=para,
                deviceID=1,
            )
        )
    else:
        list_threads.append(
            ccMultiThread(
                data_cont,
                egf_S,
                egf_S_info["snr"],
                shift_multi_S.T,
                p_or_s="s",
                padding=padding,
                para=para,
                deviceID=1,
            )
        )
    for thread in list_threads:
        try:
            thread.start()
        except:
            return
    cc_P, cc_P_info = list_threads[0].join()
    cc_S, cc_S_info = list_threads[1].join()
    #
    print("BP done, start postprocessing")
    weight_P = cc_P_info["ngood"] / (cc_P_info["ngood"] + cc_S_info["ngood"])
    weight_S = cc_P_info["ngood"] / (cc_P_info["ngood"] + cc_S_info["ngood"])
    cc_PS = cc_P * weight_P + cc_S * weight_S
    ibeg = 0
    iend = cc_PS.shape[1]
    cc_PS_info = {}
    cc_PS_info["ibeg"] = ibeg
    cc_PS_info["iend"] = iend
    # align the cc from all different search grids by shifting tt[0]-tref
    ista = 0
    if tref_array is None:
        ishft = np.round((TP[:, ista] - tref) / dt).astype(int)
    else:
        ishft = np.round((tref_array - tref) / dt).astype(int)
    for i, shft in enumerate(ishft):
        cc_PS[i, :] = np.roll(cc_PS[i, :], shft)
        cc_P[i, :] = np.roll(cc_P[i, :], shft)
        cc_S[i, :] = np.roll(cc_S[i, :], shft)
    return cc_P, cc_P_info, cc_S, cc_S_info, cc_PS, cc_PS_info, grid_info
