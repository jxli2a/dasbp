import errno
import os
import numpy as np
import pandas as pd
import utm
from datetime import datetime, timedelta
from scipy.interpolate import RBFInterpolator, griddata


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def map_name_event(method="s2l"):
    """
    Mapping name between short and long convention for event dict
    """
    map_name = {
        "ID": "event_id",
        "time": "event_time",
        "lon": "longitude",
        "lat": "latitude",
        "dep": "depth_km",
        "mag": "magnitude",
        "magtype": "magnitude_type",
        "tbeg": "begin_time",
        "tend": "end_time",
        "tref": "time_reference",
        "tbef": "time_before",
        "taft": "time_after",
        "source": "source",
    }
    if method == "s2l":
        pass
    elif method == "l2s":
        map_name = dict((v, k) for k, v in map_name.items())
    return map_name


def map_name_dasinfo(method="s2l"):
    """
    Mapping name between short and long convention for dasinfo file
    """
    map_name = {
        "ichan": "index",
        "status": "status",
        "lat": "latitude",
        "lon": "longitude",
        "ele": "elevation_m",
        "azi": "azimuth",
        "dip": "dipping angle",
    }
    if method == "s2l":
        pass
    elif method == "l2s":
        map_name = dict((v, k) for k, v in map_name.items())
    return map_name


def map_name_helper(df, func_mapper, long_or_short):
    if long_or_short == "long":
        map_name_dict = func_mapper(method="s2l")
    elif long_or_short == "short":
        map_name_dict = func_mapper(method="l2s")
    else:
        raise KeyError(f"Invalid {long_or_short}")
    # only check names that are different in long and short convention
    keys_src = {()}
    keys_des = {()}
    for key, val in map_name_dict.items():
        if key != val:
            keys_src.add(key)
            keys_des.add(val)
    # check whether the df need to be renamed given map_name dict base on df.keys()
    keys = df.keys()
    key_in_src = np.any(np.array([key in keys_src for key in keys]))
    key_in_des = np.any(np.array([key in keys_des for key in keys]))
    if key_in_src and (not key_in_des):
        df.rename(columns=map_name_dict, inplace=True)
    elif (not key_in_src) and key_in_des:
        pass
    else:
        raise RuntimeError(
            f"map_name_helper.py: The input dataframe has both long and short names"
        )


def format_catalog_dir(df):
    if isinstance(df, pd.DataFrame):
        tstr = df["time"].apply(lambda x: datetime.strftime(x, "%Y%m%d_%H%M"))
        if "mag" in df.keys():
            mstr = df["mag"].apply(lambda x: "_M{0:.1f}_".format(x))
        else:
            mstr = "_"
        return tstr + mstr + df["ID"].astype(str)
    elif isinstance(df, dict):
        tstr = datetime.strftime(df["time"], "%Y%m%d_%H%M")
        if "mag" in df.keys():
            mag = df["mag"]
            mstr = f"_M{mag:.1f}_"
        else:
            mstr = "_"
        return tstr + mstr + str(df["ID"])


def change_keyname(dict_in, map_name):
    """
    change keyname given map dict
    """
    dict_out = dict_in.copy()
    map_keys = map_name.keys()
    evt_keys = list(dict_out.keys())
    for k in evt_keys:
        if k in map_keys:
            dict_out[map_name[k]] = dict_out.pop(k)
    return dict_out


def change_event_keyname(event, method="s2l"):
    """
    s2l: short to long
    l2s: long to short
    """
    map_name = map_name_event(method=method)
    event_dict = change_keyname(event, map_name)
    return event_dict


def init_phase_window_para():
    para = {}
    para["tbef"] = 5
    para["taft"] = 10
    para["winS"] = 4
    para["winP"] = 3
    # para["SNR_threshold_S"] = 10  # dB
    # para["SNR_threshold_P"] = 5
    para["SNR_threshold_S"] = 8  # dB
    para["SNR_threshold_P"] = 4
    para["SNR_signal_beg"] = -1  # t sec before the predicted S arrival
    para["SNR_noise_win"] = 4  # t sec before event origin time
    return para


def prepare_catalog_window(catalog, time_before=30, time_after=90):
    """"""
    # default window
    catalog["tref"] = 0
    catalog["tbef"] = time_before
    catalog["taft"] = time_after
    # round up window to 0.01 sec
    catalog["tbeg"] = catalog["time"] + timedelta(seconds=1.0) * np.around(
        (catalog["tref"] - catalog["tbef"]), decimals=3
    )
    catalog["tend"] = catalog["time"] + timedelta(seconds=1.0) * np.around(
        (catalog["tref"] + catalog["taft"]), decimals=3
    )
    return catalog


def get_shift_index(TT, t_ref, dt):
    """
    get cc shift index
    Input:
    TS:      traveltime matrix [ngrid, nchannel]
    t_ref:   traveltime curve reference time point [1]
    dt:      sampling time interval
    Output:
    shift_index: [ngrid, nchannel]
    """
    shift_index = np.round((TT - t_ref) / dt).astype(int)
    return shift_index


def mask_contour_2d(dat, ix, iy, mask_threshold):
    """
    Mask 2D data outside region centered @ [iy, ix] with value less than mask_threshold
    """
    ny, nx = dat.shape
    list_ix = []
    list_iy = []
    mask = -np.ones_like(dat)
    list_ix.append(ix)
    list_iy.append(iy)
    while len(list_ix) > 0:
        # pop center
        ic_x = list_ix.pop(-1)
        ic_y = list_iy.pop(-1)
        if dat[ic_y, ic_x] > mask_threshold:
            mask[ic_y, ic_x] = 0
        else:
            mask[ic_y, ic_x] = 0
            continue
        # add adjacent
        # top
        ia_x = ic_x + 0
        ia_y = ic_y + 1
        if ia_y < ny and mask[ia_y, ia_x] == -1:
            mask[ia_y, ia_x] = 1
            list_ix.append(ia_x)
            list_iy.append(ia_y)
        # bottom
        ia_x = ic_x + 0
        ia_y = ic_y - 1
        if ia_y > 0 and mask[ia_y, ia_x] == -1:
            mask[ia_y, ia_x] = 1
            list_ix.append(ia_x)
            list_iy.append(ia_y)
        # left
        ia_x = ic_x - 1
        ia_y = ic_y + 0
        if ia_x > 0 and mask[ia_y, ia_x] == -1:
            mask[ia_y, ia_x] = 1
            list_ix.append(ia_x)
            list_iy.append(ia_y)
        # right
        ia_x = ic_x + 1
        ia_y = ic_y + 0
        if ia_x < nx and mask[ia_y, ia_x] == -1:
            mask[ia_y, ia_x] = 1
            list_ix.append(ia_x)
            list_iy.append(ia_y)
    mask[mask == -1] = 1
    return mask


def meshgrid_from_corners(corner_x, corner_y, nx, ny):
    """
    Given corners (rotated rectangle), return the meshed grids
    corner_x, corner_y: x, y location array of 4 corners
          4 --- 3
      ny  |     |
          1 --- 2
            nx
    """
    X = np.zeros([ny, nx])
    Y = np.zeros([ny, nx])
    line_l_y = np.linspace(corner_y[0], corner_y[-1], ny)
    line_l_x = np.linspace(corner_x[0], corner_x[-1], ny)
    line_r_y = np.linspace(corner_y[1], corner_y[2], ny)
    line_r_x = np.linspace(corner_x[1], corner_x[2], ny)
    for i in range(ny):
        X[i, :] = np.linspace(line_l_x[i], line_r_x[i], nx)
        Y[i, :] = np.linspace(line_l_y[i], line_r_y[i], nx)
    return X, Y


def interp2_griddata_latlon(datb, Xb, Yb, Xq, Yq):
    """
    Use scipy.interpolate.gridded to interpolate 2d surface
    datb.shape = [ny, nx]
    y: lat, x: lon
    """
    points_b = np.vstack([Yb.flatten(), Xb.flatten()]).T
    datq = griddata(points_b, datb.flatten(), (Yq, Xq), method="linear")
    return datq


def interp3_rbf_latlondep(datb, xb, yb, zb, xq, yq, zq, kernel="linear", scatter=False):
    """
    Use scipy.interpolate.RBFInterpolator to interpolate 3d volume
    datb.shape = [ny, nx, nz]
    y: lat, x: lon, z: dep [km]
    """
    Xb, Yb, Zb = np.meshgrid(xb, yb, zb)
    if not scatter:
        Xq, Yq, Zq = np.meshgrid(xq, yq, zq)
        shp = [len(yq), len(xq), len(zq)]
    else:
        Xq = xq.copy()
        Yq = yq.copy()
        Zq = zq.copy()
        shp = Xq.shape
    # change to utm coordinate
    utm0 = utm.from_latlon(np.mean(yb), np.mean(xb))
    utm_xy_b = utm.from_latlon(Yb, Xb, utm0[2], utm0[3])
    Xb = (utm_xy_b[0] - utm0[0]) / 1000
    Yb = (utm_xy_b[1] - utm0[0]) / 1000
    utm_xy_q = utm.from_latlon(Yq, Xq, utm0[2], utm0[3])
    Xq = (utm_xy_q[0] - utm0[0]) / 1000
    Yq = (utm_xy_q[1] - utm0[0]) / 1000
    # base and query points
    points_b = np.vstack([Yb.flatten(), Xb.flatten(), Zb.flatten()]).T
    points_q = np.vstack([Yq.flatten(), Xq.flatten(), Zq.flatten()]).T
    datq = RBFInterpolator(points_b, datb.flatten(), kernel=kernel)(points_q).reshape(
        shp
    )
    return datq


def interp_vol3(vol, grid_b, grid_q):
    """"""
    lonq = grid_q["lon"].flatten()
    latq = grid_q["lat"].flatten()
    depq = grid_q["dep"].flatten()
    volq = interp3_rbf_latlondep(
        vol, grid_b["xx"], grid_b["yy"], grid_b["zz"], lonq, latq, depq, scatter=True
    )
    return volq.reshape(grid_q["lon"].shape)


def get_grid_cc(cmat, grid_b, lon, lat, dep, nxwin=1, nywin=1, nzwin=1):
    """"""
    # maximum on closest grid
    ig_y = np.argmin(np.abs(grid_b["yy"] - lat))
    ig_x = np.argmin(np.abs(grid_b["xx"] - lon))
    ig_z = np.argmin(np.abs(grid_b["zz"] - dep))
    ny = len(grid_b["yy"])
    nx = len(grid_b["xx"])
    nz = len(grid_b["zz"])
    # print(ig_x, ig_y, ig_z)
    ib_y = max(0, ig_y - nywin)
    ie_y = min(ny, ig_y + nywin + 1)
    ib_x = max(0, ig_x - nxwin)
    ie_x = min(nx, ig_x + nxwin + 1)
    ib_z = max(0, ig_z - nzwin)
    ie_z = min(nz, ig_z + nzwin + 1)
    cg = cmat[ib_y:ie_y, :, :][:, ib_x:ie_x, :][:, :, ib_z:ie_z]
    ig_max = np.unravel_index(np.argmax(cg), cg.shape)
    ig_y += ig_max[0] - nywin
    ig_x += ig_max[1] - nxwin
    ig_z += ig_max[2] - nzwin
    ig_g = nz * nx * ig_y + nz * ig_x + ig_z
    return cg, ig_x, ig_y, ig_z, ig_g


def get_joint_location_time(cc_gt, ig_t, grid_b, grid_q, mask_threshold=0.6, ntwin=20):
    """"""
    shp = (len(grid_b["yy"]), len(grid_b["xx"]), len(grid_b["zz"]))
    cc_time_axis = grid_b["tt"]
    Tref_mat = grid_b["tref"].reshape(shp)
    cmat = (np.max(cc_gt[:, ig_t - ntwin : ig_t + ntwin], axis=1)).reshape(shp)
    tmat = (
        cc_time_axis[
            (np.argmax(cc_gt[:, ig_t - ntwin : ig_t + ntwin], axis=1)).reshape(shp)
            + ig_t
            - ntwin
        ]
        - Tref_mat
    )
    # interp on slice
    cmatq = interp_vol3(cmat, grid_b, grid_q)
    tmatq = interp_vol3(tmat, grid_b, grid_q)
    imax = np.where(cmatq == np.max(cmatq))
    ix = imax[1][0]
    iy = imax[0][0]
    lon = grid_q["lon"][iy, ix]
    lat = grid_q["lat"][iy, ix]
    dep = grid_q["dep"][iy, ix]
    time = tmatq[iy, ix]
    cg, ig_x, ig_y, ig_z, ig_g = get_grid_cc(
        cmat, grid_b, lon, lat, dep, nxwin=1, nywin=1, nzwin=1
    )
    ig_t = np.argmax(cc_gt[ig_g, ig_t - ntwin : ig_t + ntwin]) + ig_t - ntwin
    # peak mask
    maskq = mask_contour_2d(cmatq, ix, iy, np.max(cmatq) * mask_threshold)
    return {
        "cmat": cmat,
        "lon": lon,
        "lat": lat,
        "dep": dep,
        "time": time,
        "lonq": grid_q["lon"],
        "latq": grid_q["lat"],
        "depq": grid_q["dep"],
        "cmatq": cmatq,
        "tmatq": tmatq,
        "maskq": maskq,
        "ig": ig_g,
        "ix": ig_x,
        "iy": ig_y,
        "iz": ig_z,
        "it": ig_t,
        "lon_g": grid_b["xx"][ig_x],
        "lat_g": grid_b["yy"][ig_y],
        "dep_g": grid_b["zz"][ig_z],
        "time_g": grid_b["tt"][ig_t] - grid_b["tref"][ig_g],
    }
