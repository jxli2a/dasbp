import numpy as np
import numpy.ma as ma
import utm
import matplotlib.pyplot as plt
from matplotlib import patheffects
from .reader import read_USGS_MOMENT_RATE, read_scardec_stf


def init_publication_rcParams():
    """
    Set up true font rcParams for publication
    """
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.transparent"] = True
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    # other default parameters
    _fontsize = 16
    params = {
        'image.interpolation': 'nearest',
        'savefig.dpi': 300,  # to adjust notebook inline plot size
        'axes.labelsize': _fontsize,  # fontsize for x and y labels (was 10)
        'axes.titlesize': _fontsize,
        'font.size': _fontsize,
        'legend.fontsize': _fontsize,
        'xtick.labelsize': _fontsize,
        'ytick.labelsize': _fontsize,
        'text.usetex': False,
        'font.size': _fontsize,
    }
    matplotlib.rcParams.update(params)


init_publication_rcParams()


def imshow_two_system(
    datN,
    datS,
    infoN,
    infoS,
    perc=95,
    figsize=(14, 6),
    cmap=plt.get_cmap("seismic"),
    title=None,
    grid=True,
):
    if perc is not None:
        clipVal = np.percentile(
            np.abs(np.concatenate([datN.T, datS.T], axis=1)), perc)
        vmin = -clipVal
        vmax = clipVal
    else:
        vmin = np.min([np.min(datN), np.min(datS)])
        vmax = np.min([np.max(datN), np.max(datS)])
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
    tt_N = infoN["time_axis"]
    tt_S = infoS["time_axis"]
    ax[0].imshow(
        np.flip(datN.T, axis=1),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=[datN.shape[0], 0, tt_N[-1], tt_N[0]],
    )
    ax[0].set_xlabel("Channel number")
    ax[0].set_ylabel("Time (sec)")
    ax[0].set_title("North system")
    ax[0].grid(grid)
    ax[1].imshow(
        datS.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=[0, datS.shape[0], tt_S[-1], tt_S[0]],
    )
    ax[1].set_xlabel("Channel number")
    ax[1].set_title("South system")
    ax[1].grid(grid)
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        plt.suptitle(title)
    return fig, ax


def show_volume_slice(
    vol,
    xx=None,
    yy=None,
    zz=None,
    center="max",
    cmap=None,
    cb_label="CC",
    figsize=(10, 10),
    vmin=None,
    vmax=None,
):
    """
    show three orthorgonal slices of a volume
    vol.shape = [ny, nx, nz]
    """
    if vmax is None:
        vmax = np.max(vol)
    if vmin is None:
        vmin = np.min(vol)
    ny, nx, nz = vol.shape
    if xx is None:
        xx = np.arange(0, nx)
    if yy is None:
        yy = np.arange(0, ny)
    if zz is None:
        zz = np.arange(0, nz)
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    # vmin = vmax*0.8
    if center == "max":
        icenter = np.where(vol == vmax)
    elif center == "min":
        icenter = np.where(vol == vmin)
    elif isinstance(center, int):
        iy = int(center / nz / nx)
        ix = int((center - nz * nx * iy) / nz)
        iz = int(center - nz * nx * iy - nz * ix)
        icenter = [[iy], [ix], [iz]]
    elif len(center) == 3:
        icenter = []
        icenter.append([np.argmin(np.abs(yy - center[1]))])
        icenter.append([np.argmin(np.abs(xx - center[0]))])
        icenter.append([np.argmin(np.abs(zz - center[2]))])
    iy = icenter[0][0]
    ix = icenter[1][0]
    iz = icenter[2][0]
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    dz = zz[1] - zz[0]
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(nrows=9, ncols=9, wspace=0, hspace=0)
    ax = []
    ax.append(fig.add_subplot(gs[:6, :6]))
    ax.append(fig.add_subplot(gs[6:8, :6]))
    ax.append(fig.add_subplot(gs[:6, 6:8]))
    # subplot(2,2,1)
    extent = [xx[0] - dx / 2, xx[-1] + dx / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
    im = ax[0].imshow(
        vol[:, :, iz],
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax[0].plot(extent[:2], np.ones(2) * yy[iy], "w-")
    ax[0].plot(np.ones(2) * xx[ix], extent[2:], "w-")
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])
    # subplot(2,2,3)
    extent = [xx[0] - dx / 2, xx[-1] + dx / 2, zz[-1] + dz / 2, zz[0] - dz / 2]
    ax[1].imshow(
        vol[iy, :, :].T,
        extent=extent,
        origin="upper",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax[1].plot(extent[:2], np.ones(2) * zz[iz], "w-")
    ax[1].plot(np.ones(2) * xx[ix], extent[2:], "w-")
    # subplot(2,2,2)
    extent = [zz[0] - dz / 2, zz[-1] + dz / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
    ax[2].imshow(
        vol[:, ix, :],
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax[2].plot(extent[:2], np.ones(2) * yy[iy], "w-")
    ax[2].plot(np.ones(2) * zz[iz], extent[2:], "w-")
    ax[2].xaxis.set_ticks_position("top")
    ax[2].yaxis.set_ticks_position("right")
    # colorbar
    cbaxes = fig.add_axes([0.13, 0.08, 0.7, 0.03])
    cb = fig.colorbar(im, cax=cbaxes, orientation="horizontal", label=cb_label)
    return fig, ax


def show_xyz_on_volume_slice(x, y, z, ax, **kwargs):
    """"""
    ax[0].plot(x, y, **kwargs)
    ax[1].plot(x, z, **kwargs)
    ax[2].plot(z, y, **kwargs)


def show_cc_mat(cc_mat, cc_mat_info, center=None, event=None, event_egf=None, **kwargs):
    """
    show 3 perpendicular slices in CC block centered @ center
    center can be:
        'max' [default]
        'min'
        [lon, lat, dep]
    """
    if center is None:
        center = "max"
    if "xx" in cc_mat_info.keys():
        xx = cc_mat_info["xx"]
    elif "grid_lon" in cc_mat_info.keys():
        xx = cc_mat_info["grid_lon"]
    if "yy" in cc_mat_info.keys():
        yy = cc_mat_info["yy"]
    elif "grid_lon" in cc_mat_info.keys():
        yy = cc_mat_info["grid_lat"]
    if "zz" in cc_mat_info.keys():
        zz = cc_mat_info["zz"]
    elif "grid_lon" in cc_mat_info.keys():
        zz = cc_mat_info["grid_dep"]
    fig, ax = show_volume_slice(
        cc_mat, xx=xx, yy=yy, zz=zz, center=center, **kwargs)
    if event is not None:
        show_xyz_on_volume_slice(
            event["lon"],
            event["lat"],
            event["dep"],
            ax,
            marker="o",
            markerfacecolor="r",
            markeredgecolor="k",
        )
    if event_egf is not None:
        show_xyz_on_volume_slice(
            event_egf["lon"],
            event_egf["lat"],
            event_egf["dep"],
            ax,
            marker="o",
            markerfacecolor="b",
            markeredgecolor="k",
        )
    return fig, ax


def plot_cc1d_res(
    cc_P,
    cc_S,
    cc_PS,
    subevent,
    dt=1 / 200,
    color_P="springgreen",
    color_S="goldenrod",
    color_PS="k",
    figsize=(8, 4),
):
    time_axis = np.arange(-600, 600) * dt + subevent["time"]
    time_indx = np.arange(-600, 600) + subevent["it"]
    ig = subevent["ig"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        time_axis, cc_P[ig, time_indx], "-", color=color_P, label="P", linewidth=3.0
    )
    ax.plot(
        time_axis, cc_S[ig, time_indx], "-", color=color_S, label="S", linewidth=3.0
    )
    ax.plot(
        time_axis, cc_PS[ig, time_indx], "-", color=color_PS, label="P+S", linewidth=2
    )
    ax.legend()
    ax.grid()
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("CC")
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.axvline(x=subevent["time"], color="darkmagenta")
    return fig, ax


def plot_subevent_stf(subevent_list_all, fn_mr_usgs, fn_mr_scardec):
    """
    plot subevents time together with USGS and SCARDEC moment rate function
    """
    usgs_mr_time, usgs_mr = read_USGS_MOMENT_RATE(fn_mr_usgs)
    timeq, scardec_mr = read_scardec_stf(fn_mr_scardec)
    fig, ax_time = plt.subplots(figsize=(11, 11 / 3))
    ax_time.plot(usgs_mr_time, usgs_mr, color="gray", label="USGS", lw=2)
    ax_time.plot(timeq, scardec_mr, "-", color="blue", label="SCARDEC average")
    for i, subevt in enumerate(subevent_list_all):
        t = subevt["time"]
        ax_time.axvline(x=t, color=subevt["color"], label="S" + str(i), lw=2)
    ax_time.set_ylim(bottom=0)
    ax_time.set_xlim([-2.5, 15])
    ax_time.grid()
    ax_time.legend()
    ax_time.set_xlabel("Time (sec)")
    ax_time.set_ylabel("Moment Rate (N*m/sec)")
    return fig, ax_time


def plot_subevent_contour_on_finite_fault_mapview(ff_dict, subevent_list_all):
    """
    plot subevents back-projection image contour on top of finite fault image
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    fig.set_linewidth(3)
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Slip colorplot
    im = ax.pcolormesh(
        ff_dict["lon"],
        ff_dict["lat"],
        ff_dict["slip"],
        cmap=plt.get_cmap("YlOrBr"),
        shading="gouraud",
    )
    ax.set_aspect("equal")
    ct_subevt = []
    contour_array = np.array([0.7, 0.8, 0.9, 1.0])
    lon_shft = [0.000, 0.002, -0.02, 0.002]
    lat_shft = [0.010, -0.01, 0.002, -0.014]
    for i, subevt in enumerate(subevent_list_all):
        label = "S{0}".format(i)
        color = subevt["color"]
        hsc = ax.scatter(
            subevt["lon"],
            subevt["lat"],
            s=140,
            c=color,
            ec="k",
            cmap=plt.get_cmap("viridis"),
            label=label,
            zorder=4,
        )
        slice_FQ_mask = ma.masked_array(subevt["cmatq"], mask=subevt["maskq"])
        ct_subevt.append(
            ax.contour(
                subevt["lonq"],
                subevt["latq"],
                slice_FQ_mask,
                levels=np.max(subevt["cmatq"]) * contour_array,
                colors=color,
                corner_mask=True,
                alpha=1,
                zorder=3,
            )
        )
        ax.text(
            subevt["lon"] + lon_shft[i],
            subevt["lat"] + lat_shft[i],
            label,
            fontsize=20,
            path_effects=buffer,
        )

    ax.legend(loc="upper left")
    ax.set_xlim([-119.556, -119.44])
    ax.set_ylim([38.4, 38.6])
    ax.set_aspect("equal")
    ax.set_xlabel("Lon. (deg)", fontsize=14)
    ax.set_ylabel("Lat. (deg)", fontsize=14)
    scale_bar_lonlat(ax, length=5, location=(0.5, 0.18))
    ax.set_xticks([-119.54, -119.50, -119.46])
    ax.set_yticks(
        [
            38.4,
            38.45,
            38.5,
            38.55,
            38.6,
        ]
    )
    ax.grid()
    ax.tick_params(axis="both", labelsize=14)
    return fig, ax


def scale_bar_lonlat(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """"""
    llx0, llx1 = ax.get_xlim()
    lly0, lly1 = ax.get_ylim()
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    bar_ll0, bar_ll1 = _utm_helper(sbllx, sblly, length * 1000)
    bar_xs = [bar_ll0[1], bar_ll1[1]]
    bar_ys = [bar_ll0[0], bar_ll0[0]]
    # scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    ax.plot(bar_xs, bar_ys, color="k",
            linewidth=linewidth, path_effects=buffer)
    # scalebar label
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    ax.text(
        sbllx,
        sblly,
        str(length) + " km",
        horizontalalignment="center",
        verticalalignment="bottom",
        path_effects=buffer,
        fontsize=16,
    )


def _utm_helper(lon, lat, length, azimuth=90):
    azimuth = azimuth / 180 * np.pi
    utm_origin = utm.from_latlon(lat, lon)
    bar_ll0 = utm.to_latlon(
        utm_origin[0] - length / 2 * np.sin(azimuth),
        utm_origin[1] + length / 2 * np.cos(azimuth),
        utm_origin[2],
        utm_origin[3],
    )
    bar_ll1 = utm.to_latlon(
        utm_origin[0] + length / 2 * np.sin(azimuth),
        utm_origin[1] - length / 2 * np.cos(azimuth),
        utm_origin[2],
        utm_origin[3],
    )
    return bar_ll0, bar_ll1
