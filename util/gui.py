import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from .util import *

params = {
    "image.interpolation": "nearest",
    "savefig.dpi": 300,  
    "axes.labelsize": 12, 
    "axes.titlesize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False,
}
import matplotlib

matplotlib.rcParams.update(params)


def make_box_layout():
    return widgets.Layout(
        border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px"
    )

"""
Example:
vis = vis_bp_xyzt(
    bp_image["ccPS"],
    bp_image["ccP"],
    bp_image["ccS"],
    bp_image["grid_info"],
    event=mainshock,
    event_egf=aftershock_egf1,
)
vis
"""

class vis_bp_xyzt(widgets.VBox):
    """
    Visualization GUI for interactive analyzing backprojection results. 
    Inputs:
        - datGT:   Backprojection result data size=[ngrid, ntime]
        - datGT_P: Backprojection result data (P phase ) size=[ngrid, ntime]
        - datGT_S: Backprojection result data (S phase ) size=[ngrid, ntime]
        - data_info: should contain:
                    'tt':   time axis
                    'tref': reference time for calculating absolute origin time
        - event:   event to be detected
        - event_egf: event used for EGF
    """

    def __init__(self, datGT, datGT_P, datGT_S, data_info, event=None, event_egf=None):
        super().__init__()
        output = widgets.Output()
        self.data = datGT
        self.data_P = datGT_P
        self.data_S = datGT_S
        self.ng, self.nt = datGT.shape
        self.ig = 0
        self.xx = data_info["xx"]
        self.yy = data_info["yy"]
        self.zz = data_info["zz"]
        if "tt" in data_info.keys():
            self.tt = data_info["tt"]
            self.dt = self.tt[1] - self.tt[0]
        else:
            self.tt = np.arange(0, self.nt)
            self.dt = 1
        if "tref" in data_info.keys():
            self.tref = data_info["tref"]
        else:
            self.tref = None
        self.tlim = [self.tt[0], self.tt[-1]]
        self.event = event
        self.event_egf = event_egf
        self.nx = len(self.xx)
        self.ny = len(self.yy)
        self.nz = len(self.zz)
        self.nt = len(self.tt)
        self.ix = np.round(self.nx / 2).astype(int)
        self.iy = np.round(self.ny / 2).astype(int)
        self.iz = np.round(self.nz / 2).astype(int)
        self.it = np.round(self.nt / 2).astype(int)
        self.ntwin = 20
        self.nxwin = self.nx
        self.nywin = self.ny
        self.nzwin = self.nz
        self.vol = np.zeros((self.ny, self.nx, self.nz))
        self.update_vol()
        self.cross_yz = []
        self.cross_zx = []
        self.cross_xy = []
        self.im_yz = None
        self.im_zx = None
        self.im_xy = None
        self.im_gt = None
        self.line_gt = []
        with output:
            self.fig_xyz, self.ax_xyz, self.ax_gt = self.init_fig()

        self.init_plot()
        # controls
        self.int_slider_x = widgets.IntSlider(
            value=self.ix, min=0, max=self.nx - 1, step=1, description="index_X"
        )
        self.int_slider_y = widgets.IntSlider(
            value=self.iy, min=0, max=self.ny - 1, step=1, description="index_Y"
        )
        self.int_slider_z = widgets.IntSlider(
            value=self.iz, min=0, max=self.nz - 1, step=1, description="index_Z"
        )
        self.int_slider_t = widgets.IntSlider(
            value=self.it, min=0, max=self.nt - 1, step=1, description="index_T"
        )
        self.int_slider_g = widgets.IntSlider(
            value=self.ig, min=0, max=self.ng - 1, step=1, description="index_G"
        )
        self.text_tbar = widgets.BoundedFloatText(
            min=self.tlim[0],
            max=self.tlim[1],
            value=self.tt[self.it],
            step=self.dt,
            description="tbar",
        )
        text_tmin = widgets.BoundedFloatText(
            min=self.tt[0],
            max=self.tlim[1],
            value=self.tlim[0],
            step=0.01,
            description="tmin",
            disabled=False,
        )
        text_tmax = widgets.BoundedFloatText(
            min=self.tlim[0],
            max=self.tt[-1],
            value=self.tlim[-1],
            step=0.01,
            description="tmax",
            disabled=False,
        )
        if self.tref is not None:
            self.text_tref = widgets.BoundedFloatText(
                value=self.tref[self.ig], description="tref", disabled=True, step=0.001
            )
        else:
            self.text_tref = widgets.BoundedFloatText(
                value=0.0, description="tref", disabled=True, step=0.001
            )
        text_cmax = widgets.BoundedFloatText(
            min=0,
            max=np.max(self.data),
            value=np.max(self.data),
            step=0.01,
            description="ccmax",
            disabled=False,
        )
        text_twin = widgets.IntText(value=self.ntwin, description="twin")
        text_xwin = widgets.IntText(value=self.nxwin, description="xwin")
        text_ywin = widgets.IntText(value=self.nywin, description="ywin")
        text_zwin = widgets.IntText(value=self.nzwin, description="zwin")
        button_max = widgets.Button(description="max")
        # boxes
        control_xyz = widgets.VBox(
            [
                self.int_slider_x,
                self.int_slider_y,
                self.int_slider_z,
                self.int_slider_g,
                text_cmax,
            ]
        )
        control_gt = widgets.VBox(
            [self.int_slider_t, self.text_tbar, text_tmin, text_tmax, self.text_tref]
        )
        control_misc = widgets.VBox(
            [text_twin, text_xwin, text_ywin, text_zwin, button_max]
        )
        controls = widgets.HBox([control_xyz, control_gt, control_misc])

        controls.layout = make_box_layout()

        self.int_slider_x.observe(self.update_ix, "value")
        self.int_slider_y.observe(self.update_iy, "value")
        self.int_slider_z.observe(self.update_iz, "value")
        self.int_slider_t.observe(self.update_it, "value")
        self.int_slider_g.observe(self.update_ig, "value")
        self.text_tbar.observe(self.update_t, "value")
        text_tmin.observe(self.update_tlim_min, "value")
        text_tmax.observe(self.update_tlim_max, "value")
        text_cmax.observe(self.update_clim_max, "value")
        text_twin.observe(self.update_ntwin, "value")
        text_xwin.observe(self.update_nxwin, "value")
        text_ywin.observe(self.update_nywin, "value")
        text_zwin.observe(self.update_nzwin, "value")
        button_max.on_click(self.update_imax)
        self.children = [output, controls]

    def init_fig(self):
        fig_xyz = plt.figure(constrained_layout=False, figsize=(10, 10 * 9 / 13))
        gs = fig_xyz.add_gridspec(nrows=9, ncols=13, wspace=0, hspace=0)
        ax_xyz = []
        ax_gt = []
        # xy slice
        ax_xyz.append(fig_xyz.add_subplot(gs[:6, :6]))
        # xz slide
        ax_xyz.append(fig_xyz.add_subplot(gs[6:8, :6]))
        # yz slice
        ax_xyz.append(fig_xyz.add_subplot(gs[:6, 6:8]))
        # gt slice
        ax_gt.append(fig_xyz.add_subplot(gs[:6, 9:13]))
        # c(t) sequence
        ax_gt.append(fig_xyz.add_subplot(gs[6:8, 9:13]))

        return fig_xyz, ax_xyz, ax_gt

    def init_plot(self):
        vmax = np.max(self.data)
        vmin = 0
        vmax_all = np.max([vmax, np.max(self.data_P), np.max(self.data_S)])
        xx = self.xx
        yy = self.yy
        zz = self.zz
        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]
        dz = zz[1] - zz[0]
        # grid-time image
        self.im_gt = self.ax_gt[0].imshow(
            self.data,
            aspect="auto",
            extent=[self.tt[0], self.tt[-1], 0, self.ng],
            origin="lower",
            cmap=plt.get_cmap("seismic"),
        )
        self.ig = np.argmax(np.max(self.data, axis=1))
        self.it = np.argmax(self.data[self.ig, :])
        self.ix, self.iy, self.iz = self.ig2ixyz(self.ig)
        ix = self.ix
        iy = self.iy
        iz = self.iz
        self.update_vol()
        self.line_gt.append(
            plt.plot(self.tt, self.data[self.ig, :], "g-", label="Both")[0]
        )
        self.line_gt.append(
            plt.plot(self.tt, self.data_P[self.ig, :], "b-", label="P")[0]
        )
        self.line_gt.append(
            plt.plot(self.tt, self.data_S[self.ig, :], "r-", label="S")[0]
        )
        self.cross_g = self.ax_gt[0].plot(
            [self.tt[0], self.tt[-1]], self.ig * np.ones(2), "k-"
        )
        self.cross_t = self.ax_gt[1].plot(
            self.tt[self.it] * np.ones(2), [-vmax_all, vmax_all], "k-"
        )
        self.ax_gt[0].xaxis.set_ticks([])
        self.ax_gt[0].yaxis.set_ticks_position("right")
        self.ax_gt[1].yaxis.set_ticks_position("right")
        self.ax_gt[1].set_xlim(self.tlim)
        self.ax_gt[1].set_ylim([-vmax_all - 0.05, vmax_all + 0.05])
        self.ax_gt[1].set_xlabel("Time (sec)")
        # xyz volume slices
        # cmap = plt.get_cmap('YlOrBr')
        cmap = plt.get_cmap("viridis")
        # xy slice
        extent = [xx[0] - dx / 2, xx[-1] + dx / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
        self.im_xy = self.ax_xyz[0].imshow(
            self.vol[:, :, self.iz],
            extent=extent,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        self.cross_xy.append(
            self.ax_xyz[0].plot(extent[:2], np.ones(2) * yy[iy], "w-")[0]
        )
        self.cross_xy.append(
            self.ax_xyz[0].plot(np.ones(2) * xx[ix], extent[2:], "w-")[0]
        )
        # xz slice
        extent = [xx[0] - dx / 2, xx[-1] + dx / 2, zz[-1] + dz / 2, zz[0] - dz / 2]
        self.im_zx = self.ax_xyz[1].imshow(
            self.vol[self.iy, :, :].T,
            extent=extent,
            origin="upper",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        self.cross_zx.append(
            self.ax_xyz[1].plot(extent[:2], np.ones(2) * zz[iz], "w-")[0]
        )
        self.cross_zx.append(
            self.ax_xyz[1].plot(np.ones(2) * xx[ix], extent[2:], "w-")[0]
        )
        # yz slice
        extent = [zz[0] - dz / 2, zz[-1] + dz / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
        self.im_yz = self.ax_xyz[2].imshow(
            self.vol[:, self.ix, :],
            extent=extent,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        self.cross_yz.append(
            self.ax_xyz[2].plot(extent[:2], np.ones(2) * yy[iy], "w-")[0]
        )
        self.cross_yz.append(
            self.ax_xyz[2].plot(np.ones(2) * zz[iz], extent[2:], "w-")[0]
        )
        self.ax_xyz[0].xaxis.set_ticks([])
        self.ax_xyz[0].yaxis.set_ticks([])
        self.ax_xyz[2].xaxis.set_ticks_position("top")
        self.ax_xyz[2].yaxis.set_ticks_position("right")
        # colorbar
        cbaxes = self.fig_xyz.add_axes([0.15, 0.1, 0.4, 0.02])
        self.cb = self.fig_xyz.colorbar(
            self.im_xy, cax=cbaxes, orientation="horizontal", label="CC"
        )
        # show event and egf on the slices
        self.event_str = ""
        self.event_egf_str = ""
        if self.event is not None:
            self.show_xyz_on_slice(
                self.event["lon"],
                self.event["lat"],
                self.event["dep"],
                marker="o",
                markerfacecolor="orangered",
                markeredgecolor="k",
            )
            if "dir" in self.event.keys():
                self.event_str = self.event["dir"]
        if self.event_egf is not None:
            self.show_xyz_on_slice(
                self.event_egf["lon"],
                self.event_egf["lat"],
                self.event_egf["dep"],
                marker="o",
                markerfacecolor="darkorange",
                markeredgecolor="k",
            )
            if "dir" in self.event_egf.keys():
                self.event_egf_str = self.event_egf["dir"]
        title_str = self.get_title_str()
        self.title = self.fig_xyz.suptitle(title_str)

    def show_xyz_on_slice(self, x, y, z, **kwargs):
        """"""
        self.ax_xyz[0].plot(x, y, **kwargs)
        self.ax_xyz[1].plot(x, z, **kwargs)
        self.ax_xyz[2].plot(z, y, **kwargs)

    def update_vol(self):
        self.vol = np.max(
            self.data[:, self.it - self.ntwin : self.it + self.ntwin + 1], axis=1
        ).reshape((self.ny, self.nx, self.nz))

    def update_ix(self, change):
        self.ix = change.new
        self.update_slice_yz()
        self.ig = self.ixyz2ig(self.ix, self.iy, self.iz)
        self.update_slider_int()
        self.update_gt()

    def update_iy(self, change):
        self.iy = change.new
        self.update_slice_zx()
        self.ig = self.ixyz2ig(self.ix, self.iy, self.iz)
        self.update_slider_int()
        self.update_gt()

    def update_iz(self, change):
        self.iz = change.new
        self.update_slice_xy()
        self.ig = self.ixyz2ig(self.ix, self.iy, self.iz)
        self.update_slider_int()
        self.update_gt()

    def update_ig(self, change):
        self.ig = change.new
        self.ix, self.iy, self.iz = self.ig2ixyz(self.ig)
        self.update_slider_int()
        self.update_slice_xyzt()

    def update_ntwin(self, change):
        self.ntwin = change.new

    def update_nxwin(self, change):
        self.nxwin = change.new

    def update_nywin(self, change):
        self.nywin = change.new

    def update_nzwin(self, change):
        self.nzwin = change.new

    def update_imax(self, b):
        """
        find the local maximum index [ix, iy, iz, it] given x,y,z,t window
        """
        itbeg = np.max([self.it - self.ntwin, 0])
        itend = np.min([self.it + self.ntwin + 1, self.nt])
        ixbeg = np.max([self.ix - self.nxwin, 0])
        ixend = np.min([self.ix + self.nxwin + 1, self.nx])
        iybeg = np.max([self.iy - self.nywin, 0])
        iyend = np.min([self.iy + self.nywin + 1, self.ny])
        izbeg = np.max([self.iz - self.nzwin, 0])
        izend = np.min([self.iz + self.nzwin + 1, self.nz])
        igx, igy, igz = np.meshgrid(
            np.arange(ixbeg, ixend), np.arange(iybeg, iyend), np.arange(izbeg, izend)
        )
        igx = igx.flatten()
        igy = igy.flatten()
        igz = igz.flatten()
        igg = self.ixyz2ig(igx, igy, igz)
        igmax = np.argmax(np.max(self.data[igg, itbeg:itend], axis=1))
        self.ix = igx[igmax]
        self.iy = igy[igmax]
        self.iz = igz[igmax]
        self.ig = self.ixyz2ig(self.ix, self.iy, self.iz)
        self.it = np.argmax(self.data[self.ig, itbeg:itend]) + itbeg
        self.int_slider_x.value = self.ix
        self.int_slider_y.value = self.iy
        self.int_slider_z.value = self.iz
        self.int_slider_t.value = self.it
        self.text_tbar.value = self.tt[self.it]
        self.update_slice_xyzt()

    def update_it(self, change):
        self.it = change.new
        self.text_tbar.value = self.tt[self.it]
        self.update_slice_xyzt()

    def update_t(self, change):
        self.it = np.round((change.new - self.tt[0]) / self.dt).astype(int)
        self.int_slider_t.value = self.it
        self.update_slice_xyzt()

    def update_tlim_min(self, change):
        itmin = np.round((change.new - self.tt[0]) / self.dt).astype(int)
        itmin = np.max([0, itmin])
        self.int_slider_t.min = itmin
        self.tlim[0] = self.tt[itmin]
        self.ax_gt[0].set_xlim(left=self.tlim[0])
        self.ax_gt[1].set_xlim(left=self.tlim[0])

    def update_tlim_max(self, change):
        itmax = np.round((change.new - self.tt[0]) / self.dt).astype(int)
        itmax = np.min([self.nt - 1, itmax])
        self.int_slider_t.max = itmax
        self.tlim[1] = self.tt[itmax]
        self.ax_gt[0].set_xlim(right=self.tlim[1])
        self.ax_gt[1].set_xlim(right=self.tlim[1])

    def update_tref(self):
        if self.tref is not None:
            self.text_tref.value = self.tref[self.ig]

    def update_clim_max(self, change):
        self.im_xy.set_clim(vmax=change.new)
        self.im_yz.set_clim(vmax=change.new)
        self.im_zx.set_clim(vmax=change.new)

    def update_slice_yz(self, vmin=None, vmax=None):
        """"""
        self.cross_xy[1].set_xdata(np.ones(2) * self.xx[self.ix])
        self.cross_zx[1].set_xdata(np.ones(2) * self.xx[self.ix])
        self.im_yz.set_data(self.vol[:, self.ix, :])
        if None not in (vmin, vmax):
            self.im_yz.set_clim(vmin=vmin, vmax=vmax)

    def update_slice_zx(self, vmin=None, vmax=None):
        """"""
        self.cross_xy[0].set_ydata(np.ones(2) * self.yy[self.iy])
        self.cross_yz[0].set_ydata(np.ones(2) * self.yy[self.iy])
        self.im_zx.set_data(self.vol[self.iy, :, :].T)
        if None not in (vmin, vmax):
            self.im_zx.set_clim(vmin=vmin, vmax=vmax)

    def update_slice_xy(self, vmin=None, vmax=None):
        """"""
        self.cross_zx[0].set_ydata(np.ones(2) * self.zz[self.iz])
        self.cross_yz[1].set_xdata(np.ones(2) * self.zz[self.iz])
        self.im_xy.set_data(self.vol[:, :, self.iz])
        if None not in (vmin, vmax):
            self.im_xy.set_clim(vmin=vmin, vmax=vmax)

    def extract_slice(self):
        """"""
        fun_interp = rgi(
            (self.yy, self.xx, self.zz), self.vol, bounds_error=False, fill_value=0
        )
        vol_q = fun_interp(np.array([self.latq, self.lonq, self.depq]).T)
        return vol_q.reshape((20, 15), order="F")

    def update_slice_xyz(self):
        # vmax = np.max(self.vol)
        # vmin = np.min(self.vol)
        vmin = None
        vmax = None
        self.update_slice_yz(vmin=vmin, vmax=vmax)
        self.update_slice_xy(vmin=vmin, vmax=vmax)
        self.update_slice_zx(vmin=vmin, vmax=vmax)

    def update_gt(self):
        self.cross_g[0].set_ydata(np.ones(2) * self.ig)
        self.cross_t[0].set_xdata(np.ones(2) * self.tt[self.it])
        self.line_gt[0].set_ydata(self.data[self.ig, :])
        self.line_gt[1].set_ydata(self.data_P[self.ig, :])
        self.line_gt[2].set_ydata(self.data_S[self.ig, :])

    def update_title(self):
        text_str = self.get_title_str()
        self.title.set_text(text_str)

    def get_title_str(self):
        if self.tref is not None:
            traveltime = self.tref[self.ig]
        else:
            traveltime = 0
        title_str = "Event: {0}, EGF: {1}\n Time={2:.3f}, CC={3:.2f}".format(
            self.event_str,
            self.event_egf_str,
            self.tt[self.it] - traveltime,
            self.data[self.ig, self.it],
        )
        return title_str

    def update_slider_int(self):
        self.int_slider_t.value = self.it
        self.int_slider_x.value = self.ix
        self.int_slider_y.value = self.iy
        self.int_slider_z.value = self.iz
        self.int_slider_g.value = self.ig

    def update_slice_xyzt(self):
        """"""
        self.update_gt()
        self.update_vol()
        self.update_slice_xyz()
        self.update_title()
        self.update_tref()

    def ixyz2ig(self, ix, iy, iz):
        """
        (ix, iy, iz) to ig
        """
        return self.nz * self.nx * iy + self.nz * ix + iz

    def ig2ixyz(self, ig):
        """
        ig to (ix, iy, iz)
        """
        iy = int(ig / self.nz / self.nx)
        ix = int((ig - self.nz * self.nx * iy) / self.nz)
        iz = int(ig - self.nz * self.nx * iy - self.nz * ix)
        return ix, iy, iz
