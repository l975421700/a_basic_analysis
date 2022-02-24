

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.colors import ListedColormap

# self defined
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    plot_maxmin_points,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    regrid,
)

from a00_basic_analysis.b_module.namelist import (
    month_days,
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region example plot

era5_hr_sl_201412_10uv_slp_sst = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/hr_sl_79_present/era5_hr_sl_201412_10uv_slp_sst.nc'
)

lon = era5_hr_sl_201412_10uv_slp_sst.longitude.values
lat = era5_hr_sl_201412_10uv_slp_sst.latitude.sel(
    latitude=slice(-30, -90)).values

i_hour = 0

time = era5_hr_sl_201412_10uv_slp_sst.time[i_hour].values
pres = era5_hr_sl_201412_10uv_slp_sst.msl[i_hour, :, :].sel(
    latitude=slice(-30, -90)).values / 100
wind_u = era5_hr_sl_201412_10uv_slp_sst.u10[i_hour, :, :].sel(
    latitude=slice(-30, -90)).values
wind_v = era5_hr_sl_201412_10uv_slp_sst.v10[i_hour, :, :].sel(
    latitude=slice(-30, -90)).values

pres_interval = 10
pres_intervals = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)

fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.3]) / 2.54, fm_top=0.96, )

# contour of geopotential height
plt_pres = ax.contour(
    lon, lat, pres,
    colors='b', levels=pres_intervals, linewidths=0.2,
    transform=ccrs.PlateCarree(), clip_on=True)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_intervals, inline_spacing=10, fontsize=6)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Mean sea level pressure [hPa]'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.19), handlelength=1,
                      columnspacing=1)

# plot H/L symbols
plot_maxmin_points(
    lon, lat, pres, ax, 'max', 150, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    lon, lat, pres, ax, 'min', 150, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

# plot wind arrows
iarrow = 15
plt_quiver = ax.quiver(
    lon[::iarrow], lat[::iarrow],
    wind_u[::iarrow, ::iarrow], wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
             #  coordinates='data',
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.text(0.5, -0.2,
        str(time)[0:10] + ' ' + str(time)[11:13] + ':00 UTC',
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.0_SH 2014-12-01T00 wind and pressure.png',)



'''
stats.describe(pres, axis=None)
'''
# endregion
# =============================================================================



