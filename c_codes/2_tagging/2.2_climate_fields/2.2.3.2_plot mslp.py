

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ

# plot
import proplot as pplt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    plot_maxmin_points,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec_num,
    month_dec,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

lon = psl_zh[expid[i]]['psl']['am'].lon
lat = psl_zh[expid[i]]['psl']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm psl in ECHAM6 and ERA5

with open('scratch/cmip6/hist/psl/psl_era5_79_14_alltime.pkl', 'rb') as f:
    psl_era5_79_14_alltime = pickle.load(f)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.3_moisture_transport/' + '6.1.4.3 ' + expid[i] + ' and ERA5 am_sm psl Antarctica.png'
cbar_label1 = 'Mean sea level pressure [$hPa$]'

pltlevel = np.arange(975, 1025 + 1e-4, 2.5)
pltticks = np.arange(975, 1025 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol == 0) | (jcol == 1)):
            axs[irow, jcol] = hemisphere_plot(northextent=-30, ax_org = axs[irow, jcol])
        else:
            axs[irow, jcol].axis('off')

filter_boxes = 150
#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plot H/L symbols
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    axs[0, 0], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    axs[0, 0], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

axs[0, 1].pcolormesh(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['am'] / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['am'] / 100,
    axs[0, 1], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['am'] / 100,
    axs[0, 1], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

#-------- sm
axs[1, 0].pcolormesh(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='DJF') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='DJF') / 100,
    axs[1, 0], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='DJF') / 100,
    axs[1, 0], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='MAM') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='MAM') / 100,
    axs[1, 1], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='MAM') / 100,
    axs[1, 1], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='JJA') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='JJA') / 100,
    axs[1, 2], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='JJA') / 100,
    axs[1, 2], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='SON') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='SON') / 100,
    axs[1, 3], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_zh[expid[i]]['psl']['am'].lon,
    psl_zh[expid[i]]['psl']['am'].lat,
    psl_zh[expid[i]]['psl']['sm'].sel(season='SON') / 100,
    axs[1, 3], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

#-------- sm
axs[2, 0].pcolormesh(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='DJF') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='DJF') / 100,
    axs[2, 0], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='DJF') / 100,
    axs[2, 0], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='MAM') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='MAM') / 100,
    axs[2, 1], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='MAM') / 100,
    axs[2, 1], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='JJA') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='JJA') / 100,
    axs[2, 2], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='JJA') / 100,
    axs[2, 2], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='SON') / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='SON') / 100,
    axs[2, 3], 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    psl_era5_79_14_alltime['am'].longitude,
    psl_era5_79_14_alltime['am'].latitude,
    psl_era5_79_14_alltime['sm'].sel(season='SON') / 100,
    axs[2, 3], 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'ECHAM6', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'ERA5', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF ECHAM6', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM ECHAM6', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA ECHAM6', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON ECHAM6', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF ERA5', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM ERA5', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA ERA5', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON ERA5', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.75,aspect=40,extend='both',
    anchor=(0.5, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)





'''
stats.describe(psl_zh[expid[i]]['psl']['am'].sel(lat=slice(-30, -90)), axis=None)
stats.describe(psl_zh[expid[i]]['psl']['sm'].sel(lat=slice(-30, -90)), axis=None)
psl_zh[expid[i]]['psl']['am'].to_netcdf('scratch/test/test.nc')
psl_era5_79_14_alltime['am'].to_netcdf('scratch/test/test1.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am psl Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.6_psl/6.1.2.6 ' + expid[i] + ' psl am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=975, cm_max=1025, cm_interval1=5, cm_interval2=10, cmap='PuOr')

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plot H/L symbols
filter_boxes = 50
plot_maxmin_points(
    lon, lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    ax, 'max', filter_boxes, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    lon, lat,
    psl_zh[expid[i]]['psl']['am'] / 100,
    ax, 'min', filter_boxes, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('Mean sea level pressure (mslp) [$hPa$]', linespacing=1.5,)
fig.savefig(output_png)



'''
stats.describe(temp2_alltime[expid[i]]['am'].sel(lat=slice(-20, -90)),
               axis=None, nan_policy='omit')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot DJF-JJA psl Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.6_psl/6.1.2.6 ' + expid[i] + ' psl DJF-JJA Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3, cmap='BrBG')

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    (psl_zh[expid[i]]['psl']['sm'].sel(season='DJF') - \
        psl_zh[expid[i]]['psl']['sm'].sel(season='JJA')) / 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    psl_zh[expid[i]]['psl']['sea'][3::4],
    psl_zh[expid[i]]['psl']['sea'][1::4]
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('DJF - JJA mslp [$hPa$]', linespacing=1.5,)
fig.savefig(output_png)



'''
stats.describe(temp2_alltime[expid[i]]['am'].sel(lat=slice(-20, -90)),
               axis=None, nan_policy='omit')

'''
# endregion
# -----------------------------------------------------------------------------





