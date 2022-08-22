

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

# plot
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

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    plot_maxmin_points,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]

# region import output

i = 0
expid[i]

exp_org_o = {}
exp_org_o[expid[i]] = {}


filenames_psl_zh = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_psl_zh.nc'))

exp_org_o[expid[i]]['psl_zh'] = xr.open_mfdataset(filenames_psl_zh[120:], data_vars='minimal', coords='minimal', parallel=True)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann psl and gh

i = 0
expid[i]

psl_zh = {}
psl_zh[expid[i]] = {}

psl_zh[expid[i]]['psl'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['psl_zh'].psl)
psl_zh[expid[i]]['zh'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['psl_zh'].zh)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'wb') as f:
    pickle.dump(psl_zh[expid[i]], f)


'''
#-------------------------------- check
i = 0
expid[i]
psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

# calculate manually
def time_weighted_mean(ds):
    return ds.weighted(ds.time.dt.days_in_month).mean('time', skipna=False)

test = {}
test['mon'] = exp_org_o[expid[i]]['psl_zh'].psl.copy()
test['sea'] = exp_org_o[expid[i]]['psl_zh'].psl.resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = exp_org_o[expid[i]]['psl_zh'].psl.resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()



(psl_zh[expid[i]]['psl']['mon'].values[np.isfinite(psl_zh[expid[i]]['psl']['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(psl_zh[expid[i]]['psl']['sea'].values[np.isfinite(psl_zh[expid[i]]['psl']['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(psl_zh[expid[i]]['psl']['ann'].values[np.isfinite(psl_zh[expid[i]]['psl']['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(psl_zh[expid[i]]['psl']['mm'].values[np.isfinite(psl_zh[expid[i]]['psl']['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(psl_zh[expid[i]]['psl']['sm'].values[np.isfinite(psl_zh[expid[i]]['psl']['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(psl_zh[expid[i]]['psl']['am'].values[np.isfinite(psl_zh[expid[i]]['psl']['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann psl in ERA5

psl_era5_79_14 = xr.open_dataset('scratch/cmip6/hist/psl/psl_ERA5_mon_sl_197901_201412.nc')

psl_era5_79_14_alltime = mon_sea_ann(var_monthly=psl_era5_79_14.msl)

with open('scratch/cmip6/hist/psl/psl_era5_79_14_alltime.pkl', 'wb') as f:
    pickle.dump(psl_era5_79_14_alltime, f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm psl in ECHAM6 and ERA5

i = 0
expid[i]
psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

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
# region plot psl and u/v in ECHAM

i = 0
expid[i]

uv_plev = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl', 'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

moisture_flux = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.pkl', 'rb') as f:
    moisture_flux[expid[i]] = pickle.load(f)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.3_moisture_transport/' + '6.1.4.3 ' + expid[i] + ' am psl and 850hPa wind Antarctica.png'

plt_pres = psl_zh[expid[i]]['psl']['am'] / 100
pres_interval = 5
pres_intervals = np.arange(
    np.floor(np.min(plt_pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(plt_pres) / pres_interval + 1) * pres_interval,
    pres_interval)

pltlevel = np.arange(-6, 6 + 1e-4, 0.5)
pltticks = np.arange(-6, 6 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()


fig, ax = hemisphere_plot(
    northextent=-45,
    figsize=np.array([5.8, 8.8]) / 2.54,
    fm_bottom=0.13,
    )

plt_ctr = ax.contour(
    plt_pres.lon, plt_pres.lat, plt_pres,
    colors='b', levels=pres_intervals, linewidths=0.2,
    transform=ccrs.PlateCarree(), clip_on=True)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt='%d',
    levels=pres_intervals, inline_spacing=10, fontsize=6)
h1, _ = plt_ctr.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    bbox_to_anchor=(0.5, -0.14),
    handlelength=1, columnspacing=1)

# plot H/L symbols
plot_maxmin_points(
    plt_pres.lon, plt_pres.lat, plt_pres,
    ax, 'max', 150, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    plt_pres.lon, plt_pres.lat, plt_pres,
    ax, 'min', 150, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

# plot wind arrows
iarrow = 2
plt_quiver = ax.quiver(
    plt_pres.lon[::iarrow], plt_pres.lat[::iarrow],
    uv_plev[expid[i]]['u']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    uv_plev[expid[i]]['v']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    color='gray', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.15, Y=-0.14, U=10,
             label='10 [$m \; s^{-1}$]    850 $hPa$ wind',
             labelpos='E', labelsep=0.05,)

plt_mesh = ax.pcolormesh(
    moisture_flux[expid[i]]['meridional']['am'].lon,
    moisture_flux[expid[i]]['meridional']['am'].lat,
    moisture_flux[expid[i]]['meridional']['am'].sel(plev=85000) * 10**3,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),
    zorder = -2)

cbar1 = fig.colorbar(
    plt_mesh, ax=ax,
    fraction=0.1,
    orientation="horizontal",shrink=1,aspect=40,extend='both',
    anchor=(0.5, 0.9), ticks=pltticks)
cbar1.ax.set_xlabel('Meridional moisture flux at 850 $hPa$\n[$10^{-3} \; kg\;kg^{-1} \; m\;s^{-1}$]', linespacing=1.5)

fig.savefig(output_png)



'''
stats.describe(abs(moisture_flux[expid[i]]['meridional']['am'].sel(plev=85000, lat=slice(-60, -90)) * 10**3),
               axis=None, nan_policy='omit')

(np.isfinite(uv_plev[expid[i]]['u']['am'].sel(plev=85000)) == np.isfinite(uv_plev[expid[i]]['v']['am'].sel(plev=85000))).all()
'''
# endregion
# -----------------------------------------------------------------------------

