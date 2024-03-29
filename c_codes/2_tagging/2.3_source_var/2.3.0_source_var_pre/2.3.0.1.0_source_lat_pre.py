

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'nudged_705_6.0',
    ]
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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


'''
pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/check/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lat[expid[i]]['am'].to_netcdf('output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0/analysis/echam/check/pi_m_502_5.0.pre_weighted_lat_am.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lat am + am aprt


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat am Antarctica_contour.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-46, cm_max=-34, cm_interval1=1, cm_interval2=2, cmap='viridis',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt_data = pre_weighted_lat[expid[i]]['am'].copy()

plt_data.sel(lat=slice(90, -60)).values[:] = np.nan

plt1 = plot_t63_contourf(
    lon, lat, plt_data, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source latitude [$°\;S$]', linespacing=2)
fig.savefig(output_png)



'''

# output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat am Antarctica.png'
# output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat am Antarctica + am aprt.png'

# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

# plt1 = ax.pcolormesh(
#     lon,
#     lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# plt1 = ax.contourf(
#     lon,
#     lat,
#     plt_data,
#     levels = pltlevel,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# # plot am aprt
# pltctr1 = np.array([0.05, 0.1, 0.5, ])
# pltctr2 = np.array([1, 2, 4, ])
# plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d

# plt2 = ax.contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     plt_data.sel(lat=slice(-50, -90)),
#     levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='dotted',
# )
# ax.clabel(plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
#           levels=pltctr1, inline_spacing=10, fontsize=6,)

# plt3 = ax.contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     plt_data.sel(lat=slice(-50, -90)),
#     levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# ax.clabel(plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
#           levels=pltctr2, inline_spacing=5, fontsize=6,)

#-------------------------------- plot for Louise 13 Sep

turner_obs = pd.read_csv(
    'finse_school/data/Antarctic_site_records/Turner_obs.csv')

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am Antarctica_for_Louise.png'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54,
    llatlabel = True,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])

plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d

cplot_ice_cores(np.negative(turner_obs.lon), turner_obs.lat, ax, marker='*',
                edgecolors = 'red', zorder=3,c='red', s=5)

# plt2 = ax.contour(
#     lon, lat,
#     plt_data,
#     levels=pltctr1, colors = 'm', transform=ccrs.PlateCarree(),
#     linewidths=0.7, linestyles='dotted',
# )
# ax.clabel(plt2, inline=1, colors='m', fmt=remove_trailing_zero,
#           levels=pltctr1, inline_spacing=10, fontsize=9,)

# plt3 = ax.contour(
#     lon, lat,
#     plt_data,
#     levels=pltctr2, colors = 'm', transform=ccrs.PlateCarree(),
#     linewidths=0.7, linestyles='solid',
# )
# ax.clabel(plt3, inline=1, colors='m', fmt=remove_trailing_zero,
#           levels=pltctr2, inline_spacing=5, fontsize=9,)

plt1 = ax.pcolormesh(
    pre_weighted_lat[expid[i]]['am'].lon,
    pre_weighted_lat[expid[i]]['am'].lat,
    pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Source latitude [$°$]\n ', linespacing=2)
fig.savefig(output_png)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lat am_sm_5

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_sm_5 Antarctica.png'
cbar_label1 = 'Source latitude [$° \; S$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-50, cm_max=-30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
ctr_level = np.array([2, 4, 6, ])

nrow = 1
ncol = 5
fm_right = 2 / (5.8*ncol + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol + 2, 5.8*nrow+0.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=-60, ax_org = axs[jcol],
        l45label = False, loceanarcs = False)
    cplot_ice_cores(
        ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])

#-------- Am
plt_data = pre_weighted_lat[expid[i]]['am'].copy()
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
plt_data.sel(lat=slice(90, -60)).values[:] = np.nan
# plt_mesh1 = axs[0].pcolormesh(
#     lon, lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, plt_data, axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[0].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

plt_data = pre_weighted_lat[expid[i]]['ann'].std(
    dim='time', skipna=True, ddof=1)
plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

plt_ctr1 = axs[0].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    plt_data.sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'r', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',)
axs[0].clabel(
    plt_ctr1, inline=1, colors='r', fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10,)
plt.text(
    0.5, 1.04, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

#-------- sm
for iseason in range(len(seasons)):
    plt_data = pre_weighted_lat[expid[i]]['sm'].sel(season=seasons[iseason]).copy()
    # plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
    plt_data.sel(lat=slice(90, -60)).values[:] = np.nan
    # axs[1 + iseason].pcolormesh(
    #     lon, lat,
    #     plt_data,
    #     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plot_t63_contourf(
        lon, lat, plt_data, axs[1 + iseason],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    axs[1 + iseason].add_feature(
	    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    
    plt_data = pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == \
            seasons_last_num[iseason])
        ).std(dim='time', skipna=True, ddof=1)
    plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
    
    plt_ctr = axs[1 + iseason].contour(
        lon, lat.sel(lat=slice(-60, -90)),
        plt_data.sel(lat=slice(-60, -90)),
        levels=ctr_level, colors = 'r', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='solid',
    )
    axs[1 + iseason].clabel(
        plt_ctr, inline=1, colors='r', fmt=remove_trailing_zero,
        levels=ctr_level, inline_spacing=10,)
    plt.text(
        0.5, 1.04, seasons[iseason], transform=axs[1 + iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=20,extend='both', ticks=pltticks,
    anchor=(1.5, 0.5))
cbar1.ax.set_ylabel(cbar_label1, linespacing=2)
cbar1.ax.set_yticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])

fig.subplots_adjust(left=0.01, right = 1-fm_right, bottom = 0, top = 0.94)
fig.savefig(output_png)




'''
# aprt ctr
pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])

# plot am aprt
plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d
plt2 = axs[0].contour(
    lon, lat,
    plt_data,
    levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
axs[0].clabel(plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=5, fontsize=6,)

plt3 = axs[0].contour(
    lon, lat,
    plt_data,
    levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0].clabel(plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=5, fontsize=6,)



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lat DJF-JJA


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat DJF-JJA Antarctica.png'

pltlevel = np.arange(-6, 6 + 1e-4, 1)
pltticks = np.arange(-6, 6 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    pre_weighted_lat[expid[i]]['sm'].sel(season = 'DJF') - \
        pre_weighted_lat[expid[i]]['sm'].sel(season = 'JJA'),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][3::4,],
    pre_weighted_lat[expid[i]]['sea'][1::4,],)
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('DJF - JJA source latitude [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region cross check pre_weighted_lat am

#-------- import data
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' pre_weighted_lat am cross_check.png'
file_dir = 'output/echam-6.3.05p2-wiso/pi/'
pre_weighted_var_files = [
    # 'pi_m_406_4.7/analysis/echam/pi_m_406_4.7.pre_weighted_lat_am.nc',
    # 'pi_m_410_4.8/analysis/echam/pi_m_410_4.8.pre_weighted_lat_am.nc',
    'pi_m_503_5.0/analysis/echam/pi_m_503_5.0.pre_weighted_lat_am.nc',
]

pre_weighted_var = {}
pre_weighted_var['am_lowres'] = xr.open_dataset(
    file_dir + pre_weighted_var_files[0])
# pre_weighted_var['am_highres'] = xr.open_dataset(
#     file_dir + pre_weighted_var_files[1])

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/check/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat

#-------------------------------- plot
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PuOr',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-2.5, cm_max=2.5, cm_interval1=0.5, cm_interval2=1, cmap='PiYG',)

cbar_label1 = 'Source latitude [$°$]'
cbar_label2 = 'Differences [$°$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

# plot am values
# plt_mesh1 = axs[0].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, pre_weighted_lat[expid[i]]['am'], axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

# plot am norm - lowres
# plt_mesh2 = axs[1].pcolormesh(
#     lon, lat, pre_weighted_var['am_lowres'].pre_weighted_lat_am - \
#         pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# plot am norm - highres
# plt_mesh2 = axs[2].pcolormesh(
#     lon, lat, pre_weighted_var['am_highres'].pre_weighted_lat_am - \
#         pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

# plot lowres
# plt_mesh2 = axs[1].pcolormesh(
#     lon, lat, pre_weighted_var['am_lowres'].pre_weighted_lat_am,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh2 = plot_t63_contourf(
    lon, lat, pre_weighted_var['am_lowres'].pre_weighted_lat_am, axs[1],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

# plot am norm - lowres
# plt_mesh3 = axs[2].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['am'] - \
#         pre_weighted_var['am_lowres'].pre_weighted_lat_am,
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
plt_mesh3 = plot_t63_contourf(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - \
        pre_weighted_var['am_lowres'].pre_weighted_lat_am, axs[2],
    pltlevel2, 'both', pltnorm2, pltcmp2, ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, '(a) Scaled-flux water tracing',
    transform=axs[0].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b) Prescribed-region water tracing (10$°$ latitude bins)',
    transform=axs[1].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c) Differences: (a) - (b)',
    transform=axs[2].transAxes, ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)


echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

diff = abs(pre_weighted_lat[expid[i]]['am'] - pre_weighted_var['am_lowres'].pre_weighted_lat_am).values
np.nanmax(diff)
np.nanmean(diff)

np.average(
    diff[np.isfinite(diff)],
    weights = echam6_t63_cellarea.cell_area.values[np.isfinite(diff)],
)
np.nanmean(pre_weighted_lat[expid[i]]['am'] - pre_weighted_var['am_lowres'].pre_weighted_lat_am)


'''
#-------------------------------- framework to plot
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=10, cmap='PuOr',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2, cmap='PiYG',)

cbar_label1 = 'Source latitude [$°$]'
cbar_label2 = 'Differences in source latitude [$°$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

plt.text(
    0.5, 1.05, 'TEXT', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)

#-------- annual values
    'pi_m_406_4.7/analysis/echam/pi_m_406_4.7.pre_weighted_lat_ann.nc',
    'pi_m_410_4.8/analysis/echam/pi_m_410_4.8.pre_weighted_lat_ann.nc',

pre_weighted_var['ann_lowres'] = xr.open_dataset(
    file_dir + pre_weighted_var_files[1])
pre_weighted_var['ann_highres'] = xr.open_dataset(
    file_dir + pre_weighted_var_files[3])

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am all source properties

#-------- import data
pre_weighted_var = {}
pre_weighted_var[expid[i]] = {}

source_var = ['lat', 'lon', 'rellon', 'sst', 'rh2m', 'wind10']

prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.pre_weighted_lat.pkl',
    prefix + '.pre_weighted_lon.pkl',
    prefix + '.pre_weighted_lon.pkl',
    prefix + '.pre_weighted_sst.pkl',
    prefix + '.pre_weighted_rh2m.pkl',
    prefix + '.pre_weighted_wind10.pkl',
]

for ivar, ifile in zip(source_var, source_var_files):
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        pre_weighted_var[expid[i]][ivar] = pickle.load(f)

#-------- plot
cm_mins = [-46, 0, -180, 11, 75, 10.1, ]
cm_maxs = [-34, 360, 180, 17, 80, 11.3, ]
cm_interval1s = [1, 30, 30, 0.5, 0.5, 0.1, ]
cm_interval2s = [2, 60, 60, 1, 1, 0.2, ]
cmaps = ['viridis_r', 'BrBG', 'twilight_shifted',
         'viridis_r', 'cividis', 'magma_r', ]
cbar_labels = [
    'Source latitude [$°\;S$]', 'Source longitude [$°$]',
    'Relative source longitude [$°$]',
    'Source SST [$°C$]', 'Source rh2m [$\%$]', 'Source vel10 [$m \; s^{-1}$]',
]

# output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3 ' + expid[i] + ' pre_weighted_var am Antarctica + am aprt.png'
output_png = 'figures/test/test.png'

nrow = 2
ncol = 3

wspace = 0.02
hspace = 0.12
fm_left = 0.02
fm_bottom = hspace / nrow
fm_right = 0.98
fm_top = 0.98

# plot am aprt
pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])
plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 7.8*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    )

ipanel=0
for irow in range(nrow):
  for jcol in range(ncol):
    # if ((irow != 0) | (jcol == 2)):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-60, ax_org = axs[irow, jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[irow, jcol])
        
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        icount = irow * ncol + jcol
        ivar = source_var[icount]
        
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min = cm_mins[icount],
            cm_max = cm_maxs[icount],
            cm_interval1 = cm_interval1s[icount],
            cm_interval2 = cm_interval2s[icount],
            cmap = cmaps[icount],)
        if (ivar == 'rh2m'):
            pltcmp = pltcmp.reversed()
        
        plt_meshdata = pre_weighted_var[expid[i]][ivar]['am'].copy()
        if(ivar == 'rellon'):
            plt_meshdata = calc_lon_diff(
                pre_weighted_var[expid[i]][ivar]['am'], lon_2d)
        
        extend = 'both'
        if ((ivar == 'lon') | (ivar == 'rellon')):
            extend = 'neither'
        
        # plt_meshdata.values[
        #     echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
        
        if False: # ((ivar == 'lon') | (ivar == 'rellon')):
            plt1 = axs[irow, jcol].pcolormesh(
                lon,
                lat,
                plt_meshdata,
                norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        else:
            if(ivar == 'lat'):
                plt_meshdata.sel(lat=slice(90, -60)).values[:] = np.nan
            
            plt1 = plot_t63_contourf(
                lon, lat, plt_meshdata, axs[irow, jcol],
                pltlevel, extend, pltnorm, pltcmp, ccrs.PlateCarree(),)
        
        axs[irow, jcol].add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        # plt2 = axs[irow, jcol].contour(
        #     lon, lat.sel(lat=slice(-50, -90)),
        #     plt_data.sel(lat=slice(-50, -90)),
        #     levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
        #     linewidths=0.5, linestyles='dotted',)
        # axs[irow, jcol].clabel(
        #     plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
        #     levels=pltctr1, inline_spacing=10, fontsize=6,)

        # plt3 = axs[irow, jcol].contour(
        #     lon, lat.sel(lat=slice(-50, -90)),
        #     plt_data.sel(lat=slice(-50, -90)),
        #     levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
        #     linewidths=0.5, linestyles='solid',)
        # axs[irow, jcol].clabel(
        #     plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
        #     levels=pltctr2, inline_spacing=5, fontsize=6,)
        
        cbar = fig.colorbar(
            plt1, ax=axs[irow, jcol], aspect=30,
            format=remove_trailing_zero_pos,
            orientation="horizontal", shrink=0.9, ticks=pltticks, extend=extend,
            pad=0.05,)
        if ((irow == 0) & (jcol == 0)):
            cbar.ax.set_xticklabels(
                [remove_trailing_zero(x) for x in np.negative(pltticks)])
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel(cbar_labels[icount])
        
        print(ivar)

fig.subplots_adjust(
    left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
    wspace=wspace, hspace=hspace,)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot relative pre_weighted_lat am + am aprt


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/6.1.3.0 ' + expid[i] + ' relative pre_weighted_lat am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=10, cm_max=50, cm_interval1=4, cm_interval2=8, cmap='PiYG',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    pre_weighted_lat[expid[i]]['am'] - lat_2d,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
# cbar.ax.set_xticklabels(
#     [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Relative source latitude [$°$]',)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check pre_weighted_lat at EAIS/WAIS/AP

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)
# echam6_t63_surface_height.values[echam6_t63_geosp.SLM.values == 0] = np.nan

for imask in ['AIS', 'EAIS', 'WAIS', 'AP']:
    # imask = 'AIS'
    print('#-------- ' + imask)
    
    mask = echam6_t63_ais_mask['mask'][imask]
    
    source_lat_max = np.max(pre_weighted_lat[expid[i]]['am'].values[mask])
    source_lat_min = np.min(pre_weighted_lat[expid[i]]['am'].values[mask])
    print(np.round(source_lat_max, 1))
    print(np.round(source_lat_min, 1))


for imask in ['AIS', 'EAIS', 'WAIS', 'AP']:
    # imask = 'AIS'
    print('#-------- ' + imask)
    
    mask = echam6_t63_ais_mask['mask'][imask]
    
    source_lat = np.average(
        pre_weighted_lat[expid[i]]['am'].values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask] * \
            ocean_aprt_alltime[expid[i]]['am'].sel(var_names='lat').values[
                mask],
    )
    
    print(np.round(source_lat, 1))


imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)
mask_low = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values < 2250)

for mask in[mask_high, mask_low]:
    # print(mask)
    
    source_lat = np.average(
        pre_weighted_lat[expid[i]]['am'].values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask] * \
            ocean_aprt_alltime[expid[i]]['am'].sel(var_names='lat').values[
                mask],
    )
    
    print(np.round(source_lat, 1))


imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]
djf_jja_lat = (pre_weighted_lat[expid[i]]['sm'].sel(season = 'DJF') - \
    pre_weighted_lat[expid[i]]['sm'].sel(season = 'JJA')).compute()

np.average(
    djf_jja_lat.values[mask],
    weights = echam6_t63_cellarea.cell_area.values[mask],
)
np.mean(djf_jja_lat.values[mask])


'''
imask = 'AIS'
echam6_t63_ais_mask['mask'][imask].sum()
(mask_high).sum()
(mask_low).sum()

'''
# endregion
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# region plot pre_weighted_lat am/sm


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_sm.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'


pltlevel = np.arange(-60, 60 + 1e-4, 10)
pltticks = np.arange(-60, 60 + 1e-4, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 2)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 3
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = globe_plot(
                ax_org = axs[irow, jcol], add_grid_labels=False)
            plt.text(
                0, 1.05, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='left', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr1 = axs[0, 0].contour(
#     lon, lat,
#     pre_weighted_lat[expid[i]]['ann'].std(
#         dim='time', skipna=True),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[0, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)


for iseason in range(len(seasons)):
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    # plt_ctr = axs[1, iseason].contour(
    #     lon, lat,
    #     pre_weighted_lat[expid[i]]['sea'].sel(
    #         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == \
    #             seasons_last_num[iseason])
    #         ).std(dim='time', skipna=True),
    #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    #     linewidths=0.5, linestyles='solid',
    # )
    # axs[1, iseason].clabel(
    #     plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    #     levels=ctr_level, inline_spacing=10, fontsize=7,)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season=seasons[iseason]) - pre_weighted_lat[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    ttest_fdr_res = ttest_fdr_control(
        pre_weighted_lat[expid[i]]['sea'].sel(
            time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ),
        pre_weighted_lat[expid[i]]['ann'],
        )
    axs[2, iseason].scatter(
        x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
        s=0.5, c='k', marker='.', edgecolors='none',
        transform=ccrs.PlateCarree(),
        )
    
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')


#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][3::4,],
    pre_weighted_lat[expid[i]]['sea'][1::4,],)
axs[0, 1].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM') - pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][0::4,],
    pre_weighted_lat[expid[i]]['sea'][2::4,],)
axs[0, 2].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lat am/sm Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_sm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-10, 10 + 1e-4, 2)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

# ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(
                northextent=-50, ax_org = axs[irow, jcol])
            cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[irow, jcol])
            plt.text(
                0, 0.95, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='center', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr1 = axs[0, 0].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['ann'].std(
#         dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[0, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

for iseason in range(len(seasons)):
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    # plt_ctr = axs[1, iseason].contour(
    #     lon, lat,
    #     pre_weighted_lat[expid[i]]['sea'].sel(
    #         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == \
    #             seasons_last_num[iseason])
    #         ).std(dim='time', skipna=True),
    #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    #     linewidths=0.5, linestyles='solid',
    # )
    # axs[1, iseason].clabel(
    #     plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    #     levels=ctr_level, inline_spacing=10, fontsize=7,)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season=seasons[iseason]) - pre_weighted_lat[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    ttest_fdr_res = ttest_fdr_control(
        pre_weighted_lat[expid[i]]['sea'].sel(
            time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ),
        pre_weighted_lat[expid[i]]['ann'],
        )
    axs[2, iseason].scatter(
        x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
        s=0.5, c='k', marker='.', edgecolors='none',
        transform=ccrs.PlateCarree(),
        )
    
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')


#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][3::4,],
    pre_weighted_lat[expid[i]]['sea'][1::4,],)
axs[0, 1].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM') - pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][0::4,],
    pre_weighted_lat[expid[i]]['sea'][2::4,],)
axs[0, 2].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region

echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

am_height = echam6_t63_surface_height.values[echam6_t63_ais_mask['mask']['AIS']]

am_source_lat = pre_weighted_lat[expid[i]]['am'].values[echam6_t63_ais_mask['mask']['AIS']]

output_png = 'figures/test/trial.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.scatter(
    am_height, am_source_lat, s=3,
)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------
