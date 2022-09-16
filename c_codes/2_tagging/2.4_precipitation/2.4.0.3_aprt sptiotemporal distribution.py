

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
from statsmodels.stats import multitest
import pycircstat as circ

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
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
    mean_over_ais,
    time_weighted_mean,
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
    cplot_ttest,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset(
    'scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

'''
ocean_pre_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[i]] = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am aprt over Antarctica

#-------- basic settings

pltctr1 = np.array([0.004, 0.008, 0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

plt_data = wisoaprt_alltime[expid[i]]['am'][0].values * seconds_per_d

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' am aprt Antarctica.png'

#-------- plot

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 6.5]) / 2.54, lw=0.1,
    fm_bottom=0.1, fm_top=0.99)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt2 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)

plt3 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

plt.text(
    0.5, -0.08, 'Annual mean precipitation [$mm \; day^{-1}$]',
    transform=ax.transAxes, ha='center', va='center', rotation='horizontal')

fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily aprt PDF/histogram over Antarctica


wisoaprt_daily = wisoaprt_alltime[expid[i]]['daily'][:, 0].copy()
b_cell_area = np.broadcast_to(
    echam6_t63_cellarea.cell_area.values, wisoaprt_daily.shape)

b_mask = {}
for imask in echam6_t63_ais_mask['mask'].keys():
    # imask = 'AIS'
    b_mask[imask] = np.broadcast_to(
        echam6_t63_ais_mask['mask'][imask][None, :, :], wisoaprt_daily.shape)

wisoaprt_daily_pd = pd.DataFrame(columns=('Region', 'daily_pre', 'weights'))
wisoaprt_daily_flatten = {}
b_cell_area_flatten = {}

for imask in echam6_t63_ais_mask['mask'].keys():
    # imask = 'AIS'
    wisoaprt_daily_flatten[imask] = wisoaprt_daily.values[
        b_mask[imask] & (wisoaprt_daily.values >= 2e-8)]
    b_cell_area_flatten[imask] = b_cell_area[
        b_mask[imask] & (wisoaprt_daily.values >= 2e-8)]
    
    wisoaprt_daily_pd = pd.concat(
        [wisoaprt_daily_pd,
         pd.DataFrame(data={
             'Region': imask,
             'daily_pre': wisoaprt_daily_flatten[imask] * seconds_per_d,
             'weights': b_cell_area_flatten[imask], })],
        ignore_index=True,)
    
    print(imask)


#-------- plot PDF

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' daily precipitation PDF over AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

sns.kdeplot(
    data = wisoaprt_daily_pd,
    x= 'daily_pre',
    weights='weights',
    hue='Region',
    palette=['black', 'blue', 'red', 'magenta'],
    hue_order=['AIS', 'EAIS', 'WAIS', 'AP'],
    cut=0, clip=(0, 5),
    linewidth=1,
    common_norm=False,
    # bw_adjust=1,
)

ax.set_xticks(np.arange(0, 5 + 1e-4, 1))
ax.set_yticks(np.arange(0, 3.5 + 1e-4, 0.5))
ax.set_xlim(0, 5)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.set_xlabel('Daily precipitation [$mm \; day^{-1}$]')
ax.set_ylabel('Area-weighted PDF')

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.99)

fig.savefig(output_png)


#-------- plot histogram
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' daily precipitation histogram over AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

sns.histplot(
    x=wisoaprt_daily_flatten['AIS'] * seconds_per_d,
    weights=(b_cell_area_flatten['AIS'] / np.mean(b_cell_area_flatten['AIS'])),
    stat='count',
    bins=np.arange(0, 5 + 1e-4, 0.025), color=['lightgray', ],
    # kde=True,
)

ax.set_xlim(0, 5)
ax.set_xlabel('Daily precipitation [$mm \; day^{-1}$]')
ax.set_ylabel('Area-weighted count of grid cells')

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.95)
fig.savefig(output_png)


'''
#-------- plot histogram
# plt_hist = plt.hist(
#     x=wisoaprt_daily_flatten['AIS'] * seconds_per_d,
#     weights=b_cell_area_flatten['AIS'],
#     color=['lightgray', ],
#     bins=np.arange(0, 5 + 1e-4, 0.01),
#     # density=True,
#     rwidth=1,
# )


stats.describe(wisoaprt_daily_flatten['AIS'] * 3600 * 24, )
np.max(wisoaprt_daily.values[wisoaprt_daily.values < 2e-8])
# 6.4% of grid cells has daily precipitation

#  & (wisoaprt_daily.values * 3600 * 24 < 0.05)
    # bins=np.arange(0, 0.05 + 1e-4, 0.002),
# ax.set_xticks(np.arange(0, 0.05 + 1e-4, 0.01))
# ax.set_xticklabels(np.arange(0, 0.05 + 1e-4, 0.01), size=8)
# ax.set_yticks(np.arange(0, 6 + 1e-4, 1))
# ax.set_yticklabels(np.arange(0, 6 + 1e-4, 1), size=8)

# np.min(wisoaprt_daily_pd.daily_pre)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily aprt and daily relative pre_weighted_lon


#-------------------------------- basic settings

itime_start_djf = np.where(aprt_geo7_alltime[expid[i]]['daily'].time == np.datetime64('2058-12-01T23:52:30'))[0][0]
itime_end_son = np.where(aprt_geo7_alltime[expid[i]]['daily'].time == np.datetime64('2059-12-01T23:52:30'))[0][0]

# pltlevel = np.arange(0, 360 + 1e-4, 20)
# pltticks = np.arange(0, 360 + 1e-4, 60)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 45)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

#-------------------------------- plot
itime = 1
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' daily aprt and pre_weighted_lon.png'

pltctr1 = np.array([0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_pre = aprt_geo7_alltime[expid[i]]['daily'][itime_start_djf + itime,].sel(
    wisotype=[19, 21]).sum(dim='wisotype') * seconds_per_d

plt2 = ax.contour(
    lon, lat.sel(lat=slice(-50, -90)),
    plt_pre.sel(lat=slice(-50, -90)),
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)


plt3 = ax.contour(
    lon, lat.sel(lat=slice(-50, -90)),
    plt_pre.sel(lat=slice(-50, -90)),
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

plt_mesh = ax.pcolormesh(
    lon, lat.sel(lat=slice(-50, -90)),
    (calc_lon_diff(pre_weighted_lon[expid[i]]['daily'][
        itime_start_djf + itime,], lon_2d)).sel(lat=slice(-50, -90)),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), )

plt.text(
    0, 0.95, str(plt_pre.time.values)[5:10], transform=ax.transAxes,
    ha='left', va='bottom', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Precipitation-weighted open-oceanic\n relative source longitude [$°$]', linespacing=1.5)
fig.savefig(output_png)


'''
pre_weighted_lon[expid[i]]['daily'].time[itime_start_djf].values
aprt_geo7_alltime[expid[i]]['daily'].time[itime_start_djf].values

pre_weighted_lon[expid[i]]['daily'].time[itime_end_son].values
aprt_geo7_alltime[expid[i]]['daily'].time[itime_end_son].values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate daily aprt and daily relative pre_weighted_lon


#-------------------------------- basic settings

itime_start_djf = np.where(aprt_geo7_alltime[expid[i]]['daily'].time == np.datetime64('2058-12-01T23:52:30'))[0][0]
itime_end_son = np.where(aprt_geo7_alltime[expid[i]]['daily'].time == np.datetime64('2059-12-01T23:52:30'))[0][0]

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 45)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


output_mp4 = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' daily aprt and relative pre_weighted_lon 1201_1130.mp4'

ctr_lev1 = np.array([0.5, 1, ])
ctr_lev2 = np.array([2, 4, 8, ])

#-------------------------------- plot

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Precipitation-weighted open-oceanic\nrelative source longitude [$°$]', linespacing=1.5)

plt_objs = []

def update_frames(itime):
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_data = aprt_geo7_alltime[expid[i]]['daily'][
        itime_start_djf + itime,].sel(
            wisotype=[19, 21]).sum(dim='wisotype') * seconds_per_d
    
    plt_ctr1 = ax.contour(
        lon,
        lat.sel(lat=slice(-50, -90)),
        plt_data.sel(lat=slice(-50, -90)),
        levels=ctr_lev1, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='dotted',
    )
    plt_ctr1.__class__ = mpl.contour.QuadContourSet
    plt_lab1 = ax.clabel(
        plt_ctr1, inline=True, colors='b', fmt=remove_trailing_zero,
        levels=ctr_lev1, inline_spacing=10, fontsize=7,)
    
    plt_ctr2 = ax.contour(
        lon,
        lat.sel(lat=slice(-50, -90)),
        plt_data.sel(lat=slice(-50, -90)),
        levels=ctr_lev2, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='solid',
    )
    plt_ctr2.__class__ = mpl.contour.QuadContourSet
    plt_lab2 = ax.clabel(
        plt_ctr2, inline=True, colors='b', fmt=remove_trailing_zero,
        levels=ctr_lev2, inline_spacing=10, fontsize=7,)
    
    plt_mesh = ax.pcolormesh(
        lon,
        lat.sel(lat=slice(-50, -90)),
        (calc_lon_diff(pre_weighted_lon[expid[i]]['daily'][itime_start_djf + itime,], lon_2d)).sel(lat=slice(-50, -90)),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), )
    
    plt_txt = plt.text(
        0, 0.95, str(plt_data.time.values)[5:10], transform=ax.transAxes,
        ha='left', va='bottom', rotation='horizontal')
    
    plt_objs = plt_ctr1.collections + plt_ctr2.collections + \
        plt_lab1 + plt_lab2 + [plt_mesh, plt_txt, ]
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=366, interval=1000, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am d018

smow_d018 = 0.22279967
smow_dD = 0.3288266

do18 = (((wisoaprt_alltime[expid[i]]['am'][1] / \
    wisoaprt_alltime[expid[i]]['am'][0]) / smow_d018 - 1) * 1000).compute()


pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])

plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d

output_png = 'figures/6_awi/6.1_echam6/6.1.6_isotopes/6.1.6.0 ' + expid[i] + ' am d_018 Antarctica.png'

pltlevel = np.arange(-55, -5 + 1e-4, 5)
pltticks = np.arange(-55, -5 + 1e-4, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54,
    llatlabel = True,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt2 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=9,)

plt3 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=5, fontsize=9,)

plt1 = ax.pcolormesh(
    lon,
    lat,
    do18,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

turner_obs = pd.read_csv(
    'finse_school/data/Antarctic_site_records/Turner_obs.csv')
cplot_ice_cores(np.negative(turner_obs.lon), turner_obs.lat, ax, marker='*',
                edgecolors = 'red', zorder=3, s=5, c='red')

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Annual mean $\delta^{18}O$ [‰]\n ', linespacing=2)
fig.savefig(output_png)








'''
#-------- SMOW values
smow_standard = xr.open_dataset('/work/ollie/qigao001/startdump/wiso/calc_wiso_d/SMOW.FAC.T63.nwiso_3.nc')

(smow_standard.smow[0, 0]).max().values == (smow_standard.smow[0, 0]).min().values
(smow_standard.smow[0, 1]).max().values == (smow_standard.smow[0, 1]).min().values
(smow_standard.smow[0, 2]).max().values == (smow_standard.smow[0, 2]).min().values

#-------- check with initial plot

pltlevel = np.arange(-50, 0.01, 0.1)
pltticks = np.arange(-50, 0.01, 5)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    lon,
    lat,
    do18,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ [‰]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5,)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/trial.png',)

'''
# endregion
# -----------------------------------------------------------------------------

