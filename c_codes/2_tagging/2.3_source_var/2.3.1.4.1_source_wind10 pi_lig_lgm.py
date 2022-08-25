

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
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import pre_weighted_wind10

pre_weighted_wind10 = {}

expid = [
    'pi_m_416_4.9',
    'lig_m_000_4.9',
    'lgm_m_000_4.9',
]

exp_odir = [
    'output/echam-6.3.05p2-wiso/pi/',
    'output/echam-6.3.05p2-wiso/lig/',
    'output/echam-6.3.05p2-wiso/lgm/',
]

for i in range(len(expid)):
    # i = 0
    with open(exp_odir[i] + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.pkl', 'rb') as f:
        pre_weighted_wind10[expid[i]] = pickle.load(f)

    print(exp_odir[i] + '    ' + expid[i])


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm pre_weighted_wind10

#-------- basic set

lon = pre_weighted_wind10[expid[0]]['am'].lon
lat = pre_weighted_wind10[expid[0]]['am'].lat

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.4_wind10/' + '6.1.3.4 pi_lig_lgm ' + expid[0] + ' ' + expid[1] + ' ' + expid[2] + ' pre_weighted_wind10 am_sm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'

pltlevel = np.arange(8, 13 + 1e-4, 0.5)
pltticks = np.arange(8, 13 + 1e-4, 0.5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

nrow = 3
ncol = 5
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-45, ax_org = axs[irow, jcol])

for i in range(len(expid)):
    #-------- am
    # i = 2
    print('i = ' + str(i))
    plt_mesh1 = axs[i, 0].pcolormesh(
        lon, lat,
        pre_weighted_wind10[expid[i]]['am'],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    # plt_ctr1 = axs[i, 0].contour(
    #     lon, lat.sel(lat=slice(-45, -90)),
    #     pre_weighted_wind10[expid[i]]['ann'].std(
    #         dim='time', skipna=True).sel(lat=slice(-45, -90)),
    #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    #     linewidths=0.5, linestyles='solid',
    # )
    # axs[i, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
    #                  levels=ctr_level, inline_spacing=10, fontsize=7,)
    
    #-------- sm
    for iseason in range(len(seasons)):
        # iseason = 3
        axs[i, iseason + 1].pcolormesh(
            lon, lat,
            pre_weighted_wind10[expid[i]]['sm'].sel(season=seasons[iseason]),
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        # plt_ctr2 = axs[i, iseason + 1].contour(
        #     lon, lat.sel(lat=slice(-45, -90)),
        #     pre_weighted_wind10[expid[i]]['sea'].sel(
        #         time=(pre_weighted_wind10[expid[i]]['sea'].time.dt.month == \
        #             seasons_last_num[iseason])
        #         ).std(dim='time', skipna=True).sel(lat=slice(-45, -90)),
        #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
        #     linewidths=0.5, linestyles='solid',
        # )
        # axs[i, iseason + 1].clabel(
        #     plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
        #     levels=ctr_level, inline_spacing=10, fontsize=7,)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

for iseason in range(len(seasons)):
    # iseason = 3
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[0, 1 + iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'PI', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'LIG', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'LGM', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.02, right = 0.99, bottom = fm_bottom*0.8, top = 0.97)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm diff pre_weighted_wind10

#-------- basic set

lon = pre_weighted_wind10[expid[0]]['am'].lon
lat = pre_weighted_wind10[expid[0]]['am'].lat

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.4_wind10/' + '6.1.3.4 pi_lig_lgm ' + expid[0] + ' ' + expid[1] + ' ' + expid[2] + ' pre_weighted_wind10 am_sm diff Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'

pltlevel = np.arange(8, 13 + 1e-4, 0.5)
pltticks = np.arange(8, 13 + 1e-4, 0.5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-2, 2 + 1e-4, 0.5)
pltticks2 = np.arange(-2, 2 + 1e-4, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

# ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 3
ncol = 5
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-45, ax_org = axs[irow, jcol])

i = 0
plt_mesh1 = axs[i, 0].pcolormesh(
    lon, lat,
    pre_weighted_wind10[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr1 = axs[i, 0].contour(
#     lon, lat.sel(lat=slice(-45, -90)),
#     pre_weighted_wind10[expid[i]]['ann'].std(
#         dim='time', skipna=True).sel(lat=slice(-45, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[i, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#                  levels=ctr_level, inline_spacing=10, fontsize=7,)

#-------- sm
for iseason in range(len(seasons)):
    # iseason = 3
    axs[i, iseason + 1].pcolormesh(
        lon, lat,
        pre_weighted_wind10[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    # plt_ctr2 = axs[i, iseason + 1].contour(
    #     lon, lat.sel(lat=slice(-45, -90)),
    #     pre_weighted_wind10[expid[i]]['sea'].sel(
    #         time=(pre_weighted_wind10[expid[i]]['sea'].time.dt.month == \
    #             seasons_last_num[iseason])
    #         ).std(dim='time', skipna=True).sel(lat=slice(-45, -90)),
    #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    #     linewidths=0.5, linestyles='solid',
    # )
    # axs[i, iseason + 1].clabel(
    #     plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
    #     levels=ctr_level, inline_spacing=10, fontsize=7,)
    
    print(seasons[iseason])


for i in np.arange(1, len(expid)):
    #-------- am
    # i = 2
    print('i = ' + str(i))
    plt_mesh2 = axs[i, 0].pcolormesh(
        lon, lat,
        pre_weighted_wind10[expid[i]]['am'] - pre_weighted_wind10[expid[0]]['am'],
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    #-------- sm
    for iseason in range(len(seasons)):
        # iseason = 2
        axs[i, iseason + 1].pcolormesh(
            lon, lat,
            pre_weighted_wind10[expid[i]]['sm'].sel(season=seasons[iseason]) - \
                pre_weighted_wind10[expid[0]]['sm'].sel(season=seasons[iseason]),
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

for iseason in range(len(seasons)):
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[0, 1 + iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'PI', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'LIG - PI', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'LGM - PI', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.02, right = 0.99, bottom = fm_bottom*0.8, top = 0.97)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------

