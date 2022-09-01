

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_sst = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sst.pkl', 'rb') as f:
    pre_weighted_sst[expid[i]] = pickle.load(f)

lon = pre_weighted_sst[expid[i]]['am'].lon
lat = pre_weighted_sst[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source sst


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'


pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 1)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)
    plt.text(
        0, 1.05, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt_mesh1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt_mesh2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source sst Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'

pltlevel = np.arange(8, 20 + 1e-4, 1)
pltticks = np.arange(8, 20 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-50, ax_org = axs[jcol])
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)


'''
# ctr_level = np.array([1, 2, 3, 4, 5, ])

# plt_ctr1 = axs[0].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['ann'].std(
#         dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# plt_ctr2 = axs[1].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['sea'].sel(
#         time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == 2)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7)

# plt_ctr3 = axs[2].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['sea'].sel(
#         time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == 8)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[2].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm source sst


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_sm.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'


pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 1)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

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
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


for iseason in range(len(seasons)):
    # iseason = 3
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season=seasons[iseason]) - pre_weighted_sst[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    ttest_fdr_res = ttest_fdr_control(
        pre_weighted_sst[expid[i]]['sea'].sel(
            time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ),
        pre_weighted_sst[expid[i]]['ann'],
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
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[0, 1].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='MAM') - pre_weighted_sst[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][0::4,],
    pre_weighted_sst[expid[i]]['sea'][2::4,],)
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
# region plot am/sm source lat Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_sm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'

pltlevel = np.arange(8, 20 + 1e-4, 1)
pltticks = np.arange(8, 20 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


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
            cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[irow, jcol])
            plt.text(
                0, 0.95, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='center', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season=seasons[iseason]) - pre_weighted_sst[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    ttest_fdr_res = ttest_fdr_control(
        pre_weighted_sst[expid[i]]['sea'].sel(
            time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ),
        pre_weighted_sst[expid[i]]['ann'],
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
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[0, 1].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='MAM') - pre_weighted_sst[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][0::4,],
    pre_weighted_sst[expid[i]]['sea'][2::4,],)
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
# region plot annual mean values

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/6.1.3.2 ' + expid[i] + ' pre_weighted_sst am Antarctica.png'

pltlevel = np.arange(8, 18 + 1e-4, 1)
pltticks = np.arange(8, 18 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

plt1 = ax.pcolormesh(
    pre_weighted_sst[expid[i]]['am'].lon,
    pre_weighted_sst[expid[i]]['am'].lat,
    pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Source SST [$°C$]\n ', linespacing=2)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

