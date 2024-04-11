

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    # 'nudged_701_5.0',
    
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

wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoevap_alltime[expid[i]]['am'].lon
lat = wisoevap_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

# major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
# major_ice_core_site = major_ice_core_site.loc[
#     major_ice_core_site['age (kyr)'] > 120, ]

# with open('scratch/products/era5/evap/era5_mon_evap_1979_2021_alltime.pkl', 'rb') as f:
#     era5_mon_evap_1979_2021_alltime = pickle.load(f)

# lon_era5 = era5_mon_evap_1979_2021_alltime['am'].longitude
# lat_era5 = era5_mon_evap_1979_2021_alltime['am'].latitude
# lon_2d_era5, lat_2d_era5 = np.meshgrid(lon_era5, lat_era5,)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot fraction of evap as pre

(wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) / wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) * 100).to_netcdf('scratch/test/test0.nc')


# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot am/sm aprt and evap


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.1_evap/6.1.4.1.0_aprt_evap/6.1.4.1.0 ' + expid[i] + ' aprt and evap am_sm.png'
cbar_label1 = 'Precipitation or Evaporation [$mm \; day^{-1}$]'
# cbar_label2 = 'Evaporation [$mm \; day^{-1}$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

# pltlevel2 = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltticks2 = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
# pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)

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
        if ((irow != 0) | (jcol == 0) | (jcol == 1)):
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
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt_mesh2 = axs[0, 1].pcolormesh(
    lon, lat, wisoevap_alltime[expid[i]]['am'][0] * seconds_per_d * (-1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    
    #-------- sm pre
    axs[1, iseason].pcolormesh(
        lon, lat,
        wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(
            season=seasons[iseason]) * seconds_per_d,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm evap
    axs[2, iseason].pcolormesh(
        lon, lat,
        wisoevap_alltime[expid[i]]['sm'][:, 0].sel(
            season=seasons[iseason]) * seconds_per_d * (-1),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm pre label
    plt.text(
        0.5, 1.05, seasons[iseason] + ' precipitation',
        transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    #-------- sm evap label
    plt.text(
        0.5, 1.05, seasons[iseason] + ' evaporation',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')



plt.text(
    0.5, 1.05, 'Annual mean precipitation', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean evaporation', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)





'''
# wisoaprt_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')
# wisoevap_alltime[expid[i]]['am'].to_netcdf('scratch/test/test2.nc')

#---- std
# stats.describe(wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24, axis=None)
# (wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24 == (wisoaprt_alltime[expid[i]]['ann'][:, 0] * 3600 * 24).std(dim='time')).all()
# np.max(abs(wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24 - (wisoaprt_alltime[expid[i]]['ann'][:, 0] * 3600 * 24).std(dim='time')))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm evap


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.1_evap/6.1.4.1.0_aprt_evap/6.1.4.1.0 ' + expid[i] + ' evap am_sm.png'
cbar_label1 = 'Evaporation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in evaporation [$mm \; day^{-1}$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

pltlevel2 = np.array([-6, -4, -2, -0.5, -0.1, 0, 0.1, 0.5, 2, 4, 6,])
pltticks2 = np.array([-6, -4, -2, -0.5, -0.1, 0, 0.1, 0.5, 2, 4, 6,])
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1)

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
    lon, lat, wisoevap_alltime[expid[i]]['am'][0] * seconds_per_d * (-1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat,
        wisoevap_alltime[expid[i]]['sm'][:, 0].sel(
            season=seasons[iseason]) * seconds_per_d * (-1),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat,
        (wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season=seasons[iseason]) - \
            wisoevap_alltime[expid[i]]['am'][0]) * seconds_per_d * (-1),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    cplot_ttest(
        wisoevap_alltime[expid[i]]['sea'][:, 0].sel(
            time=(wisoevap_alltime[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ),
        wisoevap_alltime[expid[i]]['ann'][:, 0],
        axs[2, iseason], lon_2d, lat_2d,)
    
    #-------- sm pre label
    plt.text(
        0.5, 1.05, seasons[iseason],
        transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    #-------- sm evap label
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    print(seasons[iseason])

#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon, lat,
    (wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') - \
            wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='JJA')
            ) * seconds_per_d * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

cplot_ttest(
    wisoevap_alltime[expid[i]]['sea'][3::4, 0],
    wisoevap_alltime[expid[i]]['sea'][1::4, 0],
    axs[0, 1], lon_2d, lat_2d,)

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat,
    (wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') - \
            wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='SON')
            ) * seconds_per_d * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

cplot_ttest(
    wisoevap_alltime[expid[i]]['sea'][0::4, 0],
    wisoevap_alltime[expid[i]]['sea'][2::4, 0],
    axs[0, 2], lon_2d, lat_2d,)

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
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)


'''
# ttest_fdr_res = ttest_fdr_control(
#     wisoevap_alltime[expid[i]]['sea'][3::4, 0],
#     wisoevap_alltime[expid[i]]['sea'][1::4, 0],)
# axs[0, 1].scatter(
#     x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
#     s=0.5, c='k', marker='.', edgecolors='none', transform=ccrs.PlateCarree(),
#     )
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm ERA5 evap

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.1_evap/6.1.4.1.0_aprt_evap/6.1.4.1.0 era5 evap am_sm.png'
cbar_label1 = 'Evaporation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in evaporation [$mm \; day^{-1}$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

pltlevel2 = np.array([-6, -4, -2, -0.5, -0.1, 0, 0.1, 0.5, 2, 4, 6,])
pltticks2 = np.array([-6, -4, -2, -0.5, -0.1, 0, 0.1, 0.5, 2, 4, 6,])
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1)

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
    lon_era5, lat_era5, era5_mon_evap_1979_2021_alltime['am'] * (-1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon_era5, lat_era5,
        era5_mon_evap_1979_2021_alltime['sm'].sel(
            season=seasons[iseason]) * (-1),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon_era5, lat_era5,
        (era5_mon_evap_1979_2021_alltime['sm'].sel(season=seasons[iseason]) - \
            era5_mon_evap_1979_2021_alltime['am']) * (-1),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    # cplot_ttest(
    #     era5_mon_evap_1979_2021_alltime['sea'].sel(
    #         time=(era5_mon_evap_1979_2021_alltime['sea'].time.dt.month == \
    #             seasons_last_num[iseason])
    #         ),
    #     era5_mon_evap_1979_2021_alltime['ann'],
    #     axs[2, iseason], lon_2d_era5, lat_2d_era5,)
    
    #-------- sm pre label
    plt.text(
        0.5, 1.05, seasons[iseason],
        transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    #-------- sm evap label
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    print(seasons[iseason])

#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon_era5, lat_era5,
    (era5_mon_evap_1979_2021_alltime['sm'].sel(season='DJF') - \
            era5_mon_evap_1979_2021_alltime['sm'].sel(season='JJA')
            ) * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

# cplot_ttest(
#     era5_mon_evap_1979_2021_alltime['sea'][3::4, ],
#     era5_mon_evap_1979_2021_alltime['sea'][1::4, ],
#     axs[0, 1], lon_2d_era5, lat_2d_era5,)

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon_era5, lat_era5,
    (era5_mon_evap_1979_2021_alltime['sm'].sel(season='MAM') - \
            era5_mon_evap_1979_2021_alltime['sm'].sel(season='SON')
            ) * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

# cplot_ttest(
#     era5_mon_evap_1979_2021_alltime['sea'][0::4, ],
#     era5_mon_evap_1979_2021_alltime['sea'][2::4, ],
#     axs[0, 2], lon_2d_era5, lat_2d_era5,)

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
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am evap Antarctica

# output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.1_evap/6.1.4.1 ' + expid[i] + ' evap am Antarctica.png'
output_png = 'figures/test/test.png'

pltlevel = np.array([-0.1, -0.05, 0, 0.05, 0.1, 0.5, 1, 2, 4,])
pltticks = np.array([-0.1, -0.05, 0, 0.05, 0.1, 0.5, 1, 2, 4,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['am'][0] * seconds_per_d * (-1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('Evaporation [$mm \; day^{-1}$]', linespacing=1.5,)
fig.savefig(output_png)

# There is a net deposition over most of Antarctica.

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot DJF-JJA evap Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.1_evap/6.1.4.1 ' + expid[i] + ' evap DJF-JJA Antarctica.png'

pltlevel = np.arange(-100, 100 + 1e-4, 20)
pltticks = np.arange(-100, 100 + 1e-4, 40)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    100 * (wisoevap_alltime[expid[i]]['sm'].sel(season='DJF', wisotype=1) / \
        wisoevap_alltime[expid[i]]['sm'].sel(season='JJA', wisotype=1) - 1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    wisoevap_alltime[expid[i]]['sea'][3::4, 0],
    wisoevap_alltime[expid[i]]['sea'][1::4, 0],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('(DJF/JJA-1) evaporation [$\%$]', linespacing=1.5,)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# region plot aprt and evap Antarctica



#-------- basic set

lon = wisoevap_alltime[expid[i]]['am'].lon
lat = wisoevap_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt and evap am_sm Antarctica.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Evaporation/Precipitation [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-15, 15 + 1e-4, 3)
pltticks2 = np.arange(-15, 15 + 1e-4, 6)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


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
            axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    lon, lat, wisoevap_alltime[expid[i]]['am'][0] * (-100) / wisoaprt_alltime[expid[i]]['am'][0],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- sm pre
axs[1, 0].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- sm evap/pre
plt_mesh2 = axs[2, 0].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean precipitation', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean evaporation/precipitation', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF precipitation', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM precipitation', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA precipitation', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON precipitation', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF evaporation/precipitation', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM evaporation/precipitation', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA evaporation/precipitation', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON evaporation/precipitation', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------



