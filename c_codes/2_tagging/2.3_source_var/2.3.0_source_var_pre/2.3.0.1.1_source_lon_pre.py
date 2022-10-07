

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
from scipy.stats import circstd

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_lon = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

lon = pre_weighted_lon[expid[i]]['am'].lon
lat = pre_weighted_lon[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

'''
pre_weighted_sinlon = {}
pre_weighted_coslon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon.pkl', 'rb') as f:
    pre_weighted_sinlon[expid[i]] = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon.pkl', 'rb') as f:
    pre_weighted_coslon[expid[i]] = pickle.load(f)

# pre_weighted_lon[expid[i]]['am'].to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lon am/sm


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_sm.png'
cbar_label1 = 'Precipitation-weighted open-oceanic relative source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'


pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PRGn', len(pltlevel2)-1).reversed()


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
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


for iseason in range(len(seasons)):
    # iseason = 3
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat,
        calc_lon_diff(
            pre_weighted_lon[expid[i]]['sm'].sel(season=seasons[iseason]),
            lon_2d),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat,
        calc_lon_diff(
            pre_weighted_lon[expid[i]]['sm'].sel(season=seasons[iseason]),
            pre_weighted_lon[expid[i]]['am'],),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    wwtest_res = circ.watson_williams(
        pre_weighted_lon[expid[i]]['sea'].sel(
            time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])).values * np.pi / 180,
        pre_weighted_lon[expid[i]]['ann'].values * np.pi / 180,
        axis=0,
        )[0] < 0.05
    axs[2, iseason].scatter(
        x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
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

    print(seasons[iseason])


#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'),),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[0, 1].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='MAM'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='SON'),),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 5)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 11)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[0, 2].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
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
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
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
# region plot pre_weighted_lon am/sm Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_sm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic relative source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-30, 30 + 1e-4, 5)
pltticks2 = np.arange(-30, 30 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PRGn', len(pltlevel2)-1).reversed()


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
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    #-------- sm
    axs[1, iseason].pcolormesh(
        lon, lat,
        calc_lon_diff(
            pre_weighted_lon[expid[i]]['sm'].sel(season=seasons[iseason]),
            lon_2d),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm - am
    plt_mesh2 = axs[2, iseason].pcolormesh(
        lon, lat,
        calc_lon_diff(
            pre_weighted_lon[expid[i]]['sm'].sel(season=seasons[iseason]),
            pre_weighted_lon[expid[i]]['am']),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    wwtest_res = circ.watson_williams(
        pre_weighted_lon[expid[i]]['sea'].sel(
            time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])).values * np.pi / 180,
        pre_weighted_lon[expid[i]]['ann'].values * np.pi / 180,
        axis=0,
        )[0] < 0.05
    axs[2, iseason].scatter(
        x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
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
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'),),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[0, 1].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )


#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='MAM'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='SON'),),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 5)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 11)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[0, 2].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
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
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
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
# region plot pre_weighted_lon am

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon am Antarctica.png'

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 45)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('twilight_shifted', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Relative source longitude [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lon am + am aprt


#-------------------------------- add am pre

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon am Antarctica + am aprt.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-180, cm_max=180, cm_interval1=30, cm_interval2=60,
    cmap='twilight_shifted',)

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# plot am aprt
pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])
plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d

plt2 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=6,)

plt3 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=5, fontsize=6,)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Relative source longitude [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lon DJF-JJA

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon DJF-JJA Antarctica.png'

pltlevel = np.arange(-30, 30 + 1e-4, 5)
pltticks = np.arange(-30, 30 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'),),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
ax.scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('DJF - JJA source longitude [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_lon am_sm

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_sm_5 Antarctica.png'
cbar_label1 = 'Relative source longitude [$°$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-180, cm_max=180, cm_interval1=30, cm_interval2=60,
    cmap='twilight_shifted',)
ctr_level = np.array([4, 8, 12, 16, ])

nrow = 1
ncol = 5
fm_right = 2 / (5.8*ncol + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol + 2, 5.8*nrow+0.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=-50, ax_org = axs[jcol],
        l45label = False, loceanarcs = False)
    cplot_ice_cores(
        major_ice_core_site.lon, major_ice_core_site.lat, axs[jcol])

#-------- Am
plt_mesh1 = axs[0].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0].contour(
    lon, lat.sel(lat=slice(-50, -90)),
    circstd(pre_weighted_lon[expid[i]]['ann'].sel(lat=slice(-50, -90)),
            high=360, low=0, axis=0, nan_policy='omit'),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',)
axs[0].clabel(
    plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=6,)
plt.text(
    0.5, 1.04, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

#-------- sm
for iseason in range(len(seasons)):
    axs[1 + iseason].pcolormesh(
        lon, lat,
        calc_lon_diff(
            pre_weighted_lon[expid[i]]['sm'].sel(season=seasons[iseason]),
            lon_2d),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt_ctr = axs[1 + iseason].contour(
        lon, lat.sel(lat=slice(-50, -90)),
        circstd(
            pre_weighted_lon[expid[i]]['sea'].sel(
            time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ).sel(lat=slice(-50, -90)),
            high=360, low=0, axis=0, nan_policy='omit'),
        levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='solid',
    )
    axs[1 + iseason].clabel(
        plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
        levels=ctr_level, inline_spacing=10, fontsize=6,)
    
    plt.text(
        0.5, 1.04, seasons[iseason], transform=axs[1 + iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1,aspect=20,extend='neither', ticks=pltticks,
    anchor=(1.45, 0.5))
cbar1.ax.set_ylabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 1-fm_right, bottom = 0, top = 0.94)
fig.savefig(output_png)




'''
# both are the same
        circstd(calc_lon_diff(pre_weighted_lon[expid[i]]['ann'], lon_2d).sel(lat=slice(-50, -90)),
                high=180, low=-180, axis=0, nan_policy='omit'),
    circstd(pre_weighted_lon[expid[i]]['ann'].sel(lat=slice(-50, -90)),
            high=360, low=0, axis=0, nan_policy='omit'),

'''
# endregion
# -----------------------------------------------------------------------------



