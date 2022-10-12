

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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am aprt Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' aprt am Antarctica.png'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('Precipitation [$mm \; day^{-1}$]', linespacing=1.5,)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot DJF-JJA aprt Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' aprt DJF-JJA Antarctica.png'

pltlevel = np.arange(-100, 100 + 1e-4, 20)
pltticks = np.arange(-100, 100 + 1e-4, 40)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    100 * (wisoaprt_alltime[expid[i]]['sm'].sel(season='DJF', wisotype=1) / \
        wisoaprt_alltime[expid[i]]['sm'].sel(season='JJA', wisotype=1) - 1),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    wisoaprt_alltime[expid[i]]['sea'][3::4, 0],
    wisoaprt_alltime[expid[i]]['sea'][1::4, 0],
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
cbar.ax.set_xlabel('(DJF/JJA-1) precipitation [$\%$]', linespacing=1.5,)
fig.savefig(output_png)




'''
pltlevel = np.array([-8, -4, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 4, 8])
pltticks = np.array([-8, -4, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 4, 8])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1)

    # (wisoaprt_alltime[expid[i]]['sm'].sel(season='DJF', wisotype=1) - \
    #     wisoaprt_alltime[expid[i]]['sm'].sel(season='JJA', wisotype=1)) * \
    #         seconds_per_d,

stats.describe(temp2_alltime[expid[i]]['am'].sel(lat=slice(-20, -90)),
               axis=None, nan_policy='omit')

'''
# endregion
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# region plot am_sm aprt Antarctica

#-------- basic set

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt am_sm Antarctica.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in precipitation [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
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
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh2 = axs[0, 1].pcolormesh(
    lon, lat,
    (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') - 1) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat,
    (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON') - 1) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

for jcol in range(ncol):
    #-------- sm
    axs[1, jcol].pcolormesh(
        lon, lat,
        wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(
            season=seasons[jcol]) * 3600 * 24,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm/am - 1
    axs[2, jcol].pcolormesh(
        lon, lat,
        (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season=seasons[jcol]) / wisoaprt_alltime[expid[i]]['am'][0] - 1) * 100,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    print(seasons[jcol])


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF/JJA - 1', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM/SON - 1', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

for jcol in range(ncol):
    plt.text(
        0.5, 1.05, seasons[jcol], transform=axs[1, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[jcol] + '/Annual mean - 1',
        transform=axs[2, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot aprt am_sm_5

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' aprt am_sm_5 Antarctica.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

ctr_level = np.array([20, 40, 60, ])

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
    wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0].contour(
    lon, lat.sel(lat=slice(-50, -90)),
    wisoaprt_alltime[expid[i]]['ann'].std(
        dim='time', skipna=True, ddof=1).sel(lat=slice(-50, -90))[0] / \
            wisoaprt_alltime[expid[i]]['am'][0] * 100,
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
        wisoaprt_alltime[expid[i]]['sm'].sel(
            season=seasons[iseason])[0] * 3600 * 24,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt_ctr = axs[1 + iseason].contour(
        lon, lat.sel(lat=slice(-50, -90)),
        wisoaprt_alltime[expid[i]]['sea'].sel(
            time=(wisoaprt_alltime[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ).std(dim='time', skipna=True, ddof=1).sel(
                lat=slice(-50, -90))[0] / \
                wisoaprt_alltime[expid[i]]['sm'].sel(
                    season=seasons[iseason])[0] * 100,
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
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="vertical",shrink=1,aspect=20,extend='max', ticks=pltticks,
    anchor=(1.45, 0.5))
cbar1.ax.set_ylabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 1-fm_right, bottom = 0, top = 0.94)
fig.savefig(output_png)




'''

'''
# endregion
# -----------------------------------------------------------------------------









