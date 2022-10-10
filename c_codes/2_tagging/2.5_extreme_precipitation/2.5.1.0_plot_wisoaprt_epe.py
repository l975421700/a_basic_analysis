

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

wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

quantiles = {'90%': 0.9, '95%': 0.95, '99%': 0.99}

epe_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_weighted_lat.pkl', 'rb') as f:
    epe_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot 95% daily precipitation

iqtl = '95%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' daily precipitation percentile_' + iqtl[:2] + ' Antarctica.png'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7.8]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    wisoaprt_epe[expid[i]]['quantiles']['95%'] * seconds_per_d,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    iqtl + ' quantile of\ndaily precipitation [$mm \; day^{-1}$]', linespacing=1.5)
fig.savefig(output_png, dpi=1200)

'''
wisoaprt_epe[expid[i]]['frc_aprt']['am']['99%'].to_netcdf('scratch/test/test.nc')

#---- check fraction of real pre to total pre
tot_pre = (wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d).values
sum_pre = (wisoaprt_epe[expid[i]]['sum_aprt']['am']['original'] / len(wisoaprt_alltime[expid[i]]['daily'].time) * seconds_per_d).values

np.max(abs((tot_pre - sum_pre) / tot_pre))

diff = (tot_pre - sum_pre) / tot_pre
where_max = np.where(abs(diff) == np.max(abs(diff)))
tot_pre[where_max]
sum_pre[where_max]
(tot_pre[where_max] - sum_pre[where_max]) / tot_pre[where_max]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean precipitation

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' aprt am Antarctica.png'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm \; day^{-1}$]', linespacing=1.5)
fig.savefig(output_png, dpi=1200)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region percentile of 5% heaviest precipitation days of total precipitation


iqtl = '95%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' daily precipitation percentile_' + iqtl[:2] + '_frc Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=20, cm_max=40, cm_interval1=2, cm_interval2=2, cmap='PuOr',
    reversed=False)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7.8]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    wisoaprt_epe[expid[i]]['frc_aprt']['am'][iqtl] * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Contribution of 5$\%$ heaviest\nprecipitation to total precipitation [$\%$]', linespacing=1.5)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare three quantiles [90%, 95%, 99%]

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' compare quantiles frc_source_lat Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=50, cm_interval1=5, cm_interval2=5, cmap='PuOr',
    reversed=False)

pltlevel2 = np.arange(0, 10 + 1e-4, 1)
pltticks2 = np.arange(0, 10 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

nrow = 2
ncol = 3

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-50, ax_org = axs[irow, jcol])
        cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat,
                        axs[irow, jcol])
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# Contribution to total precipitation
for icount,iqtl in enumerate(quantiles.keys()):
    plt1 = axs[0, icount].pcolormesh(
        lon,
        lat,
        wisoaprt_epe[expid[i]]['frc_aprt']['am'][iqtl] * 100,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    plt2 = axs[1, icount].pcolormesh(
        lon,
        lat,
        epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    ttest_fdr_res = ttest_fdr_control(
        epe_weighted_lat[expid[i]][iqtl]['ann'],
        pre_weighted_lat[expid[i]]['ann'],)
    axs[1, icount].scatter(
        x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
        s=0.5, c='k', marker='.', edgecolors='none',
        transform=ccrs.PlateCarree(),)

for icount,iqtl in enumerate(quantiles.keys()):
    plt.text(
        0.5, 1.05, iqtl,
        transform=axs[0, icount].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar1.ax.set_xlabel('Contribution to total precipitation [$\%$]', linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar2.ax.set_xlabel('EPE source latitude anomalies [$°$]', linespacing=2)


fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.96)
fig.savefig(output_png)



'''
plt.text(
    -0.05, 0.5, 'Contribution to total precipitation [$\%$]',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')

plt.text(
    -0.05, 0.5, 'EPE source latitude anomalies [$°$]',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')


'''
# endregion
# -----------------------------------------------------------------------------


