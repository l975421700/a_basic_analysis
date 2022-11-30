

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
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

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
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
    remove_trailing_zero_pos_abs,
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
    calc_lon_diff_np,
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

# epe sources
epe_sources_sites_binned = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_sources_sites_binned.pkl', 'rb') as f:
    epe_sources_sites_binned[expid[i]] = pickle.load(f)

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

wisoaprt_masked_bin_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_masked_bin_icores.pkl',
    'rb') as f:
    wisoaprt_masked_bin_icores[expid[i]] = pickle.load(f)


# normal sources
pre_weighted_var_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
    pre_weighted_var_icores[expid[i]] = pickle.load(f)


wisoaprt_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    wisoaprt_alltime_icores[expid[i]] = pickle.load(f)



'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region precipitation fraction

for isite in stations_sites.Site:
    # isite = 'EDC'
    print('#---- ' + isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.1_frc/6.1.7.2.1 binned heavy precipitation frc at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['frc']['am'].quantiles,
        wisoaprt_masked_bin_icores[expid[i]][isite]['frc']['am'].am * 100,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.vlines(
        90,
        ymin = 0, ymax = 11, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Precipitation fraction [$\%$]', labelpad=0.5)
    # ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_ylim(0, 11)
    ax.set_yticks(np.arange(0, 10 + 1e-4, 2))
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
wisoaprt_masked_bin_icores[expid[i]][isite]['frc']['am'].am.values.sum()

for isite in stations_sites.Site:
    print('#---- ' + isite)
    max_val = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['frc']['am'].am * 100)
    if (max_val > 10):
        print(max_val)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region precipitation rate

for isite in stations_sites.Site:
    # isite = 'Taylor Dome'
    print('#---- ' + isite)
    
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    
    ymax = max_value * 1.1
    ymin = min_value
    
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.2_aprt_rate/6.1.7.2.2 binned heavy precipitation rate at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].quantiles,
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = 0, ymax = 100, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0, y = 0.4)
    # ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_yscale('log')
    ax.set_yticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(AutoMinorLocator(6))
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.25, top=0.95)
    fig.savefig(output_png)






'''
for isite in stations_sites.Site:
    # isite = 'EDC'
    print('#---- ' + isite)
    
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = np.min(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    
    
    print(min_value)
    print(max_value)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source lat

ivar = 'lat'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    # ytickmax = np.floor(ymax)
    # ytickmin = np.ceil(ymin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source latitude [$°\;S$]', labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_ylim(ymin, ymax)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
for isite in stations_sites.Site:
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source sst

ivar = 'sst'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source SST [$°C$]')
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylim(ymin, ymax)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)



'''
ivar = 'sst'
for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source rh2m

ivar = 'rh2m'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.15
    ymin = min_value - 0.15
    # ytickmax = np.ceil(ymax)
    # ytickmin = np.floor(ymin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source rh2m [$\%$]', labelpad = 1)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 0.5))
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
ivar = 'rh2m'
for isite in stations_sites.Site:
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source wind10

ivar = 'wind10'

for isite in stations_sites.Site:
    # isite = 'Taylor Dome'
    print(isite)
    
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.05
    ymin = min_value - 0.05
    # ytickmax = np.ceil(ymax)
    # ytickmin = np.floor(ymin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source wind10 [$m \; s^{-1}$]', labelpad = 0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 0.2))
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.27, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)



'''
ivar = 'wind10'
for isite in stations_sites.Site:
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source lon

ivar = 'lon'

for isite in stations_sites.Site:
    # isite = 'DOME F'
    print(isite)
    
    local_lon = stations_sites.loc[stations_sites.Site==isite].lon.values
    rel_lon = calc_lon_diff_np(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am.values,
        local_lon,)
    
    max_value = np.max(rel_lon)
    min_value = np.min(rel_lon)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    # ytickmax = np.ceil(ymax)
    # ytickmin = np.floor(ymin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        rel_lon,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        calc_lon_diff_np(
            pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values,
            local_lon,),
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Relative source longitude [$°$]', y = 0.4, labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    ax.set_ylim(ymin, ymax)
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
ivar = 'lon'
for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    local_lon = stations_sites.loc[stations_sites.Site==isite].lon.values
    rel_lon = calc_lon_diff_np(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am.values,
        local_lon,)
    
    max_value = np.max(rel_lon)
    min_value = np.min(rel_lon)
    print(np.round(max_value - min_value, 1))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source-sink distance

ivar = 'distance'

for isite in stations_sites.Site:
    # isite = 'Talos'
    print(isite)
    
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    # ytickmax = np.floor(ymax)
    # ytickmin = np.ceil(ymin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].quantiles,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am / 100,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'] / 100,
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        90,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source-sink distance [$10^{2} \; km$]', y=0.4, labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylim(ymin, ymax)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    
    ax.set_xlabel('Percentiles [$\%$]')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 100 + 1e-4, 20))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)


'''
ivar = 'distance'
for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source lat vs. precipitation rate

ivar = 'lat'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # yaxis
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source latitude [$°\;S$]', labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_ylim(ymin, ymax)
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)





'''
for isite in stations_sites.Site:
    print(isite)
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    print(np.round(max_value - min_value, 1))
    print('from ' + str(np.round(min_value, 1)) + ' to ' + \
        str(np.round(max_value, 1)))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source sst vs. precipitation rate

ivar = 'sst'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # yaxis
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source SST [$°C$]')
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylim(ymin, ymax)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source rh2m vs. precipitation rate

ivar = 'rh2m'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # yaxis
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.15
    ymin = min_value - 0.15
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source rh2m [$\%$]', labelpad = 1)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source wind10 vs. precipitation rate

ivar = 'wind10'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # yaxis
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.05
    ymin = min_value - 0.05
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source wind10 [$m \; s^{-1}$]', labelpad = 0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 0.2))
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.27, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source lon vs. precipitation rate

ivar = 'lon'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    local_lon = stations_sites.loc[stations_sites.Site==isite].lon.values
    rel_lon = calc_lon_diff_np(
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am.values,
        local_lon,)
    
    # yaxis
    max_value = np.max(rel_lon)
    min_value = np.min(rel_lon)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        rel_lon,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        calc_lon_diff_np(
            pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values,
            local_lon,),
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Relative source longitude [$°$]', y = 0.4, labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    ax.set_ylim(ymin, ymax)
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source-sink distance vs. precipitation rate

ivar = 'distance'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # yaxis
    max_value = np.max(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    min_value = np.min(epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.max(wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.0_' + ivar + '/6.1.7.2.0 binned_prerate epe_source_anomalies at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_sources_sites_binned[expid[i]][ivar][isite]['am'].am / 100,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'] / 100,
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    ax.vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    ax.set_ylabel('Source-sink distance [$10^{2} \; km$]', y=0.4, labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylim(ymin, ymax)
    # ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    
    ax.set_xlabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
    ax.set_xscale('log')
    ax.set_xticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------




