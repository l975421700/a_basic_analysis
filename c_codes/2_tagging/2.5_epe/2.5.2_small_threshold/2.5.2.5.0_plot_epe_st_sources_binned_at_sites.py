

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
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
from scipy.stats import pearsonr

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
    ten_sites_names,
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

# epe_st sources
epe_st_sources_sites_binned = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_sources_sites_binned.pkl', 'rb') as f:
    epe_st_sources_sites_binned[expid[i]] = pickle.load(f)

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

wisoaprt_masked_bin_st_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_masked_bin_st_icores.pkl',
    'rb') as f:
    wisoaprt_masked_bin_st_icores[expid[i]] = pickle.load(f)


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
wisoaprt_mask_bin_st_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_mask_bin_st_icores.pkl',
    'rb') as f:
    wisoaprt_mask_bin_st_icores[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region subset source properties against precipitation rates

# stations_sites.Site
# Sites = ['EDC', 'Halley']
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies_subset at EDC and Halley.png'

# Sites = ['DOME F', 'Vostok', 'EDML', 'WDC']
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies_subset at DOME F, Vostok, EDML, and WDC.png'


Sites = ['Rothera', 'Neumayer', "Law Dome", "Dumont d'Urville"]
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies_subset at Rothera, Neumayer, Law Dome, and Dumont.png'

ncol = 3
nrow = len(Sites)

wspace = 0.4
hspace = 0.4
fm_left = wspace / ncol * 0.7
fm_bottom = hspace / nrow * 0.6
fm_right = 1 - fm_left / 3
fm_top = 1 - fm_bottom / 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4 * ncol, 4.2 * nrow]) / 2.54,
    )

# plot panel labels
for jcol in range(ncol):
    for irow in range(nrow):
        plt.text(
            -0.2, 1.09, panel_labels[irow][:2] + str(jcol + 1) + ')',
            transform=axs[irow, jcol].transAxes)


#---------------- source latitude

ivar = 'lat'
print('#-------- ' + ivar)
jcol = 0

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source latitude [$°\;S$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- relative source longitude

ivar = 'lon'
print('#-------- ' + ivar)
jcol = 1

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    local_lon = stations_sites.loc[stations_sites.Site==isite].lon.values
    rel_lon = calc_lon_diff_np(
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am.values,
        local_lon,)
    
    # yaxis
    max_value = np.nanmax(rel_lon)
    min_value = np.nanmin(rel_lon)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        rel_lon,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        calc_lon_diff_np(
            pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values,
            local_lon,),
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel(
            'Relative source longitude [$°$]', y = 0.4, labelpad=0,)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')


#---------------- source wind10

ivar = 'wind10'
print('#-------- ' + ivar)
jcol = 2

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.05
    ymin = min_value - 0.05
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source wind10 [$m \; s^{-1}$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    # axs[irow, jcol].invert_yaxis()
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

fig.subplots_adjust(
    left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top,
    wspace=wspace, hspace=hspace)

fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region precipitation fraction

mpl.rc('font', family='Times New Roman', size=10)

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.1_frc/6.1.7.2.1 st_binned heavy precipitation frc at EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for isite in Sites:
    # isite = 'EDC'
    ax.plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].quantiles,
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].am * 100,
        '.-', lw=0.5, markersize=1.5,
        label=isite,
        )

ax.legend(
    loc='upper left', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

# ax.vlines(
#     90,
#     ymin = 0, ymax = 11, lw=0.5, linestyles='--', colors='gray')

ax.set_ylabel('Precipitation fraction [$\%$]', labelpad=0.5)
# ax.set_ylim(0, 11)
# ax.set_yticks(np.arange(0, 10 + 1e-4, 1))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Percentiles [$\%$]')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 100 + 1e-4, 10))
ax.grid(True, which = 'both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)




mpl.rc('font', family='Times New Roman', size=10)

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.1_frc/6.1.7.2.1 st_binned heavy precipitation frc_cumulative at EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for isite in Sites:
    # isite = 'EDC'
    ax.plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].quantiles,
        np.cumsum(wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].am * 100),
        '.-', lw=0.5, markersize=1.5,
        label=isite,
        )

ax.legend(
    loc='upper left', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_ylabel('Cumulative precipitation fraction [$\%$]', labelpad=0.5)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 100 + 1e-4, 10))

ax.set_xlabel('Percentiles [$\%$]')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 100 + 1e-4, 10))
ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)





for isite in stations_sites.Site:
    # isite = 'EDC'
    print('#---- ' + isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.1_frc/6.1.7.2.1 binned heavy precipitation frc at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].quantiles,
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].am * 100,
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
wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].am.values.sum()

for isite in stations_sites.Site:
    print('#---- ' + isite)
    max_val = np.max(wisoaprt_masked_bin_st_icores[expid[i]][isite]['frc']['am'].am * 100)
    if (max_val > 10):
        print(max_val)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region precipitation rate



mpl.rc('font', family='Times New Roman', size=10)

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.2_aprt_rate/6.1.7.2.2 st_binned heavy precipitation rate at EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)


ymin = 10**-3
max_value = 0

for isite in Sites:
    # isite = 'EDC'
    
    max_value = np.max((
        max_value,
        np.max(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
        ))
    
    plt_line = ax.plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].quantiles,
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        '.-', lw=0.5, markersize=1.5,
        label=isite,
        )
    
    ax.hlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        xmin = 0, xmax = 100, lw=0.5, linestyles='--',
        colors=plt_line[0].get_color())

# ax.vlines(
#     90,
#     ymin = 0, ymax = 100, lw=0.5, linestyles='--', colors='gray')

ax.legend(
    loc='upper left', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ymax = max_value * 1.1

ax.set_ylabel('Precipitation rate [$mm \; day^{-1}$]', labelpad=0.5)
ax.set_yscale('log')
# ax.set_yticks(np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
# ax.set_yticklabels(['1e-3', '1e-2', '1e-1', '1e0', '1e+1', '1e+2', ])
# ax.set_yticklabels([
#     '1e-3', '5e-3', '1e-2', '5e-2', '1e-1', '5e-1',
#     '1e0', '5e0', '1e+1', '5e+1', '1e+2', ])

ax.set_yticks(np.array([
    1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e+1, 5e+1, 1e+2, ]))
ax.set_yticklabels([
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, ])
ax.set_ylim(ymin, ymax)

ax.set_xlabel('Percentiles [$\%$]')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 100 + 1e-4, 10))
ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)






for isite in stations_sites.Site:
    # isite = 'Taylor Dome'
    print('#---- ' + isite)
    
    max_value = np.max(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    
    ymax = max_value * 1.1
    ymin = min_value
    
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2.2_aprt_rate/6.1.7.2.2 binned heavy precipitation rate at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].quantiles,
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
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
    ax.set_yticklabels([
        '1e-3', '1e-2', '1e-1', '1e0', '1e+1', '1e+2'
    ])
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
    
    max_value = np.max(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = np.min(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    
    
    print(min_value)
    print(max_value)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source properties against precipitation rates

# Sites = ['EDC', 'Halley']
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies at EDC and Halley.png'

# Sites = ['DOME F', 'Vostok', 'EDML', 'WDC']
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies at DOME F, Vostok, EDML, and WDC.png'

Sites = ['Rothera', 'Neumayer', "Law Dome", "Dumont d'Urville"]
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_st_source_anomalies at Rothera, Neumayer, Law Dome, and Dumont.png'

ncol = 6
nrow = len(Sites)

wspace = 0.4
hspace = 0.4
fm_left = wspace / ncol * 0.7
fm_bottom = hspace / nrow * 0.6
fm_right = 1 - fm_left / 3
fm_top = 1 - fm_bottom / 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4 * ncol, 4.2 * nrow]) / 2.54,
    )

# plot panel labels
for jcol in range(ncol):
    for irow in range(nrow):
        plt.text(
            -0.2, 1.09, panel_labels[irow][:2] + str(jcol + 1) + ')',
            transform=axs[irow, jcol].transAxes)


#---------------- source latitude

ivar = 'lat'
print('#-------- ' + ivar)
jcol = 0

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source latitude [$°\;S$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- relative source longitude

ivar = 'lon'
print('#-------- ' + ivar)
jcol = 1

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    local_lon = stations_sites.loc[stations_sites.Site==isite].lon.values
    rel_lon = calc_lon_diff_np(
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am.values,
        local_lon,)
    
    # yaxis
    max_value = np.nanmax(rel_lon)
    min_value = np.nanmin(rel_lon)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        rel_lon,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        calc_lon_diff_np(
            pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values,
            local_lon,),
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel(
            'Relative source longitude [$°$]', y = 0.4, labelpad=0,)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- source-sink distance

ivar = 'distance'
print('#-------- ' + ivar)
jcol = 2

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am) / 100
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am / 100,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'] / 100,
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel(
            'Source-sink distance [$10^{2} \; km$]', y=0.4, labelpad=0,)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- source sst

ivar = 'sst'
print('#-------- ' + ivar)
jcol = 3

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.25
    ymin = min_value - 0.25
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source SST [$°C$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- source rh2m

ivar = 'rh2m'
print('#-------- ' + ivar)
jcol = 4

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.15
    ymin = min_value - 0.15
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source rh2m [$\%$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    # axs[irow, jcol].invert_yaxis()
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

#---------------- source wind10

ivar = 'wind10'
print('#-------- ' + ivar)
jcol = 5

for irow, isite in enumerate(Sites):
    print('#---- ' + str(irow) + ': ' + isite)
    
    # yaxis
    max_value = np.nanmax(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    min_value = np.nanmin(epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am)
    ymax = max_value + 0.05
    ymin = min_value - 0.05
    
    # xaxis
    max_value = np.nanmax(wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am.values * seconds_per_d)
    min_value = 10**-3
    xmax = max_value * 1.1
    xmin = min_value
    
    axs[irow, jcol].plot(
        wisoaprt_masked_bin_st_icores[expid[i]][isite]['meannan']['am'].am * seconds_per_d,
        epe_st_sources_sites_binned[expid[i]][ivar][isite]['am'].am,
        '.-', lw=0.5, markersize=1.5,
        )
    plt_text = plt.text(
        0.05, 0.9, isite,
        transform=axs[irow, jcol].transAxes, color='gray',)
    axs[irow, jcol].hlines(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        xmin = 0, xmax = 100, lw=0.5, linestyles='--')
    axs[irow, jcol].vlines(
        wisoaprt_alltime_icores[expid[i]][isite]['am'].sel(wisotype=1) * seconds_per_d,
        ymin = ymin, ymax = ymax, lw=0.5, linestyles='--', colors='gray')
    
    if True: # (irow==0):
        axs[irow, jcol].set_ylabel('Source wind10 [$m \; s^{-1}$]', labelpad=0)
    axs[irow, jcol].yaxis.set_major_formatter(remove_trailing_zero_pos)
    axs[irow, jcol].set_ylim(ymin, ymax)
    # axs[irow, jcol].invert_yaxis()
    axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if True: # (irow==(nrow-1)):
        axs[irow, jcol].set_xlabel(
            'Precipitation rate [$mm \; day^{-1}$]', labelpad=0)
    axs[irow, jcol].set_xscale('log')
    axs[irow, jcol].set_xticks(
        np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, ]))
    axs[irow, jcol].set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100,])
    axs[irow, jcol].set_xlim(xmin, xmax)
    axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[irow, jcol].grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')

fig.subplots_adjust(
    left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top,
    wspace=wspace, hspace=hspace)

fig.savefig(output_png)



'''
# Sites = ['Talos', 'Taylor Dome', 'RICE', "Siple Dome"]
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_source_anomalies at TTRS.png'

# Sites = ['WDC', 'Byrd', 'James Ross', "Fletcher"]
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_source_anomalies at WBJF.png'

# Sites = ['Berkner', 'Dome A', 'DOME B', "Law Dome"]
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.2_pre_source_sites/6.1.7.2 binned_prerate epe_source_anomalies at BDDL.png'

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between source lat & wind10

epe_st_sources_sites_binned[expid[i]].keys()

isite = 'EDC'

pearsonr(
    epe_st_sources_sites_binned[expid[i]]['lat'][isite]['am'].am,
    epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am,
)

isite = 'Halley'

pearsonr(
    epe_st_sources_sites_binned[expid[i]]['lat'][isite]['am'].am,
    epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am,
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation of source properties

for isite in ['EDC', 'Halley']:
    # isite = 'EDC'
    
    for ivar2 in ['sst', 'rh2m', 'distance']:
        # ivar2 = 'sst'
        
        print('#---------------- ' + isite + ': lat vs. ' + ivar2)
        
        print(pearsonr(
            epe_st_sources_sites_binned[expid[i]]['lat'][isite]['am'].am,
            epe_st_sources_sites_binned[expid[i]][ivar2][isite]['am'].am,
        ))




# endregion
# -----------------------------------------------------------------------------



