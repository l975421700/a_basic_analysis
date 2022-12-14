

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
from scipy.stats import circstd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
    remove_trailing_zero_pos_abs
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
)

from a_basic_analysis.b_module.namelist import (
    month,
    monthini,
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

pre_weighted_var_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
    pre_weighted_var_icores[expid[i]] = pickle.load(f)


eight_sites = ['EDC', 'DOME F', 'Vostok', 'EDML',
               'Rothera', 'Halley', 'Neumayer', "Dumont d'Urville"]

wisoaprt_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    wisoaprt_alltime_icores[expid[i]] = pickle.load(f)


'''
pre_weighted_var_icores[expid[i]].keys()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source lat

ivar = 'lat'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = pre_weighted_var_icores[expid[i]][icores][ivar][
        'mon'].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        pre_weighted_var_icores[expid[i]][icores][ivar]['mm'],
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='lower center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source latitude [$°\;S$]')

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.png'
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.pdf'


for icores in Sites:
    # icores = 'EDC'
    print('#----------------' + icores)
    max_var =pre_weighted_var_icores[expid[i]][icores]['lat']['mm'].values.max()
    min_var =pre_weighted_var_icores[expid[i]][icores]['lat']['mm'].values.min()
    print(np.round(max_var - min_var, 1))



# ax.set_xticklabels(monthini)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source sst

ivar = 'sst'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.2_source_sst/6.1.8.2 ' + expid[i] + ' pre_weighted_sst mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = pre_weighted_var_icores[expid[i]][icores][ivar][
        'mon'].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        pre_weighted_var_icores[expid[i]][icores][ivar]['mm'],
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='upper center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source SST [$°C$]')

ax.set_ylabel(None)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.2_source_sst/6.1.8.2 ' + expid[i] + ' pre_weighted_sst mm icores.png'


for icores in Sites:
    # icores = 'EDC'
    print('#----------------' + icores)
    max_sst =pre_weighted_var_icores[expid[i]][icores]['sst']['mm'].values.max()
    min_sst =pre_weighted_var_icores[expid[i]][icores]['sst']['mm'].values.min()
    print(np.round(max_sst - min_sst, 1))



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm relative source lon

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

ivar = 'lon'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.1_source_lon/6.1.8.1 ' + expid[i] + ' pre_weighted_lon mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = \
        calc_lon_diff(
            pre_weighted_var_icores[expid[i]][icores][ivar]['mon'],
            t63_sites_indices[icores]['lon'],
            ).groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        calc_lon_diff(
            pre_weighted_var_icores[expid[i]][icores][ivar]['mm'],
            t63_sites_indices[icores]['lon'],
            ),
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='upper center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly relative source longitude [$°$]')

ax.set_ylabel(None)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# ax.set_yticks(np.arange(-80, 20 + 1e-4, 10))
# ax.set_ylim(-75, 15)

# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.1_source_lon/6.1.8.1 ' + expid[i] + ' pre_weighted_lon mm icores.png'


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source rh2m

ivar = 'rh2m'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.3_source_rh2m/6.1.8.3 ' + expid[i] + ' pre_weighted_rh2m mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = pre_weighted_var_icores[expid[i]][icores][ivar][
        'mon'].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        pre_weighted_var_icores[expid[i]][icores][ivar]['mm'],
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='upper center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source rh2m [$\%$]')

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.3_source_rh2m/6.1.8.3 ' + expid[i] + ' pre_weighted_rh2m mm icores.png'


for icores in Sites:
    # icores = 'EDC'
    print('#----------------' + icores)
    max_var=pre_weighted_var_icores[expid[i]][icores]['rh2m']['mm'].values.max()
    min_var=pre_weighted_var_icores[expid[i]][icores]['rh2m']['mm'].values.min()
    print(np.round(max_var - min_var, 1))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source wind10

ivar = 'wind10'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.4_source_wind10/6.1.8.4 ' + expid[i] + ' pre_weighted_wind10 mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = pre_weighted_var_icores[expid[i]][icores][ivar][
        'mon'].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        pre_weighted_var_icores[expid[i]][icores][ivar]['mm'],
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='lower center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source wind10 [$m \; s^{-1}$]')

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.4_source_wind10/6.1.8.4 ' + expid[i] + ' pre_weighted_wind10 mm icores.png'


for icores in Sites:
    # icores = 'EDC'
    print('#----------------' + icores)
    max_var=pre_weighted_var_icores[expid[i]][icores]['wind10']['mm'].values.max()
    min_var=pre_weighted_var_icores[expid[i]][icores]['wind10']['mm'].values.min()
    print(np.round(max_var - min_var, 1))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm transport distance

ivar = 'distance'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.5_transport_distance/6.1.8.5 ' + expid[i] + ' transport_distance mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = pre_weighted_var_icores[expid[i]][icores][ivar][
        'mon'].groupby('time.month').std(ddof=1) / 100
    
    ax.errorbar(
        month,
        pre_weighted_var_icores[expid[i]][icores][ivar]['mm'] / 100,
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='lower center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source-sink distance [$10^{2} \; km$]')

ax.set_ylabel(None)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.5_transport_distance/6.1.8.5 ' + expid[i] + ' transport_distance mm icores.png'


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm precipitation

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.6_pre/6.1.8.6 ' + expid[i] + ' aprt mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = wisoaprt_alltime_icores[expid[i]][icores]['mon'].sel(wisotype=1).groupby('time.month').std(ddof=1) * seconds_per_d * month_days
    
    ax.errorbar(
        month,
        wisoaprt_alltime_icores[expid[i]][icores]['mm'].sel(wisotype=1) * seconds_per_d * month_days,
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='lower center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly precipitation [$mm \; mon^{-1}$]')

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.set_yscale('log')
ax.set_yticks(np.array([5e-1, 1e0, 5e0, 1e+1, 5e+1, ]))
ax.set_yticklabels([0.5, 1, 5, 10, 50, ])

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)



'''
# Sites = eight_sites
# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.6_pre/6.1.8.6 ' + expid[i] + ' aprt mm icores.png'


(wisoaprt_alltime_icores[expid[i]]['Rothera']['mm'].sel(wisotype=1) * seconds_per_d * month_days).sum()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm precipitation

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.6_pre/6.1.8.6 ' + expid[i] + ' aprt mm EDC and Halley.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = wisoaprt_alltime_icores[expid[i]][icores]['mon'].sel(wisotype=1).groupby('time.month').std(ddof=1) * seconds_per_d * month_days
    
    ax.errorbar(
        month,
        wisoaprt_alltime_icores[expid[i]][icores]['mm'].sel(wisotype=1) * seconds_per_d * month_days,
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='lower center', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly precipitation [$mm \; mon^{-1}$]')

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.set_yscale('log')
ax.set_yticks(np.array([5e-1, 1e0, 5e0, 1e+1, 5e+1, ]))
ax.set_yticklabels([0.5, 1, 5, 10, 50, ])

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm epe days

wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

epe_days_mon = wisoaprt_epe[expid[i]]['mask']['90%'].resample(
    {'time': '1M'}).sum().compute()
epe_days_alltime = mon_sea_ann(var_monthly=epe_days_mon)


# Sites = ['EDC', 'Halley']
# output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1.0_seasonality/6.1.7.1.0 ' + expid[i] + ' monthly epe_days at EDC and Halley.png'

Sites = ['EDC', 'Halley']
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1.0_seasonality/6.1.7.1.0 ' + expid[i] + ' monthly epe_days at EDC and Halley.png'

Sites = ['DOME F', 'Vostok', 'EDML', 'WDC']
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1.0_seasonality/6.1.7.1.0 ' + expid[i] + ' monthly epe_days at DVEW.png'

Sites = ['Rothera', 'Neumayer', 'Law Dome', "Dumont d'Urville"]
output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1.0_seasonality/6.1.7.1.0 ' + expid[i] + ' monthly epe_days at RNLD.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    
    epe_days_mm = epe_days_alltime['mm'][
            :,
            t63_sites_indices[icores]['ilat'],
            t63_sites_indices[icores]['ilon']]
    epe_days_monstd = epe_days_alltime['mon'][
            :,
            t63_sites_indices[icores]['ilat'],
            t63_sites_indices[icores]['ilon']].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        np.arange(0, 12, 1),
        epe_days_mm,
        yerr = epe_days_monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores,)

ax.legend(
    loc='upper left', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly EPE days [$\#$]')
ax.set_xticks(np.arange(0, 12, 1))
ax.set_xticklabels(month)
# ax.set_xlim(-0.5, 11.5)

ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------

