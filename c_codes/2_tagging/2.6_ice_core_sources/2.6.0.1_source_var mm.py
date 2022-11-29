

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

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.png'
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.pdf'


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['lat']['mm'],
        '.-', lw=1, markersize=4,
        label=icores,)

ax.legend(
    loc='lower right', handlelength=2, framealpha = 1, ncol=1,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source latitude [$°\;S$]')
ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
for icores in eight_sites:
    # icores = 'EDC'
    print('#----------------' + icores)
    max_var =pre_weighted_var_icores[expid[i]][icores]['lat']['mm'].values.max()
    min_var =pre_weighted_var_icores[expid[i]][icores]['lat']['mm'].values.min()
    print(np.round(max_var - min_var, 1))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source sst

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.2_source_sst/6.1.8.2 ' + expid[i] + ' pre_weighted_sst mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['sst']['mm'],
        '.-', lw=1, markersize=4,
        label=icores,)

# ax.legend(
#     loc='upper right', handlelength=0.5, framealpha = 0.5, ncol=2,
#     columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source SST [$°C$]')
ax.set_ylabel(None)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
for icores in eight_sites:
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

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.1_source_lon/6.1.8.1 ' + expid[i] + ' pre_weighted_lon mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month,
        calc_lon_diff(
            pre_weighted_var_icores[expid[i]][icores]['lon']['mm'],
            t63_sites_indices[icores]['lon'],
            ),
        '.-', lw=1, markersize=4,
        label=icores,)

# ax.legend(
#     loc='upper left', handlelength=0.5, framealpha = 0.5, ncol=3,
#     columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly relative source longitude [$°$]')
ax.set_ylabel(None)
ax.set_ylim(-110, 110)
ax.set_yticks(np.arange(-90, 90 + 1e-4, 30))
ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source rh2m

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.3_source_rh2m/6.1.8.3 ' + expid[i] + ' pre_weighted_rh2m mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['rh2m']['mm'],
        '.-', lw=1, markersize=4,
        label=icores,)

# ax.legend(
#     loc='upper right', handlelength=0.5, framealpha = 0.5, ncol=2,
#     columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source rh2m [$\%$]')
ax.set_ylabel(None)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
for icores in eight_sites:
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

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.4_source_wind10/6.1.8.4 ' + expid[i] + ' pre_weighted_wind10 mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['wind10']['mm'],
        '.-', lw=1, markersize=4,
        label=icores,)

# ax.legend(
#     loc='lower center', handlelength=0.5, framealpha = 0.5, ncol=2,
#     columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source wind10 [$m \; s^{-1}$]')
ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
for icores in eight_sites:
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

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.5_transport_distance/6.1.8.5 ' + expid[i] + ' transport_distance mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month,
        pre_weighted_var_icores[expid[i]][icores]['distance']['mm'] / 100,
        '.-', lw=1, markersize=4,
        label=icores,)

# ax.legend(
#     loc='lower right', handlelength=0.5, framealpha = 0.5, ncol=2,
#     columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source-sink distance [$10^{2} \; km$]')
ax.set_ylabel(None)
# ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm precipitation

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.6_pre/6.1.8.6 ' + expid[i] + ' aprt mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in eight_sites:
    ax.plot(
        month,
        wisoaprt_alltime_icores[expid[i]][icores]['mm'].sel(wisotype=1) * seconds_per_d * month_days,
        '.-', lw=1, markersize=4,
        label=icores,)

ax.set_xlabel('Monthly precipitation [$mm \; mon^{-1}$]')
ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)
ax.set_yscale('log')

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)



'''
(wisoaprt_alltime_icores[expid[i]]['Rothera']['mm'].sel(wisotype=1) * seconds_per_d * month_days).sum()
'''
# endregion
# -----------------------------------------------------------------------------


