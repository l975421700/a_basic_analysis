

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

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.loc_indices.pkl',
    'rb') as f:
    loc_indices = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source lat

# output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.png'
output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.0_source_lat/6.1.8.0 ' + expid[i] + ' pre_weighted_lat mm icores.pdf'


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['lat']['mm'],
        '.-', lw=0.5, markersize=4,
        label=icores,)

ax.legend(
    loc='lower right', handlelength=2, framealpha = 1, ncol=2,
    columnspacing=0.5, handletextpad=0.5)

ax.set_xlabel('Monthly source latitude [$°\;S$]')
ax.set_ylabel(None)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source sst

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.2_source_sst/6.1.8.2 ' + expid[i] + ' pre_weighted_sst mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['sst']['mm'],
        '.-', lw=0.5, markersize=4,
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
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm relative source lon

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.1_source_lon/6.1.8.1 ' + expid[i] + ' pre_weighted_lon mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month,
        calc_lon_diff(
            pre_weighted_var_icores[expid[i]][icores]['lon']['mm'],
            loc_indices[icores]['lon'],
            ),
        '.-', lw=0.5, markersize=4,
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

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['rh2m']['mm'],
        '.-', lw=0.5, markersize=4,
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
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source wind10

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.4_source_wind10/6.1.8.4 ' + expid[i] + ' pre_weighted_wind10 mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month, pre_weighted_var_icores[expid[i]][icores]['wind10']['mm'],
        '.-', lw=0.5, markersize=4,
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
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm transport distance

output_png = 'figures/6_awi/6.1_echam6/6.1.8_ice_cores/6.1.8.5_transport_distance/6.1.8.5 ' + expid[i] + ' transport_distance mm icores.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for icores in pre_weighted_var_icores[expid[i]].keys():
    ax.plot(
        month,
        pre_weighted_var_icores[expid[i]][icores]['distance']['mm'] / 100,
        '.-', lw=0.5, markersize=4,
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


