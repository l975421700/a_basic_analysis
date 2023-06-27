

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs

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
import cartopy.feature as cfeature
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
    xr_par_cor,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'rb') as f:
        isotopes_EDC_800kyr_AICC = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region reconstruct past temperature

dD_2k = np.mean(isotopes_EDC_800kyr_AICC['δD H2O [‰ SMOW]'][
    isotopes_EDC_800kyr_AICC['Age [ka BP]'] <= 2])

d_excess_2k = np.mean(isotopes_EDC_800kyr_AICC['d_excess_EDC_800kyr'][
    isotopes_EDC_800kyr_AICC['Age [ka BP]'] <= 2])

d_ln_2k = np.mean(isotopes_EDC_800kyr_AICC['d_ln_EDC_800kyr'][
    isotopes_EDC_800kyr_AICC['Age [ka BP]'] <= 2])

delta_dD = isotopes_EDC_800kyr_AICC['δD H2O [‰ SMOW]'] - dD_2k
delta_d_excess = isotopes_EDC_800kyr_AICC['d_excess_EDC_800kyr'] - d_excess_2k
delta_d_ln = isotopes_EDC_800kyr_AICC['d_ln_EDC_800kyr'] - d_ln_2k

tem_rec = {}

tem_rec['Stenni'] = {}
tem_rec['echam_qg'] = {}
tem_rec['echam_qg_no_ss'] = {}

tem_rec['Stenni']['T_site'] = \
    0.16 * delta_dD + 0.44 * delta_d_excess
tem_rec['Stenni']['T_source'] = \
    0.06 * delta_dD + 0.93 * delta_d_excess

tem_rec['echam_qg']['T_site'] = \
    0.05 * delta_dD + 0.15 * delta_d_ln
tem_rec['echam_qg']['T_source'] = \
    0 * delta_dD + 0.28 * delta_d_ln

tem_rec['echam_qg_no_ss']['T_site'] = \
    0.05 * delta_dD - 0.03 * delta_d_ln
tem_rec['echam_qg_no_ss']['T_source'] = \
    0.07 * delta_dD + 0.13 * delta_d_ln

'''
isotopes_EDC_800kyr_AICC.keys()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot past temperature

#---------------- settings

#-------- variables

iT = 'T_site'
ylabel = '\u0394 $T_{site}$ [$°C$]'

iT = 'T_source'
ylabel = '\u0394 $T_{source}$ [$°C$]'

#-------- figures

#----
xaxis_max = 800
xaxis_interval = 100

#----
xaxis_max = 140
xaxis_interval = 10

#---------------- plot

output_png = 'figures/8_d-excess/8.0_records/8.0.2_reconstructions/8.0.2.0 EDC ' + iT + ' reconstructions of past ' + str(xaxis_max) + ' kyr on AICC.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    isotopes_EDC_800kyr_AICC['Age [ka BP]'].values,
    tem_rec['Stenni'][iT].values,
    c='red', lw=0.3, ls='-')

ax.plot(
    isotopes_EDC_800kyr_AICC['Age [ka BP]'].values,
    tem_rec['echam_qg'][iT].values,
    c='k', lw=0.3, ls='-')

ax.plot(
    isotopes_EDC_800kyr_AICC['Age [ka BP]'].values,
    tem_rec['echam_qg_no_ss'][iT].values,
    c='gray', lw=0.3, ls='-')

ax.set_ylabel(ylabel)
# ax.set_ylim(-75, -55)
# ax.set_yticks(np.arange(-10, 4 + 1e-4, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Age before 1950 [kyr]')
ax.set_xlim(0, xaxis_max)
ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))

# ax.spines[['right', 'top']].set_visible(False)

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.1, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot legend

fig, ax = plt.subplots(1, 1, figsize=np.array([4, 2.4]) / 2.54)

l1 = ax.plot(
    [],[], c='red', lw=0.8, ls='-')
l2 = ax.plot(
    [],[], c='k', lw=0.8, ls='-')
l3 = ax.plot(
    [],[], c='gray', lw=0.8, ls='-')
plt.legend(
    [l1[0], l2[0], l3[0],],
    ['Stenni et al. (2010)',
     'Control',
     'No supersaturation',],
    title = 'Coefficients',
    title_fontsize = 10,
    ncol=1, frameon=False,
    loc = 'center', handletextpad=0.2,)

plt.axis('off')

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

output_png = 'figures/8_d-excess/8.0_records/8.0.2_reconstructions/8.0.2.0 legend.png'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------

