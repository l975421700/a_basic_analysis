

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
    seconds_per_d,
    plot_labels,
    plot_labels_no_unit,
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

# remove the anomalous spike

isotopes_EDC_800kyr_AICC = isotopes_EDC_800kyr_AICC.drop(
    index=np.argmin(isotopes_EDC_800kyr_AICC['d_ln'])
    ).reset_index(drop=True)


isotopes_EDC_800kyr_AICC_resampled = {}

isotopes_EDC_800kyr_AICC_resampled['0.2kyr'] = \
    isotopes_EDC_800kyr_AICC.groupby(pd.cut(
        isotopes_EDC_800kyr_AICC['age'], np.arange(0, 800 + 1e-4, 0.2)
    )).mean()

isotopes_EDC_800kyr_AICC_resampled['0.2kyr']['mid_age'] = \
    np.arange(0.1, 800 + 1e-4, 0.2)


isotopes_EDC_800kyr_AICC_resampled['2kyr'] = \
    isotopes_EDC_800kyr_AICC.groupby(pd.cut(
        isotopes_EDC_800kyr_AICC['age'], np.arange(0, 800 + 1e-4, 2)
    )).mean()

isotopes_EDC_800kyr_AICC_resampled['2kyr']['mid_age'] = \
    np.arange(1, 800 + 1e-4, 2)




'''
isotopes_EDC_800kyr_AICC.iloc[np.argmin(isotopes_EDC_800kyr_AICC['d_ln'])]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot dO18, dD, d_excess, and d_ln

xaxis_max = 800
xaxis_interval = 100
xaxis_max = 140
xaxis_interval = 10


for iisotopes in ['dD', 'dO18', 'd_excess']:
    # iisotopes = 'd_ln'
    # ['dD', 'dO18', 'd_ln', 'd_excess']
    print(iisotopes)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.1_ice cores/8.0.1.1 EDC ' + iisotopes + ' of past ' + str(xaxis_max) + ' kyr on AICC.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)
    
    ax.plot(
        isotopes_EDC_800kyr_AICC['age'].values,
        isotopes_EDC_800kyr_AICC[iisotopes].values,
        c='gray', lw=0.3, ls='-', alpha=0.5)
    
    if (xaxis_max == 800):
        ax.plot(
            isotopes_EDC_800kyr_AICC_resampled['2kyr']['mid_age'].values,
            isotopes_EDC_800kyr_AICC_resampled['2kyr'][iisotopes].values,
            c='k', lw=0.5, ls='-',)
    elif (xaxis_max == 140):
        ax.plot(
            isotopes_EDC_800kyr_AICC_resampled['0.2kyr']['mid_age'].values,
            isotopes_EDC_800kyr_AICC_resampled['0.2kyr'][iisotopes].values,
            c='k', lw=0.5, ls='-',)
    
    ax.set_ylabel(plot_labels[iisotopes])
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    ax.set_xlabel('Age before 1950 [kyr]')
    ax.set_xlim(0, xaxis_max)
    ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # ax.spines[['right', 'top']].set_visible(False)
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
    fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot time intervals in AICC

# xaxis_max = 800
# xaxis_interval = 100
xaxis_max = 140
xaxis_interval = 10

time_intervals = isotopes_EDC_800kyr_AICC['age'][1:].values - isotopes_EDC_800kyr_AICC['age'][:-1].values

output_png = 'figures/8_d-excess/8.0_records/8.0.1_ice cores/8.0.1.1 time intervals on AICC in water isotopes of EDC of past ' + str(xaxis_max) + ' kyr.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    isotopes_EDC_800kyr_AICC['age'][1:].values,
    time_intervals,
    c='k', lw=0.3, ls='-')


ax.set_xlabel('Age before 1950 [kyr]')
ax.set_xlim(0, xaxis_max)
ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylim(0, 0.125)
ax.set_ylabel('Time intervals [kyr]')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

# ax.spines[['right', 'top']].set_visible(False)
ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region estimate PSD of d_ln




import scipy.signal

(frequency, psd) = scipy.signal.periodogram(
    isotopes_EDC_800kyr_AICC_resampled['2kyr']['dD'],
    1, scaling='density')

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.plot(frequency, psd)

ax.set_xlim(0, 0.1)

fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
fig.savefig('figures/test/test.png', dpi=600)


# endregion
# -----------------------------------------------------------------------------


