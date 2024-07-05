

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
    plot_labels,
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

AL21_T_source = pd.read_excel(
    'data_sources/ice_core_records/Landais_et_al_2021/data_figure_2.xlsx',
    header=0, usecols = [
        'age AICC2012 EDC ka', 'dD (EDC) permil', 'd-excess (EDC) permil',
        'Dtsite (EDC)', 'Dtsource (EDC)'])

AL21_T_source = AL21_T_source.rename(columns={
    'age AICC2012 EDC ka': 'Age',
    'dD (EDC) permil': 'dD',
    'd-excess (EDC) permil': 'd_xs',
    'Dtsite (EDC)': 'Tsite',
    'Dtsource (EDC)': 'Tsource',
})

AL21_T_source_resampled = AL21_T_source.groupby(pd.cut(
    AL21_T_source['Age'], np.arange(0, 800 + 1e-4, 2)
    )).mean()

AL21_T_source_resampled['Age'] = np.arange(1, 800 + 1e-4, 2)

'''

AL21_T_source.columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot four variables

output_png = 'figures/8_d-excess/8.0_records/8.0.2_reconstructions/8.0.2.0 AL21 EDC four variables on AICC 800 kyr.png'

fig, axs = plt.subplots(2, 1, figsize=np.array([18, 13]) / 2.54, sharex=True,
                        gridspec_kw={'hspace':0.1,})

# dD
axs[0].plot(
    AL21_T_source['Age'].values,
    AL21_T_source['dD'].values,
    c='k', lw=0.3, ls='-', alpha=0.4)
axs[0].set_ylabel(plot_labels['dD'], c='k')
axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
axs[0].yaxis.set_major_formatter(remove_trailing_zero_pos)

# Tsite
ax0 = axs[0].twinx()
ax0.plot(
    AL21_T_source['Age'].values,
    AL21_T_source['Tsite'].values,
    c='tab:blue', lw=0.3, ls='-', alpha=0.4,)
ax0.set_ylabel('Site temperature [$°C$]', c = 'tab:blue')
ax0.yaxis.set_minor_locator(AutoMinorLocator(2))
ax0.yaxis.set_major_formatter(remove_trailing_zero_pos)

# d_xs
axs[1].plot(
    AL21_T_source['Age'].values,
    AL21_T_source['d_xs'].values,
    c='k', lw=0.3, ls='-', alpha=0.4,)
axs[1].set_ylabel(plot_labels['d_xs'], c='k')
axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))
axs[1].yaxis.set_major_formatter(remove_trailing_zero_pos)

# Tsource
ax1 = axs[1].twinx()
ax1.plot(
    AL21_T_source['Age'].values,
    AL21_T_source['Tsource'].values,
    c='tab:blue', lw=0.3, ls='-', alpha=0.4,)
ax1.set_ylabel('Source temperature [$°C$]', c = 'tab:blue')
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_major_formatter(remove_trailing_zero_pos)

axs[0].plot(
    AL21_T_source_resampled['Age'].values,
    AL21_T_source_resampled['dD'].values,
    c='k', lw=1, ls='-',)
ax0.plot(
    AL21_T_source_resampled['Age'].values,
    AL21_T_source_resampled['Tsite'].values,
    c='tab:blue', lw=1, ls='-',)
axs[1].plot(
    AL21_T_source_resampled['Age'].values,
    AL21_T_source_resampled['d_xs'].values,
    c='k', lw=1, ls='-',)
ax1.plot(
    AL21_T_source_resampled['Age'].values,
    AL21_T_source_resampled['Tsource'].values,
    c='tab:blue', lw=1, ls='-',)

ax0.axhline(0, lw=0.5, ls='--')
ax1.axhline(0, lw=0.5, ls='--')

ax0.axvline(127, lw=0.5, ls='--', c='k')
ax1.axvline(127, lw=0.5, ls='--', c='k')

axs[1].set_xlabel('Age [ka]')
axs[1].set_xlim(0, 800)
axs[1].set_xticks(np.arange(0, 800 + 1e-4, 100))
axs[1].xaxis.set_minor_locator(AutoMinorLocator(2))

fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.98)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# region plot moisture source SST

xaxis_max = 800
xaxis_interval = 100
# xaxis_max = 140
# xaxis_interval = 10

output_png = 'figures/8_d-excess/8.0_records/8.0.2_reconstructions/8.0.2.0 AL21 EDC T_source reconstructions of past ' + str(xaxis_max) + ' kyr on AICC.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    AL21_T_source['age AICC2012 EDC ka'].values,
    AL21_T_source['Dtsource (EDC)'].values,
    c='k', lw=0.3, ls='-')

ax.set_ylabel(plot_labels['sst'])
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.set_xlabel('Age before 1950 [kyr]')
ax.set_xlim(0, xaxis_max)
ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


'''
np.max(abs(tem_rec[expid[i]][icores]['delta_T_source']['low'].values - tem_rec[expid[i]][icores]['delta_T_source']['high'].values))
'''
# endregion
# -----------------------------------------------------------------------------



