

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]


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
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import linregress

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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
    inversion_top,
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

i = 0

sam_mon = {}
sam_mon[expid[i]] = xr.open_dataset(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_mon[expid[0]].sam > sam_mon[expid[0]].sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_mon[expid[0]].sam < (-1 * sam_mon[expid[0]].sam.std(ddof = 1))

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.inversion_height_strength.pkl', 'rb') as f:
    inversion_height_strength = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot inversion height

isite = 'EDC'

plt_data = inversion_height_strength[isite]['mon']['Inversion height']

output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.1_sam/8.2.0.1.0 ' + expid[i] + ' histogram of monthly inversion height at ' + isite + '.png'

xmax = plt_data.mean() + plt_data.std(ddof=1) * 2
xmin = plt_data.mean() - plt_data.std(ddof=1) * 2
xlim_min = xmin - 0.5
xlim_max = xmax + 0.5

# xtickmin = np.ceil(xlim_min / 20) * 20
# xtickmax = np.floor(xlim_max / 20) * 20

fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

sns.histplot(
    plt_data,
    binwidth=4)
sns.histplot(
    plt_data[sam_posneg_ind['pos'].values],
    binwidth=4, alpha=0.5, color='red')
sns.histplot(
    plt_data[sam_posneg_ind['neg'].values],
    binwidth=4, alpha=0.5, color='black')

ax.axvline(plt_data[sam_posneg_ind['pos'].values].mean(),
           c = 'red', linewidth=0.5,)
ax.axvline(plt_data[sam_posneg_ind['neg'].values].mean(),
           c = 'black', linewidth=0.5,)

plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)

ax.set_xlabel('Inversion height [$m$]', labelpad=0.5)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
# ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 20))
ax.set_xlim(xlim_min, xlim_max)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.3, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)


'''
inversion_height_strength[isite]['mon']['Inversion height'].mean()
inversion_height_strength[isite]['mon']['Inversion height'][sam_posneg_ind['pos'].values].mean()
inversion_height_strength[isite]['mon']['Inversion height'][sam_posneg_ind['neg'].values].mean()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot inversion strength

isite = 'EDC'

plt_data = inversion_height_strength[isite]['mon']['Inversion strength']

output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.1_sam/8.2.0.1.0 ' + expid[i] + ' histogram of monthly inversion strength at ' + isite + '.png'

xmax = plt_data.mean() + plt_data.std(ddof=1) * 3
xmin = plt_data.mean() - plt_data.std(ddof=1) * 3
xlim_min = xmin - 0.5
xlim_max = xmax + 0.5

# xtickmin = np.ceil(xlim_min / 20) * 20
# xtickmax = np.floor(xlim_max / 20) * 20

fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

sns.histplot(
    plt_data,
    binwidth=1)
sns.histplot(
    plt_data[sam_posneg_ind['pos'].values],
    binwidth=1, alpha=0.5, color='red')
sns.histplot(
    plt_data[sam_posneg_ind['neg'].values],
    binwidth=1, alpha=0.5, color='black')

ax.axvline(plt_data[sam_posneg_ind['pos'].values].mean(),
           c = 'red', linewidth=0.5,)
ax.axvline(plt_data[sam_posneg_ind['neg'].values].mean(),
           c = 'black', linewidth=0.5,)

plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)

ax.set_xlabel('Inversion strength [$K$]', labelpad=0.5)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
# ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 20))
ax.set_xlim(xlim_min, xlim_max)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.3, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)


'''
inversion_height_strength[isite]['mon']['Inversion height'].mean()
inversion_height_strength[isite]['mon']['Inversion height'][sam_posneg_ind['pos'].values].mean()
inversion_height_strength[isite]['mon']['Inversion height'][sam_posneg_ind['neg'].values].mean()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region surface and inversion temperature

t_it = inversion_height_strength[isite]['mon']['Inversion temperature']
tas = inversion_height_strength[isite]['mon']['Surface temperature']
pearsonr(t_it, tas)

linearfit = linregress(
    x = tas,
    y = t_it,)

xmax_value = np.max(tas)
xmin_value = np.min(tas)
xlimmax = xmax_value + 1
xlimmin = xmin_value - 1
xtickmax = np.ceil(xlimmax)
xtickmin = np.floor(xlimmin)

ymax_value = np.max(t_it)
ymin_value = np.min(t_it)
ylimmax = ymax_value + 1
ylimmin = ymin_value - 1
ytickmax = np.ceil(ylimmax)
ytickmin = np.floor(ylimmin)

output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.1_sam/8.2.0.1.0 ' + expid[i] + ' monthly inversion vs. surface temperature at ' + isite + '.png'
fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

ax.plot(tas, t_it, '.', markersize=1.5,)

ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)

plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='black',)
plt.text(
    0.55, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
    transform=ax.transAxes, fontsize=6, linespacing=1.5)

ax.set_ylabel('$T_{inversion}$ [$K$]', labelpad=2)
ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 10))
ax.set_ylim(ylimmin, ylimmax)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('$T_{surface}$ [$K$]', labelpad=2)
ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 10))
ax.set_xlim(xlimmin, xlimmax)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.28, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)



'''
t_it = t_it.groupby('time.month') - t_it.groupby('time.month').mean()
tas = tas.groupby('time.month') - tas.groupby('time.month').mean()
pearsonr(t_it, tas)

t_it = inversion_height_strength[isite]['ann']['Inversion temperature']
tas = inversion_height_strength[isite]['ann']['Surface temperature']
pearsonr(t_it, tas)

'''
# endregion
# -----------------------------------------------------------------------------
