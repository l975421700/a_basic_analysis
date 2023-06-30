

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
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
from scipy.stats import pearsonr
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
    monthini,
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

dO18_alltime = {}
dD_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)

lon = dO18_alltime[expid[i]]['am'].lon
lat = dO18_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region scatterplot am delta D vs. delta 018 over AIS

i = 0

ais_dO18 = dO18_alltime[expid[i]]['am'].values[echam6_t63_ais_mask['mask']['AIS']]
ais_dD = dD_alltime[expid[i]]['am'].values[echam6_t63_ais_mask['mask']['AIS']]

xmax_value = np.max(ais_dO18)
xmin_value = np.min(ais_dO18)
xlimmax = xmax_value + 5
xlimmin = xmin_value - 5

ymax_value = np.max(ais_dD)
ymin_value = np.min(ais_dD)
ylimmax = ymax_value + 25
ylimmin = ymin_value - 25

linearfit = linregress(x = ais_dO18, y = ais_dD,)

output_png = 'figures/8_d-excess/8.1_controls/8.1.4_spatial_analysis/8.1.4.0 ' + expid[i] + ' am delta D vs. delta O18 over AIS.png'
fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

ax.scatter(
    ais_dO18, ais_dD, s=6, lw=0.1, facecolors='white', edgecolors='k',
    )
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope,
    lw=0.5, color='k')

plt.text(0.05, 0.9, 'AIS', transform=ax.transAxes, color='k',)

plt.text(
    0.5, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
    transform=ax.transAxes, fontsize=6, linespacing=1.5)

ax.set_ylabel('$\delta D$ [$‰$]', labelpad=2)
ax.set_yticks(np.arange(-600, 0 + 1e-4, 100))
ax.set_ylim(ylimmin, ylimmax)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('$\delta ^{18} O$ [$‰$]', labelpad=2)
ax.set_xticks(np.arange(-60, 0 + 1e-4, 10))
ax.set_xlim(xlimmin, xlimmax)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', labelsize=8)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.32, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)




log_ais_dO18 = 1000 * np.log(1 + ais_dO18 / 1000)
log_ais_dD = 1000 * np.log(1 + ais_dD / 1000)

xmax_value = np.max(log_ais_dO18)
xmin_value = np.min(log_ais_dO18)
xlimmax = xmax_value + 5
xlimmin = xmin_value - 5

ymax_value = np.max(log_ais_dD)
ymin_value = np.min(log_ais_dD)
ylimmax = ymax_value + 25
ylimmin = ymin_value - 25

linearfit = linregress(x = log_ais_dO18, y = log_ais_dD,)
polyfit2 = np.polyfit(log_ais_dO18, log_ais_dD, 2)
polyfit2_model = np.poly1d(polyfit2)

output_png = 'figures/8_d-excess/8.1_controls/8.1.4_spatial_analysis/8.1.4.0 ' + expid[i] + ' am log delta D vs. delta O18 over AIS.png'
fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

ax.scatter(
    log_ais_dO18, log_ais_dD, s=6, lw=0.1, facecolors='white', edgecolors='k',
    )
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope,
    lw=0.5, color='k')
ax.plot(
    log_ais_dO18, polyfit2_model(log_ais_dO18), lw=0.5, color='r'
)

plt.text(0.05, 0.9, 'AIS', transform=ax.transAxes, color='k',)

plt.text(
    0.3, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
    transform=ax.transAxes, fontsize=6, linespacing=1.5)

plt.text(
    0.05, 0.6,
    '$y = $' + \
        str(np.round(polyfit2[0], 4)) + '$x^2 + $' + \
        str(np.round(polyfit2[1], 2)) + '$x + $' + \
        str(np.round(polyfit2[2], 2)) + \
            '\n$R^2 = $' + \
                str(np.round(np.round(pearsonr(log_ais_dD, polyfit2_model(log_ais_dO18)).statistic ** 2, 3), 3)),
    transform=ax.transAxes, fontsize=6, linespacing=1.5, c = 'r')

ax.set_ylabel('$ln(1 + \delta D)$ [$‰$]', labelpad=2)
ax.set_yticks(np.arange(-600, 0 + 1e-4, 100))
ax.set_ylim(ylimmin, ylimmax)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('$ln(1 + \delta ^{18} O)$ [$‰$]', labelpad=2)
ax.set_xticks(np.arange(-60, 0 + 1e-4, 10))
ax.set_xlim(xlimmin, xlimmax)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', labelsize=8)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.32, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------


