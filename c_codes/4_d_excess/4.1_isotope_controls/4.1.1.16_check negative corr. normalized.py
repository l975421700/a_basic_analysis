

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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
sys.path.append('/albedo/work/user/qigao001')

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
import pingouin as pg
import statsmodels.api as sm

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
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density

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
    plot_labels_no_unit,
    plot_labels,
    time_labels,
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

d_ln_q_sfc_alltime = {}
# d_excess_q_sfc_alltime = {}
# dD_q_sfc_alltime = {}
# dO18_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    #     d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    #     dD_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    #     dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

source_var = ['lat', 'sst', 'RHsst', 'distance'] # 'lon',
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        # prefix + '.q_sfc_weighted_lon.pkl',
        prefix + '.q_sfc_weighted_sst.pkl',
        prefix + '.q_sfc_weighted_RHsst.pkl',
        prefix + '.q_sfc_transport_distance.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)

corr_sources_isotopes_q_sfc = {}
par_corr_sources_isotopes_q_sfc={}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

lon = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lat


white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

'''
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


daily_pos_lon = 30
daily_pos_lat = -40
daily_pos_ilon = np.argmin(abs(lon.values - daily_pos_lon))
daily_pos_ilat = np.argmin(abs(lat.values - daily_pos_lat))

daily_neg_lon = 30
daily_neg_lat = -50
daily_neg_ilon = np.argmin(abs(lon.values - daily_neg_lon))
daily_neg_ilat = np.argmin(abs(lat.values - daily_neg_lat))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the predicted data

ialltime = 'daily'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon].values * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln vs. source RHsst daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon].values * 100
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln vs. source RHsst daily_pos.png'

subset = np.isfinite(data1) & np.isfinite(data2)

ols_fit = sm.OLS(
    data1[subset],
    sm.add_constant(np.column_stack((
        data2[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2[subset]
rsquared = pearsonr(data1[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - data1[subset])))

eq_text = plot_labels_no_unit['d_ln'] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
        str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((data1[subset], predicted_y)))
xymin = np.min(np.concatenate((data1[subset], predicted_y)))

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

fig = plt.figure(figsize=np.array([8.8, 9.5]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

plt_scatter = ax.scatter_density(
    data1[subset],
    predicted_y,
    cmap=white_viridis, vmin=0, vmax=25,)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xticks(np.arange(-20, 40 + 1e-4, 4))
ax.set_yticks(np.arange(-20, 40 + 1e-4, 4))

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels['d_ln'],)
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels['d_ln'],)
ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.colorbar(
    plt_scatter, label='Number of points per pixel',
    ticks=np.arange(0, 25+1e-4, 5),
    orientation='horizontal', fraction=0.08, aspect=30, extend='max',)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.98)
fig.savefig(output_png)


'''
data1_normalised = ((data1 - data1.mean()) / data1.std(ddof=1))
data2_normalised = ((data2 - data2.mean()) / data2.std(ddof=1))

linearfit = linregress(
    x = data1_normalised[subset],
    y = data1_normalised[subset],)

d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].to_netcdf('scratch/test/test0.nc')

subset = np.isfinite(data1) & np.isfinite(data2)
print(pearsonr(data1[subset], data2[subset], ).statistic ** 2)
print(pearsonr(((data1 - data1.mean()) / data1.std(ddof=1))[subset], ((data2 - data2.mean()) / data2.std(ddof=1))[subset], ).statistic)
print(corr_sources_isotopes_q_sfc[expid[i]]['RHsst']['d_ln']['daily']['r'][daily_neg_ilat, daily_neg_ilon])
print(pearsonr(data1_normalised[subset], data2_normalised[subset]).statistic)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the regression deviations with source SST/lat/distance

ialltime = 'daily'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon].values * 100
output_png1 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' daily_neg deviations in predicted d_ln vs. '

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon].values * 100
# output_png1 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' daily_pos deviations in predicted d_ln vs. '

subset = np.isfinite(data1) & np.isfinite(data2)

ols_fit = sm.OLS(
    data1[subset],
    sm.add_constant(np.column_stack((data2[subset],)))).fit()

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2

predict_deviations = predicted_y - data1

#-------------------------------- plot deviations against src SST/lat/distance

for ivar in ['lat', 'sst', 'distance']:
    # ivar = 'sst'
    print('#-------------------------------- ' + ivar)
    
    output_png = output_png1 + ivar + '.png'
    
    data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
    # data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
    
    fig = plt.figure(figsize=np.array([8.8, 9.5]) / 2.54,)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
    subset1 = np.isfinite(data3) & np.isfinite(predict_deviations)
    plt_scatter = ax.scatter_density(
        data3[subset1],
        predict_deviations[subset1],
        cmap=white_viridis, vmin=0, vmax=25,)
    
    linearfit = linregress(x=data3[subset1], y=predict_deviations[subset1],)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    ax.set_xlabel(plot_labels[ivar],)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Deviations in predicted ' + plot_labels['d_ln'],)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    fig.colorbar(
        plt_scatter, label='Number of points per pixel',
        ticks=np.arange(0, 25+1e-4, 5),
        orientation='horizontal', fraction=0.08, aspect=30, extend='max',)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.98)
    fig.savefig(output_png)







'''
print(pearsonr(data1[subset], predicted_y[subset]).statistic ** 2)
print(np.sqrt(np.average(np.square(predicted_y[subset] - data1[subset]))))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the regression deviations with source SST/lat/distance

ialltime = 'daily'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon].values * 100
# data3 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# output_png1 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' daily_neg deviations in mlr predicted d_ln vs. '

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon].values * 100
data3 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
output_png1 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' daily_pos deviations in mlr predicted d_ln vs. '

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3)

ols_fit = sm.OLS(
    data1[subset],
    sm.add_constant(np.column_stack((data2[subset], data3[subset],)))).fit()

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2 + ols_fit.params[2] * data3

predict_deviations = predicted_y - data1

#-------------------------------- plot deviations against src SST/lat/distance

for ivar in ['lat', 'sst', 'distance']:
    # ivar = 'sst'
    print('#-------------------------------- ' + ivar)
    
    output_png = output_png1 + ivar + '.png'
    
    # data4 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
    data4 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
    
    fig = plt.figure(figsize=np.array([8.8, 9.5]) / 2.54,)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
    subset1 = np.isfinite(data4) & np.isfinite(predict_deviations)
    plt_scatter = ax.scatter_density(
        data4[subset1],
        predict_deviations[subset1],
        cmap=white_viridis, vmin=0, vmax=25,)
    
    linearfit = linregress(x=data4[subset1], y=predict_deviations[subset1],)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    ax.set_xlabel(plot_labels[ivar],)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Deviations in MLR predicted ' + plot_labels['d_ln'],)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    fig.colorbar(
        plt_scatter, label='Number of points per pixel',
        ticks=np.arange(0, 25+1e-4, 5),
        orientation='horizontal', fraction=0.08, aspect=30, extend='max',)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.98)
    fig.savefig(output_png)







'''
print(pearsonr(data1[subset], predicted_y[subset]).statistic ** 2)
print(np.sqrt(np.average(np.square(predicted_y[subset] - data1[subset]))))

'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot the original data

ialltime = 'daily'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon].values * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln and source RHsst daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon].values * 100
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln and source RHsst daily_pos.png'

subset = np.isfinite(data1) & np.isfinite(data2)

ols_fit = sm.OLS(
    data1[subset],
    sm.add_constant(np.column_stack((
        data2[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2[subset]
rsquared = pearsonr(data1[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - data1[subset])))

eq_text = plot_labels_no_unit['d_ln'] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
        str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

# xymax = np.max(np.concatenate((data1[subset], predicted_y)))
# xymin = np.min(np.concatenate((data1[subset], predicted_y)))

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

fig = plt.figure(figsize=np.array([8.8, 9.5]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

plt_scatter = ax.scatter_density(
    data2[subset],
    data1[subset],
    cmap=white_viridis, vmin=0, vmax=25,)

# ax.axline((0, 0), slope = 1, lw=0.5, color='k')
ax.axline(
    (0, ols_fit.params[0]), slope = ols_fit.params[1], lw=0.5,)

plt.text(
    0.05, 0.05, eq_text, transform=ax.transAxes, linespacing=2,
    va='bottom', ha='left',)

# ax.set_xticks(np.arange(-20, 40 + 1e-4, 4))
# ax.set_yticks(np.arange(-20, 40 + 1e-4, 4))

ax.set_xlabel(plot_labels['RHsst'],)
# ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel(plot_labels['d_ln'],)
# ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.colorbar(
    plt_scatter, label='Number of points per pixel',
    ticks=np.arange(0, 25+1e-4, 5),
    orientation='horizontal', fraction=0.08, aspect=30, extend='max',)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.98)
fig.savefig(output_png)


'''
data1_normalised = ((data1 - data1.mean()) / data1.std(ddof=1))
data2_normalised = ((data2 - data2.mean()) / data2.std(ddof=1))

linearfit = linregress(
    x = data1_normalised[subset],
    y = data1_normalised[subset],)

d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].to_netcdf('scratch/test/test0.nc')

subset = np.isfinite(data1) & np.isfinite(data2)
print(pearsonr(data1[subset], data2[subset], ).statistic ** 2)
print(pearsonr(((data1 - data1.mean()) / data1.std(ddof=1))[subset], ((data2 - data2.mean()) / data2.std(ddof=1))[subset], ).statistic)
print(corr_sources_isotopes_q_sfc[expid[i]]['RHsst']['d_ln']['daily']['r'][daily_neg_ilat, daily_neg_ilon])
print(pearsonr(data1_normalised[subset], data2_normalised[subset]).statistic)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the original normalised data

ialltime = 'daily'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon]
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon] * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln and source RHsst daily_neg normalized.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon]
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon] * 100
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' d_ln and source RHsst daily_pos normalized.png'

subset = np.isfinite(data1) & np.isfinite(data2)

data1_normalised = ((data1 - data1.mean()) / data1.std(ddof=1)).values
data2_normalised = ((data2 - data2.mean()) / data2.std(ddof=1)).values

ols_fit = sm.OLS(
    data1_normalised[subset],
    sm.add_constant(np.column_stack((
        data2_normalised[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2_normalised[subset]
rsquared = pearsonr(data1_normalised[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - data1_normalised[subset])))

eq_text = plot_labels_no_unit['d_ln'] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$'

# xymax = np.max(np.concatenate((data1_normalised[subset], predicted_y)))
# xymin = np.min(np.concatenate((data1_normalised[subset], predicted_y)))

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

fig = plt.figure(figsize=np.array([8.8, 9.5]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

plt_scatter = ax.scatter_density(
    data2_normalised[subset],
    data1_normalised[subset],
    cmap=white_viridis, vmin=0, vmax=25,)

# ax.axline((0, 0), slope = 1, lw=0.5, color='k')
ax.axline(
    (0, ols_fit.params[0]), slope = ols_fit.params[1], lw=0.5,)

plt.text(
    0.05, 0.05, eq_text, transform=ax.transAxes, linespacing=2,
    va='bottom', ha='left',)

# ax.set_xticks(np.arange(-20, 40 + 1e-4, 4))
# ax.set_yticks(np.arange(-20, 40 + 1e-4, 4))

ax.set_xlabel('Normalised ' + plot_labels['RHsst'],)
# ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Normalised ' + plot_labels['d_ln'],)
# ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.colorbar(
    plt_scatter, label='Number of points per pixel',
    ticks=np.arange(0, 25+1e-4, 5),
    orientation='horizontal', fraction=0.08, aspect=30, extend='max',)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.98)
fig.savefig(output_png)




'''
data1_normalised.mean()
data1_normalised.std(ddof=1)
data2_normalised[subset].mean()
data2_normalised[subset].std(ddof=1)

'''
# endregion
# -----------------------------------------------------------------------------




