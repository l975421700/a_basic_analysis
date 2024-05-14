

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
d_excess_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
        d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

source_var = ['sst', 'RHsst',]
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_sst.pkl',
        prefix + '.q_sfc_weighted_RHsst.pkl',
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

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


#-------------------------------- Daily negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- Daily positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])


#-------------------------------- Annual negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_min = np.min(annual_par_corr)
where_annual_min = np.where(annual_par_corr == annual_min)
# print(annual_min)
# print(annual_par_corr[where_annual_min[0][0], where_annual_min[1][0]])

annual_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lon.values
annual_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lat.values

annual_min_ilon = np.where(lon == annual_min_lon)[0][0]
annual_min_ilat = np.where(lat == annual_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_min_ilat, annual_min_ilon])


#-------------------------------- Annual positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_max = np.max(annual_par_corr)
where_annual_max = np.where(annual_par_corr == annual_max)
# print(annual_max)
# print(annual_par_corr[where_annual_max[0][0], where_annual_max[1][0]])

annual_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lon.values
annual_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lat.values

annual_max_ilon = np.where(lon == annual_max_lon)[0][0]
annual_max_ilat = np.where(lat == annual_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_max_ilat, annual_max_ilon])


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
# region daily_min corr. between d_ln & source SST | source RHsst

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'


#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values)
ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    sm.add_constant(np.column_stack((
        q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset] * 100,
        q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset] * 100 + ols_fit.params[2] * q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst' + \
        str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)





'''
#-------------------------------- check the correlation

d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]
q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon]
q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon]

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][daily_min_ilat, daily_min_ilon].values

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ctr_var][iisotopes][ialltime]['r'][daily_min_ilat, daily_min_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][daily_min_ilat, daily_min_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ctr_var][ivar][ialltime]['r'][daily_min_ilat, daily_min_ilon].values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region daily_max corr. between d_ln & source SST | source RHsst

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'


#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values)
ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    sm.add_constant(np.column_stack((
        q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset] * 100,
        q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset] * 100 + ols_fit.params[2] * q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
        str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST daily_max.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xticks(np.arange(2, 22 + 1e-4, 2))
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_yticks(np.arange(2, 22 + 1e-4, 2))
ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)





'''
#-------------------------------- check the correlation

d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon]
q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon]
q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon]

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][daily_max_ilat, daily_max_ilon].values

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ctr_var][iisotopes][ialltime]['r'][daily_max_ilat, daily_max_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][daily_max_ilat, daily_max_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ctr_var][ivar][ialltime]['r'][daily_max_ilat, daily_max_ilon].values



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Annual negative corr. between d_ln & source SST | source RHsst

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'


#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values)
ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset],
    sm.add_constant(np.column_stack((
        q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset] * 100,
        q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset] * 100 + ols_fit.params[2] * q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst' + \
        str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xlim(xymin-0.5, xymax+0.5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_ylim(xymin-0.5, xymax+0.5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)




'''
#-------------------------------- check the correlation

d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon]
q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon]
q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon]

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][annual_min_ilat, annual_min_ilon].values

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ctr_var][iisotopes][ialltime]['r'][annual_min_ilat, annual_min_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][annual_min_ilat, annual_min_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_min_ilat, annual_min_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_min_ilat, annual_min_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ctr_var][ivar][ialltime]['r'][annual_min_ilat, annual_min_ilon].values




'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Annual positive corr. between d_ln & source SST | source RHsst

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'


#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values)
ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset],
    sm.add_constant(np.column_stack((
        q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset] * 100,
        q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset] * 100 + ols_fit.params[2] * q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
        str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 2)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST annual_max.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xticks(np.arange(16.5, 18 + 1e-4, 0.5))
ax.set_xlim(xymin-0.5, xymax+0.5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_yticks(np.arange(16.5, 18 + 1e-4, 0.5))
ax.set_ylim(xymin-0.5, xymax+0.5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)



'''
#-------------------------------- check the correlation

d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon]
q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon]
q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon]

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][annual_max_ilat, annual_max_ilon].values

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon])
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset],
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values[subset]
).statistic
corr_sources_isotopes_q_sfc[expid[i]][ctr_var][iisotopes][ialltime]['r'][annual_max_ilat, annual_max_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][annual_max_ilat, annual_max_ilon].values

xr_par_cor(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, annual_max_ilat, annual_max_ilon].values,
    q_sfc_weighted_var[expid[i]][ivar][ialltime][:, annual_max_ilat, annual_max_ilon].values,
)
par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ctr_var][ivar][ialltime]['r'][annual_max_ilat, annual_max_ilon].values



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region daily_min corr. between d_ln & source SST | source RHsst density

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

#-------------------------------- regression

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values
# data2 = q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_min_ilat, daily_min_ilon].values
# data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon].values
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST density.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
# data2 = q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
# data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_pos_ilat, daily_pos_ilon].values
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST density daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# data2 = q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_neg_ilat, daily_neg_ilon].values
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST density daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values
data2 = q_sfc_weighted_var[expid[i]][ctr_var][ialltime][:, daily_max_ilat, daily_max_ilon].values
data3 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_max_ilat, daily_max_ilon].values
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. source RHsst and SST density daily_max.png'

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3)
ols_fit = sm.OLS(
    data1[subset],
    sm.add_constant(np.column_stack((
        data2[subset] * 100,
        data3[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * data2[subset] * 100 + ols_fit.params[2] * data3[subset]

rsquared = pearsonr(data1[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - data1[subset])))

if (ols_fit.params[2] < 0):
    eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'srcRHsst' + \
            str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
                str(np.round(ols_fit.params[0], 1)) + '$' + \
            '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
                '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'
else:
    eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
            str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
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

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
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
'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region Daily negative corr. between d_ln & local SST | local RHsst

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

#-------------------------------- Find the point
daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values) & np.isfinite(RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values) & np.isfinite(tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values)

ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    sm.add_constant(np.column_stack((
        RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset] + ols_fit.params[2] * tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'localRHsst+' + \
        str(np.round(ols_fit.params[2], 2)) + 'localSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. local RHsst and SST.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Daily positive corr. between d_ln & local SST | local RHsst

#-------------------------------- Find the point
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])



#-------------------------------- regression

subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values) & np.isfinite(RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values) & np.isfinite(tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values)
ols_fit = sm.OLS(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    sm.add_constant(np.column_stack((
        RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
        tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
        )))).fit()

# ols_fit.summary()
# ols_fit.params
# ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset] + ols_fit.params[2] * tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset]

rsquared = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y).statistic ** 2
RMSE = np.sqrt(np.average(np.square(predicted_y - d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset])))

eq_text = plot_labels_no_unit[iisotopes] + '$=' + \
    str(np.round(ols_fit.params[1], 2)) + 'localRHsst+' + \
        str(np.round(ols_fit.params[2], 2)) + 'localSST+' + \
            str(np.round(ols_fit.params[0], 1)) + '$' + \
        '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
            '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'

xymax = np.max(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y)))
xymin = np.min(np.concatenate((d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset], predicted_y)))

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.4 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + ' vs. local RHsst and SST daily_max.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon].values[subset],
    predicted_y,
    s=12, lw=0.1, facecolors='white', edgecolors='k',)

ax.axline((0, 0), slope = 1, lw=0.5, color='k')

plt.text(
    0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Simulated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_xticks(np.arange(2, 22 + 1e-4, 2))
ax.set_xlim(xymin-2, xymax+2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Estimated ' + time_labels[ialltime] + ' ' + plot_labels[iisotopes],)
ax.set_yticks(np.arange(2, 22 + 1e-4, 2))
ax.set_ylim(xymin-2, xymax+2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------

