

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
    plot_labels_no_unit,
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

d_ln_q_sfc_alltime = {}
d_excess_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
        d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

source_var = ['sst', 'RHsst', 'distance']
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
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


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot distribution of RHsst at daily_max and daily_min

ialltime = 'daily'

stats.describe(RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon])

stats.describe(RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon])

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.0 ' + expid[i] + ' histogram of daily RHsst in daily_min and daily max correlation points.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

sns.histplot(
    RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon],
    binwidth=1, color='tab:blue', alpha=0.5, label = 'Daily min'
    )
sns.histplot(
    RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon],
    binwidth=1, color='tab:orange', alpha=0.5, label = 'Daily max'
    )

ax.set_xlabel('Daily RHsst [$\%$]',)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.legend()

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
fig.savefig(output_png)



'''
subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot distribution of source RHsst at daily_max and daily_min

ialltime = 'daily'

stats.describe(q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon], nan_policy='omit')
stats.describe(q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_max_ilat, daily_max_ilon], nan_policy='omit')

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.0 ' + expid[i] + ' histogram of daily source RHsst in daily_min and daily max correlation points.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

sns.histplot(
    q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon] * 100,
    binwidth=1, color='tab:blue', alpha=0.5, label = 'Daily min'
    )
sns.histplot(
    q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_max_ilat, daily_max_ilon] * 100,
    binwidth=1, color='tab:orange', alpha=0.5, label = 'Daily max'
    )

ax.set_xlabel('Daily ' + plot_labels['RHsst'],)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.legend()

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
fig.savefig(output_png)



'''
subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot distribution of SST at daily_max and daily_min

ialltime = 'daily'

stats.describe(tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon])

stats.describe(tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon])

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.0 ' + expid[i] + ' histogram of daily SST in daily_min and daily max correlation points.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

sns.histplot(
    tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon],
    binwidth=1, color='tab:blue', alpha=0.5, label = 'Daily min'
    )
sns.histplot(
    tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon],
    binwidth=1, color='tab:orange', alpha=0.5, label = 'Daily max'
    )

ax.set_xlabel('Daily SST [$°C$]',)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.legend()

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
fig.savefig(output_png)



'''
subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot distribution of source SST at daily_max and daily_min

ialltime = 'daily'

stats.describe(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon], nan_policy='omit')
stats.describe(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon], nan_policy='omit')

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.0 ' + expid[i] + ' histogram of daily source SST in daily_min and daily max correlation points.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

sns.histplot(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon],
    binwidth=1, color='tab:blue', alpha=0.5, label = 'Daily min'
    )
sns.histplot(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon],
    binwidth=1, color='tab:orange', alpha=0.5, label = 'Daily max'
    )

ax.set_xlabel('Daily ' + plot_labels['sst'],)
ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.legend()

ax.grid(True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
fig.savefig(output_png)



'''
subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon])
pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d_ln vs. source SST

#-------------------------------- daily
ialltime = 'daily'

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.1 ' + expid[i] + ' daily d_ln vs. source SST in daily_min and daily max.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon],
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon],
    s = 6, lw=0.1, facecolors='white', edgecolors='tab:blue',
    )

subset = np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]) & np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon])
linearfit = linregress(
    x = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][subset],
    y = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][subset],)
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope, lw=0.5, color='tab:blue')

ax.scatter(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon],
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon],
    s = 6, lw=0.1, facecolors='white', edgecolors='tab:orange',)

subset = np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon]) & np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon])
linearfit = linregress(
    x = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon][subset],
    y = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][subset],)
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope, lw=0.5, color='tab:orange')

ax.set_ylabel(plot_labels['d_ln'], labelpad=2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel(plot_labels['sst'], labelpad=2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.32, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)


#-------------------------------- annual
ialltime = 'ann'

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.1 ' + expid[i] + ' annual d_ln vs. source SST in annual_min and annual max.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_min_ilat, annual_min_ilon],
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon],
    s = 6, lw=0.1, facecolors='white', edgecolors='tab:blue',
    )

subset = np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_min_ilat, annual_min_ilon]) & np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon])
linearfit = linregress(
    x = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_min_ilat, annual_min_ilon][subset],
    y = d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_min_ilat, annual_min_ilon][subset],)
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope, lw=0.5, color='tab:blue')

ax.scatter(
    q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_max_ilat, annual_max_ilon],
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon],
    s = 6, lw=0.1, facecolors='white', edgecolors='tab:orange',)

subset = np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_max_ilat, annual_max_ilon]) & np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon])
linearfit = linregress(
    x = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, annual_max_ilat, annual_max_ilon][subset],
    y = d_ln_q_sfc_alltime[expid[i]][ialltime][:, annual_max_ilat, annual_max_ilon][subset],)
ax.axline(
    (0, linearfit.intercept), slope = linearfit.slope, lw=0.5, color='tab:orange')

ax.set_ylabel(plot_labels['d_ln'], labelpad=2)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel(plot_labels['sst'], labelpad=2)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.32, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check variations of correlation with RHsst

ialltime = 'daily'

RHsst_threshold = np.arange(60, 140+1e-4, 1)
Correlation_below_RHsst = np.zeros_like(RHsst_threshold)

for iRHsst in range(len(RHsst_threshold)):
    # iRHsst = 0
    
    subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & (RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon] < RHsst_threshold[iRHsst]).values
    
    print('#---------------- ' + str(iRHsst) + ' ' + str(RHsst_threshold[iRHsst]) + ' ' + str(subset.sum()))
    
    Correlation_below_RHsst[iRHsst] = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]).statistic


# It is always around -0.6, no much variations

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check variations of correlation with source RHsst

ialltime = 'daily'

srcRHsst_threshold = np.arange(60, 90+1e-4, 1)
Correlation_below_srcRHsst = np.zeros_like(srcRHsst_threshold)

for iRHsst in range(len(srcRHsst_threshold)):
    # iRHsst = 0
    
    subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & (q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon] * 100 < srcRHsst_threshold[iRHsst]).values
    
    print('#---------------- ' + str(iRHsst) + ' ' + str(srcRHsst_threshold[iRHsst]) + ' ' + str(subset.sum()))
    
    Correlation_below_srcRHsst[iRHsst] = pearsonr(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset], q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset]).statistic


# It is always around -0.6, no much variations

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check variations of partial correlation with RHsst

ialltime = 'daily'

RHsst_threshold = np.arange(60, 140+1e-4, 1)
Partial_corr_below_RHsst = np.zeros_like(RHsst_threshold)

for iRHsst in range(len(RHsst_threshold)):
    # iRHsst = 0
    
    subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & (RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon] < RHsst_threshold[iRHsst]).values
    
    print('#---------------- ' + str(iRHsst) + ' ' + str(RHsst_threshold[iRHsst]) + ' ' + str(subset.sum()))
    
    Partial_corr_below_RHsst[iRHsst] = xr_par_cor(
        d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    )
    
    # Partial_corr_below_RHsst[iRHsst] = xr_par_cor(
    #     d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     output = 'p'
    # )


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    RHsst_threshold, Partial_corr_below_RHsst,
    s=6, lw=1, facecolors='white', edgecolors='k',)

ax.set_xlabel('Upper threshold of local RHsst [$\%$]',)
ax.set_ylabel('Partial correlation between source SST and $d_{ln}$\nwhile controlling source RHsst',)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.24, right=0.98, bottom=0.16, top=0.98)
fig.savefig('figures/test/test.png')



# When RHsst is smaller, partial correlation between d_ln and source SST while controlling source RHsst is less negative, but still significant.

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check variations of partial correlation with source RHsst

ialltime = 'daily'

srcRHsst_threshold = np.arange(60, 90+1e-4, 1)
Partial_corr_below_srcRHsst = np.zeros_like(srcRHsst_threshold)

for iRHsst in range(len(srcRHsst_threshold)):
    # iRHsst = 0
    
    subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & (q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon] * 100 < srcRHsst_threshold[iRHsst]).values
    
    print('#---------------- ' + str(iRHsst) + ' ' + str(srcRHsst_threshold[iRHsst]) + ' ' + str(subset.sum()))
    
    Partial_corr_below_srcRHsst[iRHsst] = xr_par_cor(
        d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    )
    
    # Partial_corr_below_RHsst[iRHsst] = xr_par_cor(
    #     d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    #     output = 'p'
    # )



# Partial correlation between d_ln and source SST while controlling source RHsst does not change with source RHsst.

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region relationship between d_ln and local SST

ialltime = 'daily'

# No correlation
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon],
    tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon],
)

# Positively correlated
pearsonr(
    d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon],
    tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon],
)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check variations of partial correlation with transport distance

ialltime = 'daily'

stats.describe(q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_min_ilat, daily_min_ilon], nan_policy='omit')

distance_threshold = np.arange(500, 1500+1e-4, 20)
Partial_corr_below_distance = np.zeros_like(distance_threshold)


for ithreshold in range(len(distance_threshold)):
    # iRHsst = 0
    
    subset = np.isfinite(d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & np.isfinite(q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon]).values & (q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_min_ilat, daily_min_ilon] < distance_threshold[ithreshold]).values
    
    print('#---------------- ' + str(ithreshold) + ' ' + str(distance_threshold[ithreshold]) + ' ' + str(subset.sum()))
    
    Partial_corr_below_distance[ithreshold] = xr_par_cor(
        d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
        q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon].values[subset],
    )


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    distance_threshold / 100, Partial_corr_below_distance,
    s=6, lw=1, facecolors='white', edgecolors='k',)

ax.set_xlabel('Upper threshold of source-sink distance [$× 10^2 km$]',)
ax.set_ylabel('Partial correlation between source SST and $d_{ln}$\nwhile controlling source RHsst',)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.24, right=0.95, bottom=0.16, top=0.98)
fig.savefig('figures/test/test1.png')



# When RHsst is smaller, partial correlation between d_ln and source SST while controlling source RHsst is less negative, but still significant.

# endregion
# -----------------------------------------------------------------------------



