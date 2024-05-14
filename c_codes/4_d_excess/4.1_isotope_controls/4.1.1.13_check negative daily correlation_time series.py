

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
dD_q_sfc_alltime = {}
dO18_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    #     d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
        dD_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
        dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

source_var = ['lat', 'lon', 'sst', 'RHsst', 'distance']
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        prefix + '.q_sfc_weighted_lon.pkl',
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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

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


# daily min
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and source SST

ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel(plot_labels['sst'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])

print(corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][daily_min_ilat, daily_min_ilon].values ** 2)

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]
data2 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_min_ilat, daily_min_ilon]
subset = np.isfinite(data1) & np.isfinite(data2)
pearsonr(data1[subset], data2[subset],).statistic


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and source RHsst

ialltime = 'daily'
# itimestart = 11730
itimestart = 15730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)] * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)] * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)] * 100
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)] * 100
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_max.png'

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='k',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date',)

ax.set_ylabel(plot_labels['d_ln'], c='k')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel(plot_labels['RHsst'] + ', $R = ' + str(np.round(r_value, 2)) + '$', c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''

# check partial correlation
data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

# pearsonr(data1, data2).statistic ** 2

xr_par_cor(data1, data2, data3) ** 2

xr_par_cor(data1, data3, data2) ** 2

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and source latitude

ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic


output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' lat vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.axhline(daily_min_lat, c='tab:orange', ls='--', lw=0.5)
ax2.set_ylabel(plot_labels['lat'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)



#-------------------------------- check partial correlation while controlling both source latitude and RHsst

ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data4 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)] * 100

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3) & np.isfinite(data4)

xr_par_cor(data1[subset], data2[subset], data3[subset])

dataframe = pd.DataFrame(data={
        'd_ln': data1,
        'src_sst': data2,
        'src_lat': data3,
        'src_RHsst': data4,
    })

pg.partial_corr(data=dataframe, x='d_ln', y='src_sst', covar='src_RHsst').r.values[0]
pg.partial_corr(data=dataframe, x='d_ln', y='src_sst', y_covar='src_lat').r.values[0]

'''
ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3)
pearsonr(data1[subset], data2[subset])
pearsonr(data1[subset], data3[subset])
pearsonr(data2[subset], data3[subset])

xr_par_cor(data1[subset], data2[subset], data3[subset])
xr_par_cor(data1[subset], data3[subset], data2[subset])


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and transport distance

ialltime = 'daily'
itimestart = 11730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' distance vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' distance vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' distance vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' distance vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_max.png'

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2 / 100,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel(plot_labels['distance'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3)
pearsonr(data1[subset], data2[subset])
pearsonr(data1[subset], data3[subset])

xr_par_cor(data1[subset], data2[subset], data3[subset])
xr_par_cor(data1[subset], data3[subset], data2[subset])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and local SST

ialltime = 'daily'
itimestart = 11730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local sst vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = tsw_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local sst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# data2 = tsw_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local sst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
data2 = tsw_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local sst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_max.png'


subset = np.isfinite(data1.values) & np.isfinite(data2.values)
r_value = pearsonr(data1[subset], data2[subset]).statistic


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time.dt.floor('1d'), data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel('Local SST [$°C$]', c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
tsw_alltime[expid[i]][ialltime][itimestart:(itimestart+idatalength)].to_netcdf('scratch/test/test0.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and local RHsst

ialltime = 'daily'
itimestart = 11730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][(itimestart):(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = RHsst_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][(itimestart):(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# data2 = RHsst_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][(itimestart):(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
data2 = RHsst_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][(itimestart):(itimestart+idatalength)]
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local RHsst vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_max.png'


subset = np.isfinite(data1.values) & np.isfinite(data2.values)
r_value = pearsonr(data1[subset], data2[subset]).statistic

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time.dt.floor('1d'), data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel('Local RHsst [$\%$]', c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][(itimestart-1):(itimestart+idatalength-1)]
data3 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)] * 100

subset = np.isfinite(data1) & np.isfinite(data2) & np.isfinite(data3)
pearsonr(data1, data2)
pearsonr(data1, data3)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and local precipitation

ialltime = 'daily'
itimestart = 11730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)] * seconds_per_d
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local pre vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)] * seconds_per_d
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local pre vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
# data2 = wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)] * seconds_per_d
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local pre vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_neg.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
data2 = wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)] * seconds_per_d
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' local pre vs. d_ln from ' + str(data1.time[0].values)[:10] + ' daily_max.png'

subset = np.isfinite(data1.values) & np.isfinite(data2.values)
r_value = pearsonr(data1[subset], data2[subset]).statistic


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time.dt.floor('1d'), data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel('Precipitation [$mm \; day^{-1}$]', c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
tsw_alltime[expid[i]][ialltime][itimestart:(itimestart+idatalength)].to_netcdf('scratch/test/test0.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of d_ln and dD/d18O

ialltime = 'daily'
itimestart = 11730
idatalength = 30

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = dD_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = dO18_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic


# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' dD vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' dO18 vs. d_ln from ' + str(data1.time[0].values)[:10] + '.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
# ax2.set_ylabel(plot_labels['dD'], c='tab:orange')
ax2.set_ylabel(plot_labels['dO18'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of source SST and source latitude

ialltime = 'daily'
itimestart = 11730
# itimestart = 15730
idatalength = 30

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. lat from ' + str(data1.time[0].values)[:10] + '.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_pos_ilat, daily_pos_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. lat from ' + str(data1.time[0].values)[:10] + ' daily_pos.png'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_neg_ilat, daily_neg_ilon][itimestart:(itimestart+idatalength)]
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. lat from ' + str(data1.time[0].values)[:10] + ' daily_neg.png'

# data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
# data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
# data3 = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_max_ilat, daily_max_ilon][itimestart:(itimestart+idatalength)]
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. lat from ' + str(data1.time[0].values)[:10] + ' daily_max.png'

subset = np.isfinite(data1) & np.isfinite(data2)
r_value2 = pearsonr(data1[subset], data2[subset]).statistic

subset = np.isfinite(data1) & np.isfinite(data3)
r_value3 = pearsonr(data1[subset], data3[subset]).statistic

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data2.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date',)

ax.set_ylabel(plot_labels['sst'] + ', $R = ' + str(np.round(r_value2, 2)) + '$', c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
# ax2.invert_yaxis()
ax2.plot(
    data3.time, data3,
    'o', ls='-', ms=2, lw=0.5, c='tab:red',)

# ax2.axhline(-40, ls='--', c='tab:red', lw=0.5,)
ax2.axhline(-50, ls='--', c='tab:red', lw=0.5,)

ax2.set_ylabel(plot_labels['lat'] + ', $R = ' + str(np.round(r_value3, 2)) + '$', c='tab:red')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos_abs)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------



# ann
# -----------------------------------------------------------------------------
# region plot annual time series of d_ln and source SST

ialltime = 'ann'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon]

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic


output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' sst vs. d_ln.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel(plot_labels['sst'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

print(corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'][daily_min_ilat, daily_min_ilon].values)
print(corr_sources_isotopes_q_sfc[expid[i]][ctr_var][iisotopes][ialltime]['r'][daily_min_ilat, daily_min_ilon].values)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual time series of d_ln and source RHsst

ialltime = 'ann'

data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon]
data2 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon] * 100

subset = np.isfinite(data1) & np.isfinite(data2)
r_value = pearsonr(data1[subset], data2[subset]).statistic


output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.5 ' + expid[i] + ' ' + ialltime + ' RHsst vs. d_ln.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    data1.time, data1,
    'o', ls='-', ms=2, lw=0.5, c='tab:blue',)

ax.set_xticks(data1.time[::4])
plt.xticks(rotation=30, ha='right')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date [$R = ' + str(np.round(r_value, 2)) + '$]',)

ax.set_ylabel(plot_labels['d_ln'], c='tab:blue')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.plot(
    data2.time, data2,
    'o', ls='-', ms=2, lw=0.5, c='tab:orange',)
ax2.set_ylabel(plot_labels['RHsst'], c='tab:orange')
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.25, top=0.98)
fig.savefig(output_png)


'''

# check partial correlation
data1 = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data2 = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
data3 = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

# pearsonr(data1, data2).statistic ** 2

xr_par_cor(data1, data2, data3) ** 2

xr_par_cor(data1, data3, data2) ** 2

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check correlation

ialltime = 'daily'
itimestart = 11730
idatalength = 30

daily_d_ln = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
daily_src_sst = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
daily_src_RHsst = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)] * 100
daily_src_distance = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

daily_local_sst = tsw_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]
daily_local_RHsst = RHsst_alltime[expid[i]][ialltime][:, daily_min_ilat, daily_min_ilon][itimestart:(itimestart+idatalength)]

pearsonr(daily_d_ln, daily_src_sst).statistic
pearsonr(daily_d_ln, daily_src_RHsst).statistic
pearsonr(daily_src_sst, daily_src_RHsst).statistic

pearsonr(daily_d_ln, daily_local_sst).statistic
pearsonr(daily_d_ln, daily_local_RHsst).statistic

pearsonr(daily_src_sst, daily_local_sst).statistic
pearsonr(daily_src_sst, daily_local_RHsst).statistic
pearsonr(daily_src_sst, daily_src_distance, ).statistic





# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region check source-sink distance

stats.describe(q_sfc_weighted_var[expid[i]]['distance']['daily'][:, daily_min_ilat, daily_min_ilon], nan_policy='omit')
# 1074±556
stats.describe(q_sfc_weighted_var[expid[i]]['distance']['daily'][:, daily_max_ilat, daily_max_ilon], nan_policy='omit')
# 415±111
stats.describe(q_sfc_weighted_var[expid[i]]['distance']['daily'][:, daily_neg_ilat, daily_neg_ilon], nan_policy='omit')
# 1276±734
stats.describe(q_sfc_weighted_var[expid[i]]['distance']['daily'][:, daily_pos_ilat, daily_pos_ilon], nan_policy='omit')
# 660±315

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check partial correlation

ialltime = 'daily'

for ipoint in ['daily_min', 'daily_max', 'daily_neg', 'daily_pos', ]:
    print('#-------------------------------- ' + ipoint)
    
    if (ipoint == 'daily_min'):
        daily_ilat = daily_min_ilat
        daily_ilon = daily_min_ilon
    elif (ipoint == 'daily_max'):
        daily_ilat = daily_max_ilat
        daily_ilon = daily_max_ilon
    elif (ipoint == 'daily_neg'):
        daily_ilat = daily_neg_ilat
        daily_ilon = daily_neg_ilon
    elif (ipoint == 'daily_pos'):
        daily_ilat = daily_pos_ilat
        daily_ilon = daily_pos_ilon
    
    daily_d_ln = d_ln_q_sfc_alltime[expid[i]][ialltime][:, daily_ilat, daily_ilon]
    daily_RHsst = q_sfc_weighted_var[expid[i]]['RHsst'][ialltime][:, daily_ilat, daily_ilon] * 100
    daily_sst = q_sfc_weighted_var[expid[i]]['sst'][ialltime][:, daily_ilat, daily_ilon]
    daily_lat = q_sfc_weighted_var[expid[i]]['lat'][ialltime][:, daily_ilat, daily_ilon]
    daily_distance = q_sfc_weighted_var[expid[i]]['distance'][ialltime][:, daily_ilat, daily_ilon]
    
    # print('#-------- Corr. between d_ln and srcSST, controlling srcRHsst')
    # print(np.round(xr_par_cor(daily_d_ln, daily_sst, daily_RHsst), 2))
    
    # print('#-------- Corr. between d_ln and srcRHsst, controlling srcSST')
    # print(np.round(xr_par_cor(daily_d_ln, daily_RHsst, daily_sst), 2))
    
    # print('#-------- Corr. between d_ln and srcSST, controlling srclat')
    # print(np.round(xr_par_cor(daily_d_ln, daily_sst, daily_lat), 2))
    
    print('#-------- Corr. between d_ln and srclat, controlling srcSST')
    print(np.round(xr_par_cor(daily_d_ln, daily_lat, daily_sst), 2))
    
    # print('#-------- Corr. between d_ln and srcSST, controlling distance')
    # print(np.round(xr_par_cor(daily_d_ln, daily_sst, daily_distance), 2))
    
    itimestart = 11730
    idatalength = 30
    
    # print('#-------- Corr. between d_ln and srcSST, controlling srcRHsst')
    # print(np.round(xr_par_cor(daily_d_ln[itimestart:(itimestart+idatalength)], daily_sst[itimestart:(itimestart+idatalength)], daily_RHsst[itimestart:(itimestart+idatalength)]), 2))
    
    # print('#-------- Corr. between d_ln and srcRHsst, controlling srcSST')
    # print(np.round(xr_par_cor(daily_d_ln[itimestart:(itimestart+idatalength)], daily_RHsst[itimestart:(itimestart+idatalength)], daily_sst[itimestart:(itimestart+idatalength)]), 2))
    
    # print('#-------- Corr. between d_ln and srcSST, controlling srclat')
    # print(np.round(xr_par_cor(daily_d_ln[itimestart:(itimestart+idatalength)], daily_sst[itimestart:(itimestart+idatalength)], daily_lat[itimestart:(itimestart+idatalength)]), 2))
    
    print('#-------- Corr. between d_ln and srcSST, controlling srclat')
    print(np.round(xr_par_cor(daily_d_ln[itimestart:(itimestart+idatalength)], daily_lat[itimestart:(itimestart+idatalength)], daily_sst[itimestart:(itimestart+idatalength)]), 2))
    
    # print('#-------- Corr. between d_ln and srcSST, controlling distance')
    # print(np.round(xr_par_cor(daily_d_ln[itimestart:(itimestart+idatalength)], daily_sst[itimestart:(itimestart+idatalength)], daily_distance[itimestart:(itimestart+idatalength)]), 2))




'''
    for ivar in ['RHsst', 'sst', 'lat', 'distance']:
        print('#---------------- correlation between d_ln and ' + ivar)
        
        itimestart = 11730
        idatalength = 30
        
        data1 = daily_d_ln[itimestart:(itimestart+idatalength)]
        data2 = q_sfc_weighted_var[expid[i]][ivar][ialltime][:, daily_ilat, daily_ilon][itimestart:(itimestart+idatalength)]
        
        print(np.round(pearsonr(data1, data2).statistic, 2))


    itimestart = 11730
    idatalength = 30
    data1 = daily_sst[itimestart:(itimestart+idatalength)]
    data2 = daily_lat[itimestart:(itimestart+idatalength)]
    print(np.round(pearsonr(data1, data2).statistic, 2))
    
    subset = np.isfinite(daily_sst) & np.isfinite(daily_lat)
    print(np.round(pearsonr(daily_sst[subset], daily_lat[subset]).statistic, 2))

'''
# endregion
# -----------------------------------------------------------------------------





