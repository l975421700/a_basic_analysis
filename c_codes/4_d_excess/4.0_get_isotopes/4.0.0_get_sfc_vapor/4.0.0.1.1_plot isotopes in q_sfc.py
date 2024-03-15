

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

isotopes_q_sfc_alltime = {}
isotopes_q_sfc_alltime[expid[i]] = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    isotopes_q_sfc_alltime[expid[i]]['dO18'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    isotopes_q_sfc_alltime[expid[i]]['dD'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    isotopes_q_sfc_alltime[expid[i]]['d_xs'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    isotopes_q_sfc_alltime[expid[i]]['d_ln'] = pickle.load(f)

lon = isotopes_q_sfc_alltime[expid[i]]['dO18']['am'].lon
lat = isotopes_q_sfc_alltime[expid[i]]['dO18']['am'].lat


corr_sources_isotopes_q_sfc = {}
par_corr_sources_isotopes_q_sfc={}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)


'''
dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

d_excess_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)

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
# region plot am SH

ialltime = 'am'

for iisotopes in ['dO18', 'dD', 'd_xs', 'd_ln']:
    # iisotopes = 'dO18'
    print('#-------------------------------- ' + iisotopes)
    
    print(stats.describe(isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime].sel(lat=slice(-10, -60)), axis=None))
    
    if (iisotopes == 'dO18'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=-10, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=True)
    elif (iisotopes == 'dD'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-150, cm_max=-70, cm_interval1=10, cm_interval2=20,
            cmap='viridis', reversed=True)
    elif (iisotopes == 'd_xs'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=18, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=True)
    elif (iisotopes == 'd_ln'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=30, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=True)
    
    output_png = 'figures/8_d-excess/8.2_climate/8.2.1 ' + expid[i] + ' ' + ialltime + ' ' + iisotopes + '.png'
    
    fig, ax = hemisphere_plot(northextent=-10, figsize=np.array([5.8, 7.5]) / 2.54,)
    
    plt1 = plot_t63_contourf(
        lon, lat,
        isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime],
        ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    cplot_ice_cores(daily_min_lon, daily_min_lat, ax, s=12)
    cplot_ice_cores(daily_max_lon, daily_max_lat, ax, s=12)
    
    cbar = fig.colorbar(
        plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.2,
        )
    
    cbar.ax.set_xlabel(plot_labels[iisotopes] + '\nin annual mean surface vapour', linespacing=1.5)
    fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot distribution of dO18 and dD at daily_max and daily_min

ialltime = 'daily'

for iisotopes in ['dO18', 'dD', 'd_xs', 'd_ln']:
    # iisotopes = 'dO18'
    print('#-------------------------------- ' + iisotopes)
    
    print(stats.describe(isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime][:, daily_min_ilat, daily_min_ilon]))
    print(stats.describe(isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime][:, daily_max_ilat, daily_max_ilon]))
    
    output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.3 ' + expid[i] + ' histogram of daily ' + iisotopes + ' in daily_min and daily max correlation points.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    sns.histplot(
        isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime][:, daily_min_ilat, daily_min_ilon],
        binwidth=1, color='tab:blue', alpha=0.5, label = 'Daily min'
        )
    sns.histplot(
        isotopes_q_sfc_alltime[expid[i]][iisotopes][ialltime][:, daily_max_ilat, daily_max_ilon],
        binwidth=1, color='tab:orange', alpha=0.5, label = 'Daily max'
        )
    
    ax.set_xlabel('Daily ' + plot_labels[iisotopes],)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend()
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


