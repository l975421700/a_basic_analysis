

# salloc --account=paleodyn.paleodyn --partition=mpp --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
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
# region extract am data

daily_min_srcRHsst = q_sfc_weighted_var[expid[i]]['RHsst']['am'][daily_min_ilat, daily_min_ilon].values * 100
print(daily_min_srcRHsst)
# 74%
daily_max_srcRHsst = q_sfc_weighted_var[expid[i]]['RHsst']['am'][daily_max_ilat, daily_max_ilon].values * 100
print(daily_max_srcRHsst)
# 72%

daily_min_dD = isotopes_q_sfc_alltime[expid[i]]['dD']['am'][daily_min_ilat, daily_min_ilon].values
print(daily_min_dD)
# -108‰
daily_max_dD = isotopes_q_sfc_alltime[expid[i]]['dD']['am'][daily_max_ilat, daily_max_ilon].values
print(daily_max_dD)
# -86‰

daily_min_dO18 = isotopes_q_sfc_alltime[expid[i]]['dO18']['am'][daily_min_ilat, daily_min_ilon].values
print(daily_min_dO18)
# -13.7‰
daily_max_dO18 = isotopes_q_sfc_alltime[expid[i]]['dO18']['am'][daily_max_ilat, daily_max_ilon].values
print(daily_max_dO18)
# -11.8‰


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate evaporative isotopic fluxes

#-------------------------------- Functions to calculate variables

def cal_alpha_lv(iisotopes, T):
    '''
    #-------- Input
    iisotopes:  'd18O' or 'dD'
    T:          in K.
    
    #-------- Output
    
    '''
    
    import numpy as np
    
    if (iisotopes == 'd18O'):
        alpha1 = 1137
        alpha2 = -0.4156
        alpha3 = -2.0667e-3
    elif (iisotopes == 'dD'):
        alpha1 = 24844
        alpha2 = -76.248
        alpha3 = 52.612e-3
        # alpha3 = -52.612e-3
    
    alpha_lv = np.exp(alpha1 / T**2 + alpha2 / T + alpha3)
    
    return(alpha_lv)


def cal_delta_evap(iisotopes, T, delta_vapour, RHsst):
    '''
    #-------- Input
    iisotopes:      'd18O' or 'dD'
    T:              in K.
    delta_vapour:   no unit
    RHsst:          no unit
    
    #-------- Output
    
    '''
    
    if (iisotopes == 'd18O'):
        k = 0.00475
    elif (iisotopes == 'dD'):
        k = 0.00418
    
    alpha_lv = cal_alpha_lv(iisotopes, T)
    
    delta_evap = (1 - k) / (1 - RHsst) * (1/alpha_lv - RHsst * (1 + delta_vapour)) - 1
    
    return(delta_evap)




#-------------------------------- Daily min

daily_min_dD_evap = cal_delta_evap(
    'dD', np.arange(0, 25 + 1e-4, 0.05) + zerok,
    daily_min_dD / 1000, daily_min_srcRHsst / 100,
    ) * 1000

daily_min_d18O_evap = cal_delta_evap(
    'd18O', np.arange(0, 25 + 1e-4, 0.05) + zerok,
    daily_min_dO18 / 1000, daily_min_srcRHsst / 100,
    ) * 1000

daily_min_d_xs_evap = daily_min_dD_evap - 8 * daily_min_d18O_evap

ln_dD = 1000 * np.log(1 + daily_min_dD_evap / 1000)
ln_d18O = 1000 * np.log(1 + daily_min_d18O_evap / 1000)

daily_min_d_ln_evap = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

print(pearsonr(np.arange(0, 25 + 1e-4, 0.05) + zerok, daily_min_d_xs_evap))
print(pearsonr(np.arange(0, 25 + 1e-4, 0.05) + zerok, daily_min_d_ln_evap))


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    np.arange(0, 25 + 1e-4, 0.05), daily_min_d_ln_evap,
    s=6, lw=1, facecolors='white', edgecolors='k',)

ax.set_xlabel(plot_labels['sst'],)
ax.set_ylabel(plot_labels['d_ln'],)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
fig.savefig('figures/test/test.png')


#-------------------------------- Daily max

daily_max_dD_evap = cal_delta_evap(
    'dD', np.arange(0, 25 + 1e-4, 0.05) + zerok,
    daily_max_dD / 1000, daily_max_srcRHsst / 100,
    ) * 1000

daily_max_d18O_evap = cal_delta_evap(
    'd18O', np.arange(0, 25 + 1e-4, 0.05) + zerok,
    daily_max_dO18 / 1000, daily_max_srcRHsst / 100,
    ) * 1000

daily_max_d_xs_evap = daily_max_dD_evap - 8 * daily_max_d18O_evap

ln_dD = 1000 * np.log(1 + daily_max_dD_evap / 1000)
ln_d18O = 1000 * np.log(1 + daily_max_d18O_evap / 1000)

daily_max_d_ln_evap = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

print(pearsonr(np.arange(0, 25 + 1e-4, 0.05) + zerok, daily_max_d_xs_evap))
print(pearsonr(np.arange(0, 25 + 1e-4, 0.05) + zerok, daily_max_d_ln_evap))


# endregion
# -----------------------------------------------------------------------------



