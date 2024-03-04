

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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
from scipy.stats import pearsonr

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
import matplotlib.dates as mdates

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
    monthini,
    plot_labels,
    expid_colours,
    expid_labels,
    zerok,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('data_sources/Dome_C_records/BS13_Dome_C.pkl', 'rb') as f:
    BS13_Dome_C = pickle.load(f)

isotopes_alltime_icores = {}
temp2_alltime_icores = {}
wisoaprt_alltime_icores = {}

for i in range(len(expid)):
    print('#-------------------------------- ' + expid[i])
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
        temp2_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
        wisoaprt_alltime_icores[expid[i]] = pickle.load(f)

tsurf_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'rb') as f:
    tsurf_alltime[expid[i]] = pickle.load(f)

ERA5_monthly_tp_temp2_tsurf_1979_2022 = xr.open_dataset('scratch/ERA5/ERA5_monthly_tp_temp2_tsurf_1979_2022.nc')

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')
isite = 'EDC'
site_lat = ten_sites_loc[ten_sites_loc['Site'] == isite]['lat'][0]
site_lon = ten_sites_loc[ten_sites_loc['Site'] == isite]['lon'][0]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check statistics

ivar = 'd_xs'   # ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]

mon_xdata = BS13_Dome_C['mon']['date'].values
mon_obs_var = BS13_Dome_C['mon'][ivar].values

daily_xdata = BS13_Dome_C['1d']['date'].values
daily_obs_var = BS13_Dome_C['1d'][ivar].values

am_obs_var = BS13_Dome_C['am'][ivar]

if (ivar == 'dD'):
    mon_sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
    daily_sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31'))
elif (ivar == 'd18O'):
    mon_sim_var = isotopes_alltime_icores[expid[i]]['dO18']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
    daily_sim_var = isotopes_alltime_icores[expid[i]]['dO18']['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31'))
elif (ivar == 'd_xs'):
    mon_sim_var = isotopes_alltime_icores[expid[i]]['d_excess']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
    daily_sim_var = isotopes_alltime_icores[expid[i]]['d_excess']['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31'))
elif (ivar == 'd_ln'):
    mon_sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
    daily_sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31'))
elif (ivar == 'pre'):
    mon_sim_var = wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')) * seconds_per_d
    daily_sim_var = wisoaprt_alltime_icores[expid[i]]['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31')) * seconds_per_d
    ERA5_data     = ERA5_monthly_tp_temp2_tsurf_1979_2022.tp.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2008-01-1', '2010-12-31')).values * 1000
elif (ivar == 'temp2'):
    mon_sim_var = temp2_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
    daily_sim_var = temp2_alltime_icores[expid[i]]['EDC']['daily'].sel(time=slice('2008-01-1', '2010-12-31'))
    ERA5_data     = ERA5_monthly_tp_temp2_tsurf_1979_2022.t2m.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2008-01-1', '2010-12-31')).values - zerok


#-------------------------------- check dxs

np.nanstd(mon_obs_var, ddof=1)
np.nanstd(mon_sim_var, ddof=1)


np.nanmin(mon_sim_var - mon_obs_var)
wheremin = np.where(mon_sim_var - mon_obs_var == np.nanmin(mon_sim_var - mon_obs_var))[0][0]
mon_sim_var[wheremin].values
mon_obs_var[wheremin]
mon_sim_var[wheremin].values - mon_obs_var[wheremin]
mon_xdata[wheremin]

np.nanmax(mon_sim_var - mon_obs_var)
wheremax = np.where(mon_sim_var - mon_obs_var == np.nanmax(mon_sim_var - mon_obs_var))[0][0]
mon_sim_var[wheremax].values
mon_obs_var[wheremax]
mon_sim_var[wheremax].values - mon_obs_var[wheremax]
mon_xdata[wheremax]

np.nanmean(abs(mon_sim_var - mon_obs_var))


subset = np.isfinite(mon_obs_var) & np.isfinite(mon_sim_var)
pearsonr(mon_obs_var[subset], mon_sim_var[subset]).statistic ** 2
np.sqrt(np.average(np.square(mon_obs_var[subset] - mon_sim_var[subset])))


#-------------------------------- check dln

np.nanstd(mon_obs_var, ddof=1)
np.nanstd(mon_sim_var, ddof=1)


np.nanmin(mon_sim_var - mon_obs_var)
wheremin = np.where(mon_sim_var - mon_obs_var == np.nanmin(mon_sim_var - mon_obs_var))[0][0]
mon_sim_var[wheremin].values - mon_obs_var[wheremin]
mon_xdata[wheremin]

np.nanmax(mon_sim_var - mon_obs_var)
wheremax = np.where(mon_sim_var - mon_obs_var == np.nanmax(mon_sim_var - mon_obs_var))[0][0]
mon_sim_var[wheremax].values - mon_obs_var[wheremax]
mon_xdata[wheremax]

np.nanmean(abs(mon_sim_var - mon_obs_var))


subset = np.isfinite(mon_obs_var) & np.isfinite(mon_sim_var)
pearsonr(mon_obs_var[subset], mon_sim_var[subset]).statistic ** 2
np.sqrt(np.average(np.square(mon_obs_var[subset] - mon_sim_var[subset])))


#-------------------------------- check correlation between dD and dln
subset = np.isfinite(BS13_Dome_C['mon']['dD'].values) & np.isfinite(BS13_Dome_C['mon']['d_ln'].values)
pearsonr(BS13_Dome_C['mon']['dD'].values[subset], BS13_Dome_C['mon']['d_ln'].values[subset]).statistic ** 2
pearsonr(BS13_Dome_C['mon']['dD'].values[subset], BS13_Dome_C['mon']['d_xs'].values[subset]).statistic ** 2
pearsonr(BS13_Dome_C['mon']['d18O'].values[subset], BS13_Dome_C['mon']['d_ln'].values[subset]).statistic ** 2
pearsonr(BS13_Dome_C['mon']['d18O'].values[subset], BS13_Dome_C['mon']['d_xs'].values[subset]).statistic ** 2


#-------------------------------- check simulated d18O
wheremax = np.where(abs(mon_sim_var.values - mon_obs_var) == np.nanmax(abs(mon_sim_var.values - mon_obs_var)))[0][0]
mon_sim_var.values[wheremax] - mon_obs_var[wheremax]
np.nanmax(abs(mon_sim_var.values - mon_obs_var))
mon_xdata[wheremax]
mon_sim_var.values - mon_obs_var

am_obs_var
np.average(
    isotopes_alltime_icores[expid[i]]['dO18']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')).values,
    weights=wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')).values * seconds_per_d
)


subset = np.isfinite(mon_obs_var) & np.isfinite(mon_sim_var)
pearsonr(mon_obs_var[subset], mon_sim_var[subset]).statistic ** 2
np.sqrt(np.average(np.square(mon_obs_var[subset] - mon_sim_var[subset])))


#-------------------------------- check simulated dD
wheremax = np.where(abs(mon_sim_var.values - mon_obs_var) == np.nanmax(abs(mon_sim_var.values - mon_obs_var)))[0][0]
mon_sim_var.values[wheremax] - mon_obs_var[wheremax]
np.nanmax(abs(mon_sim_var.values - mon_obs_var))
mon_xdata[wheremax]
mon_sim_var.values - mon_obs_var

am_obs_var
np.average(
    isotopes_alltime_icores[expid[i]]['dD']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')).values,
    weights=wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')).values * seconds_per_d
)


# subset = np.isfinite(mon_obs_var) & np.isfinite(mon_sim_var)
# pearsonr(mon_obs_var[subset], mon_sim_var[subset]).statistic ** 2
# np.sqrt(np.average(np.square(mon_obs_var[subset] - mon_sim_var[subset])))


#-------------------------------- check simulated pre
np.round(mon_sim_var.values - mon_obs_var, 3)
np.round(ERA5_data   - mon_obs_var, 3)

np.nanmean(BS13_Dome_C['1d']['pre']) * 365
np.mean(daily_sim_var.values) * 365
ERA5_data.mean() * 365
(ERA5_data * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,])).sum() / 3


#-------------------------------- check simulated temp2
np.max(abs(mon_sim_var.values - mon_obs_var))
np.argmax(abs(mon_sim_var.values - mon_obs_var))
mon_sim_var[np.argmax(abs(mon_sim_var.values - mon_obs_var))]
mon_obs_var[np.argmax(abs(mon_sim_var.values - mon_obs_var))]
mon_xdata[np.argmax(abs(mon_sim_var.values - mon_obs_var))]

np.max(abs(ERA5_data - mon_obs_var))
np.argmax(abs(ERA5_data - mon_obs_var))
ERA5_data[np.argmax(abs(ERA5_data - mon_obs_var))]
mon_obs_var[np.argmax(abs(ERA5_data - mon_obs_var))]
mon_xdata[np.argmax(abs(ERA5_data - mon_obs_var))]

np.max((ERA5_data - mon_obs_var))
np.argmax((ERA5_data - mon_obs_var))
ERA5_data[np.argmax((ERA5_data - mon_obs_var))]
mon_obs_var[np.argmax((ERA5_data - mon_obs_var))]
mon_xdata[np.argmax((ERA5_data - mon_obs_var))]


#-------------------------------- check simulated d_xs: okay
dD = mon_sim_var[-2]
dO18 = mon_sim_var[-2]
d_xs = mon_sim_var[-2]
d_ln = mon_sim_var[-2]

dD - 8 * dO18
d_xs

ln_dD = 1000 * np.log(1 + dD / 1000)
ln_d18O = 1000 * np.log(1 + dO18 / 1000)
ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
d_ln


#-------------------------------- check simulated monthly data: okay
idailydate_start = np.where(daily_xdata == np.datetime64('2010-11-01'))[0][0]
idailydate_end   = np.where(daily_xdata == np.datetime64('2010-12-01'))[0][0]

daily_pre = daily_sim_var[idailydate_start:idailydate_end].copy()
# daily_dD  = daily_sim_var[idailydate_start:idailydate_end].copy()
daily_dO18= daily_sim_var[idailydate_start:idailydate_end].copy()

# subset = np.isfinite(daily_pre) & np.isfinite(daily_dD)
# np.average(daily_dD[subset], weights=daily_pre[subset])
subset = np.isfinite(daily_pre) & np.isfinite(daily_dO18)
np.average(daily_dO18[subset], weights=daily_pre[subset])

mon_sim_var[-2]

daily_xdata[idailydate_start:idailydate_end]

np.mean(daily_sim_var[idailydate_start:idailydate_end])
mon_sim_var[-2]


#-------------------------------- check observed d_xs: okay
BS13_Dome_C['mon']['dD'].values[-2] - 8 * BS13_Dome_C['mon']['d18O'].values[-2]
BS13_Dome_C['mon']['d_xs'].values[-2]

ln_dD = 1000 * np.log(1 + BS13_Dome_C['mon']['dD'].values[-2] / 1000)
ln_d18O = 1000 * np.log(1 + BS13_Dome_C['mon']['d18O'].values[-2] / 1000)
ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
BS13_Dome_C['mon']['d_ln'].values[-2]


#-------------------------------- check observed monthly data: okay
idailydate_start = np.where(daily_xdata == np.datetime64('2010-11-01'))[0][0]
idailydate_end   = np.where(daily_xdata == np.datetime64('2010-12-01'))[0][0]

daily_xdata[idailydate_start:idailydate_end]

daily_pre = daily_obs_var[idailydate_start:idailydate_end].copy()
# daily_dD  = daily_obs_var[idailydate_start:idailydate_end].copy()
daily_dO18= daily_obs_var[idailydate_start:idailydate_end].copy()

# subset = np.isfinite(daily_pre) & np.isfinite(daily_dD)
# np.average(daily_dD[subset], weights=daily_pre[subset])
subset = np.isfinite(daily_pre) & np.isfinite(daily_dO18)
np.average(daily_dO18[subset], weights=daily_pre[subset])

'''
mon_xdata
mon_obs_var
mon_sim_var

daily_sim_var
'''
# endregion
# -----------------------------------------------------------------------------




