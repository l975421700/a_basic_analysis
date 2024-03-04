

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


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare ERA5 with observations

isite = 'EDC'

site_lat = ten_sites_loc[ten_sites_loc['Site'] == isite]['lat'][0]
site_lon = ten_sites_loc[ten_sites_loc['Site'] == isite]['lon'][0]

#-------------------------------- check temp2

observation = BS13_Dome_C['mon']['temp2'].values
simulation  = temp2_alltime_icores[expid[i]][isite]['mon'].sel(time=slice('2008-01-1', '2010-12-31')).values
ERA5_data   = ERA5_monthly_tp_temp2_tsurf_1979_2022.t2m.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2008-01-1', '2010-12-31')).values - zerok

pearsonr(observation, simulation).statistic ** 2
np.sqrt(np.average(np.square(simulation - observation)))

pearsonr(observation, ERA5_data).statistic ** 2
np.sqrt(np.average(np.square(ERA5_data - observation)))

#-------------------------------- check pre

observation = BS13_Dome_C['mon']['pre'].values
simulation  = wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-1', '2010-12-31')).values * seconds_per_d
ERA5_data   = ERA5_monthly_tp_temp2_tsurf_1979_2022.tp.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2008-01-1', '2010-12-31')).values * 1000

pearsonr(observation, simulation).statistic ** 2
np.sqrt(np.average(np.square(simulation - observation)))

pearsonr(observation, ERA5_data).statistic ** 2
np.sqrt(np.average(np.square(ERA5_data - observation)))


# endregion
# -----------------------------------------------------------------------------

