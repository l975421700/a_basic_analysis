

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

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
    monthini,
    plot_labels,
    expid_labels,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import Dome C records and clean it

BS13_Dome_C = {}

BS13_Dome_C['1d'] = pd.read_excel(
    'data_sources/Dome_C_records/tc-10-2415-2016-supplement.xlsx',
    header=0, skiprows=6,)

BS13_Dome_C['1d']['date'] = pd.to_datetime(BS13_Dome_C['1d'][['YEAR', 'MONTH', 'DAY', ]])

BS13_Dome_C['1d'] = BS13_Dome_C['1d'][BS13_Dome_C['1d']['YEAR'] > 2007].reset_index()

BS13_Dome_C['1d'] = BS13_Dome_C['1d'].rename(columns={
    'T_AWS(C)': 'temp2',
    'deuterium excess': 'd_xs',
    'TOTAL(mm_w.e)': 'pre',
})

BS13_Dome_C['1d'] = BS13_Dome_C['1d'][['date', 'temp2', 'dD', 'd18O', 'd_xs', 'pre', ]]

ln_dD = 1000 * np.log(1 + BS13_Dome_C['1d']['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + BS13_Dome_C['1d']['d18O'] / 1000)
BS13_Dome_C['1d']['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


def nan_average(x):
    import numpy as np
    
    masked_data = np.ma.masked_array(x, np.isnan(x))
    average = np.ma.average(masked_data, axis=0, weights=x.index.get_level_values(1))
    result = average.filled(np.nan)
    return(result[0])

#-------------------------------- mon
BS13_Dome_C['mon'] = BS13_Dome_C['1d'][['date', 'temp2', 'pre']].resample('M', on='date').mean().reset_index()

BS13_Dome_C['mon']['dD'] = BS13_Dome_C['1d'][['date', 'pre', 'dD']].set_index(
    'pre', append=True).resample('M', on='date').apply(nan_average).values

BS13_Dome_C['mon']['d18O'] = BS13_Dome_C['1d'][['date','pre','d18O']].set_index(
    'pre', append=True).resample('M', on='date').apply(nan_average).values

BS13_Dome_C['mon']['d_xs'] = BS13_Dome_C['mon']['dD'] - 8 * BS13_Dome_C['mon']['d18O']

ln_dD = 1000 * np.log(1 + BS13_Dome_C['mon']['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + BS13_Dome_C['mon']['d18O'] / 1000)
BS13_Dome_C['mon']['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


#-------------------------------- am
BS13_Dome_C['am'] = BS13_Dome_C['1d'][['date', 'temp2', 'pre']].mean()

BS13_Dome_C['am']['dD'] = nan_average(BS13_Dome_C['1d'][['dD', 'pre']].set_index('pre', append=True))

BS13_Dome_C['am']['d18O'] = nan_average(BS13_Dome_C['1d'][['d18O', 'pre']].set_index('pre', append=True))

BS13_Dome_C['am']['d_xs'] = BS13_Dome_C['am']['dD'] - 8 * BS13_Dome_C['am']['d18O']

ln_dD = 1000 * np.log(1 + BS13_Dome_C['am']['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + BS13_Dome_C['am']['d18O'] / 1000)
BS13_Dome_C['am']['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


output_file = 'data_sources/Dome_C_records/BS13_Dome_C.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(BS13_Dome_C, f)




'''
#-------------------------------- check

with open('data_sources/Dome_C_records/BS13_Dome_C.pkl', 'rb') as f:
    BS13_Dome_C = pickle.load(f)

BS13_Dome_C_1d = pd.read_excel(
    'data_sources/Dome_C_records/tc-10-2415-2016-supplement.xlsx',
    header=0, skiprows=6,)
BS13_Dome_C_1d = BS13_Dome_C_1d[BS13_Dome_C_1d['YEAR'] > 2007]

#---------------- check 1d

data1 = BS13_Dome_C['1d']['dD'].values
data2 = BS13_Dome_C_1d['dD'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

#---------------- check mon
BS13_Dome_C['mon'][BS13_Dome_C['mon']['date'] == '2008-01-31']
BS13_Dome_C_1d[(BS13_Dome_C_1d['YEAR'] == 2008) & (BS13_Dome_C_1d['MONTH'] == 1)].mean()

dD = BS13_Dome_C_1d[(BS13_Dome_C_1d['YEAR'] == 2008) & (BS13_Dome_C_1d['MONTH'] == 1)]['dD']
pre = BS13_Dome_C_1d[(BS13_Dome_C_1d['YEAR'] == 2008) & (BS13_Dome_C_1d['MONTH'] == 1)]['TOTAL(mm_w.e)']
subset = (np.isfinite(dD) & np.isfinite(pre))
np.average(dD[subset], weights=pre[subset])


#---------------- check am
BS13_Dome_C['am']
d18O = BS13_Dome_C_1d['d18O']
pre = BS13_Dome_C_1d['TOTAL(mm_w.e)']
subset = (np.isfinite(d18O) & np.isfinite(pre))
np.average(d18O[subset], weights=pre[subset])


#-------------------------------- mm
BS13_Dome_C['mm'] = BS13_Dome_C['1d'][['date', 'temp2', 'pre']].groupby(BS13_Dome_C['1d']['date'].dt.month).mean()

# BS13_Dome_C['mm']['dD']
BS13_Dome_C['1d'][['date', 'pre', 'dD']].groupby(BS13_Dome_C['1d']['date'].dt.month).apply()

# https://stackoverflow.com/questions/31521027/groupby-weighted-average-and-sum-in-pandas-dataframe

wm = lambda x: np.average(x, weights=df.loc[x.index, "adjusted_lots"])



dln_no_nan = BS13_Dome_C['1d']['d_ln'].values[np.isfinite()]
np.argmin(BS13_Dome_C['1d']['d_ln'].values)

BS13_Dome_C['1d'].columns
'''
# endregion
# -----------------------------------------------------------------------------



