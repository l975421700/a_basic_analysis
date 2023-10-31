

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

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region clean CB19 data

def hourly_mean(arrays):
    if (np.isfinite(arrays).sum() < 30):
        return(np.nan)
    else:
        return(np.nanmean(arrays))


def six_hourly_mean(arrays):
    if (np.isfinite(arrays).sum() < 3):
        return(np.nan)
    else:
        return(np.nanmean(arrays))


def daily_mean(arrays):
    if (np.isfinite(arrays).sum() < 12):
        return(np.nan)
    else:
        return(np.nanmean(arrays))

CB19_DDU = {}

CB19_DDU['1min'] = pd.read_csv(
    'data_sources/water_isotopes/CB19/Breant_2019.txt',
    sep = '\s+', header=0, skiprows=4,)

CB19_DDU['1min'] = CB19_DDU['1min'][['Date.1', 'timeUTC.1', 'humidity.1', 'd18O.1', 'dD.1', ]]

CB19_DDU['1min'] = CB19_DDU['1min'].dropna(subset=['Date.1', 'timeUTC.1',])

CB19_DDU['1min']['time'] = pd.to_datetime([x + ' ' + y for x,y in zip(CB19_DDU['1min']['Date.1'], CB19_DDU['1min']['timeUTC.1'])], dayfirst=True)

CB19_DDU['1min'] = CB19_DDU['1min'].rename(columns={
    'd18O.1': 'd18O',
    'dD.1': 'dD',
    'humidity.1': 'humidity'
})

CB19_DDU['1min'] = CB19_DDU['1min'][['time', 'humidity', 'd18O', 'dD']]

CB19_DDU['1min']['q'] = CB19_DDU['1min']['humidity'] * 18.01528 / (28.9645 * 1e6)

CB19_DDU['1h'] = CB19_DDU['1min'].resample('1h', on='time').apply(hourly_mean).reset_index()

CB19_DDU['6h'] = CB19_DDU['1h'].resample('6h', on='time').apply(six_hourly_mean).reset_index()
CB19_DDU['1d'] = CB19_DDU['1h'].resample('1d', on='time').apply(daily_mean).reset_index()

for ialltime in ['1h', '6h', '1d']:
    # ialltime = '1h'
    print('#------------------------ ' + ialltime)
    
    CB19_DDU[ialltime]['d_xs'] = CB19_DDU[ialltime]['dD'] - 8 * CB19_DDU[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + CB19_DDU[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + CB19_DDU[ialltime]['d18O'] / 1000)
    
    CB19_DDU[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
    
    CB19_DDU[ialltime]['lat'] = -66.7
    CB19_DDU[ialltime]['lon'] = 140

output_file = 'data_sources/water_isotopes/CB19/CB19_DDU.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(CB19_DDU, f)




'''
#-------------------------------- check
with open('data_sources/water_isotopes/CB19/CB19_DDU.pkl', 'rb') as f:
    CB19_DDU = pickle.load(f)

CB19_DDU_1min = pd.read_csv(
    'data_sources/water_isotopes/CB19/Breant_2019.txt',
    sep = '\s+', header=0, skiprows=4,)

np.nanmean(CB19_DDU_1min[CB19_DDU_1min['Date'] == '29/12/16']['humidity.1'])
CB19_DDU['1d'].iloc[4]



'''
# endregion
# -----------------------------------------------------------------------------


