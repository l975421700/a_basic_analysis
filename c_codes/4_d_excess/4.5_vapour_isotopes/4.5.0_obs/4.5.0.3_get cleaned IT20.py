

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
# region clean IT20 data


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


IT20_ACE = {}


#-------------------------------- latitude and longitude

IT20_ACE['loc_part1'] = pd.read_csv(
    'data_sources/water_isotopes/IT20/3379590/ACE_filtered_meteorological_data_1min.csv',
    header=0,)

IT20_ACE['loc_part1'] = IT20_ACE['loc_part1'].rename(columns={
    'date_time': 'time',
    'latitude': 'lat',
    'longitude': 'lon',})

IT20_ACE['loc_part1'] = IT20_ACE['loc_part1'][['time', 'lat', 'lon',]]

IT20_ACE['loc_part1']['time'] = pd.to_datetime(IT20_ACE['loc_part1']['time'])

IT20_ACE['loc_part1_1h'] = IT20_ACE['loc_part1'].resample('1h', on='time').mean().reset_index()

IT20_ACE['loc_part2'] = pd.read_csv(
    'data_sources/water_isotopes/IT20/ACE_track_leg0/track_leg0_hourly splitted.csv',
    names=['date', 'hour', 'lat', 'lon'])

years = [str(x)[:4] for x in IT20_ACE['loc_part2']['date']]
months = [str(x)[4:6] for x in IT20_ACE['loc_part2']['date']]
days = [str(x)[6:8] for x in IT20_ACE['loc_part2']['date']]
hours = [str(x) for x in IT20_ACE['loc_part2']['hour']]

IT20_ACE['loc_part2']['time'] = pd.to_datetime(dict(year=years, month=months, day=days, hour=hours), utc=True)

IT20_ACE['loc_part2'] = IT20_ACE['loc_part2'][['time', 'lat', 'lon']]

IT20_ACE['loc_1h'] = pd.concat(
    [IT20_ACE['loc_part2'], IT20_ACE['loc_part1_1h']],
    ignore_index=True)


#-------------------------------- data


IT20_ACE['1h'] = pd.read_csv(
    'data_sources/water_isotopes/IT20/3250790/ACE_watervapour_isotopes_SWI13_1h.csv',
    header=0,)

IT20_ACE['1h'] = IT20_ACE['1h'].rename(columns={
    'time_ISO8601': 'time',
    'd2H': 'dD',
    'H2O': 'humidity',
})

IT20_ACE['1h']['q'] = IT20_ACE['1h']['humidity'] * 18.01528 / (28.9645 * 1e6)
IT20_ACE['1h']['time'] = pd.to_datetime(IT20_ACE['1h']['time'])
IT20_ACE['1h']['time'] = IT20_ACE['1h']['time'] - pd.Timedelta(hours=1)

IT20_ACE['1h'] = IT20_ACE['1h'][['time', 'd18O', 'dD', 'humidity', 'q']]


IT20_ACE['1h'] = IT20_ACE['1h'].merge(IT20_ACE['loc_1h'], on='time')
IT20_ACE['1h'] = IT20_ACE['1h'].dropna(subset=['lat', 'lon'])

IT20_ACE['6h'] = IT20_ACE['1h'].resample('6h', on='time').apply(six_hourly_mean).reset_index()
IT20_ACE['1d'] = IT20_ACE['1h'].resample('1d', on='time').apply(daily_mean).reset_index()


for ialltime in ['1h', '6h', '1d']:
    # ialltime = '1h'
    print('#------------------------ ' + ialltime)
    
    IT20_ACE[ialltime]['d_xs'] = IT20_ACE[ialltime]['dD'] - 8 * IT20_ACE[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + IT20_ACE[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + IT20_ACE[ialltime]['d18O'] / 1000)
    
    IT20_ACE[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

output_file = 'data_sources/water_isotopes/IT20/IT20_ACE.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(IT20_ACE, f)




'''
#-------------------------------- check

with open('data_sources/water_isotopes/IT20/IT20_ACE.pkl', 'rb') as f:
    IT20_ACE = pickle.load(f)


IT20_ACE_loc1 = pd.read_csv(
    'data_sources/water_isotopes/IT20/3379590/ACE_filtered_meteorological_data_1min.csv',
    header=0,)

IT20_ACE_loc2 = pd.read_csv(
    'data_sources/water_isotopes/IT20/ACE_track_leg0/track_leg0_hourly splitted.csv',
    names=['date', 'hour', 'lat', 'lon'])

IT20_ACE_1h = pd.read_csv(
    'data_sources/water_isotopes/IT20/3250790/ACE_watervapour_isotopes_SWI13_1h.csv',
    header=0,)


IT20_ACE['1d'].iloc[20]
np.nanmean(IT20_ACE_1h[479:503]['H2O'])
np.mean(IT20_ACE_loc2[479:503]['lon'])


IT20_ACE['1h'].columns
'''
# endregion
# -----------------------------------------------------------------------------




