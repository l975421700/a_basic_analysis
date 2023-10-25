

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
# region import data

fl_cruise_data = sorted(glob.glob('data_sources/water_isotopes/BJ19/datasets/PS*_water_vapour_stable_isotopes.tab'))
fl_cruise_data = np.concatenate([fl_cruise_data[7:], fl_cruise_data[:7]])

fl_label = [y.replace('_water_vapour_stable_isotopes.tab', '') for y in [x.replace('data_sources/water_isotopes/BJ19/datasets/', '') for x in fl_cruise_data]]


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


BJ19_polarstern = {}
BJ19_polarstern['1h'] = pd.DataFrame()
BJ19_polarstern['6h'] = pd.DataFrame()
BJ19_polarstern['1d'] = pd.DataFrame()

for ilabel, ifile in zip(fl_label, fl_cruise_data):
    # ilabel = 'PS93.1'
    # ifile = 'data_sources/water_isotopes/BJ19/datasets/PS93.1_water_vapour_stable_isotopes.tab'
    print('#-------------------------------- ' + ilabel)
    print(ifile)
    
    BJ19_polarstern[ilabel] = pd.read_csv(
        ifile, sep='\t', header=0, skiprows=21,)
    
    BJ19_polarstern[ilabel]['Date/Time'] = BJ19_polarstern[ilabel]['Date/Time'].astype('datetime64[ns]')
    
    BJ19_polarstern[ilabel] = BJ19_polarstern[ilabel].rename(columns={
        'Date/Time': 'time',
        'Latitude': 'lat',
        'Longitude': 'lon',
        'Humidity spec [g/kg]': 'q',
        'δ18O H2O vapour [‰ SMOW]': 'd18O',
        'δD H2O vapour [‰ SMOW]': 'dD',
    }).drop(columns='d xs [‰]')
    
    BJ19_polarstern[ilabel + '_6h'] = BJ19_polarstern[ilabel].resample(
        '6h', on='time').apply(six_hourly_mean).reset_index()
    
    BJ19_polarstern[ilabel + '_6h'] = BJ19_polarstern[ilabel + '_6h'].dropna(
        subset=['lat', 'lon']).reset_index()
    
    BJ19_polarstern[ilabel + '_1d'] = BJ19_polarstern[ilabel].resample(
        '1d', on='time').apply(daily_mean).reset_index()
    
    BJ19_polarstern[ilabel + '_1d'] = BJ19_polarstern[ilabel + '_1d'].dropna(
        subset=['lat', 'lon']).reset_index()
    
    BJ19_polarstern['1h'] = pd.concat(
        [BJ19_polarstern['1h'],
         BJ19_polarstern[ilabel]],
        ignore_index=True,
    )
    
    BJ19_polarstern['6h'] = pd.concat(
        [BJ19_polarstern['6h'],
         BJ19_polarstern[ilabel + '_6h']],
        ignore_index=True,
    )
    
    BJ19_polarstern['1d'] = pd.concat(
        [BJ19_polarstern['1d'],
         BJ19_polarstern[ilabel + '_1d']],
        ignore_index=True,
    )

for ialltime in ['1h', '6h', '1d']:
    # ialltime = '1h'
    print('#------------------------ ' + ialltime)
    
    BJ19_polarstern[ialltime]['d_xs'] = BJ19_polarstern[ialltime]['dD'] - 8 * BJ19_polarstern[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + BJ19_polarstern[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + BJ19_polarstern[ialltime]['d18O'] / 1000)
    
    BJ19_polarstern[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

output_file = 'data_sources/water_isotopes/BJ19/BJ19_polarstern.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(BJ19_polarstern, f)




'''
#------------------------------------ check

with open('data_sources/water_isotopes/BJ19/BJ19_polarstern.pkl', 'rb') as f:
    BJ19_polarstern = pickle.load(f)

test_file = pd.read_csv(
    'data_sources/water_isotopes/BJ19/datasets/PS98_water_vapour_stable_isotopes.tab',
    sep='\t', header=0, skiprows=21,)

np.mean(test_file[105:129]['δD H2O vapour [‰ SMOW]'])
BJ19_polarstern['1d'][250:251]['dD']

test_file[:9]
BJ19_polarstern['1d'][245:251]

np.mean(test_file[513:537]['δ18O H2O vapour [‰ SMOW]'])
BJ19_polarstern['1d'][267:268]['d18O']




'''
# endregion
# -----------------------------------------------------------------------------




