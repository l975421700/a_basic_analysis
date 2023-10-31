

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
# region get cleaned FR16

FR16_Kohnen = {}

FR16_Kohnen['isotopes'] = pd.read_excel(
    'data_sources/water_isotopes/FR16/FR16_isotopes.xlsx',
    header=0,)

FR16_Kohnen['isotopes'] = FR16_Kohnen['isotopes'].rename(columns={
    'top_time': 'time',
    'top_q': 'humidity',
    'top_dD': 'dD',
    'top_dx': 'd_xs',
})

FR16_Kohnen['isotopes']['d18O'] = (FR16_Kohnen['isotopes']['dD'] - FR16_Kohnen['isotopes']['d_xs']) / 8

FR16_Kohnen['isotopes'] = FR16_Kohnen['isotopes'].drop(columns='d_xs')

FR16_Kohnen['T'] = pd.read_excel(
    'data_sources/water_isotopes/FR16/FR16_T.xlsx',
    header=0,)

FR16_Kohnen['T'] = FR16_Kohnen['T'].rename(columns={
    'weather_time': 'time',
    'weather_T2': 'temp2',
})

def daily_mean(arrays):
    if (np.isfinite(arrays).sum() < 20):
        return(np.nan)
    else:
        return(np.nanmean(arrays))

FR16_Kohnen['isotopes_1d'] = FR16_Kohnen['isotopes'].resample('1d', on='time').apply(daily_mean).reset_index()

def daily_mean(arrays):
    if (np.isfinite(arrays).sum() < 12):
        return(np.nan)
    else:
        return(np.nanmean(arrays))

FR16_Kohnen['T_1d'] = FR16_Kohnen['T'].resample('1d', on='time').apply(daily_mean).reset_index()


FR16_Kohnen['1d'] = pd.merge(
    FR16_Kohnen['isotopes_1d'],
    FR16_Kohnen['T_1d'],)

FR16_Kohnen['1d']['q'] = FR16_Kohnen['1d']['humidity'] * 18.01528 / (28.9645 * 1e6)
FR16_Kohnen['1d']['d_xs'] = FR16_Kohnen['1d']['dD']-8*FR16_Kohnen['1d']['d18O']

ln_dD = 1000 * np.log(1 + FR16_Kohnen['1d']['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + FR16_Kohnen['1d']['d18O'] / 1000)
FR16_Kohnen['1d']['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

FR16_Kohnen['1d']['lat'] = -75
FR16_Kohnen['1d']['lon'] = 0.067

output_file = 'data_sources/water_isotopes/FR16/FR16_Kohnen.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(FR16_Kohnen, f)




'''
#-------------------------------- check

with open('data_sources/water_isotopes/FR16/FR16_Kohnen.pkl', 'rb') as f:
    FR16_Kohnen = pickle.load(f)

FR16_isotopes = pd.read_excel(
    'data_sources/water_isotopes/FR16/FR16_isotopes.xlsx',
    header=0,)
FR16_T = pd.read_excel(
    'data_sources/water_isotopes/FR16/FR16_T.xlsx',
    header=0,)

FR16_isotopes[32:70].mean()
FR16_Kohnen['1d'].iloc[2]




'''
# endregion
# -----------------------------------------------------------------------------



