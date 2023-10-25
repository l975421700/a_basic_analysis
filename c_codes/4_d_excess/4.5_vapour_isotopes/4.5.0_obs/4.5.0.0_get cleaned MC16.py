

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
# region clean up MC16_Dome_C data

MC16_Dome_C = {}

MC16_Dome_C['1h'] = pd.read_excel(
    'data_sources/water_isotopes/MC16/Dataset_ACP2016-8.xlsx',
    sheet_name='Picarro', header=0, skiprows=11,)

MC16_Dome_C['1h'] = MC16_Dome_C['1h'].rename(columns={
    'Date time (UTC)': 'time',
    'HS (ppmv)': 'humidity',
    'δ18O (‰)': 'd18O',
    'δD (‰)': 'dD',
    'T 3m (°C)': 't_3m',
    'T surf (°C)': 't_surf'
    }).drop(columns='d-excess (‰)')

MC16_Dome_C['1h'] = MC16_Dome_C['1h'][:-7]
MC16_Dome_C['1h']['time'] = MC16_Dome_C['1h']['time'].dt.round('H')

MC16_Dome_C['1h']['q'] = MC16_Dome_C['1h']['humidity'] * 18.01528 / (28.9645 * 1e6)

# 6h
MC16_Dome_C['6h'] = MC16_Dome_C['1h'].resample('6h', on='time').mean().reset_index()

# 1d
MC16_Dome_C['1d'] = MC16_Dome_C['1h'].resample('1d', on='time').mean().reset_index()


for ialltime in ['1h', '6h', '1d']:
    # ialltime = '1h'
    print('#------------------------ ' + ialltime)
    
    # print(MC16_Dome_C[ialltime])
    
    MC16_Dome_C[ialltime]['d_xs'] = MC16_Dome_C[ialltime]['dD'] - 8 * MC16_Dome_C[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + MC16_Dome_C[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + MC16_Dome_C[ialltime]['d18O'] / 1000)
    
    MC16_Dome_C[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
    
    MC16_Dome_C[ialltime]['lat'] = -75.1
    MC16_Dome_C[ialltime]['lon'] = 123.35


output_file = 'data_sources/water_isotopes/MC16/MC16_Dome_C.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(MC16_Dome_C, f)




#------------------------- import data
with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

iind = np.argmax(MC16_Dome_C['1h']['d_ln'])

MC16_Dome_C['1h']['d_ln'][iind]
MC16_Dome_C['1h']['d_xs'][iind]
MC16_Dome_C['1h']['dD'][iind]
MC16_Dome_C['1h']['d18O'][iind]
MC16_Dome_C['1h']['humidity'][iind]


'''
#------------------------- check statistics

stats.describe(MC16_Dome_C['1d']['d_xs'])
stats.describe(MC16_Dome_C['1d']['d_ln'])
stats.describe(MC16_Dome_C['1d']['humidity'])
stats.describe(MC16_Dome_C['1d']['d18O'])
stats.describe(MC16_Dome_C['1d']['dD'])
stats.describe(MC16_Dome_C['1d']['t_3m'])
stats.describe(MC16_Dome_C['1d']['t_surf'])
stats.describe(MC16_Dome_C['1d']['q'])

#------------------------ check dxs
for ialltime in ['1h', '6h', '1d']:
    print('#------------------------ ' + ialltime)
    print(MC16_Dome_C[ialltime])

(MC16_Dome_C['1d']['d_xs'] == MC16_Dome_C['1d']['dD'] - 8 * MC16_Dome_C['1d']['d18O']).all()

#------------------------ check average
MC16_Dome_C['6h']['d18O'][0]
MC16_Dome_C['1h']['d18O'][:6].mean()

MC16_Dome_C['6h']['dD'][0]
MC16_Dome_C['1h']['dD'][:6].mean()


MC16_Dome_C['1d']['d18O'][0]
MC16_Dome_C['1h']['d18O'][:24].mean()

MC16_Dome_C['1d']['dD'][0]
MC16_Dome_C['1h']['dD'][:24].mean()


#------------------------ check differences with weighted average

d18O_weighted = MC16_Dome_C['1h'].set_index('humidity', append=True).resample(
    '6h', on='time').apply(
        lambda x: np.average(x, weights=x.index.get_level_values(1)))

np.nanmax(abs((MC16_Dome_C['6h']['d18O'].values - d18O_weighted['d18O'].values) / MC16_Dome_C['6h']['d18O']))

d18O_weighted = MC16_Dome_C['1h'].set_index('humidity', append=True).resample(
    '1d', on='time').apply(
        lambda x: np.average(x, weights=x.index.get_level_values(1)))

np.nanmax(abs((MC16_Dome_C['1d']['d18O'].values - d18O_weighted['d18O'].values) / MC16_Dome_C['1d']['d18O']))


dD_weighted = MC16_Dome_C['1h'].set_index('humidity', append=True).resample(
    '6h', on='time').apply(
        lambda x: np.average(x, weights=x.index.get_level_values(1)))

np.nanmax(abs((MC16_Dome_C['6h']['dD'].values - dD_weighted['dD'].values) / MC16_Dome_C['6h']['dD']))

dD_weighted = MC16_Dome_C['1h'].set_index('humidity', append=True).resample(
    '1d', on='time').apply(
        lambda x: np.average(x, weights=x.index.get_level_values(1)))

np.nanmax(abs((MC16_Dome_C['1d']['dD'].values - dD_weighted['dD'].values) / MC16_Dome_C['1d']['dD']))


# check nan values
np.isnan(MC16_Dome_C['1h']['d18O']).sum()
np.isnan(MC16_Dome_C['1d']['d18O']).sum()

np.mean(MC16_Dome_C['1h'][450:456]['dD'])
MC16_Dome_C['6h'][75:76]['dD']


'''
# endregion
# -----------------------------------------------------------------------------



