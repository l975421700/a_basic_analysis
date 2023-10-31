

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


SD21_Neumayer = {}

SD21_Neumayer['1h'] = pd.read_csv(
    'data_sources/water_isotopes/SD21/BagheriDastgerdi-etal_2021.tab',
    sep='\t', header=0, skiprows=26,)

SD21_Neumayer['1h'] = SD21_Neumayer['1h'].rename(columns={
    'Date/Time': 'time',
    'WV mix ratio [g/kg]': 'w',
    'δ18O H2O [‰ SMOW]': 'd18O',
    'δD H2O vapour [‰ SMOW]': 'dD',
})

SD21_Neumayer['1h']['time'] = SD21_Neumayer['1h']['time'].astype('datetime64[ns]')
SD21_Neumayer['1h']['q'] = SD21_Neumayer['1h']['w'] / (1 + SD21_Neumayer['1h']['w'])

SD21_Neumayer['1h'] = SD21_Neumayer['1h'][['time', 'q', 'd18O', 'dD']]

SD21_Neumayer['6h'] = SD21_Neumayer['1h'].resample('6h', on='time').apply(six_hourly_mean).reset_index()

SD21_Neumayer['1d'] = SD21_Neumayer['1h'].resample('1d', on='time').apply(daily_mean).reset_index()

for ialltime in ['1h', '6h', '1d']:
    print(ialltime)
    
    SD21_Neumayer[ialltime]['d_xs'] = SD21_Neumayer[ialltime]['dD'] - 8 * SD21_Neumayer[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + SD21_Neumayer[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + SD21_Neumayer[ialltime]['d18O'] / 1000)
    
    SD21_Neumayer[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
    
    SD21_Neumayer[ialltime]['lat'] = -70.65
    SD21_Neumayer[ialltime]['lon'] = -8.25


output_file = 'data_sources/water_isotopes/SD21/SD21_Neumayer.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(SD21_Neumayer, f)




'''
#-------------------------------- check

with open('data_sources/water_isotopes/SD21/SD21_Neumayer.pkl', 'rb') as f:
    SD21_Neumayer = pickle.load(f)

SD21_Neumayer_1h = pd.read_csv(
    'data_sources/water_isotopes/SD21/BagheriDastgerdi-etal_2021.tab',
    sep='\t', header=0, skiprows=26,)

np.nanmean(SD21_Neumayer_1h[214:238]['δ18O H2O [‰ SMOW]'])
SD21_Neumayer['1d'].iloc[9]
np.nanmean(SD21_Neumayer_1h[214:238]['δD H2O vapour [‰ SMOW]'])
'''
# endregion
# -----------------------------------------------------------------------------



