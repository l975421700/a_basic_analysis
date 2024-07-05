

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region general settings

sim_periods = ['LIG0.25_extended']

sim_folder = {
    'LIG0.25_extended': 'xpvga',
}

var_names = {
    'SAT': 'temp_mm_1_5m',
    'SIC': 'iceconc_mm_srf',
    'SST': 'temp_mm_uo',
}

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get original data

hadcm3_extended_output = {}

iperiod = 'LIG0.25_extended'
hadcm3_extended_output[iperiod] = {}
for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_extended_output[iperiod][ivar] = xr.open_dataset('scratch/share/from_rahul/data_qingang/xpvga/' + ivar + '_xpvga.nc')[var_names[ivar]].squeeze().rename(ivar).sel(time=slice('5751-01-17', '6050-12-17'))

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_output.pkl', 'wb') as f:
    pickle.dump(hadcm3_extended_output, f)



'''
#-------------------------------- check

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_output.pkl', 'rb') as f:
    hadcm3_extended_output = pickle.load(f)

hadcm3_extended_output['LIG0.25_extended']['SAT']

hadcm3_extended_output['LIG0.25_extended']['SIC']

hadcm3_extended_output['LIG0.25_extended']['SST']

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded data

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_output.pkl', 'rb') as f:
    hadcm3_extended_output = pickle.load(f)

hadcm3_extended_regridded = {}

iperiod = 'LIG0.25_extended'
hadcm3_extended_regridded[iperiod] = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_extended_regridded[iperiod][ivar] = regrid(hadcm3_extended_output[iperiod][ivar])

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_regridded.pkl', 'wb') as f:
    pickle.dump(hadcm3_extended_regridded, f)




'''
#-------------------------------- check
with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_output.pkl', 'rb') as f:
    hadcm3_extended_output = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_regridded.pkl', 'rb') as f:
    hadcm3_extended_regridded = pickle.load(f)

hadcm3_extended_regridded['LIG0.25_extended']['SAT']
hadcm3_extended_regridded['LIG0.25_extended']['SIC']
hadcm3_extended_regridded['LIG0.25_extended']['SST']

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded data and anomalies

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_regridded.pkl', 'rb') as f:
    hadcm3_extended_regridded = pickle.load(f)


hadcm3_extended_regridded_alltime = {}

iperiod = 'LIG0.25_extended'
hadcm3_extended_regridded_alltime[iperiod] = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_extended_regridded_alltime[iperiod][ivar] = mon_sea_ann(var_monthly=hadcm3_extended_regridded[iperiod][ivar], seasons = 'Q-MAR',)
    
    hadcm3_extended_regridded_alltime[iperiod][ivar]['mm'] = \
        hadcm3_extended_regridded_alltime[iperiod][ivar]['mm'].rename({'month': 'time'})
    hadcm3_extended_regridded_alltime[iperiod][ivar]['sm'] = \
        hadcm3_extended_regridded_alltime[iperiod][ivar]['sm'].rename({'month': 'time'})
    hadcm3_extended_regridded_alltime[iperiod][ivar]['am'] = \
        hadcm3_extended_regridded_alltime[iperiod][ivar]['am'].expand_dims('time', axis=0)


hadcm3_extended_regridded_alltime[iperiod + '_PI'] = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar] = mon_sea_ann(var_monthly=(hadcm3_extended_regridded[iperiod][ivar][-1200:] - hadcm3_output_regridded['PI'][ivar].values).compute(), seasons = 'Q-MAR',)
    
    hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['mm'] = \
        hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['mm'].rename({'month': 'time'})
    hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['sm'] = \
        hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['sm'].rename({'month': 'time'})
    hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['am'] = \
        hadcm3_extended_regridded_alltime[iperiod + '_PI'][ivar]['am'].expand_dims('time', axis=0)

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_regridded_alltime.pkl', 'wb') as f:
    pickle.dump(hadcm3_extended_regridded_alltime, f)




'''
#-------------------------------- check

with open('scratch/share/from_rahul/data_qingang/hadcm3_extended_regridded_alltime.pkl', 'rb') as f:
    hadcm3_extended_regridded_alltime = pickle.load(f)

hadcm3_extended_regridded_alltime.keys()

hadcm3_extended_regridded_alltime['LIG0.25_extended']['SAT'].keys()


hadcm3_extended_regridded_alltime['LIG0.25_extended_PI']['SAT'].keys()


'''
# endregion
# -----------------------------------------------------------------------------


