

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
# region get original data

hadcm3_128k_output = {}

hadcm3_128k_output['SAT'] = xr.open_dataset('scratch/share/from_rahul/data_qingang/Max_128ka_control/xkdhi.temp_mm_1_5m.monthly.nc')['temp_mm_1_5m'].squeeze().rename({
            't': 'time',
            'latitude': 'lat',
            'longitude': 'lon',
    }).sel(time=slice('2653-01-16', '2752-12-16')).rename('SAT')

hadcm3_128k_output['SST'] = xr.open_dataset('scratch/share/from_rahul/data_qingang/Max_128ka_control/xkdhi.SST.monthly.nc')['temp_mm_uo'].squeeze().rename({
            't': 'time',
            'latitude': 'lat',
            'longitude': 'lon',
    }).sel(time=slice('2653-01-16', '2752-12-16')).rename('SST')

hadcm3_128k_output['SIC'] = xr.open_dataset('scratch/share/from_rahul/data_qingang/Max_128ka_control/xkdhi.iceconc_mm_srf.monthly.nc')['iceconc_mm_srf'].squeeze().rename({
            't': 'time',
            'latitude': 'lat',
            'longitude': 'lon',
    }).sel(time=slice('2653-01-16', '2752-12-16')).rename('SIC')


with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_output.pkl', 'wb') as f:
    pickle.dump(hadcm3_128k_output, f)




'''
with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_output.pkl', 'rb') as f:
    hadcm3_128k_output = pickle.load(f)


with open('scratch/share/from_rahul/data_qingang/hadcm3_output_cleaned.pkl', 'rb') as f:
    hadcm3_output_cleaned = pickle.load(f)

hadcm3_output_cleaned['PI']['SAT']
hadcm3_128k_output['SAT']

hadcm3_output_cleaned['PI']['SST']
hadcm3_128k_output['SST']

hadcm3_output_cleaned['PI']['SIC']
hadcm3_128k_output['SIC']

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded data

with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_output.pkl', 'rb') as f:
    hadcm3_128k_output = pickle.load(f)

hadcm3_128k_regridded = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_128k_regridded[ivar] = regrid(hadcm3_128k_output[ivar])

with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded.pkl', 'wb') as f:
    pickle.dump(hadcm3_128k_regridded, f)


'''
with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded.pkl', 'rb') as f:
    hadcm3_128k_regridded = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded data

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded.pkl', 'rb') as f:
    hadcm3_128k_regridded = pickle.load(f)

hadcm3_128k_regridded_alltime = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_128k_regridded_alltime[ivar] = mon_sea_ann(var_monthly=hadcm3_128k_regridded[ivar], seasons = 'Q-MAR',)
    
    hadcm3_128k_regridded_alltime[ivar]['mm'] = hadcm3_128k_regridded_alltime[ivar]['mm'].rename({'month': 'time'})
    hadcm3_128k_regridded_alltime[ivar]['sm'] = hadcm3_128k_regridded_alltime[ivar]['sm'].rename({'month': 'time'})
    hadcm3_128k_regridded_alltime[ivar]['am'] = hadcm3_128k_regridded_alltime[ivar]['am'].expand_dims('time', axis=0)


hadcm3_128k_regridded_alltime['128k_PI'] = {}

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_128k_regridded_alltime['128k_PI'][ivar] = mon_sea_ann(var_monthly=(hadcm3_128k_regridded[ivar] - hadcm3_output_regridded['PI'][ivar].values).compute(), seasons = 'Q-MAR',)
    
    hadcm3_128k_regridded_alltime['128k_PI'][ivar]['mm'] = \
        hadcm3_128k_regridded_alltime['128k_PI'][ivar]['mm'].rename({'month': 'time'})
    hadcm3_128k_regridded_alltime['128k_PI'][ivar]['sm'] = \
        hadcm3_128k_regridded_alltime['128k_PI'][ivar]['sm'].rename({'month': 'time'})
    hadcm3_128k_regridded_alltime['128k_PI'][ivar]['am'] = \
        hadcm3_128k_regridded_alltime['128k_PI'][ivar]['am'].expand_dims('time', axis=0)

with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded_alltime.pkl', 'wb') as f:
    pickle.dump(hadcm3_128k_regridded_alltime, f)


'''
with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded_alltime.pkl', 'rb') as f:
    hadcm3_128k_regridded_alltime = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


