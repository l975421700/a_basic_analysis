

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

sim_periods = ['PI', 'LIG', 'LIG0.25']

sim_folder = {
    'PI': 'xpjaa',
    'LIG': 'xpkba',
    'LIG0.25': 'xppfa',
}

var_names = {
    'SAT': 'temp_mm_1_5m',
    'SIC': 'iceconc_mm_uo',
    'SST': 'temp_mm_uo',
}

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get original data


hadcm3_output = {}

for iperiod in ['PI', 'LIG',]:
    # iperiod = 'LIG'
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output[iperiod] = {}
    
    for ivar in ['SAT', 'SIC', 'SST']:
        # ivar = 'SAT'
        print('#---------------- ' + ivar)
        
        filelists = sorted(glob.glob('scratch/share/from_rahul/data_qingang/' + sim_folder[iperiod] + '/' + ivar + '/*'))
        
        hadcm3_output[iperiod][ivar] = xr.open_mfdataset(filelists)[
            var_names[ivar]]

var_names = {
    'SAT': 'temp_mm_1_5m',
    'SIC': 'iceconc_mm_srf',
    'SST': 'temp_mm_uo',
}

iperiod = 'LIG0.25'
hadcm3_output[iperiod] = {}
for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_output[iperiod][ivar] = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/29nov23/xppfa_' + ivar + '_3000yrs.nc')[var_names[ivar]]

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'wb') as f:
    pickle.dump(hadcm3_output, f)




'''
#-------------------------------- check

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

#---------------- PI
iperiod = 'PI'; ivar = 'SAT'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpjaa/SAT/xpjaaa@pd*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.temp_mm_1_5m.values
data2 = hadcm3_output[iperiod][ivar].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


iperiod = 'PI'; ivar = 'SIC'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpjaa/SIC/xpjaao@pf*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.iceconc_mm_uo.values
data2 = hadcm3_output[iperiod][ivar].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


iperiod = 'PI'; ivar = 'SST'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpjaa/SST/xpjaao@pf*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.temp_mm_uo.values
data2 = hadcm3_output[iperiod][ivar].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


#---------------- LIG
iperiod = 'LIG'; ivar = 'SAT'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpkba/SAT/xpkbaa@pd*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.temp_mm_1_5m.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


iperiod = 'LIG'; ivar = 'SIC'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpkba/SIC/xpkbao@pf*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.iceconc_mm_uo.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


iperiod = 'LIG'; ivar = 'SST'
filelist = sorted(glob.glob('scratch/share/from_rahul/data_qingang/xpkba/SST/xpkbao@pf*.nc'))
dataset = xr.open_mfdataset(filelist)

data1 = dataset.temp_mm_uo.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---------------- LIG
iperiod = 'LIG0.25'; ivar = 'SAT'
dataset = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/xppfa_SAT_bristol_BAS_13sep23.nc')

data1 = dataset.temp_mm_1_5m.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


iperiod = 'LIG0.25'; ivar = 'SIC'
dataset = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/xppfa_SIC_bristol_BAS_13sep23.nc')

data1 = dataset.iceconc_mm_srf.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


iperiod = 'LIG0.25'; ivar = 'SST'
dataset = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/xppfa_SST_bristol_BAS_13sep23.nc')

data1 = dataset.temp_mm_uo.values
data2 = hadcm3_output[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())






#-------------------------------- check length

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

for iperiod in ['LIG0.25']:
    # iperiod = 'LIG'
    # ['PI', 'LIG', 'LIG0.25']
    print('#-------------------------------- ' + iperiod)
    
    for ivar in ['SAT', 'SIC', 'SST']:
        # ivar = 'SAT'
        print('#---------------- ' + ivar)
        
        #---------------- length
        print(hadcm3_output[iperiod][ivar])

# hadcm3_output['LIG0.25']['SST'].time
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get cleaned data

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

hadcm3_output_cleaned = {}

iperiod = 'PI'
hadcm3_output_cleaned[iperiod] = {}
# hadcm3_output[iperiod]

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_output_cleaned[iperiod][ivar] = \
        hadcm3_output[iperiod][ivar].squeeze().rename({
            't': 'time',
            'latitude': 'lat',
            'longitude': 'lon',
    }).rename(ivar).sel(time=slice('2371-01-17', '2470-12-17'))


iperiod = 'LIG'
hadcm3_output_cleaned[iperiod] = {}
# hadcm3_output[iperiod]

for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_output_cleaned[iperiod][ivar] = \
        hadcm3_output[iperiod][ivar].squeeze().rename({
            't': 'time',
            'latitude': 'lat',
            'longitude': 'lon',
    }).rename(ivar).sel(time=slice('2270-01-17', '2369-12-17'))


iperiod = 'LIG0.25'
hadcm3_output_cleaned[iperiod] = {}
# hadcm3_output[iperiod]

hadcm3_output_cleaned[iperiod]['SAT'] = \
    hadcm3_output[iperiod]['SAT'].squeeze().rename(
        'SAT').sel(time=slice('4737-01-17', '4836-12-17'))

hadcm3_output_cleaned[iperiod]['SIC'] = \
    hadcm3_output[iperiod]['SIC'].squeeze().rename(
        'SIC').sel(time=slice('4737-01-17', '4836-12-17'))

hadcm3_output_cleaned[iperiod]['SST'] = \
    hadcm3_output[iperiod]['SST'].squeeze().rename(
        'SST')

hadcm3_output_cleaned[iperiod]['SST']['time'] = hadcm3_output[iperiod]['SAT']['time'][:-1]
hadcm3_output_cleaned[iperiod]['SST'] = \
    hadcm3_output_cleaned[iperiod]['SST'].sel(
        time=slice('4737-01-17', '4836-12-17'))

hadcm3_output_cleaned[iperiod]['SST']['lon'] = hadcm3_output_cleaned['PI']['SST']['lon']
hadcm3_output_cleaned[iperiod]['SST']['lat'] = hadcm3_output_cleaned['PI']['SST']['lat']

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_cleaned.pkl', 'wb') as f:
    pickle.dump(hadcm3_output_cleaned, f)




'''
#-------------------------------- check

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_cleaned.pkl', 'rb') as f:
    hadcm3_output_cleaned = pickle.load(f)

iperiod = 'PI'
for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    data1 = hadcm3_output[iperiod][ivar].sel(t=slice('2371-01-17', '2470-12-17')).squeeze().values
    data2 = hadcm3_output_cleaned[iperiod][ivar].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

iperiod = 'LIG'
for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    data1 = hadcm3_output[iperiod][ivar].sel(t=slice('2270-01-17', '2369-12-17')).squeeze().values
    data2 = hadcm3_output_cleaned[iperiod][ivar].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

iperiod = 'LIG0.25'
for ivar in ['SAT', 'SIC',]:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    data1 = hadcm3_output[iperiod][ivar].sel(time=slice('4737-01-17', '4836-12-17')).squeeze().values
    data2 = hadcm3_output_cleaned[iperiod][ivar].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

ivar= 'SST'
# hadcm3_output[iperiod]['SAT'].time[:-11]
data1 = hadcm3_output[iperiod][ivar][-1210:-10].squeeze().values
data2 = hadcm3_output_cleaned[iperiod][ivar].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




#-------------------------------- check length

for iperiod in ['PI', 'LIG', 'LIG0.25']:
    print('#-------------------------------- ' + iperiod)
    
    for ivar in ['SAT', 'SIC', 'SST']:
        print('#---------------- ' + ivar)
        
        #---------------- length
        print(hadcm3_output[iperiod][ivar].shape)
        print(hadcm3_output_cleaned[iperiod][ivar].shape)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded data

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_cleaned.pkl', 'rb') as f:
    hadcm3_output_cleaned = pickle.load(f)

hadcm3_output_regridded = {}

for iperiod in hadcm3_output_cleaned.keys():
    # iperiod = 'LIG0.25'
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output_regridded[iperiod] = {}
    
    for ivar in hadcm3_output_cleaned[iperiod].keys():
        # ivar = 'SST'
        print('#---------------- ' + ivar)
        
        # print(hadcm3_output_cleaned[iperiod][ivar].shape)
        
        hadcm3_output_regridded[iperiod][ivar] = regrid(hadcm3_output_cleaned[iperiod][ivar])

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'wb') as f:
    pickle.dump(hadcm3_output_regridded, f)



'''
#-------------------------------- check
with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_cleaned.pkl', 'rb') as f:
    hadcm3_output_cleaned = pickle.load(f)

iperiod = 'PI'
ivar = 'SAT'
hadcm3_output_regridded[iperiod][ivar].to_netcdf('scratch/test/test0.nc')
hadcm3_output_cleaned[iperiod][ivar].to_netcdf('scratch/test/test1.nc')


#-------------------------------- check length
with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

for iperiod in ['PI', 'LIG', 'LIG0.25']:
    print('#-------------------------------- ' + iperiod)
    
    for ivar in ['SAT', 'SIC', 'SST']:
        print('#---------------- ' + ivar)
        
        #---------------- length
        print(hadcm3_output_regridded[iperiod][ivar].shape)


# hadcm3_output_cleaned[iperiod][ivar].lat
# hadcm3_output_cleaned[iperiod][ivar].lon
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded data and anomalies

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

hadcm3_output_regridded_alltime = {}

for iperiod in ['PI', 'LIG', 'LIG0.25']:
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output_regridded_alltime[iperiod] = {}
    
    for ivar in ['SAT', 'SIC', 'SST']:
        print('#---------------- ' + ivar)
        
        hadcm3_output_regridded_alltime[iperiod][ivar] = mon_sea_ann(var_monthly=hadcm3_output_regridded[iperiod][ivar], seasons = 'Q-MAR',)
        
        hadcm3_output_regridded_alltime[iperiod][ivar]['mm'] = \
            hadcm3_output_regridded_alltime[iperiod][ivar]['mm'].rename({'month': 'time'})
        hadcm3_output_regridded_alltime[iperiod][ivar]['sm'] = \
            hadcm3_output_regridded_alltime[iperiod][ivar]['sm'].rename({'month': 'time'})
        hadcm3_output_regridded_alltime[iperiod][ivar]['am'] = \
            hadcm3_output_regridded_alltime[iperiod][ivar]['am'].expand_dims('time', axis=0)

for iperiod in ['LIG', 'LIG0.25']:
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output_regridded_alltime[iperiod + '_PI'] = {}
    
    for ivar in ['SAT', 'SIC', 'SST']:
        print('#---------------- ' + ivar)
        
        hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar] = mon_sea_ann(var_monthly=(hadcm3_output_regridded[iperiod][ivar] - hadcm3_output_regridded['PI'][ivar].values).compute(), seasons = 'Q-MAR',)
        
        hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['mm'] = \
            hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['mm'].rename({'month': 'time'})
        hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['sm'] = \
            hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['sm'].rename({'month': 'time'})
        hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['am'] = \
            hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['am'].expand_dims('time', axis=0)

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'wb') as f:
    pickle.dump(hadcm3_output_regridded_alltime, f)




'''
#-------------------------------- check
with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded.pkl', 'rb') as f:
    hadcm3_output_regridded = pickle.load(f)

for iperiod in ['PI', 'LIG', 'LIG0.25']:
    for ivar in ['SAT', 'SIC', 'SST']:
        data1 = hadcm3_output_regridded[iperiod][ivar].values
        data2 = hadcm3_output_regridded_alltime[iperiod][ivar]['mon'].values
        print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

for iperiod in ['LIG', 'LIG0.25']:
    for ivar in ['SAT', 'SIC', 'SST']:
        data1 = (hadcm3_output_regridded[iperiod][ivar] - hadcm3_output_regridded['PI'][ivar].values).values
        data2 = hadcm3_output_regridded_alltime[iperiod + '_PI'][ivar]['mon'].values
        print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

'''
# endregion
# -----------------------------------------------------------------------------
