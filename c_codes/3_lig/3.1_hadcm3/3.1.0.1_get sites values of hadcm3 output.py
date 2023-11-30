

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    mon_sea_ann,
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


#-------------------------------- import simulations

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime = pickle.load(f)

lon = hadcm3_output_regridded_alltime['PI']['SST']['am'].lon.values
lat = hadcm3_output_regridded_alltime['PI']['SST']['am'].lat.values

#-------------------------------- import reconstructions

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region settings

lig_datasets = {}
lig_anom_name = {}
lig_twosigma_name = {}

# annual_sst
lig_datasets['annual_sst'] = {
    'EC': lig_recs['EC']['SO_ann'],
    'JH': lig_recs['JH']['SO_ann'],
    'DC': lig_recs['DC']['annual_128'],
    'MC': None,}
lig_anom_name['annual_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_ann',
    'MC': None,}
lig_twosigma_name['annual_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# summer_sst
lig_datasets['summer_sst'] = {
    'EC': lig_recs['EC']['SO_jfm'],
    'JH': lig_recs['JH']['SO_jfm'],
    'DC': lig_recs['DC']['JFM_128'],
    'MC': lig_recs['MC']['interpolated'],}
lig_anom_name['summer_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_jfm',
    'MC': 'sst_anom_hadisst_jfm',}
lig_twosigma_name['summer_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# annual_sat
lig_datasets['annual_sat'] = {
    'EC': lig_recs['EC']['AIS_am'],
    'JH': None,
    'DC': None,
    'MC': None,}
lig_anom_name['annual_sat'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}
lig_twosigma_name['annual_sat'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}

# sep_sic
lig_datasets['sep_sic'] = {
    'EC': None,
    'JH': None,
    'DC': None,
    'MC': lig_recs['MC']['interpolated'],}
lig_anom_name['sep_sic'] = {
    'EC': None,
    'JH': None,
    'DC': None,
    'MC': 'sic_anom_hadisst_sep',}
lig_twosigma_name['sep_sic'] = {
    'EC': None,
    'JH': None,
    'DC': None,
    'MC': None,}


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

hadcm3_output_site_values = {}

for iproxy in ['annual_sst', 'summer_sst', 'annual_sat', 'sep_sic',]:
    # iproxy = 'annual_sst'
    print('#-------------------------------- ' + iproxy)
    
    hadcm3_output_site_values[iproxy] = {}
    
    if (iproxy == 'annual_sst'):
        ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SST']['am'].squeeze().values
        ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['am'].squeeze().values
    elif (iproxy == 'summer_sst'):
        ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SST']['sm'].sel(time=3).values
        ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['sm'].sel(time=3).values
    elif (iproxy == 'annual_sat'):
        ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SAT']['am'].squeeze().values
        ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SAT']['am'].squeeze().values
    elif (iproxy == 'sep_sic'):
        ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SIC']['mm'].sel(time=9).values * 100
        ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SIC']['mm'].sel(time=9).values * 100
    
    for irec in ['EC', 'JH', 'DC', 'MC',]:
        # irec = 'EC'
        print('#---------------- ' + irec)
        
        if (lig_datasets[iproxy][irec] is not None):
            if (lig_twosigma_name[iproxy][irec] is not None):
                hadcm3_output_site_values[iproxy][irec] = \
                    lig_datasets[iproxy][irec][[
                        'Station', 'Latitude', 'Longitude',
                        lig_anom_name[iproxy][irec],
                        lig_twosigma_name[iproxy][irec],
                        ]].rename(columns={
                            lig_anom_name[iproxy][irec]: 'rec_lig_pi',
                            lig_twosigma_name[iproxy][irec]: 'rec_lig_pi_2sigma',
                        })
            elif (lig_twosigma_name[iproxy][irec] is None):
                hadcm3_output_site_values[iproxy][irec] = \
                    lig_datasets[iproxy][irec][[
                        'Station', 'Latitude', 'Longitude',
                        lig_anom_name[iproxy][irec],
                        ]].rename(columns={
                            lig_anom_name[iproxy][irec]: 'rec_lig_pi',
                        })
            
            hadcm3_output_site_values[iproxy][irec]['sim_lig_pi'] = \
                find_multi_gridvalue_at_site(
                    hadcm3_output_site_values[iproxy][irec]['Latitude'].values,
                    hadcm3_output_site_values[iproxy][irec]['Longitude'].values,
                    lat,
                    lon,
                    ivar1,
                )
            hadcm3_output_site_values[iproxy][irec]['sim_lig0.25Sv_pi'] = \
                find_multi_gridvalue_at_site(
                    hadcm3_output_site_values[iproxy][irec]['Latitude'].values,
                    hadcm3_output_site_values[iproxy][irec]['Longitude'].values,
                    lat,
                    lon,
                    ivar2,
                )
            
            hadcm3_output_site_values[iproxy][irec]['sim_rec_lig_pi'] = \
                hadcm3_output_site_values[iproxy][irec]['sim_lig_pi'] - \
                    hadcm3_output_site_values[iproxy][irec]['rec_lig_pi']
            hadcm3_output_site_values[iproxy][irec]['sim_rec_lig0.25Sv_pi'] = \
                hadcm3_output_site_values[iproxy][irec]['sim_lig0.25Sv_pi'] - \
                    hadcm3_output_site_values[iproxy][irec]['rec_lig_pi']
            
            # print(hadcm3_output_site_values[iproxy][irec])

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values.pkl', 'wb') as f:
    pickle.dump(hadcm3_output_site_values, f)




'''

#-------------------------------- check
with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values.pkl', 'rb') as f:
    hadcm3_output_site_values = pickle.load(f)

for iproxy in hadcm3_output_site_values.keys():
    # iproxy = 'annual_sst'
    print('#-------------------------------- ' + iproxy)
    
    for irec in hadcm3_output_site_values[iproxy].keys():
        # irec = 'EC'
        print('#---------------- ' + irec)
        
        print(hadcm3_output_site_values[iproxy][irec])


#---------------- annual_sst
ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SST']['am'].squeeze().values
ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['am'].squeeze().values

iproxy = 'annual_sst'
irec = 'EC'

iproxy = 'annual_sst'
irec = 'JH'

iproxy = 'annual_sst'
irec = 'DC'

#---------------- summer_sst

ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SST']['sm'].sel(time=3).values
ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['sm'].sel(time=3).values

iproxy = 'summer_sst'
irec = 'EC'

iproxy = 'summer_sst'
irec = 'JH'

iproxy = 'summer_sst'
irec = 'DC'

iproxy = 'summer_sst'
irec = 'MC'

#---------------- annual_sat

iproxy = 'annual_sat'
irec = 'EC'

ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SAT']['am'].squeeze().values
ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SAT']['am'].squeeze().values

#---------------- sep_sic

iproxy = 'sep_sic'
irec = 'MC'

ivar1 = hadcm3_output_regridded_alltime['LIG_PI']['SIC']['mm'].sel(time=9).values
ivar2 = hadcm3_output_regridded_alltime['LIG0.25_PI']['SIC']['mm'].sel(time=9).values

print(lig_datasets[iproxy][irec][['Station', 'Latitude', 'Longitude', lig_anom_name[iproxy][irec]]])
print(hadcm3_output_site_values[iproxy][irec])

find_multi_gridvalue_at_site(
    hadcm3_output_site_values[iproxy][irec].Latitude.values,
    hadcm3_output_site_values[iproxy][irec].Longitude.values,
    lat, lon, ivar1,)
find_multi_gridvalue_at_site(
    hadcm3_output_site_values[iproxy][irec].Latitude.values,
    hadcm3_output_site_values[iproxy][irec].Longitude.values,
    lat, lon, ivar2,)



            if (lig_twosigma_name[iproxy][irec] is not None):
                print(lig_datasets[iproxy][irec][[
                    'Station', 'Latitude', 'Longitude',
                    lig_anom_name[iproxy][irec],
                    lig_twosigma_name[iproxy][irec],
                    ]])
            elif (lig_twosigma_name[iproxy][irec] is None):
                print(lig_datasets[iproxy][irec][[
                    'Station', 'Latitude', 'Longitude',
                    lig_anom_name[iproxy][irec],
                    ]])
'''
# endregion
# -----------------------------------------------------------------------------


