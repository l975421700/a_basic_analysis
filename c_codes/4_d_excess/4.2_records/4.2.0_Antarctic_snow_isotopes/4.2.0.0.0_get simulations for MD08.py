

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    ]


# -----------------------------------------------------------------------------
# region import packages

# management
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import pandas as pd

from a_basic_analysis.b_module.basic_calculations import (
    find_multi_gridvalue_at_site,
)

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


wisoaprt_alltime = {}
wisoevap_alltime = {}
dO18_alltime = {}
dD_alltime = {}
d_ln_alltime = {}
d_excess_alltime = {}
tsurf_alltime = {}


for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
        wisoevap_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'rb') as f:
        tsurf_alltime[expid[i]] = pickle.load(f)

lon = d_ln_alltime[expid[0]]['am'].lon
lat = d_ln_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


'''
wisoaprt_alltime = {}
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)
    
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region clean Valerie's data

Antarctic_snow_isotopes = pd.read_csv(
    'data_sources/ice_core_records/Antarctic_snow_isotopic_composition/Antarctic_snow_isotopic_composition_DB.tab',
    sep='\t', header=0, skiprows=97,)

Antarctic_snow_isotopes = Antarctic_snow_isotopes.rename(columns={
    'Latitude': 'lat',
    'Longitude': 'lon',
    't [°C]': 'temperature',
    'Acc rate [cm/a] (Calculated)': 'accumulation',
    'δD [‰ SMOW] (Calculated average/mean values)': 'dD',
    'δD std dev [±]': 'dD_std',
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)': 'dO18',
    'δ18O std dev [±]': 'dO18_std',
    'd xs [‰] (Calculated average/mean values)': 'd_excess',
    'd xs std dev [±] (Calculated)': 'd_excess_std',
})

Antarctic_snow_isotopes = Antarctic_snow_isotopes[[
    'lat', 'lon', 'temperature', 'accumulation', 'dD', 'dD_std', 'dO18', 'dO18_std', 'd_excess', 'd_excess_std',
]]

ln_dD = 1000 * np.log(1 + Antarctic_snow_isotopes['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + Antarctic_snow_isotopes['dO18'] / 1000)

Antarctic_snow_isotopes['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

# Antarctic_snow_isotopes = Antarctic_snow_isotopes.dropna(
#     subset=['lat', 'lon'], ignore_index=True)

'''
Antarctic_snow_isotopes.columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract simulations for obserations

Antarctic_snow_isotopes_simulations = {}

for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    Antarctic_snow_isotopes_simulations[expid[i]] = Antarctic_snow_isotopes.copy()
    
    for var_name in ['temperature', 'accumulation', 'dD', 'dD_std', 'dO18', 'dO18_std', 'd_excess', 'd_excess_std', 'd_ln', 'd_ln_std']:
        # var_name = 'd_ln'
        print('#-------- ' + var_name)
        
        if (var_name == 'temperature'):
            ivar = tsurf_alltime[expid[i]]['am']
        elif (var_name == 'accumulation'):
            ivar = ((wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) + wisoevap_alltime[expid[i]]['am'].sel(wisotype=1))  * seconds_per_d / 10 * 365).compute()
        elif (var_name == 'dD'):
            ivar = dD_alltime[expid[i]]['am']
        elif (var_name == 'dD_std'):
            ivar = dD_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
        elif (var_name == 'dO18'):
            ivar = dO18_alltime[expid[i]]['am']
        elif (var_name == 'dO18_std'):
            ivar = dO18_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
        elif (var_name == 'd_excess'):
            ivar = d_excess_alltime[expid[i]]['am']
        elif (var_name == 'd_excess_std'):
            ivar = d_excess_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
        elif (var_name == 'd_ln'):
            ivar = d_ln_alltime[expid[i]]['am']
        elif (var_name == 'd_ln_std'):
            ivar = d_ln_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
        
        Antarctic_snow_isotopes_simulations[expid[i]][var_name + '_sim'] = \
            find_multi_gridvalue_at_site(
                Antarctic_snow_isotopes_simulations[expid[i]]['lat'].values,
                Antarctic_snow_isotopes_simulations[expid[i]]['lon'].values,
                lat.values,
                lon.values,
                ivar.values,
            )
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(Antarctic_snow_isotopes_simulations[expid[i]], f)




'''
#-------------------------------- check

i = 0
Antarctic_snow_isotopes_simulations = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
    Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)

for irecord in np.arange(10, 1000, 10):
    # irecord = 100
    print('irecord: ' + str(irecord))
    
    slat = Antarctic_snow_isotopes_simulations[expid[i]]['lat'][irecord]
    slon = Antarctic_snow_isotopes_simulations[expid[i]]['lon'][irecord]
    
    if (np.isfinite(slat) & np.isfinite(slon)):
        ilat, ilon = find_ilat_ilon_general(slat, slon, lat_2d, lon_2d)
        
        if(abs(slat - lat[ilat].values) > 1.5):
            print('Site vs. grid lat: ' + str(np.round(slat, 1)) + ' vs. ' + str(np.round(lat[ilat].values, 1)))
        
        if (abs(slon - lon[ilon].values) > 2):
            print('Site vs. grid lon: ' + str(np.round(slon, 1)) + ' vs. ' + str(np.round(lon[ilon].values, 1)))
        
        data1 = Antarctic_snow_isotopes_simulations[expid[i]]['d_ln_sim'][irecord]
        data2 = d_ln_alltime[expid[i]]['am'][ilat, ilon].values
        
        if (data1 != data2):
            print('!----------- mismatch')


#-------------------------------- check identical values

i = 0
Antarctic_snow_isotopes_simulations = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
    Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)


values, counts = np.unique(Antarctic_snow_isotopes_simulations[expid[i]]['dD_sim'], return_counts=True)
wheremax = np.where(counts == np.max(counts))[0][0]
mode = values[wheremax]
number = counts[wheremax]


same_values = Antarctic_snow_isotopes_simulations[expid[i]][ (Antarctic_snow_isotopes_simulations[expid[i]]['dD_sim'] == mode) ]

print('latitude range: ' + str(np.max(same_values['lat']) - np.min(same_values['lat'])))
print('longitude range: ' + str(np.max(same_values['lon']) - np.min(same_values['lon'])))


Antarctic_snow_isotopes[ (Antarctic_snow_isotopes_simulations[expid[i]]['dD_sim'] == mode) ]['NAVG [#]']

'''
# endregion
# -----------------------------------------------------------------------------

