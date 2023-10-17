

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    'nudged_701_5.0',
    ]


# -----------------------------------------------------------------------------
# region import packages

# management
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import os

from a_basic_analysis.b_module.basic_calculations import (
    find_ilat_ilon,
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

d_ln_alltime = {}

for i in range(1):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)

lon = d_ln_alltime[expid[0]]['am'].lon
lat = d_ln_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

'''
Antarctic_snow_isotopes_simulations[expid[i]].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region group data in the same grid cell, clean all NaN

Antarctic_snow_isotopes_simulations = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
        Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)

Antarctic_snow_isotopes_sim_grouped = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    Antarctic_snow_isotopes_simulations[expid[i]] = \
        Antarctic_snow_isotopes_simulations[expid[i]].dropna(
            subset=['lat', 'lon', 'temperature', 'accumulation', 'dD', 'dO18',],
            ignore_index=True)
    
    grid_indices = np.zeros(
        (len(Antarctic_snow_isotopes_simulations[expid[i]].index)),
        dtype=np.int64)
    
    for irecord in range(len(grid_indices)):
        # irecord = 0
        # print(irecord)
        
        slat = Antarctic_snow_isotopes_simulations[expid[i]]['lat'][irecord]
        slon = Antarctic_snow_isotopes_simulations[expid[i]]['lon'][irecord]
        
        ilat, ilon = find_ilat_ilon(slat, slon, lat.values, lon.values)
        
        if (abs(lat_2d[ilat, ilon] - slat) > 1.5):
            print('lat diff.: '+str(np.round(abs(lat_2d[ilat, ilon]-slat), 1)))
        
        if (slon < 0): slon += 360
        if (abs(lon_2d[ilat, ilon] - slon) > 1.5):
            print('lon diff.: '+str(np.round(abs(lon_2d[ilat, ilon]-slon), 1)))
        
        grid_indices[irecord] = ilat * len(lon.values) + ilon
    
    Antarctic_snow_isotopes_simulations[expid[i]]['grid_indices'] = \
        grid_indices
    
    Antarctic_snow_isotopes_sim_grouped[expid[i]] = Antarctic_snow_isotopes_simulations[expid[i]].groupby('grid_indices').mean().reset_index()
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(Antarctic_snow_isotopes_sim_grouped[expid[i]], f)



'''
#-------------------------------- check



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region group data in the same grid cell, clean NaN in lat&lon

Antarctic_snow_isotopes_simulations = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
        Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)

Antarctic_snow_isotopes_sim_grouped_all = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    Antarctic_snow_isotopes_simulations[expid[i]] = \
        Antarctic_snow_isotopes_simulations[expid[i]].dropna(
            subset=['lat', 'lon', ],
            ignore_index=True)
    
    grid_indices = np.zeros(
        (len(Antarctic_snow_isotopes_simulations[expid[i]].index)),
        dtype=np.int64)
    
    for irecord in range(len(grid_indices)):
        # irecord = 0
        # print(irecord)
        
        slat = Antarctic_snow_isotopes_simulations[expid[i]]['lat'][irecord]
        slon = Antarctic_snow_isotopes_simulations[expid[i]]['lon'][irecord]
        
        ilat, ilon = find_ilat_ilon(slat, slon, lat.values, lon.values)
        
        if (abs(lat_2d[ilat, ilon] - slat) > 1.5):
            print('lat diff.: '+str(np.round(abs(lat_2d[ilat, ilon]-slat), 1)))
        
        if (slon < 0): slon += 360
        if (abs(lon_2d[ilat, ilon] - slon) > 1.5):
            print('lon diff.: '+str(np.round(abs(lon_2d[ilat, ilon]-slon), 1)))
        
        grid_indices[irecord] = ilat * len(lon.values) + ilon
    
    Antarctic_snow_isotopes_simulations[expid[i]]['grid_indices'] = \
        grid_indices
    
    Antarctic_snow_isotopes_sim_grouped_all[expid[i]] = Antarctic_snow_isotopes_simulations[expid[i]].groupby('grid_indices').mean().reset_index()
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(Antarctic_snow_isotopes_sim_grouped_all[expid[i]], f)



'''
#-------------------------------- check

for i in range(len(expid)):
    # i = 0
    print(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl')


'''
# endregion
# -----------------------------------------------------------------------------



