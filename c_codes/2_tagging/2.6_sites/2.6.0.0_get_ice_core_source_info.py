

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    ]
i = 0


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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# major_ice_core_site => stations_sites
# loc_indices => t63_sites_indices
# -----------------------------------------------------------------------------
# region import data

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# get grid information
echam6_t63_slm = xr.open_dataset(
    'scratch/others/land_sea_masks/echam6_t63_slm.nc')

lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

# import sites indices
with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)


'''

#---------------- previous code

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

lon = temp2_alltime[expid[i]]['am'].lon
lat = temp2_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

loc_indices = {}

for icores, slat, slon in zip(major_ice_core_site.Site,
                              major_ice_core_site.lat,
                              major_ice_core_site.lon, ):
    loc_indices[icores] = {}
    
    loc_indices[icores]['lat'] = slat
    loc_indices[icores]['lon'] = slon
    
    loc_indices[icores]['ilat'], loc_indices[icores]['ilon'] = \
        find_ilat_ilon(slat, slon, lat.values, lon.values)

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.loc_indices.pkl',
    'wb') as f:
    pickle.dump(loc_indices, f)

#-------- check
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.loc_indices.pkl',
    'rb') as f:
    loc_indices = pickle.load(f)

for icores in major_ice_core_site.Site:
    print(icores + ':    ' + str(loc_indices[icores]['lat']) + \
        ':    ' + str(loc_indices[icores]['lon']))
    print(lat[loc_indices[icores]['ilat']].values)
    print(lon[loc_indices[icores]['ilon']].values)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get temp2 at ice core sites

# temp2
temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)


temp2_alltime_icores = {}
temp2_alltime_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    # print('#--------' + icores)
    temp2_alltime_icores[expid[i]][icores] = {}
    
    for ialltime in temp2_alltime[expid[i]].keys():
        # print('#----' + ialltime)
        if ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'mon'
            temp2_alltime_icores[expid[i]][icores][ialltime] = \
                temp2_alltime[expid[i]][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
        elif (ialltime == 'am'):
            temp2_alltime_icores[expid[i]][icores][ialltime] = \
                temp2_alltime[expid[i]][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl',
    'wb') as f:
    pickle.dump(temp2_alltime_icores[expid[i]], f)




'''
for icores in temp2_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    temp2_am_icores = \
        np.round(temp2_alltime_icores[expid[i]][icores]['am'].values, 1)
    temp2_annstd_icores = \
        np.round(
            temp2_alltime_icores[expid[i]][icores]['ann'].std(ddof=1).values,
            1)
    print(str(temp2_am_icores) + ' ± ' + str(temp2_annstd_icores))


#-------------------------------- check
temp2_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
    temp2_alltime_icores[expid[i]] = pickle.load(f)

for icores in temp2_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
    print('grid lat:   ' + \
        str(np.round(temp2_alltime_icores[expid[i]][icores]['am'].lat.values, 2)))
    print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
    print('grid lon:   ' + \
        str(temp2_alltime_icores[expid[i]][icores]['am'].lon.values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt at ice core sites

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime_icores = {}
wisoaprt_alltime_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#--------' + icores)
    wisoaprt_alltime_icores[expid[i]][icores] = {}
    
    for ialltime in wisoaprt_alltime[expid[i]].keys():
        # print('#----' + ialltime)
        if ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'daily'
            wisoaprt_alltime_icores[expid[i]][icores][ialltime] = \
                wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
        elif (ialltime == 'am'):
            wisoaprt_alltime_icores[expid[i]][icores][ialltime] = \
                wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_alltime_icores[expid[i]], f)




'''
for icores in wisoaprt_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    wisoaprt_am_icores = \
        np.round(wisoaprt_alltime_icores[expid[i]][icores][
            'am'].sel(wisotype=1).values * seconds_per_d * 365, 1)
    wisoaprt_annstd_icores = \
        np.round((wisoaprt_alltime_icores[expid[i]][icores][
            'ann'].sel(wisotype=1)  * seconds_per_d * 365
                  ).std(ddof=1).values, 1)
    print(str(wisoaprt_am_icores) + ' ± ' + str(wisoaprt_annstd_icores))


#-------------------------------- check
wisoaprt_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    wisoaprt_alltime_icores[expid[i]] = pickle.load(f)

wisoaprt_alltime_icores[expid[i]]['EDC']['daily']

for icores in wisoaprt_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
    print('grid lat:   ' + \
        str(np.round(wisoaprt_alltime_icores[expid[i]][icores]['am'].lat.values, 2)))
    print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
    print('grid lon:   ' + \
        str(wisoaprt_alltime_icores[expid[i]][icores]['am'].lon.values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get source properties at ice core sites

pre_weighted_var = {}
pre_weighted_var[expid[i]] = {}

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']

prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.pre_weighted_lat.pkl',
    prefix + '.pre_weighted_lon.pkl',
    prefix + '.pre_weighted_sst.pkl',
    prefix + '.pre_weighted_rh2m.pkl',
    prefix + '.pre_weighted_wind10.pkl',
    prefix + '.transport_distance.pkl',
]

for ivar, ifile in zip(source_var, source_var_files):
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        pre_weighted_var[expid[i]][ivar] = pickle.load(f)


pre_weighted_var_icores = {}
pre_weighted_var_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#-------- ' + icores)
    
    pre_weighted_var_icores[expid[i]][icores] = {}
    
    for ivar in source_var:
        # ivar = 'lat'
        print('#---- ' + ivar)
        pre_weighted_var_icores[expid[i]][icores][ivar] = {}
        
        for ialltime in pre_weighted_var[expid[i]][ivar].keys():
            print('#-- ' + ialltime)
            
            if ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
                # ialltime = 'daily'
                pre_weighted_var_icores[expid[i]][icores][ivar][ialltime] = \
                    pre_weighted_var[expid[i]][ivar][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
            elif (ialltime == 'am'):
                # ialltime = 'am'
                pre_weighted_var_icores[expid[i]][icores][ivar][ialltime] = \
                    pre_weighted_var[expid[i]][ivar][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'wb') as f:
    pickle.dump(pre_weighted_var_icores[expid[i]], f)


'''
for ivar in source_var:
    print('#-------------------------------- ' + ivar)
    for icores in stations_sites.Site:
        print('#----------------' + icores)
        
        pre_weighted_var_am_icores = \
            np.round(
                pre_weighted_var_icores[expid[i]][icores][ivar]['am'].values,
                1)
        pre_weighted_var_annstd_icores = \
            np.round(
                pre_weighted_var_icores[expid[i]][icores][ivar]['ann'].std(
                    ddof=1).values,
                1)
        
        if (ivar == 'lon'):
            # ivar = 'lon'
            pre_weighted_var_annstd_icores = \
                np.round(
                    circstd(pre_weighted_var_icores[expid[i]][icores][ivar]['ann'], high=360, low=0),
                    1)
        
        if (ivar == 'distance'):
            # ivar = 'distance'
            pre_weighted_var_am_icores = \
                np.round(
                    pre_weighted_var_icores[expid[i]][icores][ivar][
                        'am'].values / 100,
                    1)
            pre_weighted_var_annstd_icores = \
                np.round(
                    (pre_weighted_var_icores[expid[i]][icores][ivar][
                        'ann'] / 100).std(ddof=1).values,
                    1)
        
        print(str(pre_weighted_var_am_icores) + ' ± ' + \
            str(pre_weighted_var_annstd_icores))


ivar = 'lon'
print('#-------------------------------- ' + ivar)
for icores in stations_sites.Site:
    print('#----------------' + icores)
    pre_weighted_var_am_icores = \
            np.round(
                calc_lon_diff(
                    pre_weighted_var_icores[expid[i]][icores][ivar]['am'],
                    t63_sites_indices[icores]['lon'],
                ).values,
                1)
    print(pre_weighted_var_am_icores)


#-------------------------------- check
pre_weighted_var_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
    pre_weighted_var_icores[expid[i]] = pickle.load(f)

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']

for icores in pre_weighted_var_icores[expid[i]].keys():
    for ivar in source_var:
        # icores = 'EDC'
        print('#----------------' + icores)
        print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
        print('grid  lat:  ' + \
            str(np.round(pre_weighted_var_icores[expid[i]][icores][ivar]['am'].lat.values, 2)))
        print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
        print('grid  lon:  ' + \
            str(pre_weighted_var_icores[expid[i]][icores][ivar]['am'].lon.values))


ivar = 'lon'
circstd(pre_weighted_var_icores[expid[i]][icores][ivar]['ann'],
        high=360, low=0)
(pre_weighted_var_icores[expid[i]][icores][ivar]['ann']).std(ddof=1).values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get delta O18, delta D, and d_ln


#-------------------------------- import data

isotopes_alltime = {}
isotopes_alltime[expid[i]] = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    isotopes_alltime[expid[i]]['dO18'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    isotopes_alltime[expid[i]]['dD'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    isotopes_alltime[expid[i]]['d_excess'] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    isotopes_alltime[expid[i]]['d_ln'] = pickle.load(f)


#-------------------------------- extract data

isotopes_alltime_icores = {}
isotopes_alltime_icores[expid[i]] = {}

for iisotope in ['dO18', 'dD', 'd_excess', 'd_ln']:
    # iisotope = 'd_ln'
    print('#---------------- ' + iisotope)
    
    isotopes_alltime_icores[expid[i]][iisotope] = {}
    
    for icores in stations_sites.Site:
        # icores = 'EDC'
        print('#-------- ' + icores)
        
        isotopes_alltime_icores[expid[i]][iisotope][icores] = {}
        
        for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime] = \
                isotopes_alltime[expid[i]][iisotope][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
        
        isotopes_alltime_icores[expid[i]][iisotope][icores]['am'] = \
            isotopes_alltime[expid[i]][iisotope]['am'][
                t63_sites_indices[icores]['ilat'],
                t63_sites_indices[icores]['ilon']]



with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl',
    'wb') as f:
    pickle.dump(isotopes_alltime_icores[expid[i]], f)



'''
#-------------------------------- check
isotopes_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
    isotopes_alltime_icores[expid[i]] = pickle.load(f)

for iisotope in ['dO18', 'dD', 'd_excess', 'd_ln']:
    # iisotope = 'd_ln'
    print('#---------------- ' + iisotope)
    
    for icores in stations_sites.Site:
        # icores = 'EDC'
        print('#----------------' + icores)
        
        # print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
        # print('grid lat:   ' + str(np.round(isotopes_alltime_icores[expid[i]][iisotope][icores]['am'].lat.values, 1)))
        # print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
        # print('grid lon:   ' + str(np.round(isotopes_alltime_icores[expid[i]][iisotope][icores]['am'].lon.values, 1)))
        
        local_lat = t63_sites_indices[icores]['lat']
        grid_lat = isotopes_alltime_icores[expid[i]][iisotope][icores]['am'].lat.values
        
        local_lon = t63_sites_indices[icores]['lon']
        grid_lon = isotopes_alltime_icores[expid[i]][iisotope][icores]['am'].lon.values
        
        if (local_lon < 0): local_lon += 360
        
        if (abs(local_lat - grid_lat) > 2):
            print('local lat:  ' + str(local_lat))
            print('grid lat:   ' + str(np.round(grid_lat, 1)))
        
        if (abs(local_lon - grid_lon) > 2):
            print('local lon:  ' + str(local_lon))
            print('grid lon:   ' + str(np.round(grid_lon, 1)))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get aprt_frc at ice core sites

aprt_frc = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc.pkl', 'rb') as f:
    aprt_frc[expid[i]] = pickle.load(f)

# aprt_frc[expid[i]].keys()

aprt_frc_alltime_icores = {}
aprt_frc_alltime_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#--------' + icores)
    aprt_frc_alltime_icores[expid[i]][icores] = {}
    
    for ialltime in aprt_frc[expid[i]]['Atlantic Ocean'].keys():
        # print('#----' + ialltime)
        if ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'daily'
            aprt_frc_alltime_icores[expid[i]][icores][ialltime] = \
                (aprt_frc[expid[i]]['Atlantic Ocean'][ialltime] + \
                    aprt_frc[expid[i]]['Indian Ocean'][ialltime] + \
                        aprt_frc[expid[i]]['Pacific Ocean'][ialltime] + \
                            aprt_frc[expid[i]]['Southern Ocean'][ialltime]).compute()[
                                :,
                                t63_sites_indices[icores]['ilat'],
                                t63_sites_indices[icores]['ilon']]
        elif (ialltime == 'am'):
            # ialltime = 'am'
            aprt_frc_alltime_icores[expid[i]][icores][ialltime] = \
                (aprt_frc[expid[i]]['Atlantic Ocean'][ialltime] + \
                    aprt_frc[expid[i]]['Indian Ocean'][ialltime] + \
                        aprt_frc[expid[i]]['Pacific Ocean'][ialltime] + \
                            aprt_frc[expid[i]]['Southern Ocean'][ialltime]).compute()[
                                t63_sites_indices[icores]['ilat'],
                                t63_sites_indices[icores]['ilon']]


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl',
    'wb') as f:
    pickle.dump(aprt_frc_alltime_icores[expid[i]], f)




'''
for icores in aprt_frc_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    aprt_frc_am_icores = \
        np.round(aprt_frc_alltime_icores[expid[i]][icores][
            'am'].values, 1)
    aprt_frc_annstd_icores = \
        np.round((aprt_frc_alltime_icores[expid[i]][icores][
            'ann']).std(ddof=1).values, 1)
    print(str(aprt_frc_am_icores) + ' ± ' + str(aprt_frc_annstd_icores))


#-------------------------------- check
aprt_frc_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl', 'rb') as f:
    aprt_frc_alltime_icores[expid[i]] = pickle.load(f)

for icores in aprt_frc_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
    print('grid lat:   ' + \
        str(np.round(aprt_frc_alltime_icores[expid[i]][icores]['am'].lat.values, 2)))
    print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
    print('grid lon:   ' + \
        str(aprt_frc_alltime_icores[expid[i]][icores]['am'].lon.values))

'''
# endregion
# -----------------------------------------------------------------------------


