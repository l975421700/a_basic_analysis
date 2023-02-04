

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp

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
    regrid,
    time_weighted_mean,
    find_ilat_ilon_general,
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


# -----------------------------------------------------------------------------
# region import data

#---------------- import reconstructions
lig_recs = {}

lig_recs['EC'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name='Capron et al. 2017', header=0, skiprows=12, nrows=47,)

lig_recs['JH'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name=' Hoffman et al. 2017', header=0, skiprows=14, nrows=37,)

lig_recs['DC'] = pd.read_csv(
    'data_sources/LIG/Chandler-Langebroek_2021_SST-anom.tab',
    sep='\t', header=0, skiprows=76,
    )

lig_recs['MC'] = pd.read_excel(
    'data_sources/LIG/Chadwick_et_al_2021/AICC2012 ages.xlsx',
    header=0,)

#-------- rename columns names
lig_recs['DC'] = lig_recs['DC'].rename(columns={'Site': 'Station'})
lig_recs['MC'] = lig_recs['MC'].rename(columns={
    'Core name': 'Station',
    'Latitude (degrees South)': 'Latitude',
    'Longitude (degrees East)': 'Longitude',})

lig_recs['MC'].Latitude = -1 * lig_recs['MC'].Latitude


'''
lig_recs['EC'].columns # ['Station', 'Latitude', 'Longitude']
lig_recs['JH'].columns # ['Station', 'Latitude', 'Longitude']
lig_recs['DC'].columns # ['Site', 'Latitude', 'Longitude']
lig_recs['MC'].columns # ['Core name', 'Latitude (degrees South)', 'Longitude (degrees East)']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices for 1*1 degree

lon = xe.util.grid_global(1, 1).lon.values
lat = xe.util.grid_global(1, 1).lat.values

lig_recs_loc_indices = {}

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    lig_recs_loc_indices[irec] = {}
    
    stations = lig_recs[irec].Station.unique()
    
    for istation in stations:
        # istation = stations[0]
        # istation = 'HM71-19'
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        lig_recs_loc_indices[irec][istation] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices.pkl', 'wb') as f:
    pickle.dump(lig_recs_loc_indices, f)


'''
#-------------------------------- check
with open('scratch/cmip6/lig/rec/lig_recs_loc_indices.pkl', 'rb') as f:
    lig_recs_loc_indices = pickle.load(f)

from haversine import haversine

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    for istation in lig_recs[irec].Station.unique():
        # istation = lig_recs[irec].Station.unique()[0]
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        glat = lat[lig_recs_loc_indices[irec][istation][0],
                   lig_recs_loc_indices[irec][istation][1]]
        glon = lon[lig_recs_loc_indices[irec][istation][0],
                   lig_recs_loc_indices[irec][istation][1]]
        
        distance = haversine([slat, slon], [glat, glon], normalize=True,)
        
        if (distance > 50):
            print(np.round(distance, 0))




    print(len(lig_recs[irec].Station))
    print(len(lig_recs[irec].Station.unique()))

        print('#---- ' + str(slat) + ', ' + str(slon))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices for HadISST

HadISST1_1 = {}
HadISST1_1['sst'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_sst.nc')
lon, lat = np.meshgrid(
    HadISST1_1['sst'].longitude.values, HadISST1_1['sst'].latitude.values,)

lig_recs_loc_indices_hadisst = {}

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'DC'
    print('#---------------- ' + irec)
    
    lig_recs_loc_indices_hadisst[irec] = {}
    
    stations = lig_recs[irec].Station.unique()
    
    for istation in stations:
        # istation = stations[0]
        # istation = 'HM71-19'
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        lig_recs_loc_indices_hadisst[irec][istation] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_hadisst.pkl', 'wb') as f:
    pickle.dump(lig_recs_loc_indices_hadisst, f)








'''
#-------------------------------- check
with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_hadisst.pkl', 'rb') as f:
    lig_recs_loc_indices_hadisst = pickle.load(f)

from haversine import haversine

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    for istation in lig_recs[irec].Station.unique():
        # istation = lig_recs[irec].Station.unique()[0]
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        
        glat = HadISST1_1['sst'].sst[
            0,
            lig_recs_loc_indices_hadisst[irec][istation][0],
            lig_recs_loc_indices_hadisst[irec][istation][1]
            ].latitude
        glon = HadISST1_1['sst'].sst[
            0,
            lig_recs_loc_indices_hadisst[irec][istation][0],
            lig_recs_loc_indices_hadisst[irec][istation][1]
            ].longitude
        
        distance = haversine([slat, slon], [glat, glon], normalize=True,)
        
        if (distance > 50):
            print(np.round(distance, 0))




# lat and lat_2d differ
lon = xe.util.grid_global(1, 1).lon.values
lat = xe.util.grid_global(1, 1).lat.values

lon_2d, lat_2d = np.meshgrid(
    HadISST1_1['sst'].longitude.values, HadISST1_1['sst'].latitude.values,)

(lon == lon_2d).all()
(lat == lat_2d).all()


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices for PMIP3 ensmean

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc')
lon, lat = np.meshgrid(
    pmip3_lig_sim['annual_sst'].longitude.values,
    pmip3_lig_sim['annual_sst'].latitude.values,)

lig_recs_loc_indices_pmip3ens = {}

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'DC'
    print('#---------------- ' + irec)
    
    lig_recs_loc_indices_pmip3ens[irec] = {}
    
    stations = lig_recs[irec].Station.unique()
    
    for istation in stations:
        # istation = stations[0]
        # istation = 'HM71-19'
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        lig_recs_loc_indices_pmip3ens[irec][istation] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_pmip3ens.pkl', 'wb') as f:
    pickle.dump(lig_recs_loc_indices_pmip3ens, f)




'''
#-------------------------------- check
with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_pmip3ens.pkl', 'rb') as f:
    lig_recs_loc_indices_pmip3ens = pickle.load(f)

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc')

from haversine import haversine

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    for istation in lig_recs[irec].Station.unique():
        # istation = lig_recs[irec].Station.unique()[0]
        print('#-------- ' + istation)
        
        lig_data = lig_recs[irec].loc[lig_recs[irec].Station == istation]
        
        slat = lig_data.Latitude.values[0]
        slon = lig_data.Longitude.values[0]
        
        
        glat = pmip3_lig_sim['annual_sst'].sst[
            lig_recs_loc_indices_pmip3ens[irec][istation][0],
            lig_recs_loc_indices_pmip3ens[irec][istation][1]
            ].latitude
        glon = pmip3_lig_sim['annual_sst'].sst[
            lig_recs_loc_indices_pmip3ens[irec][istation][0],
            lig_recs_loc_indices_pmip3ens[irec][istation][1]
            ].longitude
        
        distance = haversine([slat, slon], [glat, glon], normalize=True,)
        
        if (distance > 150):
            print(np.round(distance, 0))
'''
# endregion
# -----------------------------------------------------------------------------

