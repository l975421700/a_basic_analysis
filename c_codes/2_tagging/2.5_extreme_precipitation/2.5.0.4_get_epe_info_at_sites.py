

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
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
sys.path.append('/work/ollie/qigao001')
import gc

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
import proplot as pplt
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


# -----------------------------------------------------------------------------
# region find ilat/ilon indices for stations and core sites

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

T63GR15_jan_surf = xr.open_dataset(
    '/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_jan_surf.nc')

lon = T63GR15_jan_surf.lon
lat = T63GR15_jan_surf.lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

t63_sites_indices = {}

for icores, slat, slon in zip(stations_sites.Site,
                              stations_sites.lat,
                              stations_sites.lon, ):
    t63_sites_indices[icores] = {}
    
    t63_sites_indices[icores]['lat'] = slat
    t63_sites_indices[icores]['lon'] = slon
    
    t63_sites_indices[icores]['ilat'], t63_sites_indices[icores]['ilon'] = \
        find_ilat_ilon(slat, slon, lat.values, lon.values)

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'wb') as f:
    pickle.dump(t63_sites_indices, f)


'''
#-------- check
from haversine import haversine

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')

stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

T63GR15_jan_surf = xr.open_dataset(
    '/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_jan_surf.nc')

lon = T63GR15_jan_surf.lon
lat = T63GR15_jan_surf.lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

for icores in stations_sites.Site:
    slat = t63_sites_indices[icores]['lat']
    slon = t63_sites_indices[icores]['lon']
    # glat = lat[t63_sites_indices[icores]['ilat']].values
    # glon = lon[t63_sites_indices[icores]['ilon']].values
    glat = lat_2d[t63_sites_indices[icores]['ilat'],
                  t63_sites_indices[icores]['ilon']]
    glon = lon_2d[t63_sites_indices[icores]['ilat'],
                  t63_sites_indices[icores]['ilon']]
    
    distance = haversine([slat, slon], [glat,glon], normalize=True,)
    
    if (distance > 100):
        print(np.round(distance, 0))
        print(slat)
        print(slon)
        print(glat)
        print(glon)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import epe information

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

# set quantiles information
quantile_interval  = np.arange(1, 99 + 1e-4, 1, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

# set epe source files
source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.epe_weighted_lat.pkl',
    prefix + '.epe_weighted_lon.pkl',
    prefix + '.epe_weighted_sst.pkl',
    prefix + '.epe_weighted_rh2m.pkl',
    prefix + '.epe_weighted_wind10.pkl',
    prefix + '.transport_distance_epe.pkl',
    ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract epe sources information

epe_sources_sites = {}
epe_sources_sites[expid[i]] = {}

for ivar, ifile in zip(source_var, source_var_files):
    # ivar = 'lat'
    # ifile = 'output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0/analysis/echam/pi_m_502_5.0.epe_weighted_lat.pkl'
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f:
        epe_weighted_var = pickle.load(f)
    
    epe_sources_sites[expid[i]][ivar] = {}
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        epe_sources_sites[expid[i]][ivar][isite] = {}
        
        for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            epe_sources_sites[expid[i]][ivar][isite][ialltime] = {}
            
            for iqtl in epe_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                epe_sources_sites[expid[i]][ivar][isite][ialltime][iqtl] = \
                    epe_weighted_var[iqtl][ialltime][
                        :,
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].copy()
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        epe_sources_sites[expid[i]][ivar][isite][ialltime] = pd.DataFrame(
            columns=('iqtl', 'quantiles', 'am',))
        
        for iqtl in epe_weighted_var.keys():
            # iqtl = '90%'
            # print('#-- ' + iqtl)
            
            epe_sources_sites[expid[i]][ivar][isite][ialltime] = pd.concat([
                epe_sources_sites[expid[i]][ivar][isite][ialltime],
                pd.DataFrame(data={
                    'iqtl': iqtl,
                    'quantiles': quantiles[iqtl],
                    'am': epe_weighted_var[iqtl][ialltime][
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].values,
                    }, index=[0])],
                ignore_index=True,)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_sources_sites.pkl', 'wb') as f:
    pickle.dump(epe_sources_sites[expid[i]], f)



'''


#-------------------------------- check
epe_sources_sites = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_sources_sites.pkl', 'rb') as f:
    epe_sources_sites[expid[i]] = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

quantile_interval  = np.arange(50, 99 + 1e-4, 1, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.epe_weighted_lat.pkl',
    prefix + '.epe_weighted_lon.pkl',
    prefix + '.epe_weighted_sst.pkl',
    prefix + '.epe_weighted_rh2m.pkl',
    prefix + '.epe_weighted_wind10.pkl',
    prefix + '.transport_distance_epe.pkl',
    ]


for iind in range(6):
    # iind = 0
    ivar = source_var[iind]
    ifile = source_var_files[iind]
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f: epe_weighted_var = pickle.load(f)
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            
            for iqtl in epe_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                data1 = epe_sources_sites[expid[i]][ivar][isite][
                    ialltime][iqtl].values
                data2 = epe_weighted_var[iqtl][ialltime][
                    :,
                    t63_sites_indices[isite]['ilat'],
                    t63_sites_indices[isite]['ilon']].copy().values
                print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        # epe_sources_sites[expid[i]][ivar][isite][ialltime] = pd.DataFrame(
        #     columns=('iqtl', 'quantiles', 'am',))
        
        for iqtl in epe_weighted_var.keys():
            # iqtl = '90%'
            # print('#-- ' + iqtl)
            
            data1 = epe_sources_sites[expid[i]][ivar][isite][ialltime].loc[
                epe_sources_sites[expid[i]][ivar][isite][ialltime].iqtl == iqtl
            ].am.values[0]
            
            data2 = epe_weighted_var[iqtl][ialltime][
                t63_sites_indices[isite]['ilat'],
                t63_sites_indices[isite]['ilon']].values
            print(data1 == data2)
    
    del epe_weighted_var


# epe_weighted_var = {}
# epe_weighted_var[expid[i]] = {}
# epe_weighted_var[expid[i]][ivar] => epe_weighted_var
    # del epe_weighted_var
    # gc.collect()

'''
# endregion
# -----------------------------------------------------------------------------


