

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
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
sys.path.append('/home/users/qino')
sys.path.append('/albedo/work/user/qigao001')
import datetime

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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import multiprocessing

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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
import cartopy.feature as cfeature

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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

# import source lat, lon, and aprt

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat

# import others

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

one_degree_grid = xr.open_dataset('scratch/others/one_degree_grids_cdo.nc')


with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region attribut pre to source lat and lon

contributions2site_aprt = {}

for isite in ten_sites_loc.Site:
    # isite = 'EDC'
    # isite = 'Rothera'
    print('#-------- ' + isite)
    
    contributions=np.zeros((len(one_degree_grid.lat), len(one_degree_grid.lon)))
    
    isitelat = ten_sites_loc.lat[ten_sites_loc.Site == isite].values[0]
    isitelon = ten_sites_loc.lon[ten_sites_loc.Site == isite].values[0]
    if (isitelon < 0): isitelon += 360
    
    # print(str(isitelon) + ' & ' + str(isitelat))
    
    #---- get data
    
    daily_src_lat_isite = pre_weighted_lat[expid[i]]['daily'].sel(lat = isitelat, lon = isitelon, method = 'nearest')
    daily_src_lon_isite = pre_weighted_lon[expid[i]]['daily'].sel(lat = isitelat, lon = isitelon, method = 'nearest')
    
    daily_aprt_isite = ocean_aprt_alltime[expid[i]]['daily'].sel(var_names = 'lat').sel(lat = isitelat, lon = isitelon, method = 'nearest')
    daily_aprt_isite = daily_aprt_isite.copy().where(daily_aprt_isite > 2e-8, other=np.nan).compute()
    
    finite_data = (np.isfinite(daily_src_lat_isite) & np.isfinite(daily_src_lon_isite) & np.isfinite(daily_aprt_isite)).values
    # finite_data.sum()
    
    daily_src_lat_isite = daily_src_lat_isite[finite_data]
    daily_src_lon_isite = daily_src_lon_isite[finite_data]
    daily_aprt_isite = daily_aprt_isite[finite_data]
    
    if ((len(daily_src_lat_isite) != len(daily_aprt_isite)) | (len(daily_src_lon_isite) != len(daily_aprt_isite))):
        sys.exit("Error: length differs")
    
    #---- attribute
    
    for itime in range(len(daily_src_lat_isite)):
        # itime = 299
        
        ilat, ilon = find_ilat_ilon(
            slat=daily_src_lat_isite[itime].values,
            slon=daily_src_lon_isite[itime].values,
            lat=one_degree_grid.lat.values,
            lon=one_degree_grid.lon.values,
            )
        # print(daily_src_lat_isite[itime].values)
        # print(one_degree_grid.lat[ilat].values)
        
        # print(daily_src_lon_isite[itime].values)
        # print(one_degree_grid.lon[ilon].values)
        
        contributions[ilat, ilon] += daily_aprt_isite[itime].values
    
    contributions = contributions / np.sum(daily_aprt_isite.values)
    
    contributions2site_aprt[isite] = xr.DataArray(
        data=contributions,
        coords={'lat': one_degree_grid.lat, 'lon': one_degree_grid.lon,},
    )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_aprt.pkl', 'wb') as f:
    pickle.dump(contributions2site_aprt, f)




'''
#-------------------------------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_aprt.pkl', 'rb') as f:
    contributions2site_aprt = pickle.load(f)

# np.sum(contributions)
for isite in ten_sites_loc.Site:
    print('#-------- ' + isite)
    
    print(np.sum(contributions2site_aprt[isite]).values)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region attribute pre to source lat and lon for whole Antarctica


def process_item(igrid):
    # igrid = 15904
    # igrid = 88 * 192 + 66
    
    ilat = igrid // len(lon)
    ilon = igrid % len(lon)
    
    contribution=np.zeros((len(one_degree_grid.lat), len(one_degree_grid.lon)))
    
    daily_src_lat_isite = pre_weighted_lat[expid[i]]['daily'][:, ilat, ilon]
    daily_src_lon_isite = pre_weighted_lon[expid[i]]['daily'][:, ilat, ilon]
    daily_aprt_isite = ocean_aprt_alltime[expid[i]]['daily'].sel(var_names = 'lat')[:, ilat, ilon]
    daily_aprt_isite = daily_aprt_isite.copy().where(daily_aprt_isite > 2e-8, other=np.nan).compute()
    
    finite_data = (np.isfinite(daily_src_lat_isite) & np.isfinite(daily_src_lon_isite) & np.isfinite(daily_aprt_isite)).values
    daily_src_lat_isite = daily_src_lat_isite[finite_data]
    daily_src_lon_isite = daily_src_lon_isite[finite_data]
    daily_aprt_isite = daily_aprt_isite[finite_data]
    
    if ((len(daily_src_lat_isite) != len(daily_aprt_isite)) | (len(daily_src_lon_isite) != len(daily_aprt_isite))):
        sys.exit("Error: length differs")
    
    #---- attribute
    for itime in range(len(daily_src_lat_isite)):
        # itime = 299
        src_lat, src_lon = find_ilat_ilon(
            slat=daily_src_lat_isite[itime].values,
            slon=daily_src_lon_isite[itime].values,
            lat=one_degree_grid.lat.values,
            lon=one_degree_grid.lon.values,
            )
        
        contribution[src_lat, src_lon] += daily_aprt_isite[itime].values
    
    return(contribution)

num_processes = multiprocessing.cpu_count()
print('#-------- Number of processors: ' + str(num_processes))

begin_time = datetime.datetime.now()
print('#-------- Beginning time: ' + str(begin_time))

pool = multiprocessing.Pool(processes=num_processes)

# items = [15904, 16962, ]
items = np.where(echam6_t63_ais_mask['mask']['AIS'].flatten(order='C'))[0]

# items = []

# for isite in t63_sites_indices.keys():
#     # isite = 'EDC'
#     print(isite)
    
#     ilat = t63_sites_indices[isite]['ilat']
#     ilon = t63_sites_indices[isite]['ilon']
    
#     items.append(ilat * 192 + ilon)

contributions = pool.map(process_item, items)
pool.close()
pool.join()

print('#-------- Time taken: ' + str(datetime.datetime.now() - begin_time))

contributions2AIS_aprt = xr.DataArray(
        data=np.zeros((len(lat),
                       len(lon),
                       len(one_degree_grid.lat),
                       len(one_degree_grid.lon))),
        coords={'lat_t63': lat.values,
                'lon_t63': lon.values,
                'lat_1deg': one_degree_grid.lat.values,
                'lon_1deg': one_degree_grid.lon.values,},
    )

for icount, igrid in enumerate(items):
    print(str(icount) + ' ' + str(igrid))
    ilat = igrid // len(lon)
    ilon = igrid % len(lon)
    
    contributions2AIS_aprt[ilat, ilon, :, :] = contributions[icount]


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2AIS_aprt.pkl', 'wb') as f:
    pickle.dump(contributions2AIS_aprt, f)


'''
#-------------------------------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2AIS_aprt.pkl', 'rb') as f:
    contributions2AIS_aprt = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_aprt.pkl', 'rb') as f:
    contributions2site_aprt = pickle.load(f)

with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

for isite in ten_sites_loc.Site:
    # isite = 'EDC'
    # isite = 'Halley'
    print(isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    daily_aprt_isite = ocean_aprt_alltime[expid[i]]['daily'].sel(var_names = 'lat')[:, ilat, ilon]
    daily_aprt_isite = daily_aprt_isite.copy().where(daily_aprt_isite > 2e-8, other=np.nan).compute()
    contribution1 = contributions2site_aprt[isite].values
    contribution2 = contributions2AIS_aprt[ilat, ilon].values
    
    print(daily_aprt_isite.sum().values)
    print(contribution2.sum())
    
    contribution2 = contribution2 / contribution2.sum()
    print(contribution2.sum())
    print(contribution1.sum())
    
    # print((contribution1 == contribution2).all())
    print(np.nanmax(abs((contribution2 - contribution1) / contribution1)))

for isite in ten_sites_loc.Site:
    # isite = 'EDC'
    # isite = 'Halley'
    print(isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    print(echam6_t63_ais_mask['mask']['AIS'][ilat, ilon])


#-------------------------------- others
print(multiprocessing.cpu_count())
print(multiprocessing.get_all_start_methods())
print(multiprocessing.get_start_method())
print(multiprocessing.set_start_method())


#-------------------------------- loop over igrid
for igrid in np.where(echam6_t63_ais_mask['mask']['AIS'].flatten(order='C'))[0]:
    # igrid = 15904
    
    ilat = igrid // len(lon)
    ilon = igrid % len(lon)
    
    if (echam6_t63_ais_mask['mask']['AIS'][ilat, ilon] != True):
        print('False')


#-------------------------------- wrapper function
def parallel_processing(items):
    # global contributions2AIS_aprt
    
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    pool.map(process_item, items)
    pool.close()
    pool.join()
    
    # return(contributions2AIS_aprt)

parallel_processing(np.where(echam6_t63_ais_mask['mask']['AIS'].flatten(order='C'))[0][:20])

#-------- check for two grids

ilat = 15904 // len(lon)
ilon = 15904 % len(lon)
print(contributions2AIS_aprt[ilat, ilon, :, :].sum())

ilat = 16962 // len(lon)
ilon = 16962 % len(lon)
print(contributions2AIS_aprt[ilat, ilon, :, :].sum())

'''
# endregion
# -----------------------------------------------------------------------------



