

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
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
import datetime
import psutil

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
from geopy.distance import geodesic, great_circle
from haversine import haversine, haversine_vector

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

lp_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lp_weighted_lon.pkl', 'rb') as f:
    lp_weighted_lon[expid[i]] = pickle.load(f)

lp_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lp_weighted_lat.pkl', 'rb') as f:
    lp_weighted_lat[expid[i]] = pickle.load(f)

lon = lp_weighted_lon[expid[i]]['10%']['am'].lon
lat = lp_weighted_lon[expid[i]]['10%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get transport distance

transport_distance_lp = {}
transport_distance_lp[expid[i]] = {}

begin_time = datetime.datetime.now()
print(begin_time)

for iqtl in lp_weighted_lon[expid[i]].keys(): # ['90%', '95%']:
    transport_distance_lp[expid[i]][iqtl] = {}
    
    for ialltime in ['ann', 'am']:
        print(iqtl + ' - ' + ialltime)
        transport_distance_lp[expid[i]][iqtl][ialltime] = \
            lp_weighted_lat[expid[i]][iqtl][ialltime].copy().rename(
                'transport_distance_lp')
        transport_distance_lp[expid[i]][iqtl][ialltime][:] = 0
        
        if (ialltime in ['mon', 'sea', 'ann']):
            print(ialltime)
            
            years = np.unique(
                transport_distance_lp[expid[i]][iqtl][ialltime].time.dt.year)
            
            for iyear in years:
                # iyear = 2010
                print(str(iyear) + ' / ' + str(years[-1]))
                
                time_indices = np.where(
                    transport_distance_lp[expid[i]][iqtl][
                        ialltime].time.dt.year == iyear)
                
                b_lon_2d = np.broadcast_to(
                    lon_2d,
                    transport_distance_lp[expid[i]][iqtl][ialltime][
                        time_indices].shape,
                    )
                b_lat_2d = np.broadcast_to(
                    lat_2d,
                    transport_distance_lp[expid[i]][iqtl][ialltime][
                        time_indices].shape,
                    )
                b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
                b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
                local_pairs = [[x, y] for x, y in \
                    zip(b_lat_2d_flatten, b_lon_2d_flatten)]
                
                lon_src_flatten = lp_weighted_lon[expid[i]][iqtl][ialltime][
                    time_indices].values.reshape(-1, 1).copy()
                lat_src_flatten = lp_weighted_lat[expid[i]][iqtl][ialltime][
                    time_indices].values.reshape(-1, 1).copy()
                source_pairs = [[x, y] for x, y in \
                    zip(lat_src_flatten, lon_src_flatten)]
                
                transport_distance_lp[expid[i]][iqtl][ialltime][time_indices] = haversine_vector(
                    local_pairs, source_pairs, normalize=True).reshape(
                        transport_distance_lp[expid[i]][iqtl][ialltime][time_indices].shape)
                print(datetime.datetime.now() - begin_time)
        elif (ialltime in ['mm', 'sm', 'am']):
            print(ialltime)
            b_lon_2d = np.broadcast_to(
                lon_2d, lp_weighted_lon[expid[i]][iqtl][ialltime].shape, )
            b_lat_2d = np.broadcast_to(
                lat_2d, lp_weighted_lat[expid[i]][iqtl][ialltime].shape, )
            b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
            b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
            local_pairs = [[x, y] for x, y in \
                zip(b_lat_2d_flatten, b_lon_2d_flatten)]
            
            lon_src_flatten = lp_weighted_lon[expid[i]][iqtl][
                ialltime].values.reshape(-1, 1).copy()
            lat_src_flatten = lp_weighted_lat[expid[i]][iqtl][
                ialltime].values.reshape(-1, 1).copy()
            source_pairs = [[x, y] for x, y in \
                zip(lat_src_flatten, lon_src_flatten)]
            
            transport_distance_lp[expid[i]][iqtl][ialltime][:] = haversine_vector(
                local_pairs, source_pairs, normalize=True).reshape(
                    transport_distance_lp[expid[i]][iqtl][ialltime].shape)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance_lp.pkl', 'wb') as f:
    pickle.dump(transport_distance_lp[expid[i]], f)


'''
#-------------------------------- check
transport_distance_lp = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance_lp.pkl', 'rb') as f:
    transport_distance_lp[expid[i]] = pickle.load(f)

lp_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lp_weighted_lon.pkl', 'rb') as f:
    lp_weighted_lon[expid[i]] = pickle.load(f)

lp_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lp_weighted_lat.pkl', 'rb') as f:
    lp_weighted_lat[expid[i]] = pickle.load(f)

lon = lp_weighted_lon[expid[i]]['10%']['am'].lon
lat = lp_weighted_lon[expid[i]]['10%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

iqtl = '10%'
ilat = 40
ilon = 90

for ialltime in ['ann']:
    # ialltime = 'mm'
    itime = -4
    
    local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
    source = [
        lp_weighted_lat[expid[i]][iqtl][ialltime][itime, ilat, ilon].values,
        lp_weighted_lon[expid[i]][iqtl][ialltime][itime, ilat, ilon].values,]
    
    print(haversine(local, source, normalize=True))
    print(transport_distance_lp[expid[i]][iqtl][ialltime][itime, ilat, ilon].values)

ialltime = 'am'
local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
source = [
    lp_weighted_lat[expid[i]][iqtl][ialltime][ilat, ilon].values,
    lp_weighted_lon[expid[i]][iqtl][ialltime][ilat, ilon].values,]

print(haversine(local, source, normalize=True))
print(transport_distance_lp[expid[i]][iqtl][ialltime][ilat, ilon].values)
'''
# endregion
# -----------------------------------------------------------------------------

