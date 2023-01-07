

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


#-------------------------------- check
transport_distance_epe_st = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance_epe_st_binned.pkl', 'rb') as f:
    transport_distance_epe_st[expid[i]] = pickle.load(f)

epe_st_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_lon_binned.pkl', 'rb') as f:
    epe_st_weighted_lon[expid[i]] = pickle.load(f)

epe_st_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_lat_binned.pkl', 'rb') as f:
    epe_st_weighted_lat[expid[i]] = pickle.load(f)

lon = epe_st_weighted_lon[expid[i]]['90.5%']['am'].lon
lat = epe_st_weighted_lon[expid[i]]['90.5%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

iqtl = '90.5%'
ilat = 40
ilon = 90

for ialltime in ['ann']:
    # ialltime = 'mm'
    itime = -4
    
    local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
    source = [
        epe_st_weighted_lat[expid[i]][iqtl][ialltime][itime, ilat, ilon].values,
        epe_st_weighted_lon[expid[i]][iqtl][ialltime][itime, ilat, ilon].values,]
    
    print(haversine(local, source, normalize=True))
    print(transport_distance_epe_st[expid[i]][iqtl][ialltime][itime, ilat, ilon].values)

ialltime = 'am'
local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
source = [
    epe_st_weighted_lat[expid[i]][iqtl][ialltime][ilat, ilon].values,
    epe_st_weighted_lon[expid[i]][iqtl][ialltime][ilat, ilon].values,]

print(haversine(local, source, normalize=True))
print(transport_distance_epe_st[expid[i]][iqtl][ialltime][ilat, ilon].values)
