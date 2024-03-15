

# salloc --account=paleodyn.paleodyn --partition=mpp --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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


q_sfc_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lat.pkl', 'rb') as f:
    q_sfc_weighted_lat[expid[i]] = pickle.load(f)

q_sfc_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lon.pkl', 'rb') as f:
    q_sfc_weighted_lon[expid[i]] = pickle.load(f)

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

lon = q_sfc_weighted_lat[expid[i]]['am'].lon
lat = q_sfc_weighted_lat[expid[i]]['am'].lat

one_degree_grid = xr.open_dataset('scratch/others/one_degree_grids_cdo.nc')



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


par_corr_sources_isotopes_q_sfc={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)


#-------------------------------- Daily negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- Daily positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])


#-------------------------------- Annual negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_min = np.min(annual_par_corr)
where_annual_min = np.where(annual_par_corr == annual_min)
# print(annual_min)
# print(annual_par_corr[where_annual_min[0][0], where_annual_min[1][0]])

annual_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lon.values
annual_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lat.values

annual_min_ilon = np.where(lon == annual_min_lon)[0][0]
annual_min_ilat = np.where(lat == annual_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_min_ilat, annual_min_ilon])


#-------------------------------- Annual positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_max = np.max(annual_par_corr)
where_annual_max = np.where(annual_par_corr == annual_max)
# print(annual_max)
# print(annual_par_corr[where_annual_max[0][0], where_annual_max[1][0]])

annual_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lon.values
annual_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lat.values

annual_max_ilon = np.where(lon == annual_max_lon)[0][0]
annual_max_ilat = np.where(lat == annual_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_max_ilat, annual_max_ilon])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region attribute q_sfc to source lat and lon

contributions2site_q_sfc = {}

for isite in ['daily_min', 'daily_max']:
    # isite = 'daily_min'
    print('#-------------------------------- ' + isite)
    
    if (isite == 'daily_min'):
        isitelat = daily_min_lat
        isitelon = daily_min_lon
    elif (isite == 'daily_max'):
        isitelat = daily_max_lat
        isitelon = daily_max_lon
    
    contributions=np.zeros((len(one_degree_grid.lat), len(one_degree_grid.lon)))
    
    
    #---------------- get data
    
    daily_src_lat_isite = q_sfc_weighted_lat[expid[i]]['daily'].sel(lat = isitelat, lon = isitelon, method = 'nearest').copy()
    daily_src_lon_isite = q_sfc_weighted_lon[expid[i]]['daily'].sel(lat = isitelat, lon = isitelon, method = 'nearest').copy()
    
    daily_q_sfc_isite = ocean_q_sfc_alltime[expid[i]]['daily'].sel(var_names = 'lat').sel(lat = isitelat, lon = isitelon, method = 'nearest').copy()
    
    subset = (np.isfinite(daily_src_lat_isite) & np.isfinite(daily_src_lon_isite) & np.isfinite(daily_q_sfc_isite)).values
    
    daily_src_lat_isite = daily_src_lat_isite[subset]
    daily_src_lon_isite = daily_src_lon_isite[subset]
    daily_q_sfc_isite   = daily_q_sfc_isite[subset]
    
    
    #---------------- attribute
    
    for itime in range(len(daily_src_lat_isite)):
        
        ilat, ilon = find_ilat_ilon(
            slat=daily_src_lat_isite[itime].values,
            slon=daily_src_lon_isite[itime].values,
            lat=one_degree_grid.lat.values,
            lon=one_degree_grid.lon.values,
            )
        
        contributions[ilat, ilon] += daily_q_sfc_isite[itime].values
    
    contributions = contributions / np.sum(daily_q_sfc_isite.values)
    
    contributions2site_q_sfc[isite] = xr.DataArray(
        data=contributions,
        coords={'lat': one_degree_grid.lat, 'lon': one_degree_grid.lon,},
    )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_q_sfc.pkl', 'wb') as f:
    pickle.dump(contributions2site_q_sfc, f)




'''
#-------------------------------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_q_sfc.pkl', 'rb') as f:
    contributions2site_q_sfc = pickle.load(f)

# np.sum(contributions)
for isite in ['daily_min', 'daily_max']:
    print('#-------------------------------- ' + isite)
    
    print(np.sum(contributions2site_q_sfc[isite]).values)
'''
# endregion
# -----------------------------------------------------------------------------
