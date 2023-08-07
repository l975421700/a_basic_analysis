

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_605_5.5',
    'pi_606_5.6',
    'pi_609_5.7',
    ]


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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress

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
import cartopy.feature as cfeature
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
    regrid,
    mean_over_ais,
    time_weighted_mean,
    find_ilat_ilon,
    find_ilat_ilon_general,
    find_multi_gridvalue_at_site,
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
    plot_labels,
    expid_colours,
    expid_labels,
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
    cplot_ttest,
    xr_par_cor,
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


Antarctic_snow_isotopes_simulations = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
        Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)


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
# region group data in the same grid cell

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
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped.pkl', 'wb') as f:
        pickle.dump(Antarctic_snow_isotopes_sim_grouped[expid[i]], f)



'''
#-------------------------------- check



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region group data in the same grid cell

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
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl', 'wb') as f:
        pickle.dump(Antarctic_snow_isotopes_sim_grouped_all[expid[i]], f)



'''
#-------------------------------- check

for i in range(len(expid)):
    # i = 0
    print(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl')


'''
# endregion
# -----------------------------------------------------------------------------



