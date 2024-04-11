

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
from scipy.stats import pearsonr
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
    plot_labels_no_unit,
    plot_labels,
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

source_var = ['lat', 'lon', 'sst', 'RHsst', 'distance']
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        prefix + '.q_sfc_weighted_lon.pkl',
        prefix + '.q_sfc_weighted_sst.pkl',
        prefix + '.q_sfc_weighted_RHsst.pkl',
        prefix + '.q_sfc_transport_distance.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)




'''
q_sfc_transport_distance = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_transport_distance.pkl', 'rb') as f:
    q_sfc_transport_distance[expid[i]] = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


par_corr_sources_isotopes_q_sfc={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

lon = par_corr_sources_isotopes_q_sfc[expid[i]]['d_ln']['sst']['RHsst']['daily']['r'].lon
lat = par_corr_sources_isotopes_q_sfc[expid[i]]['d_ln']['sst']['RHsst']['daily']['r'].lat

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
# region plot am transport distance over the open ocean

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.2 ' + expid[i] + ' am q_sfc transport distance.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='viridis',)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7]) / 2.54,)

plt1 = plot_t63_contourf(
    q_sfc_transport_distance[expid[i]]['am'].lon,
    q_sfc_transport_distance[expid[i]]['am'].lat,
    q_sfc_transport_distance[expid[i]]['am'] / 100,
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel(plot_labels['distance'], linespacing=2)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check transport distance at daily_min and daily_max


stats.describe(q_sfc_transport_distance[expid[i]]['daily'][:, daily_min_ilat, daily_min_ilon], nan_policy='omit')
np.nanmean(q_sfc_transport_distance[expid[i]]['daily'][:, daily_min_ilat, daily_min_ilon])
np.nanstd(q_sfc_transport_distance[expid[i]]['daily'][:, daily_min_ilat, daily_min_ilon], ddof=1)
q_sfc_transport_distance[expid[i]]['am'][daily_min_ilat, daily_min_ilon]
# 1074±556 km
# 997±556 km

stats.describe(q_sfc_transport_distance[expid[i]]['daily'][:, daily_max_ilat, daily_max_ilon], nan_policy='omit')
np.nanmean(q_sfc_transport_distance[expid[i]]['daily'][:, daily_max_ilat, daily_max_ilon])
np.nanstd(q_sfc_transport_distance[expid[i]]['daily'][:, daily_max_ilat, daily_max_ilon], ddof=1)
q_sfc_transport_distance[expid[i]]['am'][daily_max_ilat, daily_max_ilon]
# 415±111 km
# 385±111 km

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am relative source latitude of q_sfc

q_sfc_weighted_var[expid[i]]['lat']['daily'].std(dim='time', ddof=1).to_netcdf('scratch/test/test0.nc')

(q_sfc_weighted_var[expid[i]]['lat']['am'] - q_sfc_weighted_var[expid[i]]['lat']['am'].lat).to_netcdf('scratch/test/test0.nc')

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.2 ' + expid[i] + ' am q_sfc relative source latitude.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='RdBu',)

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['am'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['am'].lat,
    q_sfc_weighted_var[expid[i]]['lat']['am'] - q_sfc_weighted_var[expid[i]]['lat']['am'].lat,
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.12,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('Relative source latitude [$°$]', linespacing=2)

fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot std of daily source latitude of q_sfc

q_sfc_weighted_var[expid[i]]['lat']['daily'].std(dim='time', ddof=1).to_netcdf('scratch/test/test0.nc')

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.2 ' + expid[i] + ' daily q_sfc relative source latitude std.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=6, cm_interval1=0.5, cm_interval2=1, cmap='Oranges',
    reversed=False,)

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['am'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['am'].lat,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].std(dim='time', ddof=1),
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.12,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('Standard deviation of daily source latitude [$°$]')

fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------







