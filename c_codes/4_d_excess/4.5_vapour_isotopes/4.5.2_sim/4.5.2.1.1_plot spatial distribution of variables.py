# check


# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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
from scipy.stats import pearsonr
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath
from metpy.calc import pressure_to_height_std
from metpy.units import units

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    remove_trailing_zero_pos_abs,
    ticks_labels,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
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
    plot_labels,
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

dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

d_excess_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

q_sfc_weighted_var = {}
q_sfc_weighted_var[expid[i]] = {}

for src_var in ['lat', 'sst', 'rh2m', 'wind10']:
    print('#--------------------------------' + src_var)
    src_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + src_var + '.pkl'
    print(src_file)
    
    with open(src_file, 'rb') as f:
        q_sfc_weighted_var[expid[i]][src_var] = pickle.load(f)

q_sfc_transport_distance = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_transport_distance.pkl', 'rb') as f:
    q_sfc_transport_distance[expid[i]] = pickle.load(f)

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

lon = dD_q_sfc_alltime[expid[i]]['am'].lon
lat = dD_q_sfc_alltime[expid[i]]['am'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am surface vapour variables

for var_name in ['distance']:
    # var_name = 'dD'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 'lat', 'sst', 'rh2m', 'wind10']
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 'dD'):
        var = dD_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-470, cm_max=-70, cm_interval1=25, cm_interval2=50,
            cmap='viridis', reversed=False)
    elif (var_name == 'd18O'):
        var = dO18_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=-10, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_xs'):
        var = d_excess_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-3, cm_max=24, cm_interval1=1.5, cm_interval2=3,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_ln'):
        var = d_ln_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=31, cm_interval1=1.5, cm_interval2=3,
            cmap='viridis', reversed=False)
    elif (var_name == 'q'): #g/kg
        var = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['am'].sel(
            lev=47, lat=slice(-17, -90)) * 1000
        pltlevel = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltticks = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = cm.get_cmap('viridis', len(pltlevel)-1)
    elif (var_name == 'lat'):
        var = q_sfc_weighted_var[expid[i]]['lat']['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-70, cm_max=-10, cm_interval1=3, cm_interval2=6,
            cmap='viridis', reversed=False)
    elif (var_name == 'sst'):
        var = q_sfc_weighted_var[expid[i]]['sst']['am'].sel(lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=27, cm_interval1=1.5, cm_interval2=3,
            cmap='viridis', reversed=False)
    elif (var_name == 'rh2m'):
        var = q_sfc_weighted_var[expid[i]]['rh2m']['am'].sel(
            lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=65, cm_max=110, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'wind10'):
        var = q_sfc_weighted_var[expid[i]]['wind10']['am'].sel(
            lat=slice(-17, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=5, cm_max=11, cm_interval1=0.25, cm_interval2=0.5,
            cmap='viridis', reversed=False)
    elif (var_name == 'distance'):
        var = q_sfc_transport_distance[expid[i]]['am'].sel(
            lat=slice(-17, -90)) / 100
        # stats.describe(var, axis=None)
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=70, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.1 ' + expid[i] + ' am_sfc ' + var_name + '.png'
    
    fig, ax = hemisphere_plot(
        northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
    
    plt_mesh = plot_t63_contourf(
        lon, lat.sel(lat=slice(-17, -90)), var, ax,
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.05,
        )
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_xlabel('Annual mean surface ' + plot_labels[var_name], linespacing=1.5, size=8)
    if (var_name == 'lat'):
        cbar.ax.invert_xaxis()
        cbar.ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    
    fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am relative source latitude

q_sfc_weighted_var = {}
q_sfc_weighted_var[expid[i]] = {}

for src_var in ['lat',]:
    print('#--------------------------------' + src_var)
    src_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + src_var + '.pkl'
    print(src_file)
    
    with open(src_file, 'rb') as f:
        q_sfc_weighted_var[expid[i]][src_var] = pickle.load(f)

lon = q_sfc_weighted_var[expid[i]][src_var]['am'].lon
lat = q_sfc_weighted_var[expid[i]][src_var]['am'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.1 ' + expid[i] + ' am_sfc relative source latitude.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='BrBG',
    asymmetric=True,)

fig, ax = hemisphere_plot(
    northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

# stats.describe(q_sfc_weighted_var[expid[i]][src_var]['am'] - lat.sel(lat=slice(-17, -90)), axis=None)
plt_mesh = plot_t63_contourf(
    lon, lat, q_sfc_weighted_var[expid[i]][src_var]['am'] - lat, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('Annual mean surface relative Source latitude [$°$]', linespacing=1.5, size=7)

fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am relative source sst

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

q_sfc_weighted_var = {}
q_sfc_weighted_var[expid[i]] = {}

for src_var in ['sst',]:
    print('#--------------------------------' + src_var)
    src_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + src_var + '.pkl'
    print(src_file)
    
    with open(src_file, 'rb') as f:
        q_sfc_weighted_var[expid[i]][src_var] = pickle.load(f)

lon = q_sfc_weighted_var[expid[i]][src_var]['am'].lon
lat = q_sfc_weighted_var[expid[i]][src_var]['am'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.1 ' + expid[i] + ' am_sfc relative source sst.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=8, cm_interval1=2, cm_interval2=2, cmap='BrBG',
    asymmetric=True,)

fig, ax = hemisphere_plot(
    northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt_mesh = plot_t63_contourf(
    lon, lat, q_sfc_weighted_var[expid[i]][src_var]['am'] - tsw_alltime[expid[i]]['am'], ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('Annual mean surface relative Source SST [$°\;C$]', linespacing=1.5, size=7)

fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am contributions of each region to q

q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)

lon = q_geo7_sfc_frc_alltime[expid[i]]['am'].lon
lat = q_geo7_sfc_frc_alltime[expid[i]]['am'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

for iregion in q_geo7_sfc_frc_alltime[expid[i]]['am'].geo_regions.values:
    # iregion = 'Open Ocean'
    print('#-------------------------------- ' + iregion)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.1_region_contributions/8.3.1.1.0 ' + expid[i] + ' am_sfc ' + iregion + ' contributions.png'
    
    if (iregion in ['AIS', 'Land excl. AIS', 'SH seaice']):
        cm_min = 0
        cm_max = 40
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'magma'
        reverse = True
        expand = 'max'
    if (iregion in ['Open Ocean']):
        cm_min = 40
        cm_max = 100
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'viridis'
        reverse = True
        expand = 'min'
    if (iregion in ['Pacific Ocean', 'Southern Ocean', 'Indian Ocean', 'Atlantic Ocean',]):
        cm_min = 0
        cm_max = 100
        cm_interval1 = 10
        cm_interval2 = 20
        cmap = 'cividis'
        reverse = True
        expand = 'neither'
    
    var = q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(
        geo_regions=iregion, lat=slice(-17, -90))
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=cm_min, cm_max=cm_max,
        cm_interval1=cm_interval1, cm_interval2=cm_interval2,
        cmap=cmap, reversed=reverse)
    
    fig, ax = hemisphere_plot(
        northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
    
    plt_mesh = plot_t63_contourf(
        lon, lat.sel(lat=slice(-17, -90)), var, ax,
        pltlevel, expand, pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend=expand,
        pad=0.05,
        )
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_xlabel('Contribution to surface q from ' + iregion + ' [$\%$]', linespacing=1.5, size=8)
    
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate daily varibales


itime_start = np.datetime64('2021-12-01')
itime_end   = np.datetime64('2022-03-01')

north_extent = -30

for var_name in ['lat',]:
    # var_name = 'dD'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 'lat', 'sst', 'rh2m', 'wind10', 'distance']
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 'dD'):
        var = dD_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-500, cm_max=-100, cm_interval1=25, cm_interval2=50,
            cmap='viridis', reversed=False)
    elif (var_name == 'd18O'):
        var = dO18_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-80, cm_max=-10, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_xs'):
        var = d_excess_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=30, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_ln'):
        var = d_ln_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=35, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'q'): #g/kg
        var = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'].sel(
            lev=47, lat=slice(north_extent + 2, -90)) * 1000
        pltlevel = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltticks = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = cm.get_cmap('viridis', len(pltlevel)-1)
    elif (var_name == 'lat'):
        var = q_sfc_weighted_var[expid[i]]['lat']['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-70, cm_max=-20, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'sst'):
        var = q_sfc_weighted_var[expid[i]]['sst']['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=27, cm_interval1=1.5, cm_interval2=3,
            cmap='viridis', reversed=False)
    elif (var_name == 'rh2m'):
        var = q_sfc_weighted_var[expid[i]]['rh2m']['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=65, cm_max=110, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'wind10'):
        var = q_sfc_weighted_var[expid[i]]['wind10']['daily'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=16, cm_interval1=0.5, cm_interval2=2,
            cmap='viridis', reversed=False)
    elif (var_name == 'distance'):
        var = q_sfc_transport_distance[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90)) / 100
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=20,
            cmap='viridis', reversed=False)
    
    # print(stats.describe(var, axis=None, nan_policy='omit'))
    
    itime_start_idx = np.argmin(abs(var.time.values - itime_start))
    itime_end_idx = np.argmin(abs(var.time.values - itime_end))
    plt_data = var[itime_start_idx:itime_end_idx].compute().copy()
    
    start_time = str(plt_data.time.values[0])[:10]
    end_time   = str(plt_data.time.values[-1])[:10]
    print('Start time: ' + start_time)
    print('End time:   ' + end_time)
    
    output_mp4 = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.2_animation/8.3.1.2.0 ' + expid[i] + ' daily_sfc ' + var_name + ' ' + start_time + ' to ' + end_time + '.mp4'
    
    fig, ax = hemisphere_plot(northextent=north_extent, fm_top=0.92,)
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
    
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
        format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.15,
        )
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('Daily surface ' + plot_labels[var_name], linespacing=1.5)
    
    plt_objs = []
    
    def update_frames(itime):
        # itime = 0
        global plt_objs
        for plt_obj in plt_objs:
            plt_obj.remove()
        plt_objs = []
        
        plt_mesh = ax.pcolormesh(
            plt_data.lon,
            plt_data.lat,
            plt_data[itime],
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), )
        
        plt_txt = plt.text(
            0.5, 1, str(plt_data.time[itime].values)[:10],
            transform=ax.transAxes,
            ha='center', va='bottom', rotation='horizontal')
        
        plt_objs = [plt_mesh, plt_txt, ]
        return(plt_objs)
    
    ani = animation.FuncAnimation(
        fig, update_frames, frames=itime_end_idx - itime_start_idx,
        interval=500, blit=False)
    
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region am precipitation and surface vapour variables, and their differences

#-------------------------------- import data

dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

d_excess_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    d_excess_alltime[expid[i]] = pickle.load(f)

d_ln_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    d_ln_alltime[expid[i]] = pickle.load(f)

pre_weighted_var = {}
pre_weighted_var[expid[i]] = {}

source_var = ['lat', 'sst', 'rh2m', 'wind10', 'distance']
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.pre_weighted_lat.pkl',
    prefix + '.pre_weighted_sst.pkl',
    prefix + '.pre_weighted_rh2m.pkl',
    prefix + '.pre_weighted_wind10.pkl',
    prefix + '.transport_distance.pkl',
]

for ivar, ifile in zip(source_var, source_var_files):
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        pre_weighted_var[expid[i]][ivar] = pickle.load(f)

#-------------------------------- plot data
north_extent = -60

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'lat', 'sst', 'rh2m', 'wind10', 'distance', ]:
    # var_name = 'd18O'
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 'dD'):
        sfc_data = dD_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = dD_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-470, cm_max=-150, cm_interval1=20, cm_interval2=40,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-140, cm_max=0, cm_interval1=10, cm_interval2=20,
            cmap='viridis', reversed=False)
    elif (var_name == 'd18O'):
        sfc_data = dO18_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = dO18_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=-20, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-18, cm_max=0, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_xs'):
        sfc_data = d_excess_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = d_excess_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=20, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=0, cm_max=18, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_ln'):
        sfc_data = d_ln_q_sfc_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = d_ln_alltime[expid[i]]['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=8, cm_max=24, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=0, cm_max=12, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
    elif (var_name == 'lat'):
        sfc_data = q_sfc_weighted_var[expid[i]]['lat']['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = pre_weighted_var[expid[i]]['lat']['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-50, cm_max=-36, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-18, cm_max=2, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
    elif (var_name == 'sst'):
        sfc_data = q_sfc_weighted_var[expid[i]]['sst']['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = pre_weighted_var[expid[i]]['sst']['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-9, cm_max=2, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
    elif (var_name == 'rh2m'):
        sfc_data = q_sfc_weighted_var[expid[i]]['rh2m']['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = pre_weighted_var[expid[i]]['rh2m']['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=76, cm_max=90, cm_interval1=1, cm_interval2=2,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=0, cm_max=5, cm_interval1=0.5, cm_interval2=1,
            cmap='viridis', reversed=False)
    elif (var_name == 'wind10'):
        sfc_data = q_sfc_weighted_var[expid[i]]['wind10']['am'].sel(lat=slice(north_extent + 2, -90))
        pre_data = pre_weighted_var[expid[i]]['wind10']['am'].sel(lat=slice(north_extent + 2, -90))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=6, cm_max=10.5, cm_interval1=0.25, cm_interval2=0.5,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=0, cm_interval1=0.1, cm_interval2=0.2,
            cmap='viridis', reversed=False)
    elif (var_name == 'distance'):
        sfc_data = q_sfc_transport_distance[expid[i]]['am'].sel(
            lat=slice(north_extent + 2, -90)) / 100
        pre_data = pre_weighted_var[expid[i]]['distance']['am'].sel(lat=slice(north_extent + 2, -90)) / 100
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=5, cm_max=65, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-28, cm_max=10, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
    
    # print(stats.describe(np.concatenate((sfc_data.values[echam6_t63_ais_mask['mask']['AIS'][-17:, :]], pre_data.values[echam6_t63_ais_mask['mask']['AIS'][-17:, :]])), axis=None, nan_policy='omit'))
    # print(stats.describe((sfc_data.values - pre_data.values)[echam6_t63_ais_mask['mask']['AIS'][-17:, :]], axis=None, nan_policy='omit'))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.3_sfc_vs_pre/8.3.1.3.0 ' + expid[i] + ' am_sfc_pre ' + var_name + '.png'
    
    cbar_label = 'Annual mean ' + plot_labels[var_name]
    cbar_label2 = 'Differences: (a) - (b)'
    
    nrow = 1
    ncol = 3
    fm_bottom = 2.5 / (5.8*nrow + 2)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
        subplot_kw={'projection': ccrs.SouthPolarStereo()},
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)
    
    ipanel=0
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(northextent=north_extent,ax_org = axs[jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
        plt.text(
            0.05, 0.975, panel_labels[ipanel],
            transform=axs[jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
    
    plt1 = plot_t63_contourf(
        sfc_data.lon,
        sfc_data.lat,
        sfc_data,
        axs[0],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    plot_t63_contourf(
        pre_data.lon,
        pre_data.lat,
        pre_data,
        axs[1],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    plt2 = plot_t63_contourf(
        pre_data.lon,
        pre_data.lat,
        sfc_data - pre_data,
        axs[2],
        pltlevel2, 'both', pltnorm2, pltcmp2, ccrs.PlateCarree(),)
    
    for jcol in range(ncol):
        axs[jcol].add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    
    plt.text(
        0.5, 1.05, 'Surface vapour', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, 'Precipitation', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(a) - (b)', transform=axs[2].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar1 = fig.colorbar(
        plt1, ax=axs,
        orientation="horizontal",shrink=0.5,aspect=40,
        anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
    cbar1.ax.set_xlabel(cbar_label, linespacing=2)
    
    cbar2 = fig.colorbar(
        plt2, ax=axs,
        orientation="horizontal",shrink=0.5,aspect=40,
        anchor=(1.1,-2.2),ticks=pltticks2)
    cbar2.ax.set_xlabel(cbar_label2, linespacing=2)
    
    fig.subplots_adjust(
        left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
    fig.savefig(output_png)



'''
    # print(stats.describe(sfc_data.values[echam6_t63_ais_mask['mask']['AIS'][-17:, :]], axis=None, nan_policy='omit'))
    # print(stats.describe(pre_data.values[echam6_t63_ais_mask['mask']['AIS'][-17:, :]], axis=None, nan_policy='omit'))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am evaporation in ECHAM6 and ERA5 and their diff


wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

with open('scratch/ERA5/evap/ERA5_evap_1979_2022_alltime.pkl', 'rb') as f:
    ERA5_evap_1979_2022_alltime = pickle.load(f)

#---------------------------- global plot

plt_data1 = wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001
plt_data2 = ERA5_evap_1979_2022_alltime['am'] * 365
# plt_data3 = regrid(plt_data1) - regrid(plt_data2)
plt_data3 = (regrid(plt_data1) - regrid(plt_data2)) / abs(regrid(plt_data2)) * 100

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=0, cm_interval1=0.1, cm_interval2=0.2, cmap='viridis',)
# pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
#     cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='PiYG',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='PiYG',)

# output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am evap differences.png'
output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am evap differences_percentage.png'

cbar_label1 = 'Evaporation [$m \; year^{-1}$]'
# cbar_label2 = 'Differences [$m \; year^{-1}$]'
cbar_label2 = 'Differences [$\%$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

plt_mesh1 = plot_t63_contourf(
    plt_data1.lon,
    plt_data1.lat,
    plt_data1,
    axs[0], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh2 = axs[1].contourf(
    plt_data2.longitude,
    plt_data2.latitude,
    plt_data2,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_mesh3 = axs[2].contourf(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    levels = pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, '(a): ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b): ERA5', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, '(c): (a) - (b)', transform=axs[2].transAxes,
#     ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c): (a - b) / |b|', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)



'''
print(stats.describe(plt_data1, axis=None))
print(stats.describe(plt_data2, axis=None))
print(stats.describe(plt_data3, axis=None))

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)

# mm/s * 86400 s/d * 365 d/yr * 0.001 m/mm
print(stats.describe((wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001).values[echam6_t63_ais_mask['mask']['AIS']]))
# m/d * 365 d/yr = m/yr, positive:downward
print(stats.describe((ERA5_evap_1979_2022_alltime['am'] * 365).values[era5_ais_mask['mask']['AIS']]))

print(stats.describe((wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001).values.flatten()))
print(stats.describe((ERA5_evap_1979_2022_alltime['am'] * 365).values.flatten()))


(wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001).to_netcdf('scratch/test/test.nc')
(ERA5_evap_1979_2022_alltime['am'] * 365).to_netcdf('scratch/test/test2.nc')

with open('scratch/ERA5/evap/ERA5_evap_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_evap_2013_2022_alltime = pickle.load(f)
# m/hour * 24 hour/d * 365 d/yr = m/yr, positive:downward
print(stats.describe((ERA5_evap_2013_2022_alltime['am'] * 24 * 365).values[era5_ais_mask['mask']['AIS']]))
(ERA5_evap_2013_2022_alltime['am'] * 24 * 365).to_netcdf('scratch/test/test1.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am evaporation in ECHAM6 and ERA5 and their diff_SH

wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

with open('scratch/ERA5/evap/ERA5_evap_1979_2022_alltime.pkl', 'rb') as f:
    ERA5_evap_1979_2022_alltime = pickle.load(f)

#---------------------------- global plot

plt_data1 = wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001
plt_data2 = ERA5_evap_1979_2022_alltime['am'] * 365
plt_data3 = (regrid(plt_data1) - regrid(plt_data2)) / abs(regrid(plt_data2)) * 100

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.6, cm_max=0.04, cm_interval1=0.04, cm_interval2=0.08, cmap='BrBG',
    asymmetric=True,)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='PiYG',)

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am evap differences_percentage_SH.png'

cbar_label1 = 'Evaporation [$m \; year^{-1}$]'
cbar_label2 = 'Differences [$\%$]'

nrow = 1
ncol = 3
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    # cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])

plt_mesh1 = plot_t63_contourf(
    plt_data1.lon,
    plt_data1.lat,
    plt_data1,
    axs[0], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh2 = axs[1].contourf(
    plt_data2.longitude,
    plt_data2.latitude,
    plt_data2,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_mesh3 = axs[2].contourf(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    levels = pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, '(a): ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b): ERA5', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c): (a - b) / |b|', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(1.1,-2.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)





with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')
era5_cellarea = xr.open_dataset('scratch/ERA5/temp2/ERA5_cellarea.nc')

echam6_evap = wisoevap_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d * 365 * 0.001
era5_evap = ERA5_evap_1979_2022_alltime['am'] * 365

# np.mean(echam6_evap.values[echam6_t63_ais_mask['mask']['AIS']])
np.average(
    echam6_evap.values[echam6_t63_ais_mask['mask']['AIS']],
    weights=echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)

# np.mean(era5_evap.values[era5_ais_mask['mask']['AIS']])
np.average(
    era5_evap.values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)



'''
'''
# endregion
# -----------------------------------------------------------------------------




