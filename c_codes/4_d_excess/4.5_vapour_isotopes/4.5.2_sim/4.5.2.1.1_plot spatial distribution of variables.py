
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

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

lon = dD_q_sfc_alltime[expid[i]]['am'].lon
lat = dD_q_sfc_alltime[expid[i]]['am'].lat


ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q',
                 'lat', 'sst', 'rh2m', 'wind10']:
    # var_name = 'dD'
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
    cm_min=-4, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG',
    asymmetric=True,)

fig, ax = hemisphere_plot(
    northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

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
