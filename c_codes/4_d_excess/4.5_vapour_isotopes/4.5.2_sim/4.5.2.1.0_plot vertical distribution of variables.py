
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

dD_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
    dD_q_alltime[expid[i]] = pickle.load(f)

dO18_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl', 'rb') as f:
    dO18_q_alltime[expid[i]] = pickle.load(f)

d_ln_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_alltime.pkl', 'rb') as f:
    d_ln_q_alltime[expid[i]] = pickle.load(f)

d_excess_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_alltime.pkl', 'rb') as f:
    d_excess_q_alltime[expid[i]] = pickle.load(f)

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)


q_weighted_var = {}
q_weighted_var[expid[i]] = {}

for src_var in ['lat', 'sst', 'rh2m', 'wind10']:
    print('#--------------------------------' + src_var)
    src_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_' + src_var + '.pkl'
    print(src_file)
    
    with open(src_file, 'rb') as f:
        q_weighted_var[expid[i]][src_var] = pickle.load(f)


ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

lon = dD_q_alltime[expid[i]]['am'].lon
lat = dD_q_alltime[expid[i]]['am'].lat
plevs = dD_q_alltime[expid[i]]['am'].plev

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q',
                 'lat', 'sst', 'rh2m', 'wind10']:
    # var_name = 'lat'
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 'dD'):
        var = dD_q_alltime[expid[i]]['am'].weighted(
            wiso_q_plev_alltime[expid[i]]['q16o']['am'].fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-580, cm_max=-80, cm_interval1=25, cm_interval2=50,
            cmap='viridis', reversed=False)
    elif (var_name == 'd18O'):
        var = dO18_q_alltime[expid[i]]['am'].weighted(
            wiso_q_plev_alltime[expid[i]]['q16o']['am'].fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-80, cm_max=-10, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_xs'):
        var = d_excess_q_alltime[expid[i]]['am'].weighted(
            wiso_q_plev_alltime[expid[i]]['q16o']['am'].fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=40, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_ln'):
        var = d_ln_q_alltime[expid[i]]['am'].weighted(
            wiso_q_plev_alltime[expid[i]]['q16o']['am'].fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=30, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'q'): #g/kg
        var = (wiso_q_plev_alltime[expid[i]]['q16o']['am'] * 1000
               ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltticks = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = cm.get_cmap('viridis', len(pltlevel)-1)
    elif (var_name == 'lat'):
        var = q_weighted_var[expid[i]]['lat']['am'].weighted(
            ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=0, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    elif (var_name == 'sst'):
        var = q_weighted_var[expid[i]]['sst']['am'].weighted(
            ocean_q_alltime[expid[i]]['am'].sel(var_names='sst').fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=30, cm_interval1=1, cm_interval2=3,
            cmap='viridis', reversed=False)
    elif (var_name == 'rh2m'):
        var = q_weighted_var[expid[i]]['rh2m']['am'].weighted(
            ocean_q_alltime[expid[i]]['am'].sel(var_names='rh2m').fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=68, cm_max=100, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
    elif (var_name == 'wind10'):
        var = q_weighted_var[expid[i]]['wind10']['am'].weighted(
            ocean_q_alltime[expid[i]]['am'].sel(var_names='wind10').fillna(0)
            ).mean(dim='lon').sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=5.5, cm_max=10.5, cm_interval1=0.5, cm_interval2=1,
            cmap='viridis', reversed=False)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.0 ' + expid[i] + ' am_zm_SH ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
    plt_mesh = ax.contourf(
        lat.sel(lat=slice(3, -90)),
        plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
        var,
        norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both',)
    
    ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
    ax.set_xlim(0, -88.57)
    ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    
    ax.invert_yaxis()
    ax.set_ylim(1000, 200)
    ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
    ax.set_ylabel('Pressure [$hPa$]')
    
    ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks,
        pad=0.1, fraction=0.04, anchor=(0.5, -1),
        )
    
    cbar.ax.set_xlabel('Zonal-averaged annual mean ' + plot_labels[var_name],)
    if (var_name == 'lat'):
        cbar.ax.invert_xaxis()
        cbar.ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    
    # 2nd y-axis
    height = np.round(
        pressure_to_height_std(
            pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
    ax2 = ax.twinx()
    ax2.invert_yaxis()
    ax2.set_ylim(1000, 200)
    ax2.set_yticks(np.arange(1000, 200 - 1e-4, -100))
    ax2.set_yticklabels(height.magnitude, c = 'gray')
    ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')
    
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.98)
    fig.savefig(output_png)






'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SH plot am contribution of each region to zm atmospheric humidity

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

lat = q_geo7_alltiime[expid[i]]['am'].lat
plevs = q_geo7_alltiime[expid[i]]['am'].plev

q_geo7_alltiime[expid[i]]['am_zm'] = q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').compute()

for iregion in ['AIS', 'Land excl. AIS', 'Atlantic Ocean',
                'Indian Ocean', 'Pacific Ocean', 'SH seaice',
                'Southern Ocean', 'Open Ocean',]:
    # iregion = 'Open Ocean'
    print('#-------------------------------- ' + iregion)
    
    if (iregion in ['AIS', 'Land excl. AIS', 'SH seaice',]):
        cm_min = 0
        cm_max = 30
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'magma'
        reverse = True
        expand = 'max'
    if (iregion in ['Open Ocean']):
        cm_min = 60
        cm_max = 100
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'viridis'
        reverse = True
        expand = 'min'
    if (iregion in ['Pacific Ocean', 'Southern Ocean', 'Indian Ocean', 'Atlantic Ocean',]):
        cm_min = 0
        cm_max = 50
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'cividis'
        reverse = True
        expand = 'max'
    
    var = (q_geo7_alltiime[expid[i]]['am_zm'].sel(geo_regions=iregion) / \
        q_geo7_alltiime[expid[i]]['am_zm'].sel(geo_regions='Sum') * 100
        ).sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4))
    
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=cm_min, cm_max=cm_max,
        cm_interval1=cm_interval1, cm_interval2=cm_interval2,
        cmap=cmap, reversed=reverse)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.1_region_contributions/8.3.1.1.0 ' + expid[i] + ' am_zm_SH ' + iregion + ' contributions.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
    plt_mesh = ax.contourf(
        lat.sel(lat=slice(3, -90)),
        plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
        var,
        norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend=expand,)
    
    ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
    ax.set_xlim(0, -88.57)
    ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    
    ax.invert_yaxis()
    ax.set_ylim(1000, 200)
    ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
    ax.set_ylabel('Pressure [$hPa$]')
    
    ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks,
        pad=0.1, fraction=0.04, anchor=(0.5, -1),
        )
    
    cbar.ax.set_xlabel('Contribution to zonal-averaged annual mean q from ' + iregion + ' [$\%$]',)
    
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region NH plot am contribution of each region to zm atmospheric humidity

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

lat = q_geo7_alltiime[expid[i]]['am'].lat
plevs = q_geo7_alltiime[expid[i]]['am'].plev

q_geo7_alltiime[expid[i]]['am_zm'] = q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').compute()

for iregion in ['Land excl. AIS', 'Atlantic Ocean',
                'Indian Ocean', 'Pacific Ocean',
                'Open Ocean',]:
    # iregion = 'AIS'
    print('#-------------------------------- ' + iregion)
    
    if (iregion in ['Land excl. AIS',]):
        cm_min = 0
        cm_max = 50
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'magma'
        reverse = True
        expand = 'max'
    if (iregion in ['Open Ocean']):
        cm_min = 50
        cm_max = 100
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'viridis'
        reverse = True
        expand = 'min'
    if (iregion in ['Pacific Ocean', 'Indian Ocean', 'Atlantic Ocean',]):
        cm_min = 0
        cm_max = 50
        cm_interval1 = 5
        cm_interval2 = 10
        cmap = 'cividis'
        reverse = True
        expand = 'max'
    
    var = (q_geo7_alltiime[expid[i]]['am_zm'].sel(geo_regions=iregion) / \
        q_geo7_alltiime[expid[i]]['am_zm'].sel(geo_regions='Sum') * 100
        ).sel(lat=slice(90, -3), plev=slice(1e+5, 2e+4))
    
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=cm_min, cm_max=cm_max,
        cm_interval1=cm_interval1, cm_interval2=cm_interval2,
        cmap=cmap, reversed=reverse)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.1_region_contributions/8.3.1.1.0 ' + expid[i] + ' am_zm_NH ' + iregion + ' contributions.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
    plt_mesh = ax.contourf(
        lat.sel(lat=slice(90, -3)),
        plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
        var,
        norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend=expand,)
    
    # ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
    # ax.set_xlim(0, -88.57)
    ax.set_xticks(np.arange(90, 0 - 1e-4, -10))
    ax.set_xlim(88.57, 0)
    ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    
    ax.invert_yaxis()
    ax.set_ylim(1000, 200)
    ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
    ax.set_ylabel('Pressure [$hPa$]')
    
    ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks,
        pad=0.1, fraction=0.04, anchor=(0.5, -1),
        )
    
    cbar.ax.set_xlabel('Contributions to zonal-averaged annual mean q from ' + iregion + ' [$\%$]',)
    
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------




