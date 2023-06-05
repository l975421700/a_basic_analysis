

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
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
# sys.path.append('/work/ollie/qigao001')

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

#---- import d_ln

d_ln_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)

lon = d_ln_alltime[expid[i]]['am'].lon
lat = d_ln_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

#---- import precipitation sources

source_var = ['latitude', 'SST', 'rh2m', 'wind10']
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
(sam_mon[expid[0]].sam.values == sam_mon[expid[3]].sam.values).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d_ln & source latitude

ivar = 'latitude'

cor_d_ln_ivar = {}
cor_d_ln_ivar_p = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    cor_d_ln_ivar[expid[i]] = xr.corr(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').compute()
    
    cor_d_ln_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').values
    
    cor_d_ln_ivar[expid[i]].values[cor_d_ln_ivar_p[expid[i]] > 0.05] = np.nan


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.0 pi_600_3 corr. d_ln and ' + ivar + ' mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat, cor_d_ln_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: source '+ivar+' & $d_{ln}$', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d_ln & source SST

ivar = 'SST'

cor_d_ln_ivar = {}
cor_d_ln_ivar_p = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    cor_d_ln_ivar[expid[i]] = xr.corr(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').compute()
    
    cor_d_ln_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').values
    
    cor_d_ln_ivar[expid[i]].values[cor_d_ln_ivar_p[expid[i]] > 0.05] = np.nan


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.0 pi_600_3 corr. d_ln and ' + ivar + ' mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat, cor_d_ln_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: source '+ivar+' & $d_{ln}$', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d_ln & source rh2m

ivar = 'rh2m'

cor_d_ln_ivar = {}
cor_d_ln_ivar_p = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    cor_d_ln_ivar[expid[i]] = xr.corr(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').compute()
    
    cor_d_ln_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').values
    
    cor_d_ln_ivar[expid[i]].values[cor_d_ln_ivar_p[expid[i]] > 0.05] = np.nan


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.0 pi_600_3 corr. d_ln and ' + ivar + ' mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat, cor_d_ln_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: source '+ivar+' & $d_{ln}$', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d_ln & source wind10

ivar = 'wind10'

cor_d_ln_ivar = {}
cor_d_ln_ivar_p = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    cor_d_ln_ivar[expid[i]] = xr.corr(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').compute()
    
    cor_d_ln_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
            pre_weighted_var[expid[i]][ivar]['mm'],
        dim='time').values
    
    cor_d_ln_ivar[expid[i]].values[cor_d_ln_ivar_p[expid[i]] > 0.05] = np.nan


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.0 pi_600_3 corr. d_ln and ' + ivar + ' mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat, cor_d_ln_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: source '+ivar+' & $d_{ln}$', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------
