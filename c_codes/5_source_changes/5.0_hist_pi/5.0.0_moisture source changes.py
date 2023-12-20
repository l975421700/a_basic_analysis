

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'hist_700_5.0',
    ]
# i = 0


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

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

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import cartopy.feature as cfeature

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    remove_trailing_zero_pos,
    remove_trailing_zero_pos_abs,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    regional_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    time_weighted_mean,
    regrid,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
    plot_labels,
)

from a_basic_analysis.b_module.source_properties import (
    calc_lon_diff,
    source_properties,
    sincoslon_2_lon,
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

pre_weighted_var = {}

for i in range(len(expid)):
    print('#-------------------------------- ' + str(i) + ' ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', ]
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_lon.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS plot

for ivar in ['lat', 'lon', 'sst', 'wind10']:
    # ivar = 'lat'
    print('#-------------------------------- ' + ivar)
    
    plt_data1 = pre_weighted_var[expid[0]][ivar]['am']
    plt_data2 = pre_weighted_var[expid[1]][ivar]['am']
    
    if (ivar != 'lon'):
        plt_data3 = plt_data1 - plt_data2
    else:
        plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    
    output_png = 'figures/2_source_changes/2.0_hist_pi/2.0.0 ' + expid[0] + ' vs. ' + expid[1] + ' Antarctica am source ' + ivar + '.png'
    
    if (ivar == 'lat'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-46, cm_max=-34, cm_interval1=1, cm_interval2=2,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1,
            cmap='PiYG',)
    elif (ivar == 'lon'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=360, cm_interval1=30, cm_interval2=60,
            cmap='twilight_shifted',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
            cmap='PRGn',)
    elif (ivar == 'sst'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=11, cm_max=17, cm_interval1=0.5, cm_interval2=1,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=0.5,
            cmap='BrBG',)
    elif (ivar == 'wind10'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=7, cm_max=11, cm_interval1=0.4, cm_interval2=0.8,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.4, cm_interval2=0.8,
            cmap='BrBG',)
    
    nrow = 1
    ncol = 3
    fm_bottom = 2.5 / (5.8*nrow + 2)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
        subplot_kw={'projection': ccrs.SouthPolarStereo()},
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)
    
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    
    if (ivar != 'lon'):
        plt_mesh1 = plot_t63_contourf(
            plt_data1.lon,
            plt_data1.lat.sel(lat=slice(-60 + 2, -90)),
            plt_data1.sel(lat=slice(-60 + 2, -90)),
            axs[0], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh1 = axs[0].pcolormesh(
            plt_data1.lon, plt_data1.lat, plt_data1,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh2 = plot_t63_contourf(
            plt_data2.lon,
            plt_data2.lat.sel(lat=slice(-60 + 2, -90)),
            plt_data2.sel(lat=slice(-60 + 2, -90)),
            axs[1], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh3 = plot_t63_contourf(
            plt_data3.lon,
            plt_data3.lat.sel(lat=slice(-60 + 2, -90)),
            plt_data3.sel(lat=slice(-60 + 2, -90)),
            axs[2], pltlevel2, 'both', pltnorm2, pltcmp2, ccrs.PlateCarree(),)
    else:
        plt_mesh3 = axs[2].pcolormesh(
            plt_data3.lon, plt_data3.lat, plt_data3,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    axs[0].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    axs[1].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    axs[2].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    
    plt.text(
        0.5, 1.05, '(a) PI', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(b) HIST', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(c) Differences: (a - b)', transform=axs[2].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar1 = fig.colorbar(
        plt_mesh1, ax=axs,
        orientation="horizontal",shrink=0.5, aspect=25, extend='both',
        anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos_abs, )
    cbar1.ax.set_xlabel(plot_labels[ivar], linespacing=2)
    
    cbar2 = fig.colorbar(
        plt_mesh3, ax=axs,
        orientation="horizontal",shrink=0.5,aspect=25, extend='both',
        anchor=(1.1,-2.2),ticks=pltticks2, format=remove_trailing_zero_pos,)
    cbar2.ax.set_xlabel('Differences', linespacing=2)
    
    fig.subplots_adjust(
        left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

