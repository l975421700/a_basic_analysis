

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'hist_700_5.0',
    ]
i = 0


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


scaled_flux_var_UM = {}

scaled_flux_var_UM['lat'] = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Lat.nc')['mean_evap_source_latitude']
scaled_flux_var_UM['lon'] = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Lon.nc')['mean_evap_source_longitude']
scaled_flux_var_UM['sst'] = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_SST.nc')['mean_evap_source_sst']
scaled_flux_var_UM['wind10'] = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Wind.nc')['mean_evap_source_wind__k_1_']


tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)





'''
tsw_alltime[expid[i]]['mon'].sel(time='1990-01-31T23:52:30').to_netcdf('scratch/test/test0.nc')
(UM_sst.surface_temperature - zerok).to_netcdf('scratch/test/test1.nc')

UM_sst = xr.open_mfdataset(glob.glob('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/monthly_sst/stash24/*.nc'))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region global plot

for ivar in ['lat', 'lon', 'sst', 'wind10']:
    # ivar = 'lat'
    print('#-------------------------------- ' + ivar)
    
    plt_data1 = pre_weighted_var[expid[i]][ivar]['am']
    plt_data2 = scaled_flux_var_UM[ivar]
    if (ivar == 'sst'):
        plt_data2 = plt_data2 - zerok
    elif (ivar == 'lon'):
        plt_data2.values[plt_data2.values<0] += 360
    
    if (ivar != 'lon'):
        plt_data3 = regrid(plt_data1) - regrid(plt_data2)
    else:
        plt_sinlon = np.sin(plt_data1 / 180 * np.pi)
        plt_coslon = np.cos(plt_data1 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data1 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_sinlon = np.sin(plt_data2 / 180 * np.pi)
        plt_coslon = np.cos(plt_data2 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data2 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM am source ' + ivar + '.png'
    
    # print(stats.describe(plt_data1.values, axis=None, nan_policy='omit'))
    # print(stats.describe(plt_data3.values, axis=None, nan_policy='omit'))
    
    if (ivar == 'lat'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20,
            cmap='PuOr',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3,
            cmap='PiYG',)
    elif (ivar == 'lon'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=360, cm_interval1=20, cm_interval2=60,
            cmap='twilight_shifted',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
            cmap='PRGn',)
    elif (ivar == 'sst'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=8, cm_max=30, cm_interval1=1, cm_interval2=2,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1,
            cmap='BrBG',)
    elif (ivar == 'wind10'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=12, cm_interval1=1, cm_interval2=1,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2.4, cm_max=2.4, cm_interval1=0.4, cm_interval2=0.8,
            cmap='BrBG',)
    
    nrow = 1
    ncol = 3
    fm_right = 1 - 4 / (8.8*ncol + 4)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)
    
    if (ivar != 'lon'):
        plt_mesh1 = plot_t63_contourf(
            plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh1 = axs[0].pcolormesh(
            plt_data1.lon, plt_data1.lat, plt_data1,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh2 = plot_t63_contourf(
            plt_data2.longitude, plt_data2.latitude, plt_data2, axs[1],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh3 = axs[2].contourf(
            plt_data3.lon, plt_data3.lat, plt_data3,
            levels=pltlevel2, extend='both',
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    else:
        plt_mesh3 = axs[2].pcolormesh(
            plt_data3.lon, plt_data3.lat, plt_data3,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    plt.text(
        0.5, 1.05, '(a) ECHAM6', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(b) UM', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(c) Differences: (a - b)', transform=axs[2].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar2 = fig.colorbar(
        plt_mesh3, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(0.8, 0.5), ticks=pltticks2)
    cbar2.ax.set_ylabel('Differences', linespacing=1.5)
    
    cbar1 = fig.colorbar(
        plt_mesh1, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(3.2, 0.5), ticks=pltticks)
    # cbar1.ax.set_ylabel(plot_labels[ivar], linespacing=1.5)
    cbar1.ax.set_ylabel('Source latitude [$°$]', linespacing=1.5)
    
    fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
    fig.savefig(output_png)



'''
pre_weighted_lat = xr.open_dataset('/home/users/qino/scratch/share/for_alison/pre_weighted_lat.nc')
ocean_precipitation = xr.open_dataset('/home/users/qino/scratch/share/for_alison/ocean_precipitation.nc')
plt_data1 = pre_weighted_lat.pre_weighted_lat.weighted(
    ocean_precipitation['__xarray_dataarray_variable__']
).mean(dim='time')

scaled_flux_Lat = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Lat.nc')

plt_data1 = pre_weighted_lat.pre_weighted_lat.weighted(
    ocean_precipitation['__xarray_dataarray_variable__']
).mean(dim='time')
plt_data2 = scaled_flux_Lat.mean_evap_source_latitude

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS plot

for ivar in ['lat', 'lon', 'sst', 'wind10']:
    # ivar = 'lat'
    print('#-------------------------------- ' + ivar)
    
    plt_data1 = pre_weighted_var[expid[i]][ivar]['am']
    plt_data2 = scaled_flux_var_UM[ivar]
    if (ivar == 'sst'):
        plt_data2 = plt_data2 - zerok
    elif (ivar == 'lon'):
        plt_data2.values[plt_data2.values<0] += 360
    
    if (ivar != 'lon'):
        plt_data3 = regrid(plt_data1) - regrid(plt_data2)
    else:
        plt_sinlon = np.sin(plt_data1 / 180 * np.pi)
        plt_coslon = np.cos(plt_data1 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data1 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_sinlon = np.sin(plt_data2 / 180 * np.pi)
        plt_coslon = np.cos(plt_data2 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data2 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM Antarctica am source ' + ivar + '.png'
    
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
            plt_data2.longitude,
            plt_data2.latitude.sel(latitude=slice(-90, -60 + 2)),
            plt_data2.sel(latitude=slice(-90, -60 + 2)),
            axs[1], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh3 = axs[2].contourf(
            plt_data3.lon,
            plt_data3.lat,
            plt_data3,
            levels=pltlevel2, extend='both',
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    else:
        plt_mesh3 = axs[2].pcolormesh(
            plt_data3.lon, plt_data3.lat, plt_data3,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    axs[0].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    axs[1].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    axs[2].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    
    plt.text(
        0.5, 1.05, '(a) ECHAM6', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(b) UM', transform=axs[1].transAxes,
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


# -----------------------------------------------------------------------------
# region global plot no diff.

for ivar in ['lon', 'sst', 'wind10']:
    # ivar = 'lat'
    print('#-------------------------------- ' + ivar)
    
    plt_data1 = pre_weighted_var[expid[i]][ivar]['am']
    plt_data2 = scaled_flux_var_UM[ivar]
    if (ivar == 'sst'):
        plt_data2 = plt_data2 - zerok
    elif (ivar == 'lon'):
        plt_data2.values[plt_data2.values<0] += 360
    
    if (ivar != 'lon'):
        plt_data3 = regrid(plt_data1) - regrid(plt_data2)
    else:
        plt_sinlon = np.sin(plt_data1 / 180 * np.pi)
        plt_coslon = np.cos(plt_data1 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data1 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_sinlon = np.sin(plt_data2 / 180 * np.pi)
        plt_coslon = np.cos(plt_data2 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data2 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM am source ' + ivar + ' no_diff.png'
    
    # print(stats.describe(plt_data1.values, axis=None, nan_policy='omit'))
    # print(stats.describe(plt_data3.values, axis=None, nan_policy='omit'))
    
    if (ivar == 'lat'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20,
            cmap='PuOr',)
    elif (ivar == 'lon'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=360, cm_interval1=20, cm_interval2=60,
            cmap='twilight_shifted',)
    elif (ivar == 'sst'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=8, cm_max=30, cm_interval1=1, cm_interval2=2,
            cmap='viridis_r',)
    elif (ivar == 'wind10'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=12, cm_interval1=1, cm_interval2=1,
            cmap='viridis_r',)
    
    nrow = 1
    ncol = 2
    fm_right = 1 - 2 / (8.8*ncol + 2)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([8.8*ncol + 2, 5*nrow]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)
    
    if (ivar != 'lon'):
        plt_mesh1 = plot_t63_contourf(
            plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh1 = axs[0].pcolormesh(
            plt_data1.lon, plt_data1.lat, plt_data1,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh2 = plot_t63_contourf(
            plt_data2.longitude, plt_data2.latitude, plt_data2, axs[1],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    plt.text(
        0.5, 1.05, '(a) ECHAM6', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(b) UM', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar1 = fig.colorbar(
        plt_mesh1, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(1.2, 0.5), ticks=pltticks)
    cbar1.ax.set_ylabel(plot_labels[ivar], linespacing=1.5)
    # cbar1.ax.set_ylabel('Source latitude [$°$]', linespacing=1.5)
    
    fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
    fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regional plot - Asia

extent=[60, 150, 0, 60]

for ivar in ['lat', 'sst', 'wind10']:
    # ivar = 'lat'
    # 'lon',
    print('#-------------------------------- ' + ivar)
    
    plt_data1 = pre_weighted_var[expid[i]][ivar]['am']
    plt_data2 = scaled_flux_var_UM[ivar]
    if (ivar == 'sst'):
        plt_data2 = plt_data2 - zerok
    elif (ivar == 'lon'):
        plt_data2.values[plt_data2.values<0] += 360
    
    if (ivar != 'lon'):
        plt_data3 = regrid(plt_data1) - regrid(plt_data2)
    else:
        plt_sinlon = np.sin(plt_data1 / 180 * np.pi)
        plt_coslon = np.cos(plt_data1 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data1 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_sinlon = np.sin(plt_data2 / 180 * np.pi)
        plt_coslon = np.cos(plt_data2 / 180 * np.pi)
        plt_sinlon_rgd = regrid(plt_sinlon, grid_spacing=0.1)
        plt_coslon_rgd = regrid(plt_coslon, grid_spacing=0.1)
        plt_data2 = sincoslon_2_lon(plt_sinlon_rgd, plt_coslon_rgd, )
        
        plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM am source ' + ivar + ' Asia.png'
    
    # print(stats.describe(plt_data1.values, axis=None, nan_policy='omit'))
    # print(stats.describe(plt_data3.values, axis=None, nan_policy='omit'))
    
    if (ivar == 'lat'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-10, cm_max=50, cm_interval1=5, cm_interval2=10,
            cmap='PuOr', asymmetric=True,)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3,
            cmap='PiYG',)
    elif (ivar == 'lon'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=360, cm_interval1=20, cm_interval2=60,
            cmap='twilight_shifted',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
            cmap='PRGn',)
    elif (ivar == 'sst'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=15, cm_max=30, cm_interval1=1, cm_interval2=1,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1,
            cmap='BrBG',)
    elif (ivar == 'wind10'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=5, cm_max=11, cm_interval1=1, cm_interval2=1,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2.4, cm_max=2.4, cm_interval1=0.4, cm_interval2=0.8,
            cmap='BrBG',)
    
    nrow = 1
    ncol = 3
    fm_right = 1 - 4 / (7.8*ncol + 4)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([7.8*ncol + 4, 5.6*nrow]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = regional_plot(
            ax_org = axs[jcol], extent=extent,
            figsize = np.array([7.8, 5.2]) / 2.54, ticks_and_labels=False)
        # globe_plot(ax_org = axs[jcol], add_grid_labels=False)
    
    if (ivar != 'lon'):
        plt_mesh1 = plot_t63_contourf(
            plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh1 = axs[0].pcolormesh(
            plt_data1.lon, plt_data1.lat, plt_data1,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh2 = plot_t63_contourf(
            plt_data2.longitude, plt_data2.latitude, plt_data2, axs[1],
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    else:
        plt_mesh2 = axs[1].pcolormesh(
            plt_data2.lon, plt_data2.lat, plt_data2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    if (ivar != 'lon'):
        plt_mesh3 = axs[2].contourf(
            plt_data3.lon, plt_data3.lat, plt_data3,
            levels=pltlevel2, extend='both',
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    else:
        plt_mesh3 = axs[2].pcolormesh(
            plt_data3.lon, plt_data3.lat, plt_data3,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    plt.text(
        0.5, 1.05, '(a) ECHAM6', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(b) UM', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, '(c) Differences: (a - b)', transform=axs[2].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar2 = fig.colorbar(
        plt_mesh3, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(0.8, 0.5), ticks=pltticks2)
    cbar2.ax.set_ylabel('Differences', linespacing=1.5)
    
    cbar1 = fig.colorbar(
        plt_mesh1, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(3.2, 0.5), ticks=pltticks)
    cbar1.ax.set_ylabel(plot_labels[ivar], linespacing=1.5)
    # cbar1.ax.set_ylabel('Source latitude [$°$]', linespacing=1.5)
    
    fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.01, top = 0.92)
    fig.savefig(output_png)



'''
extent=[60, 150, 0, 60]
fig, ax = regional_plot(extent=extent, figsize = np.array([7.8, 5.2]) / 2.54, ticks_and_labels=False)
fig.savefig('figures/test/trial1.png')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sst forcings

# UM_sst = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/monthly_sst/stash24/da072a.pm1990jan.nc')
# UM_sst = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/monthly_sst/stash24/da072a.pm1990feb.nc')
UM_sst = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/monthly_sst/stash24/da072a.pm1990dec.nc')

for ivar in ['sst_forcing']:
    # ivar = 'lat'
    print('#-------------------------------- ' + ivar)
    
    # plt_data1 = tsw_alltime[expid[i]]['mon'].sel(time='1990-01-31T23:52:30')
    # plt_data1 = tsw_alltime[expid[i]]['mon'].sel(time='1990-02-28T23:52:30')
    plt_data1 = tsw_alltime[expid[i]]['mon'].sel(time='1990-12-31T23:52:30')
    plt_data2 = (UM_sst.surface_temperature - zerok)
    plt_data3 = regrid(plt_data1) - regrid(plt_data2)
    
    # output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM sst forcing 199001.png'
    # output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM sst forcing 199002.png'
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.7_ECHAM6_vs_UM/6.1.3.7.0 ' + expid[i] + ' vs. UM sst forcing 199012.png'
    
    if (ivar == 'sst_forcing'):
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=32, cm_interval1=1, cm_interval2=2,
            cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1,
            cmap='BrBG',)
    
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
        plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    plt_mesh2 = plot_t63_contourf(
        plt_data2.longitude, plt_data2.latitude, plt_data2, axs[1],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    plt_mesh3 = axs[2].contourf(
        plt_data3.lon, plt_data3.lat, plt_data3,
        levels=pltlevel2, extend='both',
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    axs[0].add_feature(
        cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
    axs[1].add_feature(
        cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
    axs[2].add_feature(
        cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
    
    plt.text(
        0.5, 1.05, '(a) ECHAM6', transform=axs[0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, '(b) UM', transform=axs[1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, '(c) Differences: (a - b)', transform=axs[2].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cbar2 = fig.colorbar(
        plt_mesh3, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(0.8, 0.5), ticks=pltticks2)
    cbar2.ax.set_ylabel('Differences', linespacing=1.5)
    
    cbar1 = fig.colorbar(
        plt_mesh1, ax=axs,
        orientation="vertical",shrink=1.2,aspect=25,extend='both',
        anchor=(3.2, 0.5), ticks=pltticks)
    # cbar1.ax.set_ylabel('SST in Jan 1990 [$°C$]', linespacing=1.5)
    # cbar1.ax.set_ylabel('SST in Feb 1990 [$°C$]', linespacing=1.5)
    cbar1.ax.set_ylabel('SST in Dec 1990 [$°C$]', linespacing=1.5)
    
    fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
    fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


