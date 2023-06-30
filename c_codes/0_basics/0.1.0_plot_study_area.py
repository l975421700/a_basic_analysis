

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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
    cplot_lon180,
    cplot_wind_vectors,
    cplot_lon180_quiver,
    cplot_lon180_ctr,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 600, ]
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

ais_imbie2 = gpd.read_file(
    'data_sources/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

bedmap_tif = xr.open_rasterio(
    'data_sources/products/bedmap2_tiff/bedmap2_surface.tif')
bedmap_height = (bedmap_tif.values.squeeze().copy()).astype(np.float32)
bedmap_height[bedmap_height == 32767] = np.nan
bedmap_transform = ccrs.epsg(3031)

echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)
# echam6_t63_surface_height.values[echam6_t63_geosp.SLM.values == 0] = np.nan

era5_mon_sl_20_gph = xr.open_dataset(
    'data_sources/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_20_gph.nc')
era5_topograph = mpcalc.geopotential_to_height(
    era5_mon_sl_20_gph.z[0, :, :] * units('meter ** 2 / second ** 2'))


'''
stats.describe(bedmap_height, axis=None, nan_policy='omit')

(echam6_t63_surface_height.values <= 1).sum()
(echam6_t63_surface_height.values <= 0).sum()
geopotential_to_height(echam6_t63_geosp.GEOSP)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS + height + ice cores + divisions

output_png = 'figures/1_study_area/1.0_AIS_height_icecores_division.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=4500, cm_interval1=250, cm_interval2=500, cmap='viridis',
    reversed=False)

nrow = 1
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 8.8*nrow + 1]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.35, 'wspace': 0.35},)

axs[0] = hemisphere_plot(
    northextent=-60, ax_org = axs[0], plot_scalebar=True,
    add_grid_labels=True, l45label = False, loceanarcs = True,
    llatlabel = True)
axs[1] = hemisphere_plot(
    northextent=-60, ax_org = axs[1], plot_scalebar=True,
    add_grid_labels=True, l45label = False, loceanarcs = True)

ipanel=0
for jcol in range(ncol):
    plt.text(
        0, 1.12, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot AIS divisions
    plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='red', facecolor='none', linewidths=1, zorder=10)
    plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='blue', facecolor='none', linewidths=1, zorder=10)
    plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='m', facecolor='none', linewidths=1, zorder=10)
    
    ipanel += 1

axs[0].contourf(
    bedmap_tif.x.values,
    bedmap_tif.y.values,
    bedmap_height, levels=pltlevel, extend='max',
    norm=pltnorm, cmap=pltcmp,transform=bedmap_transform,)
# plt1 = plot_t63_contourf(
#     lon, lat, plt_data, ax,
#     pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[0].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

# plt_mesh = axs[1].pcolormesh(
#     echam6_t63_surface_height.lon,
#     echam6_t63_surface_height.lat,
#     echam6_t63_surface_height.values,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    echam6_t63_surface_height.lon,
    echam6_t63_surface_height.lat,
    echam6_t63_surface_height.values, axs[1],
    pltlevel, 'max', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[1].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)


plt.text(
    0.5, 1.12, 'Bedmap2', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.12,'ECHAM6 T63', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    anchor=(-0.3, 0.28),
    )
cbar.ax.set_xlabel('Antarctic surface height [$m$]', linespacing=2)

# axs[0]
patch1 = mpatches.Patch(
    edgecolor='blue', facecolor='none', label='EAIS', lw=1)
patch2 = mpatches.Patch(
    edgecolor='red', facecolor='none', label='WAIS', lw=1)
patch3 = mpatches.Patch(
    edgecolor='m', facecolor='none', label='AP', lw=1)
line1 = Line2D([0], [0], label='Atlantic sector', lw=2, linestyle='-',
               color='gray')
line2 = Line2D([0], [0], label='Indian sector', lw=2, linestyle=':',
               color='gray')
line3 = Line2D([0], [0], label='Pacific sector', lw=2, linestyle='--',
               color='gray')
plt.legend(handles=[patch1, patch2, patch3, line1, line2, line3],
           ncol=2, frameon=False, columnspacing=1,
           bbox_to_anchor=(1.1, -0.05),
           )

fig.subplots_adjust(left=0.08, right = 0.94, bottom = 0.23, top = 0.92)
fig.savefig(output_png)


'''
    # cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat,
    #                 axs[jcol], s=5)

# for irow in range(major_ice_core_site.shape[0]):
#     # irow = 0
#     axs[1].text(
#         major_ice_core_site.lon[irow], major_ice_core_site.lat[irow]+1,
#         major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
#         fontsize = 6, color='black')


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot location of core sites and stations

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,)

ten_sites_names = [
    'EDC', 'DOME F', 'Vostok', 'EDML', 'WDC',
    'Rothera', 'Halley', 'Neumayer', 'Law Dome', "Dumont d'Urville"]

# ten_sites_loc = stations_sites.loc[
#     [(isite in ten_sites_names) for isite in stations_sites.Site.values],
#     ].reset_index(drop=True)
# ten_sites_loc.to_pickle('data_sources/others/ten_sites_loc.pkl')


ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

output_png = 'figures/test/trial.png'

fig, ax = hemisphere_plot(northextent=-60)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax, s=6,
                zorder=4, lw=0.75)

for irow in range(Antarctic_stations.shape[0]):
    ax.text(
        Antarctic_stations.lon[irow]+2, Antarctic_stations.lat[irow]+1,
        Antarctic_stations.Site[irow], transform=ccrs.PlateCarree(),
        fontsize = 3, color='black')

fig.savefig(output_png)



'''
# output_png = 'figures/1_study_area/1.0_Antarctic core sites and stations.png'
# cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax, s=5,
#                 zorder=4,)
# cplot_ice_cores(Antarctic_stations.lon, Antarctic_stations.lat, ax, s=5,
#                 marker='s', zorder=4)

# for irow in range(major_ice_core_site.shape[0]):
#     ax.text(
#         major_ice_core_site.lon[irow]+2, major_ice_core_site.lat[irow]+1,
#         major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
#         fontsize = 3, color='black')

# # plot AIS divisions
# plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
#     ax=ax, transform=ccrs.epsg(3031),
#     edgecolor='red', facecolor='none', linewidths=0.1, zorder=2)
# plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
#     ax=ax, transform=ccrs.epsg(3031),
#     edgecolor='blue', facecolor='none', linewidths=0.1, zorder=2)
# plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
#     ax=ax, transform=ccrs.epsg(3031),
#     edgecolor='m', facecolor='none', linewidths=0.1, zorder=2)





major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')

stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot precipitation over AIS from ERA5

# import data

with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_tp_1979_2021_alltime = pickle.load(f)

#-------- PAGES text

output_png = "figures/1_study_area/Antarctic precipitation in ERA5 79_21.png"

mpl.rc('font', family='Times New Roman', size=12)
fig, ax = hemisphere_plot(
    northextent=-60,
    figsize=np.array([9, 11]) / 2.54,
    )

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax, s=15, lw=1)

pltlevel = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltticks = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('cividis', len(pltlevel)-1).reversed()

plt_ctr = ax.contourf(
    era5_mon_tp_1979_2021_alltime['am'].longitude,
    era5_mon_tp_1979_2021_alltime['am'].latitude.sel(latitude=slice(-59, -90)),
    era5_mon_tp_1979_2021_alltime['am'].sel(latitude=slice(-59, -90)) * 365,
    levels=pltlevel,extend='max',
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_ctr, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='max',
    pad=0.04, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Antarctic precipitation in ERA5 [$mm/yr$]',
                   linespacing=1.5, fontsize=14, labelpad=8)

fig.savefig(output_png)


#-------- PAGES front cover

pltlevel = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltticks = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)

output_png = "figures/1_study_area/Antarctic precipitation in ERA5 79_21_frontcover_viridis.pdf"
pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

output_png = "figures/1_study_area/Antarctic precipitation in ERA5 79_21_frontcover_orange.pdf"
pltcmp = cm.get_cmap('Oranges', len(pltlevel)-1)

output_png = "figures/1_study_area/Antarctic precipitation in ERA5 79_21_frontcover_viridis.eps"
pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

mpl.rc('font', family='Arial', size=8)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([6.5, 7.5]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_ctr = ax.contourf(
    era5_mon_tp_1979_2021_alltime['am'].longitude,
    era5_mon_tp_1979_2021_alltime['am'].latitude.sel(latitude=slice(-59, -90)),
    era5_mon_tp_1979_2021_alltime['am'].sel(latitude=slice(-59, -90)) * 365,
    levels=pltlevel,extend='max',
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_ctr, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.1, format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Precipitation in ERA5 ($mm/yr$)', linespacing=1.5, size=10)

# ax.get_grid
fig.savefig(output_png, dpi=600)



'''
# # plot topography
# plt_ctr1 = ax.contour(
#     era5_topograph.longitude,
#     era5_topograph.latitude.sel(latitude=slice(-59, -90)),
#     era5_topograph.sel(latitude=slice(-59, -90)) / 1000,
#     levels=[1, 2, 3, 4], transform=ccrs.PlateCarree(),
#     colors='blue', linewidths=0.3,)
# ax.clabel(
#     plt_ctr1, inline=1, colors='blue', fmt=remove_trailing_zero,
#     levels=[2, 3, 4], inline_spacing=1,)

#-------- Antarctic mask

with open('scratch/products/era5/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)

np.round(tp_era5_mean_over_ais['am'].values[0] * 365, 0)


pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])




#-------- Plot for PAGES

with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_tp_1979_2021_alltime = pickle.load(f)

# plot
output_png = "figures/1_study_area/Antarctic precipitation in ERA5 79_21.png"

mpl.rc('font', family='Arial', size=8)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([6.5, 7.5]) / 2.54,
    add_grid_labels=True, plot_scalebar=True,
    fm_left=0.16, fm_right=0.86, fm_bottom=0.1, fm_top=0.95,
    sb_location=(-0.14, -0.12), sb_barheight=150, llatlabel = False)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

pltlevel = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltticks = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

plt_ctr = ax.contourf(
    era5_mon_tp_1979_2021_alltime['am'].longitude,
    era5_mon_tp_1979_2021_alltime['am'].latitude.sel(latitude=slice(-59, -90)),
    era5_mon_tp_1979_2021_alltime['am'].sel(latitude=slice(-59, -90)) * 365,
    levels=pltlevel,extend='max',
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# # plot topography
# plt_ctr1 = ax.contour(
#     era5_topograph.longitude,
#     era5_topograph.latitude.sel(latitude=slice(-59, -90)),
#     era5_topograph.sel(latitude=slice(-59, -90)) / 1000,
#     levels=[1, 2, 3, 4], transform=ccrs.PlateCarree(),
#     colors='blue', linewidths=0.3,)
# ax.clabel(
#     plt_ctr1, inline=1, colors='blue', fmt=remove_trailing_zero,
#     levels=[2, 3, 4], inline_spacing=1,)

ax.add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_ctr, ax=ax, aspect=30,
    orientation="horizontal", shrink=1.3, ticks=pltticks, extend='max',
    pad=0.14, fraction=0.1, format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Precipitation in ERA5 ($mm/yr$)', linespacing=1.5, size=10)

# ax.get_grid
fig.savefig(output_png)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS height from Bedmap2


# plot
output_png = "figures/1_study_area/Antarctic surface height in Bedmap2.png"

mpl.rc('font', family='Times New Roman', size=12)
fig, ax = hemisphere_plot(
    northextent=-60, plot_scalebar=True,
    fm_left=0.15, fm_right=0.86, fm_bottom=0.1, fm_top=0.95,
    figsize=np.array([10, 11]) / 2.54, add_grid_labels=True,
    sb_location=(-0.14, -0.12), sb_barheight=200, )

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax, s=15, lw=1)

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=4500, cm_interval1=500, cm_interval2=1000, cmap='viridis',
    reversed=False)

# plot AIS divisions
plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor='none', linewidths=1, zorder=10)
plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='blue', facecolor='none', linewidths=1, zorder=10)
plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='m', facecolor='none', linewidths=1, zorder=10)

plt_ctr = ax.contourf(
    bedmap_tif.x.values,
    bedmap_tif.y.values,
    bedmap_height,
    levels=pltlevel,extend='max',
    norm=pltnorm, cmap=pltcmp,transform=bedmap_transform,)

ax.add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_ctr, ax=ax, aspect=30,
    orientation="horizontal", shrink=1.3, ticks=pltticks, extend='max',
    pad=0.14, fraction=0.1, format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Antarctic surface height in Bedmap2 [$m$]',
                   linespacing=1.5, fontsize=14, labelpad=8)

fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------

