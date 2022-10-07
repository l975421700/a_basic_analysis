

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

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
echam6_t63_surface_height.values[echam6_t63_geosp.SLM.values == 0] = np.nan



'''
stats.describe(bedmap_height, axis=None, nan_policy='omit')

(echam6_t63_surface_height.values <= 1).sum()
(echam6_t63_surface_height.values <= 0).sum()
geopotential_to_height(echam6_t63_geosp.GEOSP)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot global map ----

fig, ax = framework_plot1("global")
ax.background_img(name='natural_earth', resolution='high')
fig.savefig('figures/01_study_area/natural_earth.png', dpi=1200)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot Antarctica surface height in Bedmap2

bedmap_surface = rh.fetch_bedmap2(datasets=['surface'])
# stats.describe(
#     bedmap_surface.surface.values, axis = None, nan_policy = 'omit')

# from affine import Affine
# bedmap_transform = Affine(*bedmap_tif.attrs["transform"])
bedmap_transform = ccrs.epsg(3031)

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs = transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2], rotate_labels = False,
    )
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

demlevel = np.arange(0, 4000.1, 20)
demticks = np.arange(0, 4000.1, 1000)

plt_dem = ax.pcolormesh(
    bedmap_tif.x.values, bedmap_tif.y.values,
    bedmap_surface.surface.values,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(demlevel)), rasterized=True,
    transform=bedmap_transform,)
cbar = fig.colorbar(
    plt_dem, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=demticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface height [m] in Bedmap2")

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/01_study_area/Surface height in Bedmap2.png')


'''
# https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/always_circular_stereo.html#sphx-glr-gallery-lines-and-polygons-always-circular-stereo-py

# https://www.fatiando.org/rockhound/latest/gallery/bedmap2.html#sphx-glr-gallery-bedmap2-py


ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.background_img(name='natural_earth', resolution='high',
                  extent=[-180, 180, -90, -60])

# https://stackoverflow.com/questions/45302485/matplotlib-focus-on-specific-lon-lat-using-spstere-projection
# http://neichin.github.io/personalweb/writing/Cartopy-shapefile/
# https://www.fatiando.org/rockhound/latest/gallery/bedmap2.html


from cartopy.mpl.ticker import LongitudeFormatter
projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs = transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2],
    rotate_labels = False,
    # xpadding=0, ypadding=0,
    xformatter=LongitudeFormatter(degree_symbol='° '),
    )
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/0_test/trial.png')


'''



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot Antarctica bed height in Bedmap2

bedmap_bed = rh.fetch_bedmap2(datasets=['bed'])
# stats.describe(
#     bedmap_surface.surface.values, axis = None, nan_policy = 'omit')

bedmap_tif = xr.open_rasterio('data_source/bedmap2_tiff/bedmap2_bed.tif')

# from affine import Affine
# bedmap_transform = Affine(*bedmap_tif.attrs["transform"])
bedmap_transform = ccrs.epsg(3031)

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs=transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2], rotate_labels=False,
)
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

demlevel = np.arange(-2800, 2800.1, 20)
demticks = np.arange(-2800, 2800.1, 700)

cmp_top = cm.get_cmap('Blues_r', int(np.floor(len(demlevel) / 2)))
cmp_bottom = cm.get_cmap('Reds', int(np.floor(len(demlevel) / 2)))
cmp_colors = np.vstack(
    (cmp_top(np.linspace(0, 1, int(np.floor(len(demlevel) / 2)))),
     [1, 1, 1, 1],
     cmp_bottom(np.linspace(0, 1, int(np.floor(len(demlevel) / 2))))))
cmp_cmap = ListedColormap(cmp_colors, name='RedsBlues_r')


plt_dem = ax.pcolormesh(
    bedmap_tif.x.values, bedmap_tif.y.values, bedmap_bed.bed.values,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cmp_cmap, rasterized=True, transform=bedmap_transform,)
cbar = fig.colorbar(
    plt_dem, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.08,
    shrink=1.2, aspect=25, ticks=demticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Bed height above sea level [m] in Bedmap2")

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/01_study_area/01.00.02 Bed height in Bedmap2.png')


'''
plt_theta = ax.pcolormesh(
    lon, lat, theta100, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS boundaries, ice core sites

# Load data
with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)

one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')
ais_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

fig, ax = hemisphere_plot(
    northextent=-60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98, add_atlas=False,)

# plt_polygon = ais_imbie2.plot(
#     ax=ax, transform=ccrs.epsg(3031), cmap="viridis")

plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor='none', linewidths=0.5, zorder=2)
plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='blue', facecolor='none', linewidths=0.5, zorder=2)
plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='m', facecolor='none', linewidths=0.5, zorder=2)

ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['eais_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightblue')))),
    transform=ccrs.PlateCarree())
ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['wais_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightpink')))),
    transform=ccrs.PlateCarree())
ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['ap_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightgray')))),
    transform=ccrs.PlateCarree())


coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/1_study_area/01.00.03 Antarctic Ice Sheets.png')


#### methods to extract EAIS mask
'''

#2 points in polygon path obtained in geometry.exterior.coords
eais_mask, eais_mask01 = points_in_polygon(
    lon, lat, Path([(9.74499797821047, -90)] + list(
        ais_imbie2.to_crs(4326).geometry[2].exterior.coords) + \
        [(9.74499797821047, -90)]))

#3 extract path from plotted contours
east_ctr = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor=None,
)
east_ctr.collections[0].get_paths()
'''


'''
ax.pcolormesh(
    lon, lat, one_degree_grids_cdo_area.cell_area,
    cmap='viridis', rasterized=True, transform=ccrs.PlateCarree(),
)

ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['eais_mask01'], colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid',
    interpolation='none')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['wais_mask01'], colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['ap_mask01'], colors='m', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')


# one_degree_grids_cdo_area.cell_area.values[ais_masks['eais_mask']].sum() / 10**6
# ais_imbie2.geometry[2].area / 10**6
# one_degree_grids_cdo_area.cell_area.values[ais_masks['wais_mask']].sum() / 10**6
# ais_imbie2.geometry[1].area / 10**6
# one_degree_grids_cdo_area.cell_area.values[ais_masks['ap_mask']].sum() / 10**6
# ais_imbie2.geometry[3].area / 10**6
# 9761642.045593847, 9620521.647740476, 2110804.571811703, 2038875.6430063567, 232127.50065176177, 232678.32105463636


# plot ANT_Rignot_Basins_IMBIE2
ant_basins_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp'
)
plt_polygon = ant_basins_imbie2.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis")

# plot MEaSUREs Antarctic Boundaries
basins_IMBIE_ais = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409164/Basins_IMBIE_Antarctica_v02.shp'
)
basins_IMBIE_ais_dis = basins_IMBIE_ais.dissolve('Regions')

plt_polygon1 = basins_IMBIE_ais.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis",
    # column="Regions",
    )

plt_polygon2 = basins_IMBIE_ais_dis.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis",
    )


coastline_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409161/Coastline_Antarctica_v02.shp'
)
plt_polygon = coastline_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

iceshelf_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409162/IceShelf_Antarctica_v02.shp'
)
plt_polygon = iceshelf_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

groundingline_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409163/GroundingLine_Antarctica_v02.shp'
)
plt_polygon = groundingline_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

basins_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409165/Basins_Antarctica_v02.shp'
)
plt_polygon = basins_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

iceboundaries_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409166/IceBoundaries_Antarctica_v02.shp'
)
plt_polygon = iceboundaries_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))


# Plot Bedmap2
bedmap_tif = xr.open_rasterio(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface.tif')
surface_height_bedmap = bedmap_tif.values.copy().astype(np.float64)
# surface_height_bedmap[surface_height_bedmap == 32767] = np.nan
surface_height_bedmap[surface_height_bedmap == 32767] = 0
bedmap_transform = ccrs.epsg(3031)
pltlevel_sh = np.arange(0, 4000.1, 1)
pltticks_sh = np.arange(0, 4000.1, 1000)
plt_cmp = ax.pcolormesh(
    bedmap_tif.x, bedmap_tif.y,
    surface_height_bedmap[0, :, :],
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=bedmap_transform,)

plt_ctr = ax.contour(
    bedmap_tif.x, bedmap_tif.y, surface_height_bedmap[0, :, :], levels=0,
    colors='red', rasterized=True, transform=bedmap_transform,
    linewidths=0.25)

# Save Bedmap2 contours
from matplotlib.path import Path
import pickle
plt_ctr.collections[0].get_paths()
with open(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface_contour.pkl',
    'wb') as f:
    pickle.dump(plt_ctr.collections[0].get_paths(), f)

# Plot BAS ADD file
import geopandas as gpd
hr_coastline_polygon_add = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
plt_polygon = hr_coastline_polygon_add.plot(ax=ax, transform=ccrs.epsg(3031))

with open(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface_contour.pkl',
    'rb') as f:
    bedmap2_surface_contour = pickle.load(f)
polygon = [
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
]
poly_path = Path(polygon)


import shapefile
reader = shapefile.Reader(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
for shape in list(reader.iterShapes()):
    npoints = len(shape.points)  # total points
    nparts = len(shape.parts)  # total parts

    if nparts == 1:
        x_lon = np.zeros((len(shape.points), 1))
        y_lat = np.zeros((len(shape.points), 1))
        for ip in range(len(shape.points)):
            x_lon[ip] = shape.points[ip][0]
            y_lat[ip] = shape.points[ip][1]
        plt.plot(x_lon, y_lat, 'red', linewidth=0.25)

    else:   # loop over parts of each shape, plot separately
        for ip in range(nparts):
            i0 = shape.parts[ip]
            if ip < nparts-1:
                i1 = shape.parts[ip+1]-1
            else:
                i1 = npoints
            seg = shape.points[i0:i1+1]
            x_lon = np.zeros((len(seg), 1))
            y_lat = np.zeros((len(seg), 1))
            for ip in range(len(seg)):
                x_lon[ip] = seg[ip][0]
                y_lat[ip] = seg[ip][1]
            plt.plot(x_lon, y_lat, 'red', linewidth=0.25)

# identify problem in SCAR ADD

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)
ax.set_extent((-180, 180, -90, -60), crs=transform)

hr_coastline_polygon_add = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
plt_polygon = hr_coastline_polygon_add.plot(ax=ax,)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/0_test/trial.png')


# Zwally_Antarctic_Ice_Sheets
zwally_ais = pd.read_csv(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Zwally_Basins/Zwally_Antarctic_Ice_Sheets.txt',
    sep='\s+', header=None, names=['lat', 'lon', 'ice_sheet_id'], skiprows=9,
    # dtype={'ice_sheet_id': str},
    )

# zwally_ais_gdf = gpd.GeoDataFrame(
#     zwally_ais, geometry=gpd.points_from_xy(zwally_ais.lon, zwally_ais.lat))

geom_ap = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 28],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 28]))
polygon_ap = gpd.GeoDataFrame(geometry=[geom_ap])

geom_wais = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 29],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 29]))
polygon_wais = gpd.GeoDataFrame(geometry=[geom_wais])

geom_eais = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 30],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 30]))
polygon_eais = gpd.GeoDataFrame(geometry=[geom_eais])

polygon_ap.plot(ax=ax, color='red', zorder=3, transform=ccrs.PlateCarree())
polygon_wais.plot(ax=ax, color='blue', zorder=3, transform=ccrs.PlateCarree())
polygon_eais.plot(ax=ax, color='grey', zorder=3, transform=ccrs.PlateCarree())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS boundaries, ice core sites, surface height


fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 5.8]) / 2.54,
    fm_bottom=0.01, lw=0.1)

plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor='none', linewidths=0.15, zorder=2)
plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='blue', facecolor='none', linewidths=0.15, zorder=2)
plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='m', facecolor='none', linewidths=0.15, zorder=2)

ax.scatter(
    x = major_ice_core_site.lon, y = major_ice_core_site.lat,
    s=3, c='none', linewidths=0.5, marker='o',
    transform=ctp.crs.PlateCarree(), edgecolors = 'black',
    )

for irow in range(major_ice_core_site.shape[0]):
    # irow = 0
    ax.text(major_ice_core_site.lon[irow], major_ice_core_site.lat[irow]+1,
            major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
            fontsize = 6, color='black')

fig.savefig('figures/1_study_area/trial1.png')





'''
# pltlevel = np.arange(0, 32.01, 0.1)
# pltticks = np.arange(0, 32.01, 4)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
# pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

# plt_cmp = ax.pcolormesh(
#     x,
#     y,
#     z,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# cbar = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
#     orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
#     pad=0.02, fraction=0.2,
#     )
# cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS + height + ice cores + divisions


output_png = 'figures/1_study_area/1.0_AIS_height_icecores_division.png'


pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=4500, cm_interval1=250, cm_interval2=500, cmap='PuOr',
    reversed=False)

nrow = 1
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 8.8*nrow + 1]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.35, 'wspace': 0.35},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=-60, ax_org = axs[jcol], plot_scalebar=True,
        add_grid_labels=True, l45label = False)
    plt.text(
        0, 1.12, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat,
                    axs[jcol], s=5)
    
    # plot AIS divisions
    plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='red', facecolor='none', linewidths=0.5, zorder=2)
    plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='blue', facecolor='none', linewidths=0.5, zorder=2)
    plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='m', facecolor='none', linewidths=0.5, zorder=2)
    
    ipanel += 1

for irow in range(major_ice_core_site.shape[0]):
    # irow = 0
    axs[0].text(
        major_ice_core_site.lon[irow], major_ice_core_site.lat[irow]+1,
        major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
        fontsize = 6, color='black')

axs[0].pcolormesh(
    bedmap_tif.x.values,
    bedmap_tif.y.values,
    bedmap_height,
    norm=pltnorm, cmap=pltcmp,transform=bedmap_transform,)

plt_mesh = axs[1].pcolormesh(
    echam6_t63_surface_height.lon,
    echam6_t63_surface_height.lat,
    echam6_t63_surface_height.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.12, 'Bedmap2', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.12,'ECHAM6 T63', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='max',
    anchor=(0.5, 0.28),
    )
cbar.ax.set_xlabel('Antarctic surface height [$m$]', linespacing=2)

fig.subplots_adjust(left=0.08, right = 0.94, bottom = 0.23, top = 0.92)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


