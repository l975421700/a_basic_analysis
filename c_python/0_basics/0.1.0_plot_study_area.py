

# =============================================================================
# region import packages


# basic library
import numpy as np
import xarray as xr
import datetime
import glob
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# plot
import matplotlib.path as mpath
from matplotlib.path import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
mpl.rcParams['figure.dpi'] = 600
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'data_source/bg_cartopy'
from matplotlib.colors import ListedColormap

# from matplotlib import font_manager as fm
# fontprop_tnr = fm.FontProperties(fname='data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
# mpl.get_backend()
# mpl.rcParams['backend'] = 'Qt4Agg'  #
# plt.rcParams["font.serif"] = ["Times New Roman"]

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from geopy import distance
import rasterio as rio
import rockhound as rh
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pickle

# add ellipse
from scipy import linalg
from scipy import stats
from sklearn import mixture
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# self defined
from a00_basic_analysis.b_module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot1,
    hemisphere_plot,
)

from a00_basic_analysis.b_module.basic_calculations import (
    create_ais_mask,
)
# endregion
# =============================================================================


# =============================================================================
# region plot global map ----

fig, ax = framework_plot1("global")
ax.background_img(name='natural_earth', resolution='high')
fig.savefig('figures/01_study_area/natural_earth.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Antarctica surface height in Bedmap2

bedmap_surface = rh.fetch_bedmap2(datasets=['surface'])
# stats.describe(
#     bedmap_surface.surface.values, axis = None, nan_policy = 'omit')

bedmap_tif = xr.open_rasterio('data_source/bedmap2_tiff/bedmap2_surface.tif')

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
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
# region plot Antarctic boundaries

# (lon, lat, eais_mask, eais_mask01, \
#     wais_mask, wais_mask01, ap_mask, ap_mask01) = create_ais_mask()

# ais_masks = {'lon': lon, 'lat': lat, 'eais_mask': eais_mask,
#              'eais_mask01': eais_mask01, 'wais_mask': wais_mask,
#              'wais_mask01': wais_mask01, 'ap_mask': ap_mask,
#              'ap_mask01': ap_mask01,}

# with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'wb') as handle:
#     pickle.dump(ais_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)

one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')
ais_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

one_degree_grids_cdo_area.cell_area.values[ais_masks['eais_mask']].sum() / 10**6
ais_imbie2.geometry[2].area / 10**6
one_degree_grids_cdo_area.cell_area.values[ais_masks['wais_mask']].sum() / 10**6
ais_imbie2.geometry[1].area / 10**6
one_degree_grids_cdo_area.cell_area.values[ais_masks['ap_mask']].sum() / 10**6
ais_imbie2.geometry[3].area / 10**6
# 9761642.045593847, 9620521.647740476, 2110804.571811703, 2038875.6430063567, 232127.50065176177, 232678.32105463636

fig, ax = hemisphere_plot(
    northextent=-60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98, add_atlas=False,)

plt_polygon = ais_imbie2.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis")

# ax.pcolormesh(
#     lon, lat, one_degree_grids_cdo_area.cell_area,
#     cmap='viridis', rasterized=True, transform=ccrs.PlateCarree(),
# )

ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['eais_mask01'], colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['wais_mask01'], colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['ap_mask01'], colors='gray', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')

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
# =============================================================================



