

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
)

# endregion
# =============================================================================


# =============================================================================
# region plot global and study area ----

fig, ax = framework_plot1("global")
ax.background_img(name='natural_earth', resolution='high')
fig.savefig('figures/01_study_area/natural_earth.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Antarctica surface height

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

borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(borders, zorder=2)

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

ax.background_img(name='natural_earth', resolution='high',
                  extent=[-180, 180, -90, -60])

# https://stackoverflow.com/questions/45302485/matplotlib-focus-on-specific-lon-lat-using-spstere-projection
# http://neichin.github.io/personalweb/writing/Cartopy-shapefile/
# https://www.fatiando.org/rockhound/latest/gallery/bedmap2.html

'''



# endregion
# =============================================================================


# =============================================================================
# region plot Antarctica surface height

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




