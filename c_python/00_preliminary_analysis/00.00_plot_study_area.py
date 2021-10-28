

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
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'data_source/bg_cartopy'
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

# add ellipse
from scipy import linalg
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
fig.savefig('figures/00_test/natural_earth.png', dpi=1200)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Antarctica

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8]) / 2.54, subplot_kw={'projection': ccrs.SouthPolarStereo()}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs=ccrs.PlateCarree())

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(borders, zorder=2)

ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.25, zorder=2,
             color='gray', alpha=0.5, linestyle='--')

# Compute a circle in axes coordinates, which we can use as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

figure_margin = {
    'left': 0.12, 'right': 0.96, 'bottom': 0.05, 'top': 0.995}
fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])
fig.savefig('figures/00_test/00_trial.png')


'''
# https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/always_circular_stereo.html#sphx-glr-gallery-lines-and-polygons-always-circular-stereo-py

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

'''
# endregion
# =============================================================================

