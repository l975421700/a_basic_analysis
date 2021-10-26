

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


