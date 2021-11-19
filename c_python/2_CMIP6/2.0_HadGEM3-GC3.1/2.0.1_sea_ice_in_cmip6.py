

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
# fontprop_tnr = fm.FontProperties(fname='bas_palaeoclim_qino/others/TimesNewRoman.ttf')
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

# endregion
# =============================================================================


# =============================================================================
# region plot NH sea ice in 1850

dir_vittoria = '/gws/nopw/j04/pmip4_vol1/users/vittoria/'

fl_si = np.array(sorted(
    glob.glob(dir_vittoria + 'seaice_annual_uba937_' + '*.nc')
))

fs_si = xr.open_mfdataset(
    fl_si, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')

seaicearea = fs_si.aice[0, :, :].copy().values
seaicearea[np.where(seaicearea < 0)] = np.nan
seaicearea[np.where(seaicearea > 1)] = np.nan

fig, ax = hemisphere_plot(southextent=30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    fs_si.aice.lon, fs_si.aice.lat, seaicearea,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Sea ice area")

fig.savefig('figures/00_test/trial')


'''
# ax.add_feature(cfeature.LAND, zorder=3)
'''
# endregion
# =============================================================================





