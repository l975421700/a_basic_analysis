

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
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
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion
# =============================================================================


# =============================================================================
# region get basic information

pi_final_qg_tag4_1y_echam_am = xr.open_dataset(
    '/work/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/pi_final_qg_tag4_1y/analysis/echam/pi_final_qg_tag4_1y_2000_2003.01_echam.am.nc')
lon = pi_final_qg_tag4_1y_echam_am.lon.values
lat = pi_final_qg_tag4_1y_echam_am.lat.values
slm = pi_final_qg_tag4_1y_echam_am.slm.squeeze()


ntag = 4

tagmap_nhsh_sl = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.int8)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# sh land
tagmap_nhsh_sl.tagmap.sel(level=4, lat=slice(0, -90))[:, :] = \
    slm.sel(lat=slice(0, -90)).values

# sh sea
tagmap_nhsh_sl.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = \
    1 - slm.sel(lat=slice(0, -90)).values

# nh land
tagmap_nhsh_sl.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = \
    slm.sel(lat=slice(90, 0)).values

# nh sea
tagmap_nhsh_sl.tagmap.sel(level=7, lat=slice(90, 0))[:, :] = \
    1 - slm.sel(lat=slice(90, 0)).values

tagmap_nhsh_sl.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_sl.nc', mode='w')


'''
tagmap = xr.open_dataset('/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap.nc')

# check
tagmap_nhsh_sl = xr.open_dataset(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_sl.nc',)
np.max(tagmap_nhsh_sl.tagmap[3, :, :] + tagmap_nhsh_sl.tagmap[4, :, :] + tagmap_nhsh_sl.tagmap[5, :, :] + tagmap_nhsh_sl.tagmap[6, :, :])



'''


# (np.max(np.abs(pi_final_qg_tag4_1y_echam_am.lat.values - tagmap.lat.values)))
# endregion
# =============================================================================



