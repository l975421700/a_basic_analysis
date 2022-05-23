

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
# region tagmap_nhsh

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values

ntag = 2

tagmap_nhsh = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_nhsh.tagmap.sel(level=slice(1, 3))[:, :] = 1

# sh
tagmap_nhsh.tagmap.sel(level=4, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_nhsh.tagmap.sel(level=5, lat=slice(90, 0))[:, :] = 1


tagmap_nhsh.to_netcdf(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_nhsh.nc')

'''
# check
tagmap_nhsh = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_nhsh.nc')
np.max(tagmap_nhsh.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_nhsh.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nhsh_sl

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 4

tagmap_nhsh_sl = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_nhsh_sl.tagmap.sel(level=slice(1, 3))[:, :] = 1

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
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_nhsh_sl.nc',)


'''
# check
tagmap_nhsh_sl = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_nhsh_sl.nc',)
np.max(tagmap_nhsh_sl.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_nhsh_sl.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region tagmap_g_1_0

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values

ntag = 2

tagmap_g_1_0 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_g_1_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# sh
tagmap_g_1_0.tagmap.sel(level=4)[:, :] = 1

# nh
tagmap_g_1_0.tagmap.sel(level=5)[:, :] = 0


tagmap_g_1_0.to_netcdf(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_g_1_0.nc')

'''
# check
tagmap_g_1_0 = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_g_1_0.nc')
np.max(tagmap_g_1_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_g_1_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_ls_0

esacci_echam6_t63_trim = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 3

tagmap_ls_0 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_ls_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# sea
tagmap_ls_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1

# 0
tagmap_ls_0.tagmap.sel(level=6)[:, :] = 0


tagmap_ls_0.to_netcdf(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc',)


'''
# check
tagmap_ls_0 = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc',)
np.max(tagmap_ls_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_ls_0_5

esacci_echam6_t63_trim = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 3 + 3 + 3 + 3 + 3

tagmap_ls_0_5 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_ls_0_5.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
for i in np.arange(4, 18, 3):
    tagmap_ls_0_5.tagmap.sel(level=i).values[np.isnan(analysed_sst)] = 1

# sea
for i in np.arange(5, 18, 3):
    tagmap_ls_0_5.tagmap.sel(level=i).values[np.isfinite(analysed_sst)] = 1

# 0
# sea
for i in np.arange(6, 18.1, 3):
    tagmap_ls_0_5.tagmap.sel(level=i).values[:, :] = 0


tagmap_ls_0_5.to_netcdf(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0_5.nc',)


'''
# check
tagmap_ls_0_5 = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0_5.nc',)
np.max(tagmap_ls_0_5.tagmap[3:6, :, :].sum(axis=0))
np.min(tagmap_ls_0_5.tagmap[3:6, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_ls_15_0

esacci_echam6_t63_trim = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 17 + 0 + 0 + 0

tagmap_ls_15_0 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_ls_15_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_15_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_15_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_15_0.to_netcdf(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_15_0.nc',)


'''
# check
tagmap_ls_15_0 = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_15_0.nc',)
np.max(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# =============================================================================
