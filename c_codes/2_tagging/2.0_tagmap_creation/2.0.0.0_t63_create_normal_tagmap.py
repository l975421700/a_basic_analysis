

# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region tagmap_nhsh

echam_t63_slm = xr.open_dataset(
    'scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
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
    'startdump/tagging/tagmap/tagmap_nhsh.nc')

'''
# check
tagmap_nhsh = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_nhsh.nc')
np.max(tagmap_nhsh.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_nhsh.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_nhsh_sl

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst


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
tagmap_nhsh_sl.tagmap.sel(level=4, lat=slice(0, -90)).values[
    np.isnan(analysed_sst.sel(lat=slice(0, -90)))] = 1

# sh sea
tagmap_nhsh_sl.tagmap.sel(level=5, lat=slice(0, -90)).values[
    np.isfinite(analysed_sst.sel(lat=slice(0, -90)))] = 1

# nh land
tagmap_nhsh_sl.tagmap.sel(level=6, lat=slice(90, 0)).values[
    np.isnan(analysed_sst.sel(lat=slice(90, 0)))] = 1

# nh sea
tagmap_nhsh_sl.tagmap.sel(level=7, lat=slice(90, 0)).values[
    np.isfinite(analysed_sst.sel(lat=slice(90, 0)))] = 1

tagmap_nhsh_sl.to_netcdf(
    'startdump/tagging/tagmap/tagmap_nhsh_sl.nc',)


'''
# check
tagmap_nhsh_sl = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_nhsh_sl.nc',)
np.max(tagmap_nhsh_sl.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_nhsh_sl.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
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
    'startdump/tagging/tagmap/tagmap_ls_0.nc',)


'''
# check
tagmap_ls_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_0.nc',)
np.max(tagmap_ls_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_0_5

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
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
    'startdump/tagging/tagmap/tagmap_ls_0_5.nc',)


'''
# check
tagmap_ls_0_5 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_0_5.nc',)
np.max(tagmap_ls_0_5.tagmap[3:6, :, :].sum(axis=0))
np.min(tagmap_ls_0_5.tagmap[3:6, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_15_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
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
    'startdump/tagging/tagmap/tagmap_ls_15_0.nc',)


'''
# check
tagmap_ls_15_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_15_0.nc',)
np.max(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_5_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 0 + 0 + 0 + 7

tagmap_ls_5_0 = xr.Dataset(
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

tagmap_ls_5_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_5_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_5_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_5_0.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_5_0.nc',)


'''
# check
tagmap_ls_15_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_15_0.nc',)
np.max(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_15_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_10

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 2 + 0 + 0 + 0 + 0

tagmap_10 = xr.Dataset(
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

tagmap_10.tagmap.sel(level=slice(1, 3))[:, :] = 1

# 1
tagmap_10.tagmap.sel(level=4).values[:, :] = 1
# 0
tagmap_10.tagmap.sel(level=5).values[:, :] = 0


tagmap_10.to_netcdf(
    'startdump/tagging/tagmap/tagmap_10.nc',)


'''
# check
tagmap_10 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_10.nc',)
np.max(tagmap_10.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_10.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_55

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 2 + 0 + 0 + 0 + 0

tagmap_55 = xr.Dataset(
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

tagmap_55.tagmap.sel(level=slice(1, 3))[:, :] = 1

# 1
tagmap_55.tagmap.sel(level=4).values[:, :] = 0.5
# 0
tagmap_55.tagmap.sel(level=5).values[:, :] = 0.5


tagmap_55.to_netcdf(
    'startdump/tagging/tagmap/tagmap_55.nc',)


'''
# check
tagmap_55 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_55.nc',)
np.max(tagmap_55.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_55.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_37

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 2 + 0 + 0 + 0 + 0

tagmap_37 = xr.Dataset(
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

tagmap_37.tagmap.sel(level=slice(1, 3))[:, :] = 1

# 1
tagmap_37.tagmap.sel(level=4).values[:, :] = 0.3
# 0
tagmap_37.tagmap.sel(level=5).values[:, :] = 0.7


tagmap_37.to_netcdf(
    'startdump/tagging/tagmap/tagmap_37.nc',)


'''
# check
tagmap_37 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_37.nc',)
np.max(tagmap_37.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_37.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_l_18latbin

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

ntag = 19
latbins = np.arange(-90, 90.1, 10)

tagmap_l_18latbin = xr.Dataset(
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

tagmap_l_18latbin.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_l_18latbin.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1


for i in range(ntag-1):
    # i=0
    tagmap_l_18latbin.tagmap.sel(
        level=5+i, lat=slice(latbins[i+1], latbins[i])).values[
            np.isfinite(analysed_sst.sel(lat=slice(latbins[i+1], latbins[i])).values)] = 1

tagmap_l_18latbin.to_netcdf(
    'startdump/tagging/tagmap/tagmap_l_18latbin.nc',)


'''
# check
tagmap_l_18latbin = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_l_18latbin.nc',)
np.max(tagmap_l_18latbin.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_l_18latbin.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_11

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2

tagmap_ls_11 = xr.Dataset(
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

tagmap_ls_11.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
for i in np.arange(4, 25, 2):
    tagmap_ls_11.tagmap.sel(level=i).values[np.isnan(analysed_sst)] = 1

# sea
for i in np.arange(5, 26, 2):
    tagmap_ls_11.tagmap.sel(level=i).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_11.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_11.nc',)


'''
# check
tagmap_ls_11 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_11.nc',)
np.max(tagmap_ls_11.tagmap[3:6, :, :].sum(axis=0))
np.min(tagmap_ls_11.tagmap[3:6, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_13_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 0 + 15 + 0 + 0

tagmap_ls_13_0 = xr.Dataset(
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

tagmap_ls_13_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_13_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_13_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_13_0.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_13_0.nc',)


'''
# check
tagmap_ls_13_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_13_0.nc',)
np.max(tagmap_ls_13_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_13_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_8_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 0 + 10 + 0 + 0

tagmap_ls_8_0 = xr.Dataset(
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

tagmap_ls_8_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_8_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_8_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_8_0.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_8_0.nc',)


'''
# check
tagmap_ls_8_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_8_0.nc',)
np.max(tagmap_ls_8_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_8_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_7_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 0 + 9 + 0 + 0

tagmap_ls_7_0 = xr.Dataset(
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

tagmap_ls_7_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_7_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_7_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_7_0.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_7_0.nc',)


'''
# check
tagmap_ls_7_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_7_0.nc',)
np.max(tagmap_ls_7_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_7_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_0_6

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 3 + 3 + 3 + 3 + 3 + 3

tagmap_ls_0_6 = xr.Dataset(
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

tagmap_ls_0_6.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
for i in np.arange(4, 21, 3):
    tagmap_ls_0_6.tagmap.sel(level=i).values[np.isnan(analysed_sst)] = 1

# sea
for i in np.arange(5, 21, 3):
    tagmap_ls_0_6.tagmap.sel(level=i).values[np.isfinite(analysed_sst)] = 1

# 0
for i in np.arange(6, 21.1, 3):
    tagmap_ls_0_6.tagmap.sel(level=i).values[:, :] = 0


tagmap_ls_0_6.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_0_6.nc',)


'''
# check
tagmap_ls_0_6 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_0_6.nc',)
np.max(tagmap_ls_0_6.tagmap[:, :, :].sum(axis=0))
np.min(tagmap_ls_0_6.tagmap[:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 2

tagmap_ls = xr.Dataset(
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

tagmap_ls.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# sea
tagmap_ls.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1

tagmap_ls.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls.nc',)


'''
# check
tagmap_ls = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls.nc',)
np.max(tagmap_ls.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_l_95latbin

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

ntag = 96

tagmap_l_95latbin = xr.Dataset(
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

tagmap_l_95latbin.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_l_95latbin.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1


for i in range(ntag-1):
    # i=0
    tagmap_l_95latbin.tagmap.sel(level=5+i, lat=lat[i]).values[
        np.isfinite(analysed_sst.sel(lat=lat[i]).values)] = 1

tagmap_l_95latbin.to_netcdf(
    'startdump/tagging/tagmap/tagmap_l_95latbin.nc',)


'''
# check
tagmap_l_95latbin = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_l_95latbin.nc',)
np.max(tagmap_l_95latbin.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_l_95latbin.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_lsh_48latbin

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

ntag = 49

tagmap_lsh_48latbin = xr.Dataset(
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

tagmap_lsh_48latbin.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_lsh_48latbin.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
tagmap_lsh_48latbin.tagmap.sel(level=4, lat=slice(0, -90)).values[:, :] = 1

for i in range(ntag-1):
    # i=0
    tagmap_lsh_48latbin.tagmap.sel(level=5+i, lat=lat[i]).values[
        np.isfinite(analysed_sst.sel(lat=lat[i]).values)] = 1

tagmap_lsh_48latbin.to_netcdf(
    'startdump/tagging/tagmap/tagmap_lsh_48latbin.nc',)


'''
# check
tagmap_lsh_48latbin = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_lsh_48latbin.nc',)
np.max(tagmap_lsh_48latbin.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_lsh_48latbin.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_lnh_48latbin

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

ntag = 49

tagmap_lnh_48latbin = xr.Dataset(
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

tagmap_lnh_48latbin.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_lnh_48latbin.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
tagmap_lnh_48latbin.tagmap.sel(level=4, lat=slice(90, 0)).values[:, :] = 1

for i in range(ntag-1):
    # i=0
    tagmap_lnh_48latbin.tagmap.sel(level=5+i, lat=lat[i+48]).values[
        np.isfinite(analysed_sst.sel(lat=lat[i+48]).values)] = 1

tagmap_lnh_48latbin.to_netcdf(
    'startdump/tagging/tagmap/tagmap_lnh_48latbin.nc',)

'''
# check
tagmap_lnh_48latbin = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_lnh_48latbin.nc',)
tagmap_lsh_48latbin = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_lsh_48latbin.nc',)
tagmap_ls_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_0.nc',)
stats.describe(tagmap_lnh_48latbin.tagmap[3:, ].sum(axis=0), axis=None)
stats.describe(tagmap_lsh_48latbin.tagmap[3:, ].sum(axis=0), axis=None)
stats.describe(tagmap_lnh_48latbin.tagmap[4:, ].sum(axis=0) + tagmap_lsh_48latbin.tagmap[4:, ].sum(axis=0) + tagmap_ls_0.tagmap[3, ], axis=None)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_ls_57_0

esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

ntag = 0 + 59 + 0 + 0 + 0

tagmap_ls_57_0 = xr.Dataset(
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

tagmap_ls_57_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
tagmap_ls_57_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
# sea
tagmap_ls_57_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = 1


tagmap_ls_57_0.to_netcdf(
    'startdump/tagging/tagmap/tagmap_ls_57_0.nc',)


'''
# check
tagmap_ls_57_0 = xr.open_dataset(
    'startdump/tagging/tagmap/tagmap_ls_57_0.nc',)
np.max(tagmap_ls_57_0.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_ls_57_0.tagmap[3:, :, :].sum(axis=0))
'''
# endregion
# -----------------------------------------------------------------------------
