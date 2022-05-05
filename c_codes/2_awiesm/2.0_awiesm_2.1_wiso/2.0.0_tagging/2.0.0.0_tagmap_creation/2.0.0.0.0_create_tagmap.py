

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
np.min(tagmap_nhsh_sl.tagmap[3, :, :] + tagmap_nhsh_sl.tagmap[4, :, :] + tagmap_nhsh_sl.tagmap[5, :, :] + tagmap_nhsh_sl.tagmap[6, :, :])


'''


# (np.max(np.abs(pi_final_qg_tag4_1y_echam_am.lat.values - tagmap.lat.values)))
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nh_s

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_nh_s = xr.Dataset(
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

# nh sea
tagmap_nh_s.tagmap.sel(level=4, lat=slice(90, 0))[:, :] = \
    1 - slm.sel(lat=slice(90, 0)).values

tagmap_nh_s.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nh_s.nc', mode='w')


'''
cdo --reduce_dim -selvar,slm /work/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/pi_final_qg_tag4_1y_0/analysis/echam/pi_final_qg_tag4_1y_0_2000_2003.01_echam.am.nc /work/ollie/qigao001/output/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc

'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nh_l

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_nh_l = xr.Dataset(
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

# nh land
tagmap_nh_l.tagmap.sel(level=4, lat=slice(90, 0))[:, :] = \
    slm.sel(lat=slice(90, 0)).values

tagmap_nh_l.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nh_l.nc', mode='w')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_sl

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 5

tagmap_g_nhsh_sl = xr.Dataset(
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

# globe
tagmap_g_nhsh_sl.tagmap.sel(level=4)[:, :] = 1

# sh land
tagmap_g_nhsh_sl.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = \
    slm.sel(lat=slice(0, -90)).values

# sh sea
tagmap_g_nhsh_sl.tagmap.sel(level=6, lat=slice(0, -90))[:, :] = \
    1 - slm.sel(lat=slice(0, -90)).values

# nh land
tagmap_g_nhsh_sl.tagmap.sel(level=7, lat=slice(90, 0))[:, :] = \
    slm.sel(lat=slice(90, 0)).values

# nh sea
tagmap_g_nhsh_sl.tagmap.sel(level=8, lat=slice(90, 0))[:, :] = \
    1 - slm.sel(lat=slice(90, 0)).values



tagmap_g_nhsh_sl.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_sl.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_g = xr.Dataset(
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

tagmap_g.tagmap.sel(level=4)[:, :] = 1

tagmap_g.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g2

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 2

tagmap_g2 = xr.Dataset(
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

tagmap_g2.tagmap.sel(level=4)[:, :] = 1
tagmap_g2.tagmap.sel(level=5)[:, :] = 0.5

tagmap_g2.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g2.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g2i

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 2

tagmap_g2i = xr.Dataset(
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

tagmap_g2i.tagmap.sel(level=4)[:, :] = 1
tagmap_g2i.tagmap.sel(level=5)[:, :] = 1

tagmap_g2i.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g2i.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_ew

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 5

tagmap_g_nhsh_ew = xr.Dataset(
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

# globe
tagmap_g_nhsh_ew.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_g_nhsh_ew.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_g_nhsh_ew.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

# east
tagmap_g_nhsh_ew.tagmap.sel(level=7, lon=slice(0, 179.99))[:, :] = 1

# west
tagmap_g_nhsh_ew.tagmap.sel(level=8, lon=slice(180, 360))[:, :] = 1

tagmap_g_nhsh_ew.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_ew.nc')

'''
# check
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=4), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=5) + tagmap_g_nhsh_ew.tagmap.sel(level=6), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=7) + tagmap_g_nhsh_ew.tagmap.sel(level=8), axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_p1

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 8

tagmap_g_nhsh_p1 = xr.Dataset(
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

# globe
tagmap_g_nhsh_p1.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_g_nhsh_p1.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_g_nhsh_p1.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

# sh + 1
tagmap_g_nhsh_p1.tagmap.sel(level=7)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=5) + 1

# nh + 1
tagmap_g_nhsh_p1.tagmap.sel(level=8)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=6) + 1

# globe + 1
tagmap_g_nhsh_p1.tagmap.sel(level=9)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4) + 1

# globe -0.5
tagmap_g_nhsh_p1.tagmap.sel(level=10)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4) -0.5

# globe
tagmap_g_nhsh_p1.tagmap.sel(level=11)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4)



tagmap_g_nhsh_p1.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_p1.nc')




'''
# check
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=4), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=5) + tagmap_g_nhsh_ew.tagmap.sel(level=6), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=7) + tagmap_g_nhsh_ew.tagmap.sel(level=8), axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_1p

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 7

tagmap_g_nhsh_1p = xr.Dataset(
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

# globe
tagmap_g_nhsh_1p.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_g_nhsh_1p.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_g_nhsh_1p.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

# globe but 1 point
tagmap_g_nhsh_1p.tagmap.sel(level=7)[:, :] = 1
tagmap_g_nhsh_1p.tagmap.sel(level=7)[48, 92] = 0

# 1 point
tagmap_g_nhsh_1p.tagmap.sel(level=8)[48, 92] = 1

# zero fields
tagmap_g_nhsh_1p.tagmap.sel(level=9)[:, :] = 0

# globe
tagmap_g_nhsh_1p.tagmap.sel(level=10)[:, :] = 1

tagmap_g_nhsh_1p.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_1p.nc')

'''
# check
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=4), axis=None)
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=5) + tagmap_g_nhsh_1p.tagmap.sel(level=6), axis=None)
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=7) + tagmap_g_nhsh_1p.tagmap.sel(level=8), axis=None)
'''
# endregion
# =============================================================================

