

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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
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
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# region import data

# import 1-day model simulation
pi_echam_1d_t63 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_d_410_4.7/analysis/echam/pi_d_410_4.7.01_echam.nc')

# land sea mask info
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region create tsw binned tagmap pi_binned_tsw_tagmap

sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))

ntag = len(sstbins)

pi_binned_tsw_tagmap = xr.Dataset(
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

pi_binned_tsw_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_tsw_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned sst
for j in np.arange(5, len(sstbins)+4):
    pi_binned_tsw_tagmap.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & ((pi_echam_1d_t63.tsw[0] - 273.15) > sstbins[j-5]).values & ((pi_echam_1d_t63.tsw[0] - 273.15) <= sstbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_tsw_tagmap.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_tsw_tagmap.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_tsw_tagmap.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0


pi_binned_tsw_tagmap.to_netcdf('startdump/tagging/tagmap/pi_binned_tsw_tagmap.nc',)


'''
#-------- check

pi_binned_tsw_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_tsw_tagmap.nc',)
np.min(pi_binned_tsw_tagmap.tagmap).values >= 0
np.max(pi_binned_tsw_tagmap.tagmap).values <= 1
np.max(abs(pi_binned_tsw_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region 2nd create tsw binned tagmap pi_binned_tsw_tagmap_1

sstbins = np.concatenate((np.array([-100]), np.arange(-1, 31.1, 1), np.array([100])))

ntag = len(sstbins)

pi_binned_tsw_tagmap_1 = xr.Dataset(
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

pi_binned_tsw_tagmap_1.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_tsw_tagmap_1.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned sst
for j in np.arange(5, len(sstbins)+4):
    pi_binned_tsw_tagmap_1.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & ((pi_echam_1d_t63.tsw[0] - 273.15) > sstbins[j-5]).values & ((pi_echam_1d_t63.tsw[0] - 273.15) <= sstbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_tsw_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_tsw_tagmap_1.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_tsw_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0


pi_binned_tsw_tagmap_1.to_netcdf('startdump/tagging/tagmap/pi_binned_tsw_tagmap_1.nc',)


'''
#-------- check

pi_binned_tsw_tagmap_1 = xr.open_dataset('startdump/tagging/tagmap/pi_binned_tsw_tagmap_1.nc',)
np.min(pi_binned_tsw_tagmap_1.tagmap).values >= 0
np.max(pi_binned_tsw_tagmap_1.tagmap).values <= 1
np.max(abs(pi_binned_tsw_tagmap_1.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region create lat binned tagmap pi_binned_lat_tagmap

latbins = np.arange(-90, 90.1, 10)
b_lat = np.broadcast_to(lat[:, None], analysed_sst.shape)

ntag = len(latbins)

pi_binned_lat_tagmap = xr.Dataset(
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

pi_binned_lat_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_lat_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned lat
for j in np.arange(5, len(latbins)+4):
    pi_binned_lat_tagmap.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (b_lat > latbins[j-5]) & (b_lat <= latbins[j-4])] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_lat_tagmap.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_lat_tagmap.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_lat_tagmap.tagmap.sel(level=slice(5, ntag+3)).values[where_sea_ice] = 0


pi_binned_lat_tagmap.to_netcdf('startdump/tagging/tagmap/pi_binned_lat_tagmap.nc',)

'''
#-------- check

pi_binned_lat_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lat_tagmap.nc',)
np.min(pi_binned_lat_tagmap.tagmap).values >= 0
np.max(pi_binned_lat_tagmap.tagmap).values <= 1
np.max(abs(pi_binned_lat_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0
'''
# endregion
# =============================================================================


# =============================================================================
# region 2nd create lat binned tagmap pi_binned_lat_tagmap_1

latbins = np.arange(-90, 90.1, 5)
b_lat = np.broadcast_to(lat[:, None], analysed_sst.shape)

ntag = len(latbins)

pi_binned_lat_tagmap_1 = xr.Dataset(
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

pi_binned_lat_tagmap_1.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_lat_tagmap_1.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned lat
for j in np.arange(5, len(latbins)+4):
    pi_binned_lat_tagmap_1.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (b_lat > latbins[j-5]) & (b_lat <= latbins[j-4])] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_lat_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_lat_tagmap_1.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_lat_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).values[where_sea_ice] = 0


pi_binned_lat_tagmap_1.to_netcdf('startdump/tagging/tagmap/pi_binned_lat_tagmap_1.nc',)

'''
#-------- check

pi_binned_lat_tagmap_1 = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lat_tagmap_1.nc',)
np.min(pi_binned_lat_tagmap_1.tagmap).values >= 0
np.max(pi_binned_lat_tagmap_1.tagmap).values <= 1
np.max(abs(pi_binned_lat_tagmap_1.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0
'''
# endregion
# =============================================================================


# =============================================================================
# region create rh2m binned tagmap pi_binned_rh2m_tagmap

rh2mbins = np.concatenate((np.array([0]), np.arange(0.55, 1.051, 0.05), np.array([2])))

ntag = len(rh2mbins)

pi_binned_rh2m_tagmap = xr.Dataset(
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

pi_binned_rh2m_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_rh2m_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned rh2m
for j in np.arange(5, len(rh2mbins)+4):
    pi_binned_rh2m_tagmap.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (pi_echam_1d_t63.rh2m[0] > rh2mbins[j-5]).values & (pi_echam_1d_t63.rh2m[0] <= rh2mbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_rh2m_tagmap.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_rh2m_tagmap.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_rh2m_tagmap.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0


pi_binned_rh2m_tagmap.to_netcdf('startdump/tagging/tagmap/pi_binned_rh2m_tagmap.nc',)


'''
#-------- check

pi_binned_rh2m_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_rh2m_tagmap.nc',)
np.min(pi_binned_rh2m_tagmap.tagmap).values >= 0
np.max(pi_binned_rh2m_tagmap.tagmap).values <= 1
np.max(abs(pi_binned_rh2m_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region 2nd create rh2m binned tagmap pi_binned_rh2m_tagmap_1

rh2mbins = np.concatenate((np.array([0]), np.arange(0.52, 1.081, 0.02), np.array([2])))

ntag = len(rh2mbins)

pi_binned_rh2m_tagmap_1 = xr.Dataset(
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

pi_binned_rh2m_tagmap_1.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_rh2m_tagmap_1.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned rh2m
for j in np.arange(5, len(rh2mbins)+4):
    pi_binned_rh2m_tagmap_1.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (pi_echam_1d_t63.rh2m[0] > rh2mbins[j-5]).values & (pi_echam_1d_t63.rh2m[0] <= rh2mbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_rh2m_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_rh2m_tagmap_1.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_rh2m_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0

pi_binned_rh2m_tagmap_1.to_netcdf('startdump/tagging/tagmap/pi_binned_rh2m_tagmap_1.nc',)


'''
#-------- check

pi_binned_rh2m_tagmap_1 = xr.open_dataset('startdump/tagging/tagmap/pi_binned_rh2m_tagmap_1.nc',)
np.min(pi_binned_rh2m_tagmap_1.tagmap).values >= 0
np.max(pi_binned_rh2m_tagmap_1.tagmap).values <= 1
np.max(abs(pi_binned_rh2m_tagmap_1.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region create wind10 binned tagmap pi_binned_wind10_tagmap

wind10bins = np.concatenate((np.array([0]), np.arange(1, 16.1, 1), np.array([100])))

ntag = len(wind10bins)

pi_binned_wind10_tagmap = xr.Dataset(
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

pi_binned_wind10_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_wind10_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned wind10
for j in np.arange(5, len(wind10bins)+4):
    pi_binned_wind10_tagmap.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (pi_echam_1d_t63.wind10[0] > wind10bins[j-5]).values & (pi_echam_1d_t63.wind10[0] <= wind10bins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_wind10_tagmap.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_wind10_tagmap.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_wind10_tagmap.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0


pi_binned_wind10_tagmap.to_netcdf('startdump/tagging/tagmap/pi_binned_wind10_tagmap.nc',)


'''
#-------- check

pi_binned_wind10_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_wind10_tagmap.nc',)
np.min(pi_binned_wind10_tagmap.tagmap).values >= 0
np.max(pi_binned_wind10_tagmap.tagmap).values <= 1
np.max(abs(pi_binned_wind10_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region 2nd create wind10 binned tagmap pi_binned_wind10_tagmap_1

wind10bins = np.concatenate((np.array([0]), np.arange(0.5, 16.51, 0.5), np.array([100])))

ntag = len(wind10bins)

pi_binned_wind10_tagmap_1 = xr.Dataset(
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

pi_binned_wind10_tagmap_1.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_wind10_tagmap_1.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned wind10
for j in np.arange(5, len(wind10bins)+4):
    pi_binned_wind10_tagmap_1.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (pi_echam_1d_t63.wind10[0] > wind10bins[j-5]).values & (pi_echam_1d_t63.wind10[0] <= wind10bins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_wind10_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_wind10_tagmap_1.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_wind10_tagmap_1.tagmap.sel(level=slice(5, ntag+3)).values[
    where_sea_ice] = 0


pi_binned_wind10_tagmap_1.to_netcdf('startdump/tagging/tagmap/pi_binned_wind10_tagmap_1.nc',)


'''
#-------- check

pi_binned_wind10_tagmap_1 = xr.open_dataset('startdump/tagging/tagmap/pi_binned_wind10_tagmap_1.nc',)
np.min(pi_binned_wind10_tagmap_1.tagmap).values >= 0
np.max(pi_binned_wind10_tagmap_1.tagmap).values <= 1
np.max(abs(pi_binned_wind10_tagmap_1.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

'''
# endregion
# =============================================================================


# =============================================================================
# region create lon binned tagmap pi_binned_lon_tagmap

# lonbins = np.concatenate((np.array([-1]), np.arange(20, 340+1e-4, 20), np.array([361])))
# outfile = 'startdump/tagging/tagmap/pi_binned_lon_tagmap.nc'
lonbins = np.concatenate((np.array([-1]), np.arange(10, 350+1e-4, 10), np.array([361])))
outfile = 'startdump/tagging/tagmap/pi_binned_lon_tagmap_1.nc'

b_lon = np.broadcast_to(lon[None, :], analysed_sst.shape)

ntag = len(lonbins)

pi_binned_lon_tagmap = xr.Dataset(
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

pi_binned_lon_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_binned_lon_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# binned lon
for j in np.arange(5, len(lonbins)+4):
    pi_binned_lon_tagmap.tagmap.sel(level=j).values[
        np.isfinite(analysed_sst).values & (b_lon > lonbins[j-5]) & (b_lon <= lonbins[j-4])] = 1

where_sea_ice = np.broadcast_to((pi_echam_1d_t63.seaice[0].values > 0), pi_binned_lon_tagmap.tagmap.sel(level=slice(5, ntag+3)).shape)

pi_binned_lon_tagmap.tagmap.sel(level=4).values[(pi_echam_1d_t63.seaice[0].values > 0)] = 1
pi_binned_lon_tagmap.tagmap.sel(level=slice(5, ntag+3)).values[where_sea_ice] = 0

pi_binned_lon_tagmap.to_netcdf(outfile,)


'''
#-------- check

pi_binned_lon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lon_tagmap.nc',)
np.min(pi_binned_lon_tagmap.tagmap).values >= 0
np.max(pi_binned_lon_tagmap.tagmap).values <= 1
np.max(abs(pi_binned_lon_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0
'''
# endregion
# =============================================================================


# =============================================================================
# region combine binned tag map with pi_geo_tagmap


#---- tsw

! cdo merge startdump/tagging/tagmap/pi_binned_tsw_tagmap.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_tsw_tagmap_a.nc

! cdo merge startdump/tagging/tagmap/pi_binned_tsw_tagmap_1.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_tsw_tagmap_1_a.nc


#---- rh2m

! cdo merge startdump/tagging/tagmap/pi_binned_rh2m_tagmap.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_rh2m_tagmap_a.nc

! cdo merge startdump/tagging/tagmap/pi_binned_rh2m_tagmap_1.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_rh2m_tagmap_1_a.nc


#---- wind10

! cdo merge startdump/tagging/tagmap/pi_binned_wind10_tagmap.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_wind10_tagmap_a.nc

! cdo merge startdump/tagging/tagmap/pi_binned_wind10_tagmap_1.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_wind10_tagmap_1_a.nc


#---- lat

! cdo merge startdump/tagging/tagmap/pi_binned_lat_tagmap.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_lat_tagmap_a.nc

! cdo merge startdump/tagging/tagmap/pi_binned_lat_tagmap_1.nc -sellevel,4/10 startdump/tagging/tagmap/pi_geo_tagmap.nc startdump/tagging/tagmap/pi_binned_lat_tagmap_1_a.nc


#---- lon

! cdo merge startdump/tagging/tagmap/pi_geo_tagmap.nc -sellevel,4/22 startdump/tagging/tagmap/pi_binned_lon_tagmap.nc startdump/tagging/tagmap/pi_binned_lon_tagmap_a.nc

! cdo merge startdump/tagging/tagmap/pi_geo_tagmap.nc -sellevel,4/40 startdump/tagging/tagmap/pi_binned_lon_tagmap_1.nc startdump/tagging/tagmap/pi_binned_lon_tagmap_1_a.nc

# endregion
# =============================================================================



