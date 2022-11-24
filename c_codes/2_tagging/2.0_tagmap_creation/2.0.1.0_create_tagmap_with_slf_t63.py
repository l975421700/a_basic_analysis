

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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.path import Path

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

#-------------------------------- get land sea mask

# T63 slm and slf
T63GR15_jan_surf = xr.open_dataset(
    '/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_jan_surf.nc')

# trimmed slm
esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

# 1 means land
t63_slf = T63GR15_jan_surf.SLF.values
# 1 means land
t63_slm = T63GR15_jan_surf.SLM.values
# True means land
# t63_slm_trimmed = np.isnan(analysed_sst)


#-------------------------------- get ocean basin divisions

lon2, lat2 = np.meshgrid(lon, lat)
coors = np.hstack((lon2.reshape(-1, 1), lat2.reshape(-1, 1)))

atlantic_path1 = Path([
    (0, 90), (100, 90), (100, 30), (20, 30), (20, -90), (0, -90), (0, 90)])
atlantic_path2 = Path([
    (360, 90), (360, -90), (290, -90), (290, 8), (279, 8), (270, 15),
    (260, 20), (260, 90), (360, 90)])
pacific_path = Path([
    (100, 90), (260, 90), (260, 20), (270, 15), (279, 8), (290, 8), (290, -90),
    (140, -90), (140, -30), (130, -30), (130, -10), (100, 0), (100, 30),
    (100, 90)])
indiano_path = Path([
    (100, 30), (100, 0), (130, -10), (130, -30), (140, -30), (140, -90),
    (20, -90), (20, 30), (100, 30)])

atlantic_mask1 = atlantic_path1.contains_points(coors, radius = -0.5).reshape(lon2.shape)
atlantic_mask2 = atlantic_path2.contains_points(coors).reshape(lon2.shape)
atlantic_mask = atlantic_mask1 | atlantic_mask2
pacific_mask = pacific_path.contains_points(coors).reshape(lon2.shape)
indiano_mask = indiano_path.contains_points(coors).reshape(lon2.shape)


#-------------------------------- get model output from the 1st time step
t63_1st_output = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_d_500_wiso/unknown/pi_d_500_wiso_200001.01_echam.nc')


'''

np.min(t63_slf[t63_slm == 1])
np.max(t63_slf[t63_slm == 0])

(atlantic_mask | pacific_mask | indiano_mask).all()
(atlantic_mask & pacific_mask & indiano_mask).any()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create geo tagmap

ntag = 7

pi_geo_tagmap = xr.Dataset(
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

pi_geo_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# AIS
pi_geo_tagmap.tagmap.sel(level=3+ 1).values[
    (t63_slm == 1) & (lat < -60)[:, None] ] = 1

# Land exclusive AIS
pi_geo_tagmap.tagmap.sel(level=3+ 2).values[
    (t63_slm == 1) & (lat >= -60)[:, None] ] = 1

# Atlantic ocean and sea ice, > 50° S
pi_geo_tagmap.tagmap.sel(level=3+ 3).values[
    (t63_slm == 0) & (lat >= -50)[:, None] & atlantic_mask] = 1

# Indian ocean and sea ice, > 50° S
pi_geo_tagmap.tagmap.sel(level=3+ 4).values[
    (t63_slm == 0) & (lat >= -50)[:, None] & indiano_mask] = 1

# Pacific ocean and sea ice, > 50° S
pi_geo_tagmap.tagmap.sel(level=3+ 5).values[
    (t63_slm == 0) & (lat >= -50)[:, None] & pacific_mask] = 1

# SH sea ice, <= 50° S
pi_geo_tagmap.tagmap.sel(level=3+ 6).values[
    (t63_slm == 0) & (lat < -50)[:, None] & \
        (t63_1st_output.seaice[0,].values > 0)] = 1

# SO open ocean, <= 50° S
pi_geo_tagmap.tagmap.sel(level=3+ 7).values[
    (t63_slm == 0) & (lat < -50)[:, None] & \
        (t63_1st_output.seaice[0,].values < 1)] = 1

pi_geo_tagmap.to_netcdf('startdump/tagging/tagmap/pi_geo_tagmap.nc',)


'''
1. AIS
2. Land excl. AIS
3. Atlantic ocean and sea ice, > 50° S
4. Indian ocean and sea ice, > 50° S
5. Pacific ocean and sea ice, > 50° S
6. SH sea ice, <= 50° S
7. SO open ocean, <= 50° S


pi_geo_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_geo_tagmap.nc')
test = xr.open_dataset('startdump/tagging/archived/pi_geo_tagmap.nc')
pi_geo_tagmap.sel(level=slice(4, 5)).tagmap.sum(axis=0).sum()
test.sel(level=slice(4, 6)).tagmap.sum(axis=0).sum()

diff = pi_geo_tagmap.sel(level=slice(4, 5)).tagmap.sum(axis=0) - \
    test.sel(level=slice(4, 6)).tagmap.sum(axis=0)
diff.to_netcdf('scratch/test/test.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tsw scaled tagmap pi_tsw_tagmap

minsst = zerok - 5
maxsst = zerok + 45

ntag = 3

pi_tsw_tagmap = xr.Dataset(
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

pi_tsw_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land or sea ice
pi_tsw_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# water, scaled sst
pi_tsw_tagmap.tagmap.sel(level=5).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (t63_1st_output.tsw[0,].values - minsst) / (maxsst - minsst), 0, 1)[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

# complementary set
pi_tsw_tagmap.tagmap.sel(level=6).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_tsw_tagmap.tagmap.sel(level=5).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

pi_tsw_tagmap.to_netcdf('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)


'''
#-------- check

pi_tsw_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)
np.min(pi_tsw_tagmap.tagmap).values >= 0
np.max(pi_tsw_tagmap.tagmap).values <= 1

i = 60
j = 60
t63_1st_output.tsw[0, i, j].values
pi_tsw_tagmap.tagmap[4, i, j].values
(t63_1st_output.tsw[0, i, j].values - 268.15) / 50

#-------- check 2
pi_tsw_tagmap.tagmap.sel(level=4).values[7, 73]
pi_tsw_tagmap.tagmap.sel(level=5).values[7, 73]
pi_tsw_tagmap.tagmap.sel(level=6).values[7, 73]
t63_slf[7, 73]
t63_1st_output.seaice[0,].values[7, 73]
(t63_1st_output.tsw[0, :, :].values[7, 73] - 268.15) / 50
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create rh2m scaled tagmap pi_rh2m_tagmap

minrh2m = 0
maxrh2m = 1.6

ntag = 3

pi_rh2m_tagmap = xr.Dataset(
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

pi_rh2m_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land and sea ice
pi_rh2m_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# water, scaled rh2m
pi_rh2m_tagmap.tagmap.sel(level=5).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (t63_1st_output.rh2m[0,].values - minrh2m) / (maxrh2m - minrh2m), 0, 1)[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

# complementary set
pi_rh2m_tagmap.tagmap.sel(level=6).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_rh2m_tagmap.tagmap.sel(level=5).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

pi_rh2m_tagmap.to_netcdf('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)


'''
#-------- check

pi_rh2m_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)
np.min(pi_rh2m_tagmap.tagmap).values >= 0
np.max(pi_rh2m_tagmap.tagmap).values <= 1

i = 60
j = 60
t63_1st_output.rh2m[0, i, j].values
pi_rh2m_tagmap.tagmap[4, i, j].values
(t63_1st_output.rh2m[0, i, j].values - 0) / 1.6

#-------- check 2
pi_rh2m_tagmap.tagmap.sel(level=4).values[7, 73]
pi_rh2m_tagmap.tagmap.sel(level=5).values[7, 73]
pi_rh2m_tagmap.tagmap.sel(level=6).values[7, 73]
t63_slf[7, 73]
t63_1st_output.seaice[0,].values[7, 73]
(t63_1st_output.rh2m[0, :, :].values[7, 73] - 0) / 1.6
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create wind10 scaled tagmap pi_wind10_tagmap

minwind10 = 0
maxwind10 = 28

ntag = 3

pi_wind10_tagmap = xr.Dataset(
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

pi_wind10_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land and sea ice
pi_wind10_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# water, scaled wind10
pi_wind10_tagmap.tagmap.sel(level=5).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (t63_1st_output.wind10[0,].values - minwind10) / (maxwind10 - minwind10), 0, 1)[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

# complementary set
pi_wind10_tagmap.tagmap.sel(level=6).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_wind10_tagmap.tagmap.sel(level=5).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

pi_wind10_tagmap.to_netcdf('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)


'''
#-------- check

pi_wind10_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)
np.min(pi_wind10_tagmap.tagmap).values >= 0
np.max(pi_wind10_tagmap.tagmap).values <= 1

i = 60
j = 60
t63_1st_output.wind10[0, i, j].values
pi_wind10_tagmap.tagmap[4, i, j].values
(t63_1st_output.wind10[0, i, j].values - 0) / 28

#-------- check 2
pi_wind10_tagmap.tagmap.sel(level=4).values[7, 73]
pi_wind10_tagmap.tagmap.sel(level=5).values[7, 73]
pi_wind10_tagmap.tagmap.sel(level=6).values[7, 73]
t63_slf[7, 73]
t63_1st_output.seaice[0,].values[7, 73]
(t63_1st_output.wind10[0, :, :].values[7, 73] - 0) / 28
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create lat scaled tagmap pi_lat_tagmap

minlat = -90
maxlat = 90

ntag = 3

pi_lat_tagmap = xr.Dataset(
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

pi_lat_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land and sea ice
pi_lat_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# water, scaled lat
pi_lat_tagmap.tagmap.sel(level=5).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (lat2 - minlat) / (maxlat - minlat), 0, 1)[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

# complementary set
pi_lat_tagmap.tagmap.sel(level=6).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_lat_tagmap.tagmap.sel(level=5).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

pi_lat_tagmap.to_netcdf('startdump/tagging/tagmap/pi_lat_tagmap.nc',)


'''
#-------- check

pi_lat_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lat_tagmap.nc',)
np.min(pi_lat_tagmap.tagmap).values >= 0
np.max(pi_lat_tagmap.tagmap).values <= 1

i = 60
j = 60
lat2[i, j]
pi_lat_tagmap.tagmap[4, i, j].values
(lat2[i, j] + 90) / 180

#-------- check 2
pi_lat_tagmap.tagmap.sel(level=4).values[7, 73]
pi_lat_tagmap.tagmap.sel(level=5).values[7, 73]
pi_lat_tagmap.tagmap.sel(level=6).values[7, 73]
t63_slf[7, 73]
lat2[7, 73]
(lat2[7, 73] + 90) / 180
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create sin- and cos-lon scaled tagmap pi_sincoslon_tagmap

min_sincoslon = -1
max_sincoslon = 1

ntag = 6

pi_sincoslon_tagmap = xr.Dataset(
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

pi_sincoslon_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land and sea ice
pi_sincoslon_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1
pi_sincoslon_tagmap.tagmap.sel(level=7).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# water, scaled sincoslon
pi_sincoslon_tagmap.tagmap.sel(level=5).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (np.sin(lon2 * np.pi / 180.) - min_sincoslon) / \
            (max_sincoslon - min_sincoslon), 0, 1)[
                (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]
pi_sincoslon_tagmap.tagmap.sel(level=8).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = np.clip(
        (np.cos(lon2 * np.pi / 180.) - min_sincoslon) / \
            (max_sincoslon - min_sincoslon), 0, 1)[
                (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

# complementary set
pi_sincoslon_tagmap.tagmap.sel(level=6).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_sincoslon_tagmap.tagmap.sel(level=5).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]
pi_sincoslon_tagmap.tagmap.sel(level=9).values[
    (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)] = \
        1 - pi_sincoslon_tagmap.tagmap.sel(level=8).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1)]

pi_sincoslon_tagmap.to_netcdf('startdump/tagging/tagmap/pi_sincoslon_tagmap.nc',)


'''
#-------- check

pi_sincoslon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_sincoslon_tagmap.nc',)
np.min(pi_sincoslon_tagmap.tagmap).values >= 0
np.max(pi_sincoslon_tagmap.tagmap).values <= 1

i = 60
j = 60
lon2[i, j]
pi_sincoslon_tagmap.tagmap[4, i, j].values
(np.sin(lon2[i, j] * np.pi / 180.) + 1) / 2
pi_sincoslon_tagmap.tagmap[7, i, j].values
(np.cos(lon2[i, j] * np.pi / 180.) + 1) / 2

#-------- check 2
pi_sincoslon_tagmap.tagmap.sel(level=4).values[7, 73]
pi_sincoslon_tagmap.tagmap.sel(level=5).values[7, 73]
pi_sincoslon_tagmap.tagmap.sel(level=6).values[7, 73]
t63_slf[7, 73]
lon2[7, 73]
(np.sin(lon2[7, 73] * np.pi / 180.) + 1) / 2
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pi_6tagmap: combine six tagmaps for scaled approach

! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_lat_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_tsw_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_rh2m_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_wind10_tagmap.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' -sellevel,4/9 'startdump/tagging/tagmap/pi_sincoslon_tagmap.nc' 'startdump/tagging/tagmap/pi_6tagmap.nc'

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create lat binned tagmap pi_binned_lat_tagmap

latbins = np.arange(-90, 90.1, 10)
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

pi_binned_lat_tagmap.tagmap.sel(level=slice(1, 3))[:] = 1

# land or sea ice
pi_binned_lat_tagmap.tagmap.sel(level=4).values[
    (t63_slm == 1) | (t63_1st_output.seaice[0,].values > 0)] = 1

# binned lat
for j in np.arange(5, len(latbins)+4):
    pi_binned_lat_tagmap.tagmap.sel(level=j).values[
        (t63_slm == 0) & (t63_1st_output.seaice[0,].values < 1) & \
            (lat2 > latbins[j-5]) & (lat2 <= latbins[j-4])] = 1

pi_binned_lat_tagmap.to_netcdf('startdump/tagging/tagmap/pi_binned_lat_tagmap.nc',)

'''
'''
# endregion
# -----------------------------------------------------------------------------


