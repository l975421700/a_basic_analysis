

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
from matplotlib.path import Path
import matplotlib.patches as patches
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
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
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# region import data

# import 1-day model simulation
pi_echam_1d_t63 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_310_4.1/analysis/echam/pi_echam6_1d_310_4.1.01_echam.nc')

# land sea mask info
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

'''
# As long as friac > 0, tsw is parameterized as 271.38 K
stats.describe(pi_echam_1d_t63.tsw[0, :, :].values[(pi_echam_1d_t63.friac[0, :, :].values > 0) & np.isfinite(analysed_sst.values)])
'''
# endregion
# =============================================================================


# =============================================================================
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

pi_tsw_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_tsw_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled sst
pi_tsw_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.tsw[0, :, :].values[np.isfinite(analysed_sst)] - minsst) / (maxsst - minsst), 0, 1)

# complementary set
pi_tsw_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_tsw_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_tsw_tagmap.to_netcdf('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)


'''
#-------- check

pi_tsw_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)
stats.describe(pi_tsw_tagmap.tagmap, axis=None)
np.max(pi_tsw_tagmap.tagmap[3:, :, :].sum(axis=0))
np.min(pi_tsw_tagmap.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.tsw[0, i, j].values
pi_tsw_tagmap.tagmap[4, i, j].values
pi_tsw_tagmap.tagmap[5, i, j].values

(pi_echam_1d_t63.tsw[0, i, j].values - 268.15) / 50

#-------- another check
pi_tsw_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)
pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''
# endregion
# =============================================================================


# =============================================================================
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

pi_rh2m_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_rh2m_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled rh2m
pi_rh2m_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.rh2m[0, :, :].values[np.isfinite(analysed_sst)] - minrh2m) / (maxrh2m - minrh2m), 0, 1)


# complementary set
pi_rh2m_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_rh2m_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_rh2m_tagmap.to_netcdf('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)


'''
#-------- check

pi_rh2m_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)
stats.describe(pi_rh2m_tagmap.tagmap, axis=None)
np.max(pi_rh2m_tagmap.tagmap[3:, :, :].sum(axis=0))
np.min(pi_rh2m_tagmap.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.rh2m[0, i, j].values
pi_rh2m_tagmap.tagmap[4, i, j].values
pi_rh2m_tagmap.tagmap[5, i, j].values

pi_echam_1d_t63.rh2m[0, i, j].values / 1.6

#-------- another check
# pi_rh2m_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)
# pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''

# endregion
# =============================================================================


# =============================================================================
# region create wind10 scaled tagmap pi_wind10_tagmap [-14, 28]


minwind10 = -14
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

pi_wind10_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_wind10_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled wind10
pi_wind10_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.wind10[0, :, :].values[np.isfinite(analysed_sst)] - minwind10) / (maxwind10 - minwind10), 0, 1)


# complementary set
pi_wind10_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_wind10_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_wind10_tagmap.to_netcdf('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)


'''
#-------- check

pi_wind10_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)
stats.describe(pi_wind10_tagmap.tagmap, axis=None)
np.max(pi_wind10_tagmap.tagmap[3:, :, :].sum(axis=0))
np.min(pi_wind10_tagmap.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.wind10[0, i, j].values
pi_wind10_tagmap.tagmap[4, i, j].values
pi_wind10_tagmap.tagmap[5, i, j].values

(pi_echam_1d_t63.wind10[0, i, j].values +14) / 42

#-------- another check
# pi_wind10_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)
# pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''

# endregion
# =============================================================================


# =============================================================================
# region create wind10 scaled tagmap pi_wind10_tagmap_0 [0, 28]


minwind10 = 0
maxwind10 = 28

ntag = 3

pi_wind10_tagmap_0 = xr.Dataset(
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

pi_wind10_tagmap_0.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_wind10_tagmap_0.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled wind10
pi_wind10_tagmap_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.wind10[0, :, :].values[np.isfinite(analysed_sst)] - minwind10) / (maxwind10 - minwind10), 0, 1)


# complementary set
pi_wind10_tagmap_0.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_wind10_tagmap_0.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_wind10_tagmap_0.to_netcdf('startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',)


'''
#-------- check

pi_wind10_tagmap_0 = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',)
stats.describe(pi_wind10_tagmap_0.tagmap, axis=None)
np.max(pi_wind10_tagmap_0.tagmap[3:, :, :].sum(axis=0))
np.min(pi_wind10_tagmap_0.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.wind10[0, i, j].values
pi_wind10_tagmap_0.tagmap[4, i, j].values
pi_wind10_tagmap_0.tagmap[5, i, j].values

(pi_echam_1d_t63.wind10[0, i, j].values) / 28

#-------- another check
# pi_wind10_tagmap_0 = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',)
# pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''

# endregion
# =============================================================================


# =============================================================================
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

pi_lat_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_lat_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled lat
pi_lat_tagmap.tagmap.sel(level=5).values[:, :] = ((lat - minlat)/(maxlat - minlat))[:, None]
pi_lat_tagmap.tagmap.sel(level=5).values[np.isnan(analysed_sst)] = 0

# complementary set
pi_lat_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_lat_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_lat_tagmap.to_netcdf('startdump/tagging/tagmap/pi_lat_tagmap.nc',)


'''
#-------- check

pi_lat_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lat_tagmap.nc',)
stats.describe(pi_lat_tagmap.tagmap, axis=None)
np.max(pi_lat_tagmap.tagmap[3:, :, :].sum(axis=0))
np.min(pi_lat_tagmap.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.lat[i].values
pi_lat_tagmap.tagmap[4, i, j].values
pi_lat_tagmap.tagmap[5, i, j].values

(pi_echam_1d_t63.lat[i].values + 90) / 180

#-------- another check
# pi_lat_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lat_tagmap.nc',)
# pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''

# endregion
# =============================================================================


# =============================================================================
# region create lon scaled tagmap pi_lon_tagmap


minlon = 0
maxlon = 360

ntag = 3

pi_lon_tagmap = xr.Dataset(
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

pi_lon_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_lon_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled lon
pi_lon_tagmap.tagmap.sel(level=5).values[:, :] = ((lon - minlon)/(maxlon - minlon))[None, :]
pi_lon_tagmap.tagmap.sel(level=5).values[np.isnan(analysed_sst)] = 0

# complementary set
pi_lon_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_lon_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]


pi_lon_tagmap.to_netcdf('startdump/tagging/tagmap/pi_lon_tagmap.nc',)


'''
#-------- check

pi_lon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lon_tagmap.nc',)
stats.describe(pi_lon_tagmap.tagmap, axis=None)
np.max(pi_lon_tagmap.tagmap[3:, :, :].sum(axis=0))
np.min(pi_lon_tagmap.tagmap[3:, :, :].sum(axis=0))


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.lon[j].values
pi_lon_tagmap.tagmap[4, i, j].values
pi_lon_tagmap.tagmap[5, i, j].values

(pi_echam_1d_t63.lon[j].values) / 360

#-------- another check
# pi_lon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lon_tagmap.nc',)
# pi_wiso_1d_t63 = xr.open_dataset('???01_wiso.nc')
'''

# endregion
# =============================================================================


# =============================================================================
# region pi_five_scaled_tagmap: combine all tagmaps for scaled approach

! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_lat_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_lon_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_tsw_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_rh2m_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_wind10_tagmap_0.nc' 'startdump/tagging/tagmap/pi_five_scaled_tagmap.nc'

'''
! cdo -setlevel,1/3 -sellevel,4/6 startdump/tagging/tagmap/pi_tsw_tagmap.nc test.nc


#-------------------------------- check

pi_tagmap_files = [
    'startdump/tagging/tagmap/pi_lat_tagmap.nc',
    'startdump/tagging/tagmap/pi_lon_tagmap.nc',
    'startdump/tagging/tagmap/pi_tsw_tagmap.nc',
    'startdump/tagging/tagmap/pi_rh2m_tagmap.nc',
    'startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',
    'startdump/tagging/tagmap/pi_five_scaled_tagmap.nc',
]

pi_tagmaps = {}

for ifile in range(len(pi_tagmap_files)):
    pi_tagmaps[ifile] = xr.open_dataset(pi_tagmap_files[ifile])

(pi_tagmaps[5].tagmap[0:6] == pi_tagmaps[0].tagmap[0:6]).all()
(pi_tagmaps[5].tagmap[6:9] == pi_tagmaps[1].tagmap[3:6]).all()
(pi_tagmaps[5].tagmap[9:12] == pi_tagmaps[2].tagmap[3:6]).all()
(pi_tagmaps[5].tagmap[12:15] == pi_tagmaps[3].tagmap[3:6]).all()
(pi_tagmaps[5].tagmap[15:18] == pi_tagmaps[4].tagmap[3:6]).all()

'''
# endregion
# =============================================================================


# =============================================================================
# region

# endregion
# =============================================================================

