

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

# import 1-day model simulation
pi_echam_1d_t63 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_d_410_4.7/analysis/echam/pi_d_410_4.7.01_echam.nc')

# land sea mask info
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst




'''
# As long as friac > 0, tsw is parameterized as 271.38 K
stats.describe(pi_echam_1d_t63.tsw[-1, :, :].values[(pi_echam_1d_t63.seaice[-1, :, :].values > 0) & np.isfinite(analysed_sst.values)])
stats.describe(pi_echam_1d_t63.seaice - pi_echam_1d_t63.friac, axis=None)
test = pi_echam_1d_t63.seaice - pi_echam_1d_t63.friac
test.to_netcdf('test.nc',)


sst_amip_alex = xr.open_dataset('startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc')
sic_amip_alex = xr.open_dataset('startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')
wisosw_d_amip_alex = xr.open_dataset('startdump/model_input/pi/alex/T63_wisosw_d.nc')

# stats.describe(sst_amip_alex.sst[-1, :, :].values[(sic_amip_alex.sic[-1, :, :].values > 0) & np.isfinite(analysed_sst.values)])

# unit.20
pi_sst_t63 = xr.open_dataset('/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_piControl-LR_sst_1880-2379.nc')
# unit.25
pi_wisosw_d_t63 = xr.open_dataset('/home/ollie/mwerner/model_input/ECHAM6-wiso/PI_ctrl/T63/T63_wisosw_d.nc')
# unit.96
pi_sic_t63 = xr.open_dataset('/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_piControl-LR_sic_1880-2379.nc')


#-------------------------------- plot sea ice in Jan
# sic_amip_alex = xr.open_dataset('startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')

#---- Antarctic plot

# plt_x = sic_amip_alex.sic[0,].lon
# plt_y = sic_amip_alex.sic[0,].lat
# plt_z = sic_amip_alex.sic[0,]

# pltlevel = np.arange(0, 100.01, 10)
# pltticks = np.arange(0, 100.01, 20)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
# pltcmp = cm.get_cmap('Blues', len(pltlevel))

# cbar_label = 'Precipitation-weighted source SST [$\%$]\n '
# output_png = '/work/ollie/qigao001/figures/test.png'

# fig, ax = hemisphere_plot(northextent=-45)
# plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
#                         norm=pltnorm, cmap=pltcmp,)

# cbar = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
#     orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
#     pad=0.02, fraction=0.2,
#     )
# cbar.ax.set_xlabel(cbar_label, linespacing=2)
# fig.savefig(output_png)
'''
# endregion
# -----------------------------------------------------------------------------


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

pi_tsw_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_tsw_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled sst
pi_tsw_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.tsw[0, :, :].values[np.isfinite(analysed_sst)] - minsst) / (maxsst - minsst), 0, 1)

# complementary set
pi_tsw_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_tsw_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_tsw_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_tsw_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_tsw_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_tsw_tagmap.to_netcdf('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)


'''
#-------- check

pi_tsw_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_tsw_tagmap.nc',)
np.min(pi_tsw_tagmap.tagmap).values >= 0
np.max(pi_tsw_tagmap.tagmap).values <= 1
np.max(abs(pi_tsw_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.tsw[0, i, j].values
pi_tsw_tagmap.tagmap[4, i, j].values
(pi_echam_1d_t63.tsw[0, i, j].values - 268.15) / 50

#-------- check 2
pi_tsw_tagmap.tagmap.sel(level=4).values[7, 73]
pi_tsw_tagmap.tagmap.sel(level=5).values[7, 73]
pi_tsw_tagmap.tagmap.sel(level=6).values[7, 73]
(pi_echam_1d_t63.tsw[0, :, :].values[7, 73] - 268.15) / 50
pi_echam_1d_t63.friac[0, :, :].values[7, 73]
pi_echam_1d_t63.seaice[0, :, :].values[7, 73]

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

pi_rh2m_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_rh2m_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled rh2m
pi_rh2m_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)] = np.clip((pi_echam_1d_t63.rh2m[0, :, :].values[np.isfinite(analysed_sst)] - minrh2m) / (maxrh2m - minrh2m), 0, 1)


# complementary set
pi_rh2m_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_rh2m_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_rh2m_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_rh2m_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_rh2m_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_rh2m_tagmap.to_netcdf('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)


'''
#-------- check

pi_rh2m_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_rh2m_tagmap.nc',)
np.min(pi_rh2m_tagmap.tagmap).values >= 0
np.max(pi_rh2m_tagmap.tagmap).values <= 1
np.max(abs(pi_rh2m_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.rh2m[0, i, j].values
pi_rh2m_tagmap.tagmap[4, i, j].values
pi_echam_1d_t63.rh2m[0, i, j].values / 1.6

#-------- check 2

pi_rh2m_tagmap.tagmap.sel(level=4).values[7, 73]
pi_rh2m_tagmap.tagmap.sel(level=5).values[7, 73]
pi_rh2m_tagmap.tagmap.sel(level=6).values[7, 73]
(pi_echam_1d_t63.rh2m[0, :, :].values[7, 73]) / 1.6
pi_echam_1d_t63.friac[0, :, :].values[7, 73]
pi_echam_1d_t63.seaice[0, :, :].values[7, 73]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_wind10_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_wind10_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_wind10_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_wind10_tagmap.to_netcdf('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)


'''
#-------- check

pi_wind10_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap.nc',)
np.min(pi_wind10_tagmap.tagmap).values >= 0
np.max(pi_wind10_tagmap.tagmap).values <= 1
np.max(abs(pi_wind10_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.wind10[0, i, j].values
pi_wind10_tagmap.tagmap[4, i, j].values
(pi_echam_1d_t63.wind10[0, i, j].values +14) / 42

#-------- check 2

pi_wind10_tagmap.tagmap.sel(level=4).values[7, 73]
pi_wind10_tagmap.tagmap.sel(level=5).values[7, 73]
pi_wind10_tagmap.tagmap.sel(level=6).values[7, 73]
(pi_echam_1d_t63.wind10[0, :, :].values[7, 73] + 14) / 42
pi_echam_1d_t63.friac[0, :, :].values[7, 73]
pi_echam_1d_t63.seaice[0, :, :].values[7, 73]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_wind10_tagmap_0.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_wind10_tagmap_0.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_wind10_tagmap_0.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_wind10_tagmap_0.to_netcdf('startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',)


'''
#-------- check

pi_wind10_tagmap_0 = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',)
np.min(pi_wind10_tagmap_0.tagmap).values >= 0
np.max(pi_wind10_tagmap_0.tagmap).values <= 1
np.max(abs(pi_wind10_tagmap_0.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0

i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.wind10[0, i, j].values
pi_wind10_tagmap_0.tagmap[4, i, j].values
(pi_echam_1d_t63.wind10[0, i, j].values +0) / 28

#-------- check 2

pi_wind10_tagmap_0.tagmap.sel(level=4).values[7, 73]
pi_wind10_tagmap_0.tagmap.sel(level=5).values[7, 73]
pi_wind10_tagmap_0.tagmap.sel(level=6).values[7, 73]
(pi_echam_1d_t63.wind10[0, :, :].values[7, 73] + 0) / 28
pi_echam_1d_t63.friac[0, :, :].values[7, 73]
pi_echam_1d_t63.seaice[0, :, :].values[7, 73]

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

pi_lat_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# land
pi_lat_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1

# scaled lat
pi_lat_tagmap.tagmap.sel(level=5).values[:, :] = ((lat - minlat)/(maxlat - minlat))[:, None]
pi_lat_tagmap.tagmap.sel(level=5).values[np.isnan(analysed_sst)] = 0

# complementary set
pi_lat_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_lat_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_lat_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_lat_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_lat_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_lat_tagmap.to_netcdf('startdump/tagging/tagmap/pi_lat_tagmap.nc',)


'''
#-------- check

pi_lat_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lat_tagmap.nc',)
np.min(pi_lat_tagmap.tagmap).values >= 0
np.max(pi_lat_tagmap.tagmap).values <= 1
np.max(abs(pi_lat_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.lat[i].values
pi_lat_tagmap.tagmap[4, i, j].values
(pi_echam_1d_t63.lat[i].values + 90) / 180

#-------- check 2

pi_lat_tagmap.tagmap.sel(level=4).values[7, 73]
pi_lat_tagmap.tagmap.sel(level=5).values[7, 73]
pi_lat_tagmap.tagmap.sel(level=6).values[7, 73]
(pi_lat_tagmap.lat.values[7] + 90) / 180
pi_echam_1d_t63.friac.values[0, 7, 73]
pi_echam_1d_t63.seaice.values[0, 7, 73]

'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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

where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_lon_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_lon_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_lon_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


pi_lon_tagmap.to_netcdf('startdump/tagging/tagmap/pi_lon_tagmap.nc',)


'''
#-------- check

pi_lon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lon_tagmap.nc',)
np.min(pi_lon_tagmap.tagmap).values >= 0
np.max(pi_lon_tagmap.tagmap).values <= 1
np.max(abs(pi_lon_tagmap.tagmap[3:, :, :].sum(axis=0) - 1)).values == 0


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.lon[j].values
pi_lon_tagmap.tagmap[4, i, j].values
(pi_echam_1d_t63.lon[j].values) / 360

#-------- check 2

pi_lon_tagmap.tagmap.sel(level=4).values[7, 73]
pi_lon_tagmap.tagmap.sel(level=5).values[7, 73]
pi_lon_tagmap.tagmap.sel(level=6).values[7, 73]
(pi_lon_tagmap.lon.values[73] + 0) / 360
pi_echam_1d_t63.friac.values[0, 7, 73]
pi_echam_1d_t63.seaice.values[0, 7, 73]

'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tag map for NH/SH land, Antarctica, NH/SH ocean, NH/SH sea ice

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

pi_geo_tagmap.tagmap.sel(level=slice(1, 3))[:, :] = 1

# NH land
pi_geo_tagmap.tagmap.sel(level=4).values[
    np.isnan(analysed_sst) & (lat > 0)[:, None] ] = 1

# SH land, excl. Antarctica
pi_geo_tagmap.tagmap.sel(level=5).values[
    np.isnan(analysed_sst) & ((lat < 0) & (lat > -60))[:, None] ] = 1

# Antarctica
pi_geo_tagmap.tagmap.sel(level=6).values[
    np.isnan(analysed_sst) & (lat < -60)[:, None] ] = 1

# NH ocean, excl. sea ice
pi_geo_tagmap.tagmap.sel(level=7).values[
    (pi_geo_tagmap.tagmap.sel(level=slice(4, 6)).sum(axis=0) == 0) & (lat > 0)[:, None] & (pi_echam_1d_t63.seaice[0, :, :].values == 0)] = 1

# NH sea ice
pi_geo_tagmap.tagmap.sel(level=8).values[
    (pi_geo_tagmap.tagmap.sel(level=slice(4, 6)).sum(axis=0) == 0) & (lat > 0)[:, None] & (pi_echam_1d_t63.seaice[0, :, :].values > 0)] = 1

# SH ocean, excl. sea ice
pi_geo_tagmap.tagmap.sel(level=9).values[
    (pi_geo_tagmap.tagmap.sel(level=slice(4, 6)).sum(axis=0) == 0) & (lat < 0)[:, None] & (pi_echam_1d_t63.seaice[0, :, :].values == 0)] = 1

# SH sea ice
pi_geo_tagmap.tagmap.sel(level=10).values[
    (pi_geo_tagmap.tagmap.sel(level=slice(4, 6)).sum(axis=0) == 0) & (lat < 0)[:, None] & (pi_echam_1d_t63.seaice[0, :, :].values > 0)] = 1

pi_geo_tagmap.to_netcdf('startdump/tagging/tagmap/pi_geo_tagmap.nc',)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region pi_6tagmap: combine six tagmaps for scaled approach and sea ice one

! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_lat_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_lon_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_tsw_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_rh2m_tagmap.nc' -sellevel,4/6 'startdump/tagging/tagmap/pi_wind10_tagmap_0.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_6tagmap.nc'

! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_lat_tagmap.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_lat_tagmap_a.nc'
! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_lon_tagmap.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_lon_tagmap_a.nc'
! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_tsw_tagmap.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_tsw_tagmap_a.nc'
! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_rh2m_tagmap.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_rh2m_tagmap_a.nc'
! cdo merge -sellevel,1/6 'startdump/tagging/tagmap/pi_wind10_tagmap_0.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' 'startdump/tagging/tagmap/pi_wind10_tagmap_0_a.nc'


! cdo merge -sellevel,1/6 'startdump/tagging/archived/pi_lat_tagmap.nc' -sellevel,4/6 'startdump/tagging/archived/pi_tsw_tagmap.nc' -sellevel,4/6 'startdump/tagging/archived/pi_rh2m_tagmap.nc' -sellevel,4/6 'startdump/tagging/archived/pi_wind10_tagmap_0.nc' -sellevel,4/10 'startdump/tagging/tagmap/pi_geo_tagmap.nc' -sellevel,4/9 'startdump/tagging/archived/pi_sincoslon_tagmap.nc' 'startdump/tagging/tagmap/pi_6tagmap_2.nc'

'''
#-------------------------------- check

pi_tagmap_files = [
    'startdump/tagging/tagmap/pi_lat_tagmap.nc',
    'startdump/tagging/tagmap/pi_lon_tagmap.nc',
    'startdump/tagging/tagmap/pi_tsw_tagmap.nc',
    'startdump/tagging/tagmap/pi_rh2m_tagmap.nc',
    'startdump/tagging/tagmap/pi_wind10_tagmap_0.nc',
    'startdump/tagging/tagmap/pi_geo_tagmap.nc',
    'startdump/tagging/tagmap/pi_6tagmap.nc',
]

pi_tagmaps = {}

for ifile in range(len(pi_tagmap_files)):
    pi_tagmaps[ifile] = xr.open_dataset(pi_tagmap_files[ifile])

(pi_tagmaps[6].tagmap[0:6] == pi_tagmaps[0].tagmap[0:6]).all()
(pi_tagmaps[6].tagmap[6:9] == pi_tagmaps[1].tagmap[3:6]).all()
(pi_tagmaps[6].tagmap[9:12] == pi_tagmaps[2].tagmap[3:6]).all()
(pi_tagmaps[6].tagmap[12:15] == pi_tagmaps[3].tagmap[3:6]).all()
(pi_tagmaps[6].tagmap[15:18] == pi_tagmaps[4].tagmap[3:6]).all()
(pi_tagmaps[6].tagmap[18:25] == pi_tagmaps[5].tagmap[3:10]).all()
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

# land
pi_sincoslon_tagmap.tagmap.sel(level=4).values[np.isnan(analysed_sst)] = 1
pi_sincoslon_tagmap.tagmap.sel(level=7).values[np.isnan(analysed_sst)] = 1

# scaled sin lon
pi_sincoslon_tagmap.tagmap.sel(level=5).values[:, :] = ((np.sin(lon * np.pi / 180.) - min_sincoslon)/(max_sincoslon - min_sincoslon))[None, :]
pi_sincoslon_tagmap.tagmap.sel(level=5).values[np.isnan(analysed_sst)] = 0

# scaled cos lon
pi_sincoslon_tagmap.tagmap.sel(level=8).values[:, :] = ((np.cos(lon * np.pi / 180.) - min_sincoslon)/(max_sincoslon - min_sincoslon))[None, :]
pi_sincoslon_tagmap.tagmap.sel(level=8).values[np.isnan(analysed_sst)] = 0


# complementary set
pi_sincoslon_tagmap.tagmap.sel(level=6).values[np.isfinite(analysed_sst)] = 1 - pi_sincoslon_tagmap.tagmap.sel(level=5).values[np.isfinite(analysed_sst)]
pi_sincoslon_tagmap.tagmap.sel(level=9).values[np.isfinite(analysed_sst)] = 1 - pi_sincoslon_tagmap.tagmap.sel(level=8).values[np.isfinite(analysed_sst)]


where_sea_ice = (pi_echam_1d_t63.seaice[0, :, :].values > 0)
pi_sincoslon_tagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
pi_sincoslon_tagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
pi_sincoslon_tagmap.tagmap.sel(level=6).values[where_sea_ice] = 0
pi_sincoslon_tagmap.tagmap.sel(level=7).values[where_sea_ice] = 1
pi_sincoslon_tagmap.tagmap.sel(level=8).values[where_sea_ice] = 0
pi_sincoslon_tagmap.tagmap.sel(level=9).values[where_sea_ice] = 0


pi_sincoslon_tagmap.to_netcdf('startdump/tagging/tagmap/pi_sincoslon_tagmap.nc',)


'''
#-------- check

pi_sincoslon_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_sincoslon_tagmap.nc',)
np.min(pi_sincoslon_tagmap.tagmap).values >= 0
np.max(pi_sincoslon_tagmap.tagmap).values <= 1
np.max(abs(pi_sincoslon_tagmap.tagmap[3:, :, :].sum(axis=0) - 2)).values == 0


i = 60
j = 60
np.isfinite(analysed_sst[i, j].values)
pi_echam_1d_t63.lon[j].values
pi_sincoslon_tagmap.tagmap[4, i, j].values
(np.sin(pi_echam_1d_t63.lon[j].values * np.pi / 180) + 1)/2
pi_sincoslon_tagmap.tagmap[7, i, j].values
(np.cos(pi_echam_1d_t63.lon[j].values * np.pi / 180) + 1)/2

#-------- check 2

pi_sincoslon_tagmap.tagmap.sel(level=4).values[7, 73]
pi_sincoslon_tagmap.tagmap.sel(level=5).values[7, 73]
pi_sincoslon_tagmap.tagmap.sel(level=6).values[7, 73]
(np.sin(pi_echam_1d_t63.lon[73].values * np.pi / 180) + 1)/2
pi_echam_1d_t63.friac.values[0, 7, 73]
pi_echam_1d_t63.seaice.values[0, 7, 73]

'''

# endregion

! cdo merge -sellevel,1/10 startdump/tagging/tagmap/pi_geo_tagmap.nc -sellevel,4/9 startdump/tagging/tagmap/pi_sincoslon_tagmap.nc startdump/tagging/tagmap/pi_sincoslon_tagmap_a.nc

# -----------------------------------------------------------------------------




