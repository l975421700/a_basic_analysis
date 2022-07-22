

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region division of ocean basins

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
slm = echam6_t63_slm.slm


atlantic_path1 = Path([(0, 90), (100, 90), (100, 30), (20, 30), (20, -90), (0, -90), (0, 90)])

atlantic_path2 = Path([(360, 90), (360, -90), (290, -90), (290, 8), (279, 8), (270, 15), (260, 20), (260, 90), (360, 90)])

pacific_path = Path([(100, 90), (260, 90), (260, 20), (270, 15), (279, 8), (290, 8), (290, -90), (140, -90), (140, -30), (130, -30), (130, -10), (100, 0), (100, 30), (100, 90)])

indiano_path = Path([(100, 30), (100, 0), (130, -10), (130, -30), (140, -30), (140, -90), (20, -90), (20, 30), (100, 30)])


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    slm.lon, slm.lat, slm, transform=ccrs.PlateCarree(),
    norm=pltnorm, cmap=pltcmp,)

ax.add_patch(patches.PathPatch(atlantic_path1, fill=False, ec='red', lw=1, alpha = 0.5))
ax.add_patch(patches.PathPatch(atlantic_path2, fill=False, ec='red', lw=1,
             transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(pacific_path, fill=False, ec='m', lw=1,
             transform=ccrs.PlateCarree(), zorder=2, alpha = 0.5))
ax.add_patch(patches.PathPatch(indiano_path, fill=False, ec='yellow', lw=1,
             transform=ccrs.PlateCarree(), zorder=3, alpha = 0.5))


gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.25, zorder=2,
                  color='gray', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180.1, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90.1, 10))

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
# fig.savefig('figures/0_test/trial.png')
fig.savefig('figures/3_tagging/3.0_tagmap_creation/3.0.0_division_of_ocean_basins.png')


'''
/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/fx/sftlf/gn/v20200909/sftlf_fx_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn.nc

coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)


import geopandas as gpd
shpfile = gpd.read_file('/home/users/qino/bas_palaeoclim_qino/others/NaturalEarth/ne_10m_ocean/ne_10m_ocean.shp')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region division of continents

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
slm = echam6_t63_slm.slm


greenland_path = Path([(310, 90), (285, 77.5), (315, 57.5), (350, 57.5), (350, 90), (310, 90)])

northamerica_path = Path([(310, 90), (285, 77.5), (315, 57.5), (315, 15), (280, 15), (280, 5), (190, 5), (190, 90), (310, 90)])

southamerica_path = Path([(315, 15), (280, 15), (280, 5), (190, 5), (190, -60), (330, -60), (330, 15), (315, 15)])

southafrica_path1 = Path([(345, 57.5), (345, 36), (360, 36), (360, -60), (330, -60), (330, 15), (315, 15), (315, 57.5), (345, 57.5)])

southafrica_path2 = Path([(0, 36), (0, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -40), (80, -40), (80, -60), (0, -60), (0, 36)])

austria_path = Path([(70, -11), (130, -11), (130, 5), (190, 5), (190, -60), (80, -60), (80, -40), (70, -40), (70, -11)])

euroasia_path1 = Path([(350, 90), (360, 90), (360, 36), (345, 36), (345, 57.5), (350, 57.5), (350, 90)])

euroasia_path2 = Path([(0, 90), (0, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -11), (130, -11), (130, 5), (190, 5), (190, 90), (0, 90)])


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    slm.lon, slm.lat, slm, transform=ccrs.PlateCarree(),
    norm=pltnorm, cmap=pltcmp,)

ax.add_patch(patches.PathPatch(greenland_path, fill=False, ec='red', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(northamerica_path, fill=False, ec='blue', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(southamerica_path, fill=False, ec='gray', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(southafrica_path1, fill=False, ec='magenta', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(southafrica_path2, fill=False, ec='cyan', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(austria_path, fill=False, ec='yellow', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(euroasia_path1, fill=False, ec='black', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
ax.add_patch(patches.PathPatch(euroasia_path2, fill=False, ec='green', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))


gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.25, zorder=2,
                  color='gray', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180.1, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90.1, 10))

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/3_tagging/3.0_tagmap_creation/3.0.1_division_of_continents.png')

'''
# northamerica_pts1 = [(310, 90), (285, 77.5), (315, 57.5), (315, 45), (190, 45), (190, 90), (310, 90)]
# northamerica_path1 = Path(northamerica_pts1)
# northamerica_pts2 = [(315, 45), (315, 15), (280, 15), (280, 5), (190, 5), (190, 45), (315, 45)]
# northamerica_path2 = Path(northamerica_pts2)
# ax.add_patch(patches.PathPatch(northamerica_path1, fill=False, ec='blue', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
# ax.add_patch(patches.PathPatch(northamerica_path2, fill=False, ec='yellow', lw=1, transform=ccrs.PlateCarree(), alpha = 0.5))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_echam6_t63_0

# inputfile = 'bas_palaeoclim_qino/scratch/cmip6/hist/sst/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc'
inputfile = '/work/ollie/qigao001/startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc'

# outputfile = 'bas_palaeoclim_qino/startdump/tagmap/tagmap_echam6_t63_0.nc'
outputfile = '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_echam6_t63_0.nc'

# import data
esacci_echam6_t63_trim = xr.open_dataset(inputfile)
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

# echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
# slm = echam6_t63_slm.slm
# (lat == echam6_t63_slm.lat.values).all()
# lon = echam6_t63_slm.lon
# lat = echam6_t63_slm.lat
# slm = echam6_t63_slm.slm

# get latlon info
lon2, lat2 = np.meshgrid(lon, lat)
coors = np.hstack((lon2.reshape(-1, 1), lat2.reshape(-1, 1)))

# define paths and masks
atlantic_path1 = Path([(0, 90), (100, 90), (100, 30), (20, 30), (20, -90), (0, -90), (0, 90)])

atlantic_path2 = Path([(360, 90), (360, -90), (290, -90), (290, 8), (279, 8), (270, 15), (260, 20), (260, 90), (360, 90)])

pacific_path = Path([(100, 90), (260, 90), (260, 20), (270, 15), (279, 8), (290, 8), (290, -90), (140, -90), (140, -30), (130, -30), (130, -10), (100, 0), (100, 30), (100, 90)])

indiano_path = Path([(100, 30), (100, 0), (130, -10), (130, -30), (140, -30), (140, -90), (20, -90), (20, 30), (100, 30)])

atlantic_mask1 = atlantic_path1.contains_points(coors, radius = -0.5).reshape(lon2.shape)
atlantic_mask2 = atlantic_path2.contains_points(coors).reshape(lon2.shape)
pacific_mask = pacific_path.contains_points(coors).reshape(lon2.shape)
indiano_mask = indiano_path.contains_points(coors).reshape(lon2.shape)


greenland_path = Path([(310, 90), (285, 77.5), (315, 57.5), (350, 57.5), (350, 90), (310, 90)])

northamerica_path = Path([(310, 90), (285, 77.5), (315, 57.5), (315, 15), (280, 15), (280, 5), (190, 5), (190, 90), (310, 90)])

southamerica_path = Path([(315, 15), (280, 15), (280, 5), (190, 5), (190, -60), (330, -60), (330, 15), (315, 15)])

southafrica_path1 = Path([(345, 57.5), (345, 36), (360, 36), (360, -60), (330, -60), (330, 15), (315, 15), (315, 57.5), (345, 57.5)])

southafrica_path2 = Path([(-0.001, 36), (-0.001, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -40), (80, -40), (80, -60), (-0.001, -60), (-0.001, 36)])

austria_path = Path([(70, -11), (130, -11), (130, 5), (190, 5), (190, -60), (80, -60), (80, -40), (70, -40), (70, -11)])

euroasia_path1 = Path([(350, 90), (360, 90), (360, 36), (345, 36), (345, 57.5), (350, 57.5), (350, 90)])

euroasia_path2 = Path([(-0.001, 90), (-0.001, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -11), (130, -11), (130, 5), (190, 5), (190, 90), (-0.001, 90)])

greenland_lmask = greenland_path.contains_points(coors).reshape(lon2.shape)
northamerica_lmask = northamerica_path.contains_points(coors).reshape(lon2.shape)
southamerica_lmask = southamerica_path.contains_points(coors).reshape(lon2.shape)
southafrica_lmask1 = southafrica_path1.contains_points(coors).reshape(lon2.shape)
southafrica_lmask2 = southafrica_path2.contains_points(coors).reshape(lon2.shape)
austria_lmask = austria_path.contains_points(coors).reshape(lon2.shape)
euroasia_lmask1 = euroasia_path1.contains_points(coors).reshape(lon2.shape)
euroasia_lmask2 = euroasia_path2.contains_points(coors).reshape(lon2.shape)

# southafrica_lmask2.sum()
# 1782 -> 1722
# euroasia_lmask2.sum()
# 4454 -> 4489

# 1 global, 4 NH/SH land/sea, 38 ocean basins, 9 land basins
ntag = 1 + 4 + (15 + 14 + 9) + 9

tagmap_echam6_t63_0 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3.01, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# tag global evaporation
tagmap_echam6_t63_0.tagmap.sel(level=4).values[:, :] = 1

# sh land
tagmap_echam6_t63_0.tagmap.sel(level=5, lat=slice(0, -90)).values[
    np.isnan(analysed_sst.sel(lat=slice(0, -90)).values)] = 1

# sh sea
tagmap_echam6_t63_0.tagmap.sel(level=6, lat=slice(0, -90)).values[
    np.isfinite(analysed_sst.sel(lat=slice(0, -90)).values)] = 1

# nh land
tagmap_echam6_t63_0.tagmap.sel(level=7, lat=slice(90, 0)).values[
    np.isnan(analysed_sst.sel(lat=slice(90, 0)).values)] = 1

# nh sea
tagmap_echam6_t63_0.tagmap.sel(level=8, lat=slice(90, 0)).values[
    np.isfinite(analysed_sst.sel(lat=slice(90, 0)).values)] = 1


#### Atlantic ocean, 9-23
atlantic_lat_ubs = np.concatenate((90, np.arange(70, -60.1, -10)), axis=None)
atlantic_lat_lbs = np.concatenate((np.arange(70, -60.1, -10), -90), axis=None)

for i in range(len(atlantic_lat_ubs)):
    tagmap_echam6_t63_0.tagmap.sel(level=(9+i)).values[
    (lat2 <= atlantic_lat_ubs[i]) & \
        (lat2 > atlantic_lat_lbs[i]) & \
            atlantic_mask1 & (np.isfinite(analysed_sst))
    ] = 1
    tagmap_echam6_t63_0.tagmap.sel(level=(9+i)).values[
    (lat2 <= atlantic_lat_ubs[i]) & \
        (lat2 > atlantic_lat_lbs[i]) & \
            atlantic_mask2 & (np.isfinite(analysed_sst))
    ] = 1

#### Pacific ocean, 24-37
pacific_lat_ubs = np.concatenate((90, np.arange(60, -60.1, -10)), axis=None)
pacific_lat_lbs = np.concatenate((np.arange(60, -60.1, -10), -90), axis=None)

for i in range(len(pacific_lat_ubs)):
    tagmap_echam6_t63_0.tagmap.sel(level=(9+len(atlantic_lat_ubs)+i)).values[
    (lat2 <= pacific_lat_ubs[i]) & \
        (lat2 > pacific_lat_lbs[i]) & \
            pacific_mask & (np.isfinite(analysed_sst))
    ] = 1

#### Indian ocean, 38-47
indiano_lat_ubs = np.concatenate((30, np.arange(10, -60.1, -10)), axis=None)
indiano_lat_lbs = np.concatenate((np.arange(10, -60.1, -10), -90), axis=None)

for i in range(len(indiano_lat_ubs)):
    tagmap_echam6_t63_0.tagmap.sel(level=(
        9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+i)).values[
    (lat2 <= indiano_lat_ubs[i]) & \
        (lat2 > indiano_lat_lbs[i]) & \
            indiano_mask & (np.isfinite(analysed_sst))
    ] = 1

# Antarctica
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 0)).values[(lat2 < -60) & (np.isnan(analysed_sst))] = 1

# South America
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 1)).values[southamerica_lmask & (np.isnan(analysed_sst))] = 1

# North America, below 45 degree north
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 2)).values[(lat2 <= 45) & northamerica_lmask & (np.isnan(analysed_sst))] = 1

# North America, above 45 degree north
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 3)).values[(lat2 > 45) & northamerica_lmask & (np.isnan(analysed_sst))] = 1

# Greenland
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 4)).values[greenland_lmask & (np.isnan(analysed_sst))] = 1

# Euroasia, above 45 degree north
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 5)).values[(lat2 > 45) & euroasia_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 5)).values[(lat2 > 45) & euroasia_lmask2 & (np.isnan(analysed_sst))] = 1

# Euroasia, below 45 degree north
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 6)).values[(lat2 <= 45) & euroasia_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 6)).values[(lat2 <= 45) & euroasia_lmask2 & (np.isnan(analysed_sst))] = 1

# South Africa
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 7)).values[southafrica_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 7)).values[southafrica_lmask2 & (np.isnan(analysed_sst))] = 1

# Austria
tagmap_echam6_t63_0.tagmap.sel(level=(
    9+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 8)).values[austria_lmask & (np.isnan(analysed_sst))] = 1

tagmap_echam6_t63_0.to_netcdf(outputfile)


'''
# check
tagmap_echam6_t63_0 = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_echam6_t63_0.nc')
stats.describe(tagmap_echam6_t63_0.tagmap[3, :, :], axis=None)
stats.describe(tagmap_echam6_t63_0.tagmap[4:8, :, :].sum(axis=0), axis=None)
stats.describe(tagmap_echam6_t63_0.tagmap[8:55, :, :].sum(axis=0), axis=None)

# (tagmap_echam6_t63_0.tagmap[8:55, :, :].sum(axis=0) == 2).sum()
test = tagmap_echam6_t63_0.tagmap[8:55, :, :].sum(axis=0)
test.to_netcdf('/home/users/qino/bas_palaeoclim_qino/others/test/test.nc')



np.max(tagmap_echam6_t63_0.tagmap.sel(level=5) + tagmap_echam6_t63_0.tagmap.sel(level=6) + tagmap_echam6_t63_0.tagmap.sel(level=7) + tagmap_echam6_t63_0.tagmap.sel(level=8))

# check
np.sum(atlantic_mask1) + np.sum(atlantic_mask2) + np.sum(pacific_mask) + np.sum(indiano_mask)
(atlantic_mask1 + atlantic_mask2 + pacific_mask + indiano_mask).all()

# check sum of ocean points
np.sum(np.isfinite(analysed_sst))
np.sum(tagmap_echam6_t63_0.tagmap.sel(level=slice(9, 47)))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create tagmap_echam6_t63_1_47

inputfile = '/work/ollie/qigao001/startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc'

outputfile = '/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_echam6_t63_1_47.nc'

# import data
esacci_echam6_t63_trim = xr.open_dataset(inputfile)
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

# get latlon info
lon2, lat2 = np.meshgrid(lon, lat)
coors = np.hstack((lon2.reshape(-1, 1), lat2.reshape(-1, 1)))

# define paths and masks
atlantic_path1 = Path([(0, 90), (100, 90), (100, 30), (20, 30), (20, -90), (0, -90), (0, 90)])

atlantic_path2 = Path([(360, 90), (360, -90), (290, -90), (290, 8), (279, 8), (270, 15), (260, 20), (260, 90), (360, 90)])

pacific_path = Path([(100, 90), (260, 90), (260, 20), (270, 15), (279, 8), (290, 8), (290, -90), (140, -90), (140, -30), (130, -30), (130, -10), (100, 0), (100, 30), (100, 90)])

indiano_path = Path([(100, 30), (100, 0), (130, -10), (130, -30), (140, -30), (140, -90), (20, -90), (20, 30), (100, 30)])

atlantic_mask1 = atlantic_path1.contains_points(coors, radius = -0.5).reshape(lon2.shape)
atlantic_mask2 = atlantic_path2.contains_points(coors).reshape(lon2.shape)
pacific_mask = pacific_path.contains_points(coors).reshape(lon2.shape)
indiano_mask = indiano_path.contains_points(coors).reshape(lon2.shape)


greenland_path = Path([(310, 90), (285, 77.5), (315, 57.5), (350, 57.5), (350, 90), (310, 90)])

northamerica_path = Path([(310, 90), (285, 77.5), (315, 57.5), (315, 15), (280, 15), (280, 5), (190, 5), (190, 90), (310, 90)])

southamerica_path = Path([(315, 15), (280, 15), (280, 5), (190, 5), (190, -60), (330, -60), (330, 15), (315, 15)])

southafrica_path1 = Path([(345, 57.5), (345, 36), (360, 36), (360, -60), (330, -60), (330, 15), (315, 15), (315, 57.5), (345, 57.5)])

southafrica_path2 = Path([(-0.001, 36), (-0.001, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -40), (80, -40), (80, -60), (-0.001, -60), (-0.001, 36)])

austria_path = Path([(70, -11), (130, -11), (130, 5), (190, 5), (190, -60), (80, -60), (80, -40), (70, -40), (70, -11)])

euroasia_path1 = Path([(350, 90), (360, 90), (360, 36), (345, 36), (345, 57.5), (350, 57.5), (350, 90)])

euroasia_path2 = Path([(-0.001, 90), (-0.001, 38), (12, 38), (12, 34), (30, 34), (44, 12.5), (70, 12.5), (70, -11), (130, -11), (130, 5), (190, 5), (190, 90), (-0.001, 90)])

greenland_lmask = greenland_path.contains_points(coors).reshape(lon2.shape)
northamerica_lmask = northamerica_path.contains_points(coors).reshape(lon2.shape)
southamerica_lmask = southamerica_path.contains_points(coors).reshape(lon2.shape)
southafrica_lmask1 = southafrica_path1.contains_points(coors).reshape(lon2.shape)
southafrica_lmask2 = southafrica_path2.contains_points(coors).reshape(lon2.shape)
austria_lmask = austria_path.contains_points(coors).reshape(lon2.shape)
euroasia_lmask1 = euroasia_path1.contains_points(coors).reshape(lon2.shape)
euroasia_lmask2 = euroasia_path2.contains_points(coors).reshape(lon2.shape)

# southafrica_lmask2.sum()
# 1782 -> 1722
# euroasia_lmask2.sum()
# 4454 -> 4489

# 38 ocean basins, 9 land basins
ntag = (15 + 14 + 9) + 9

tagmap_echam6_t63_1_47 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3.01, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_echam6_t63_1_47.tagmap.sel(level=slice(1, 3))[:, :] = 1

#### Atlantic ocean, 4-18
atlantic_lat_ubs = np.concatenate((90, np.arange(70, -60.1, -10)), axis=None)
atlantic_lat_lbs = np.concatenate((np.arange(70, -60.1, -10), -90), axis=None)

for i in range(len(atlantic_lat_ubs)):
    tagmap_echam6_t63_1_47.tagmap.sel(level=(4+i)).values[
    (lat2 <= atlantic_lat_ubs[i]) & \
        (lat2 > atlantic_lat_lbs[i]) & \
            atlantic_mask1 & (np.isfinite(analysed_sst))
    ] = 1
    tagmap_echam6_t63_1_47.tagmap.sel(level=(4+i)).values[
    (lat2 <= atlantic_lat_ubs[i]) & \
        (lat2 > atlantic_lat_lbs[i]) & \
            atlantic_mask2 & (np.isfinite(analysed_sst))
    ] = 1

#### Pacific ocean, 19-32
pacific_lat_ubs = np.concatenate((90, np.arange(60, -60.1, -10)), axis=None)
pacific_lat_lbs = np.concatenate((np.arange(60, -60.1, -10), -90), axis=None)

for i in range(len(pacific_lat_ubs)):
    tagmap_echam6_t63_1_47.tagmap.sel(level=(4+len(atlantic_lat_ubs)+i)).values[
    (lat2 <= pacific_lat_ubs[i]) & \
        (lat2 > pacific_lat_lbs[i]) & \
            pacific_mask & (np.isfinite(analysed_sst))
    ] = 1

#### Indian ocean, 33-42
indiano_lat_ubs = np.concatenate((30, np.arange(10, -60.1, -10)), axis=None)
indiano_lat_lbs = np.concatenate((np.arange(10, -60.1, -10), -90), axis=None)

for i in range(len(indiano_lat_ubs)):
    tagmap_echam6_t63_1_47.tagmap.sel(level=(
        4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+i)).values[
    (lat2 <= indiano_lat_ubs[i]) & \
        (lat2 > indiano_lat_lbs[i]) & \
            indiano_mask & (np.isfinite(analysed_sst))
    ] = 1

# Antarctica
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 0)).values[(lat2 < -60) & (np.isnan(analysed_sst))] = 1

# South America
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 1)).values[southamerica_lmask & (np.isnan(analysed_sst))] = 1

# North America, below 45 degree north
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 2)).values[(lat2 <= 45) & northamerica_lmask & (np.isnan(analysed_sst))] = 1

# North America, above 45 degree north
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 3)).values[(lat2 > 45) & northamerica_lmask & (np.isnan(analysed_sst))] = 1

# Greenland
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 4)).values[greenland_lmask & (np.isnan(analysed_sst))] = 1

# Euroasia, above 45 degree north
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 5)).values[(lat2 > 45) & euroasia_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 5)).values[(lat2 > 45) & euroasia_lmask2 & (np.isnan(analysed_sst))] = 1

# Euroasia, below 45 degree north
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 6)).values[(lat2 <= 45) & euroasia_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 6)).values[(lat2 <= 45) & euroasia_lmask2 & (np.isnan(analysed_sst))] = 1

# South Africa
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 7)).values[southafrica_lmask1 & (np.isnan(analysed_sst))] = 1
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 7)).values[southafrica_lmask2 & (np.isnan(analysed_sst))] = 1

# Austria
tagmap_echam6_t63_1_47.tagmap.sel(level=(
    4+len(atlantic_lat_ubs)+len(pacific_lat_ubs)+len(indiano_lat_ubs) + 8)).values[austria_lmask & (np.isnan(analysed_sst))] = 1

tagmap_echam6_t63_1_47.to_netcdf(outputfile)


'''
# check
tagmap_echam6_t63_1_47 = xr.open_dataset(outputfile)
np.max(tagmap_echam6_t63_1_47.tagmap[3:, :, :].sum(axis=0))
np.min(tagmap_echam6_t63_1_47.tagmap[3:, :, :].sum(axis=0))

# check
np.sum(atlantic_mask1) + np.sum(atlantic_mask2) + np.sum(pacific_mask) + np.sum(indiano_mask)
(atlantic_mask1 + atlantic_mask2 + pacific_mask + indiano_mask).all()

# check sum of ocean points
np.sum(np.isfinite(analysed_sst))
np.sum(tagmap_echam6_t63_1_47.tagmap.sel(level=slice(4, 41)))

'''
# endregion
# -----------------------------------------------------------------------------




