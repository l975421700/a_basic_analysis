

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
# region division of ocean basins

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
slm = echam6_t63_slm.slm

atlantic_pts1 = [(0, 90), (100, 90), (100, 30),
                 (20, 30), (20, -90), (0, -90), (0, 90)]
atlantic_path1 = Path(atlantic_pts1)

atlantic_pts2=[(360, 90), (360, -90), (290, -90), (290, 0),
               (260, 20), (260, 90), (360, 90)]
atlantic_path2 = Path(atlantic_pts2)

pacific_pts = [(100, 90), (260, 90), (260, 20), (290, 0), (290, -90),
               (140, -90), (140, -30), (100, 30), (100, 90)]
pacific_path = Path(pacific_pts)

indiano_pts = [(100, 30), (140, -30), (140, -90),
               (20, -90), (20, 30), (100, 30)]
indiano_path = Path(indiano_pts)


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    slm.lon, slm.lat, slm, transform=ccrs.PlateCarree(),
    norm=pltnorm, cmap=pltcmp,)

ax.add_patch(patches.PathPatch(atlantic_path1, fill=False, ec='red', lw=4))
ax.add_patch(patches.PathPatch(atlantic_path2, fill=False, ec='red', lw=4,
             transform=ccrs.PlateCarree()))
ax.add_patch(patches.PathPatch(pacific_path, fill=False, ec='blue', lw=3,
             transform=ccrs.PlateCarree(), zorder=2))
ax.add_patch(patches.PathPatch(indiano_path, fill=False, ec='yellow', lw=2,
             transform=ccrs.PlateCarree(), zorder=3))
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
# =============================================================================


# =============================================================================
# region division of continents

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
slm = echam6_t63_slm.slm

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    slm.lon, slm.lat, slm, transform=ccrs.PlateCarree(),
    norm=pltnorm, cmap=pltcmp,)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/3_tagging/3.0_tagmap_creation/3.0.1_division_of_continents.png')

# endregion
# =============================================================================


# =============================================================================
# region create a tagmap

# import data
esacci_echam6_t63_trim = xr.open_dataset('bas_palaeoclim_qino/scratch/cmip6/hist/sst/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
analysed_sst = esacci_echam6_t63_trim.analysed_sst

echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam6_t63_slm.lon
lat = echam6_t63_slm.lat
slm = echam6_t63_slm.slm

# get latlon info
lon2, lat2 = np.meshgrid(lon.values, lat.values)
coors = np.hstack((lon2.reshape(-1, 1), lat2.reshape(-1, 1)))

# define paths and masks
atlantic_path1 = Path([(0, 90), (100, 90), (100, 30),
                       (20, 30), (20, -90), (0, -90), (0, 90)])
atlantic_mask1 = atlantic_path1.contains_points(
    coors, radius = -0.5).reshape(lon2.shape)

atlantic_path2 = Path([(360, 90), (360, -90), (290, -90), (290, 0),
                       (260, 20), (260, 90), (360, 90)])
atlantic_mask2 = atlantic_path2.contains_points(coors).reshape(lon2.shape)

pacific_path = Path([(100, 90), (260, 90), (260, 20), (290, 0), (290, -90),
                     (140, -90), (140, -30), (100, 30), (100, 90)])
pacific_mask = pacific_path.contains_points(coors).reshape(lon2.shape)

indiano_path = Path([(100, 30), (140, -30), (140, -90),
                     (20, -90), (20, 30), (100, 30)])
indiano_mask = indiano_path.contains_points(coors).reshape(lon2.shape)


# 1 global, 4 NH/SH land/sea, 39 ocean basins,
ntag = 1 + 4 + 39

tagmap_echam6_t63_0 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+4, 1, dtype='int32'),
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

# Atlantic, 
tagmap_echam6_t63_0.tagmap.sel(level=9).values[
    atlantic_mask1 & (lat2 > 60) & (np.isfinite(analysed_sst.values))
    ] = 1
tagmap_echam6_t63_0.tagmap.sel(level=9).values[
    atlantic_mask2 & (lat2 > 60) & (np.isfinite(analysed_sst.values))
    ] = 1



tagmap_echam6_t63_0.to_netcdf(
    'bas_palaeoclim_qino/startdump/tagmap/tagmap_echam6_t63_0.nc', mode='w')


'''
np.max(tagmap_echam6_t63_0.tagmap.sel(level=5) + tagmap_echam6_t63_0.tagmap.sel(level=6) + tagmap_echam6_t63_0.tagmap.sel(level=7) + tagmap_echam6_t63_0.tagmap.sel(level=8))

# check
np.sum(atlantic_mask1) + np.sum(atlantic_mask2) + np.sum(pacific_mask) + np.sum(indiano_mask)
(atlantic_mask1 + atlantic_mask2 + pacific_mask + indiano_mask).all()

'''
# endregion
# =============================================================================


