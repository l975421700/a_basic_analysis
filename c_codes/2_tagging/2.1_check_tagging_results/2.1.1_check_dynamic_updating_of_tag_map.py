

exp_odir = '/albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_1d_802_6.1',]
# ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0,  3, 0]
ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0,  0, 30]
kwiso2 = 3

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
from matplotlib.path import Path

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
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

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '*.01_echam.nc'))
    filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '*.01_wiso.nc'))
    filenames_surf = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*.01_surf.nc'))
    exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam)
    exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso)
    exp_org_o[expid[i]]['surf'] = xr.open_mfdataset(filenames_surf)


lon = exp_org_o[expid[i]]['echam'].lon.values
lat = exp_org_o[expid[i]]['echam'].lat.values

slm3 = exp_org_o[expid[i]]['echam'].slm.values
seaice3 = exp_org_o[expid[i]]['echam'].seaice.values

zqklevw = exp_org_o[expid[i]]['surf'].zqklevw.values
zqsw = exp_org_o[expid[i]]['surf'].zqsw.values

# broadcast lat/lon
lat3 = np.broadcast_to(lat[None, :, None], seaice3.shape)
lon3 = np.broadcast_to(lon[None, None, :], seaice3.shape)

'''
# T63 slm and slf
T63GR15_jan_surf = xr.open_dataset('/albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/pi_1d_801_6.0/input/echam/unit.24')

# 1 means land
t63_slm = T63GR15_jan_surf.SLM.values

pi_6tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_6tagmap.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on lat

itag      = 5
kstart = kwiso2 + sum(ntags[:itag])

minlat = -90
maxlat = 90

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled lat
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (lat3 - minlat) / (maxlat - minlat), 0, 1)[(slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]


#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on tsw

itag      = 7
kstart = kwiso2 + sum(ntags[:itag])

minsst = zerok - 5
maxsst = zerok + 45

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)


# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled tsw
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (exp_org_o[expid[i]]['echam'].tsw.values-minsst)/(maxsst - minsst),0,1)[
        (slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]

#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on rh2m

itag      = 8
kstart = kwiso2 + sum(ntags[:itag])

minrh2m = 0
maxrh2m = 1.6

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)


# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled tsw
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (exp_org_o[expid[i]]['echam'].rh2m.values-minrh2m)/(maxrh2m-minrh2m),0,1)[
        (slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]

#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on wind10

itag      = 9
kstart = kwiso2 + sum(ntags[:itag])

minwind10 = 0
maxwind10 = 28

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled tsw
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (exp_org_o[expid[i]]['echam'].wind10.values-minwind10) / \
        (maxwind10 - minwind10),0,1)[(slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]

#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

# satisfied from the second time step
np.max(abs(ptagmap.tagmap.values[1:] - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values[1:]))


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on sinlon

itag      = 11
kstart = kwiso2 + sum(ntags[:itag])

min_sincoslon = -1
max_sincoslon = 1

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled lat
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (np.sin(lon3 * np.pi / 180.) - min_sincoslon) / \
        (max_sincoslon - min_sincoslon), 0, 1)[(slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]


#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on coslon

itag      = 12
kstart = kwiso2 + sum(ntags[:itag])

min_sincoslon = -1
max_sincoslon = 1

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled lat
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (np.cos(lon3 * np.pi / 180.) - min_sincoslon) / \
        (max_sincoslon - min_sincoslon), 0, 1)[(slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]


#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check pi_geo_tagmap

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

kstart = 15

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  7, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 7+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# AIS
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) & (lat3 < -60)] = 1

# Land exclusive AIS
ptagmap.tagmap.sel(level=2).values[(slm3 == 1) & (lat3 >= -60)] = 1

# Atlantic ocean and sea ice, > 50° S
ptagmap.tagmap.sel(level=3).values[
    (slm3 == 0) & (lat3 >= -50) & atlantic_mask[None, :, :]] = 1

# Indian ocean and sea ice, > 50° S
ptagmap.tagmap.sel(level=4).values[
    (slm3 == 0) & (lat3 >= -50) & indiano_mask[None, :, :]] = 1

# Pacific ocean and sea ice, > 50° S
ptagmap.tagmap.sel(level=5).values[
    (slm3 == 0) & (lat3 >= -50) & pacific_mask[None, :, :]] = 1

# SH sea ice, <= 50° S
ptagmap.tagmap.sel(level=6).values[
    (slm3 == 0) & (lat3 < -50) & (seaice3 > 0)] = 1

# SO open ocean, <= 50° S
ptagmap.tagmap.sel(level=7).values[
    (slm3 == 0) & (lat3 < -50) & (seaice3 < 1)] = 1


#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(16, 22))).all().values

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on RHsst

itag      = 14
kstart = kwiso2 + sum(ntags[:itag])

minRHsst = 0
maxRHsst = 1.4

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros((len(exp_org_o[expid[i]]['wiso'].tagmap.time),
                  3, len(lat), len(lon)),
                 dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": np.arange(1, 3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# land and sea ice
ptagmap.tagmap.sel(level=1).values[(slm3 == 1) | (seaice3 > 0) ] = 1

# water, scaled lat
ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)] = np.clip(
    (zqklevw/zqsw - minRHsst) / (maxRHsst - minRHsst), 0, 1)[(slm3 == 0) & (seaice3 < 1)]

# complementary set
ptagmap.tagmap.sel(level=3).values[(slm3 == 0) & (seaice3 < 1)] = 1 - \
    ptagmap.tagmap.sel(level=2).values[(slm3 == 0) & (seaice3 < 1)]


#---- check
(ptagmap.tagmap.values == exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values).all()

np.max(abs(ptagmap.tagmap.values - exp_org_o[expid[i]]['wiso'].tagmap.sel(
    wisotype=slice(kstart+1, kstart+3)).values))

'''
'''
# endregion
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check binning tagmap based on tsw

# tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_tsw_tagmap_a.nc')
# sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))
tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_tsw_tagmap_1_a.nc')
sstbins = np.concatenate((np.array([-100]), np.arange(-1, 31.1, 1), np.array([100])))

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)


# land
ptagmap.tagmap.sel(level=4).values[island] = 1

# binned sst
for j in np.arange(5, len(sstbins)+4):
    ptagmap.tagmap.sel(level=j).values[
        isocean & ((exp_org_o[expid[i]]['echam'].tsw - 273.15) > sstbins[j-5]).values & ((exp_org_o[expid[i]]['echam'].tsw - 273.15) <= sstbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((exp_org_o[expid[i]]['echam'].seaice.values > 0)[:, None, :, :], exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(5, len(sstbins)+3)).shape)

ptagmap.tagmap.sel(level=4).values[(exp_org_o[expid[i]]['echam'].seaice.values > 0)] = 1
ptagmap.tagmap.sel(level=slice(5, len(sstbins)+3)).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, len(sstbins)+3)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, len(sstbins)+3)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(sstbins)+3)).values, axis=None)

(ptagmap.tagmap.sel(level=slice(4, len(sstbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(sstbins)+3)).values).sum()

i0, i1, i2, i3 = np.where(ptagmap.tagmap.sel(level=slice(4, len(sstbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(sstbins)+3)).values)

exp_org_o[expid[i]]['echam'].tsw.values[i0, i2, i3] - 273.15


#-------- check differences

stats.describe(tagmap.tagmap[3:-7, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:-7, :, :], axis=None)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check binning tagmap based on rh2m

# tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_rh2m_tagmap_a.nc')
# rh2mbins = np.concatenate((np.array([0]), np.arange(0.55, 1.051, 0.05), np.array([2])))
tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_rh2m_tagmap_1_a.nc')
rh2mbins = np.concatenate((np.array([0]), np.arange(0.52, 1.081, 0.02), np.array([2])))

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)


# land
ptagmap.tagmap.sel(level=4).values[island] = 1

# binned sst
for j in np.arange(5, len(rh2mbins)+4):
    ptagmap.tagmap.sel(level=j).values[
        isocean & (exp_org_o[expid[i]]['echam'].rh2m > rh2mbins[j-5]).values & (exp_org_o[expid[i]]['echam'].rh2m <= rh2mbins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((exp_org_o[expid[i]]['echam'].seaice.values > 0)[:, None, :, :], exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(5, len(rh2mbins)+3)).shape)

ptagmap.tagmap.sel(level=4).values[(exp_org_o[expid[i]]['echam'].seaice.values > 0)] = 1
ptagmap.tagmap.sel(level=slice(5, len(rh2mbins)+3)).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, len(rh2mbins)+3)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, len(rh2mbins)+3)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(rh2mbins)+3)).values, axis=None)

(ptagmap.tagmap.sel(level=slice(4, len(rh2mbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(rh2mbins)+3)).values).sum()

i0, i1, i2, i3 = np.where(ptagmap.tagmap.sel(level=slice(4, len(rh2mbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(rh2mbins)+3)).values)

exp_org_o[expid[i]]['echam'].rh2m.values[i0, i2, i3]


#-------- check differences

stats.describe(tagmap.tagmap[3:-7, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:-7, :, :], axis=None)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check binning tagmap based on wind10

# tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_wind10_tagmap_a.nc')
# wind10bins = np.concatenate((np.array([0]), np.arange(1, 16.1, 1), np.array([100])))
tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_wind10_tagmap_1_a.nc')
wind10bins = np.concatenate((np.array([0]), np.arange(0.5, 16.51, 0.5), np.array([100])))

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)


# land
ptagmap.tagmap.sel(level=4).values[island] = 1

# binned sst
for j in np.arange(5, len(wind10bins)+4):
    ptagmap.tagmap.sel(level=j).values[
        isocean & (exp_org_o[expid[i]]['echam'].wind10 > wind10bins[j-5]).values & (exp_org_o[expid[i]]['echam'].wind10 <= wind10bins[j-4]).values] = 1

where_sea_ice = np.broadcast_to((exp_org_o[expid[i]]['echam'].seaice.values > 0)[:, None, :, :], exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(5, len(wind10bins)+3)).shape)

ptagmap.tagmap.sel(level=4).values[(exp_org_o[expid[i]]['echam'].seaice.values > 0)] = 1
ptagmap.tagmap.sel(level=slice(5, len(wind10bins)+3)).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, len(wind10bins)+3)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, len(wind10bins)+3))[1:] - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(wind10bins)+3)).values[1:], axis=None)

(ptagmap.tagmap.sel(level=slice(4, len(wind10bins)+3))[1:] != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(wind10bins)+3)).values[1:]).sum()

i0, i1, i2, i3 = np.where(ptagmap.tagmap.sel(level=slice(4, len(wind10bins)+3))[1:] != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(wind10bins)+3)).values[1:])

exp_org_o[expid[i]]['echam'].wind10.values[1:][i0, i2, i3]


#-------- check differences

stats.describe(tagmap.tagmap[3:-7, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[1, 3:-7, :, :], axis=None)
(tagmap.tagmap[3:-7, :, :] != exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:-7, :, :]).sum()
'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check binning tagmap based on lat

# tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lat_tagmap_a.nc')
# latbins = np.arange(-90, 90.1, 10)
tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lat_tagmap_1_a.nc')
latbins = np.arange(-90, 90.1, 5)
b_lat = np.broadcast_to(lat[None, :, None], exp_org_o[expid[i]]['echam'].tsw.shape)

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)


# land
ptagmap.tagmap.sel(level=4).values[island] = 1

# binned sst
for j in np.arange(5, len(latbins)+4):
    ptagmap.tagmap.sel(level=j).values[
        isocean & (b_lat > latbins[j-5]) & (b_lat <= latbins[j-4])] = 1

where_sea_ice = np.broadcast_to((exp_org_o[expid[i]]['echam'].seaice.values > 0)[:, None, :, :], exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(5, len(latbins)+3)).shape)

ptagmap.tagmap.sel(level=4).values[(exp_org_o[expid[i]]['echam'].seaice.values > 0)] = 1
ptagmap.tagmap.sel(level=slice(5, len(latbins)+3)).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, len(latbins)+3)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, len(latbins)+3)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(latbins)+3)).values, axis=None)

(ptagmap.tagmap.sel(level=slice(4, len(latbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(latbins)+3)).values).sum()


#-------- check differences

stats.describe(tagmap.tagmap[3:-7, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:-7, :, :], axis=None)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check binning tagmap based on lon

# tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lon_tagmap_a.nc')
# lonbins = np.concatenate((np.array([-1]), np.arange(20, 340+1e-4, 20), np.array([361])))
tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_lon_tagmap_1_a.nc')
lonbins = np.concatenate((np.array([-1]), np.arange(10, 350+1e-4, 10), np.array([361])))

b_lon = np.broadcast_to(lon[None, None, :], exp_org_o[expid[i]]['echam'].tsw.shape)

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)


# land
ptagmap.tagmap.sel(level=11).values[island] = 1

# binned sst
for j in np.arange(12, len(lonbins)+11):
    ptagmap.tagmap.sel(level=j).values[
        isocean & (b_lon > lonbins[j-12]) & (b_lon <= lonbins[j-11])] = 1

where_sea_ice = np.broadcast_to((exp_org_o[expid[i]]['echam'].seaice.values > 0)[:, None, :, :], exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(12, len(lonbins)+10)).shape)

ptagmap.tagmap.sel(level=11).values[(exp_org_o[expid[i]]['echam'].seaice.values > 0)] = 1
ptagmap.tagmap.sel(level=slice(12, len(lonbins)+10)).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(11, len(lonbins)+10)).sum(dim='level') - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(11, len(lonbins)+10)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(11, len(lonbins)+10)).values, axis=None)


#-------- check differences

stats.describe(tagmap.tagmap[10:, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 10:, :, :], axis=None)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check binning tagmap based on RHsst

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_binned_RHsst_tagmap.nc')
RHsstbins = np.concatenate((np.arange(0, 1.401, 0.05), np.array([2])))

ptagmap = xr.Dataset(
    {"tagmap": (
        ("time", "level", "lat", "lon"),
        np.zeros(exp_org_o[expid[i]]['wiso'].tagmap.shape, dtype=np.double)),
     },
    coords={
        "time": exp_org_o[expid[i]]['wiso'].tagmap.time.values,
        "level": exp_org_o[expid[i]]['wiso'].tagmap.wisotype.values,
        "lat": exp_org_o[expid[i]]['wiso'].tagmap.lat.values,
        "lon": exp_org_o[expid[i]]['wiso'].tagmap.lon.values,
    }
)

ptagmap.tagmap.sel(level=4).values[(slm3 == 1) | (seaice3 > 0)] = 1

for j in np.arange(5, len(RHsstbins)+4):
    ptagmap.tagmap.sel(level=j).values[
        ((slm3 == 0) & (seaice3 < 1)) & \
            (zqklevw/zqsw > RHsstbins[j-5]) & (zqklevw/zqsw <= RHsstbins[j-4])] = 1

(ptagmap.tagmap.sel(level=slice(4, len(RHsstbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(RHsstbins)+3)).values).sum().values

i0, i1, i2, i3 = np.where(ptagmap.tagmap.sel(level=slice(4, len(RHsstbins)+3)) != exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, len(RHsstbins)+3)).values)

seaice3[i0, i2, i3]
slm3[i0, i2, i3]



'''
'''
# endregion
# -----------------------------------------------------------------------------

