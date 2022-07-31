

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


expid = [
    'pi_d_437_4.10',
    ]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import data

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    
    exp_org_o[expid[i]] = {}
    
    exp_org_o[expid[i]]['echam'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst

island = np.broadcast_to(np.isnan(analysed_sst), exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)
isocean = np.broadcast_to(np.isfinite(analysed_sst), exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check scale of tagmap based on lat

minlat = -90
maxlat = 90

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lat_tagmap_a.nc')

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

# scaled lat
b_lat = np.broadcast_to(exp_org_o[expid[i]]['echam'].lat.values[None, :, None], exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)

ptagmap.tagmap.sel(level=5).values[isocean] = np.clip( (b_lat - minlat) / (maxlat - minlat), 0, 1)[isocean]

# complementary set
ptagmap.tagmap.sel(level=6).values[isocean] = 1 - ptagmap.tagmap.sel(level=5).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
ptagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, 6)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values, axis=None)

test = ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(4, 6)).values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values[wheremax]

#-------- check differences
stats.describe(tagmap.tagmap[3:6, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:6, :, :], axis=None)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on lon

minlon = 0
maxlon = 360

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_lon_tagmap_a.nc')

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

# scaled lon
b_lon = np.broadcast_to(exp_org_o[expid[i]]['echam'].lon.values[None, None, :], exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)
ptagmap.tagmap.sel(level=5).values[isocean] = np.clip( (b_lon - minlon) / (maxlon - minlon), 0, 1)[isocean]

# complementary set
ptagmap.tagmap.sel(level=6).values[isocean] = 1 - ptagmap.tagmap.sel(level=5).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
ptagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, 6)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values, axis=None)

test = ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(4, 6)).values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values[wheremax]

#-------- check differences
stats.describe(tagmap.tagmap[3:6, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:6, :, :], axis=None)


'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on tsw

minsst = zerok - 5
maxsst = zerok + 45

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_tsw_tagmap_a.nc')

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

# scaled sst
ptagmap.tagmap.sel(level=5).values[isocean] = np.clip( (exp_org_o[expid[i]]['echam'].tsw.values[isocean] - minsst) / (maxsst - minsst), 0, 1)

# complementary set
ptagmap.tagmap.sel(level=6).values[isocean] = 1 - ptagmap.tagmap.sel(level=5).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
# np.min(exp_org_o[expid[i]]['echam'].seaice.values[where_sea_ice])
ptagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, 6)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values, axis=None)

test = ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(4, 6)).values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values[wheremax]

#-------- check differences
stats.describe(tagmap.tagmap[3:6, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:6, :, :], axis=None)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on rh2m

minrh2m = 0
maxrh2m = 1.6

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_rh2m_tagmap_a.nc')

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

# scaled rh2m
ptagmap.tagmap.sel(level=5).values[isocean] = np.clip( (exp_org_o[expid[i]]['echam'].rh2m.values[isocean] - minrh2m) / (maxrh2m - minrh2m), 0, 1)

# complementary set
ptagmap.tagmap.sel(level=6).values[isocean] = 1 - ptagmap.tagmap.sel(level=5).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
ptagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(4, 6)).sum(axis=1) - 1)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values, axis=None)

test = ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(4, 6)).values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values[wheremax]

#-------- check differences
stats.describe(tagmap.tagmap[3:6, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:6, :, :], axis=None)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on wind10

minwind10 = 0
maxwind10 = 28

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_wind10_tagmap_0_a.nc')

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

# scaled wind10
ptagmap.tagmap.sel(level=5).values[isocean] = np.clip( (exp_org_o[expid[i]]['echam'].wind10.values[isocean] - minwind10) / (maxwind10 - minwind10), 0, 1)

# complementary set
ptagmap.tagmap.sel(level=6).values[isocean] = 1 - ptagmap.tagmap.sel(level=5).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
ptagmap.tagmap.sel(level=4).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=5).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=6).values[where_sea_ice] = 0


#-------- check tagmap complementarity

np.max(abs(ptagmap.tagmap.sel(level=slice(4, 6)).sum(axis=1) - 1)) == 0


#-------- check differences

stats.describe(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values, axis=None)
stats.describe(ptagmap.tagmap.sel(level=slice(4, 6))[1:] - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values[1:], axis=None)
# stats.describe(abs(ptagmap.tagmap.sel(level=slice(4, 6)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6)).values), axis=None)

test = ptagmap.tagmap.sel(level=slice(4, 6))[1:] - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6))[1:].values
# test.to_netcdf('scratch/test/test.nc')
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(4, 6))[1:].values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 6))[1:].values[wheremax]


# exp_org_o[expid[i]]['echam'].wind10.values.shape


#-------- check differences
stats.describe(tagmap.tagmap[3:6, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:6, :, :], axis=None)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check scale of tagmap based on sincoslon

min_sincoslon = -1
max_sincoslon = 1
b_lon = np.broadcast_to(exp_org_o[expid[i]]['echam'].lon.values[None, None, :], exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_sincoslon_tagmap_a.nc')

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
ptagmap.tagmap.sel(level=14).values[island] = 1

# scaled sin lon
ptagmap.tagmap.sel(level=12).values[isocean] = ((np.sin(b_lon * np.pi / 180.) - min_sincoslon)/(max_sincoslon - min_sincoslon))[isocean]

# scaled cos lon
ptagmap.tagmap.sel(level=15).values[isocean] = ((np.cos(b_lon * np.pi / 180.) - min_sincoslon)/(max_sincoslon - min_sincoslon))[isocean]

# complementary set
ptagmap.tagmap.sel(level=13).values[isocean] = 1 - ptagmap.tagmap.sel(level=12).values[isocean]
ptagmap.tagmap.sel(level=16).values[isocean] = 1 - ptagmap.tagmap.sel(level=15).values[isocean]

where_sea_ice = (exp_org_o[expid[i]]['echam'].seaice[:, :, :].values > 0)
ptagmap.tagmap.sel(level=11).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=12).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=13).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=14).values[where_sea_ice] = 1
ptagmap.tagmap.sel(level=15).values[where_sea_ice] = 0
ptagmap.tagmap.sel(level=16).values[where_sea_ice] = 0



#-------- check tagmap complementarity
np.max(abs(ptagmap.tagmap.sel(level=slice(11, 16)).sum(axis=1) - 2)) == 0

#-------- check differences
stats.describe(ptagmap.tagmap.sel(level=slice(11, 16)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(11, 16)).values, axis=None)

test = ptagmap.tagmap.sel(level=slice(11, 16)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(11, 16)).values
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test.values[wheremax]
ptagmap.tagmap.sel(level=slice(11, 16)).values[wheremax]
exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(11, 16)).values[wheremax]

#-------- check differences
stats.describe(tagmap.tagmap[10:16, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 10:16, :, :], axis=None)


'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check pi_geo_tagmap

b_lat = np.broadcast_to(exp_org_o[expid[i]]['echam'].lat.values[None, :, None], exp_org_o[expid[i]]['wiso'].tagmap[:, 0].shape)

tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_geo_tagmap.nc')

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


# NH land
ptagmap.tagmap.sel(level=4).values[island & (b_lat > 0)] = 1

# SH land, excl. Antarctica
ptagmap.tagmap.sel(level=5).values[island & (b_lat < 0) & (b_lat > -60)] = 1

# Antarctica
ptagmap.tagmap.sel(level=6).values[island & (b_lat < -60)] = 1

# NH ocean, excl. sea ice
ptagmap.tagmap.sel(level=7).values[isocean & (b_lat > 0) & (exp_org_o[expid[i]]['echam'].seaice[:].values == 0)] = 1

# NH sea ice
ptagmap.tagmap.sel(level=8).values[isocean & (b_lat > 0) & (exp_org_o[expid[i]]['echam'].seaice[:].values > 0)] = 1

# SH ocean, excl. sea ice
ptagmap.tagmap.sel(level=9).values[isocean & (b_lat < 0) & (exp_org_o[expid[i]]['echam'].seaice[:].values == 0)] = 1

# SH sea ice
ptagmap.tagmap.sel(level=10).values[isocean & (b_lat < 0) & (exp_org_o[expid[i]]['echam'].seaice[:].values > 0)] = 1


#-------- check tagmap complementarity

np.max(abs(ptagmap.tagmap.sel(level=slice(4, 10)).sum(axis=1) - 1)) == 0


#-------- check differences

stats.describe(ptagmap.tagmap.sel(level=slice(4, 10)) - exp_org_o[expid[i]]['wiso'].tagmap.sel(wisotype=slice(4, 10)).values, axis=None)


#-------- check differences
stats.describe(tagmap.tagmap[3:10, :, :] - exp_org_o[expid[i]]['wiso'].tagmap.values[0, 3:10, :, :], axis=None)

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



