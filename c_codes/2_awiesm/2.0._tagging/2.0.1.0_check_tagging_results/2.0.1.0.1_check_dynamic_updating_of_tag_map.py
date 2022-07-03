

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
# =============================================================================
# region import output pi_echam6_1d*

exp_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    'pi_echam6_1d_159_3.59',
    ]

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    
    exp_org_o[expid[i]] = {}
    
    ## echam
    exp_org_o[expid[i]]['echam'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check update of tagmap based on tsw

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_15_0.nc')
sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values


for j in np.arange(4, 3+len(sstbins)):
    # j=4
    ptagmap.tagmap[:, j, :, :].values[np.where((ptagmap.tagmap[:, 3, :, :] == 0) & ((exp_org_o[expid[i]]['echam'].tsw - 273.15) > sstbins[j-4]) & ((exp_org_o[expid[i]]['echam'].tsw - 273.15) <= sstbins[j-3]))] = 1

# tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)
# bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()
# number of grid cells that differ
(ptagmap.tagmap != exp_org_o[expid[i]]['wiso'].tagmap.values).sum()

i1, i2, i3, i4 = np.where(ptagmap.tagmap != exp_org_o[expid[i]]['wiso'].tagmap.values)
exp_org_o[expid[i]]['echam'].tsw.values[i1, i3, i4] - 273.15


'''
# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check update of tagmap based on rh2m

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_7_0.nc')
rh2mbins = np.concatenate((np.array([0]), np.arange(0.72, 0.841, 0.02), np.array([2])))


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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values


for j in np.arange(4, 3+len(rh2mbins)):
    # j=4
    ptagmap.tagmap[:, j, :, :].values[np.where((ptagmap.tagmap[:, 3, :, :] == 0) & (exp_org_o[expid[i]]['echam'].rh2m > rh2mbins[j-4]) & (exp_org_o[expid[i]]['echam'].rh2m <= rh2mbins[j-3]))] = 1

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')

# inequal tagmap
((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[0, 4, :, :] != 0).sum()

# rh2m at inequal tagmap
exp_org_o[expid[i]]['echam'].rh2m.values[np.where((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[:, 4, :, :] != 0)]

# statistics of rh2m at inequal tagmap
stats.describe(exp_org_o[expid[i]]['echam'].rh2m.values[np.where((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[:, 9, :, :] != 0)], axis=None)

'''
exp_org_o[expid[i]]['wiso'].tagmap
exp_org_o[expid[i]]['echam'].tsw
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check update of tagmap based on wind10m

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_8_0.nc')
wind10bins = np.concatenate((np.array([0]), np.arange(4, 11.1, 1), np.array([100])))


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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values


for j in np.arange(4, 3+len(wind10bins)):
    # j=4
    ptagmap.tagmap[:, j, :, :].values[np.where((ptagmap.tagmap[:, 3, :, :] == 0) & (exp_org_o[expid[i]]['echam'].wind10 > wind10bins[j-4]) & (exp_org_o[expid[i]]['echam'].wind10 <= wind10bins[j-3]))] = 1

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')

# inequal tagmap
((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[0, 4, :, :] != 0).sum()

# wind10 at inequal tagmap
exp_org_o[expid[i]]['echam'].wind10.values[np.where((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[:, 4, :, :] != 0)]

# statistics of wind10 at inequal tagmap
stats.describe(exp_org_o[expid[i]]['echam'].wind10.values[np.where((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[:, 9, :, :] != 0)], axis=None)

'''
exp_org_o[expid[i]]['wiso'].tagmap
exp_org_o[expid[i]]['echam'].tsw
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check scale of tagmap based on lat

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values

ptagmap.tagmap[:, 4, :, :] = ((exp_org_o[expid[i]]['wiso'].tagmap.lat.values + 90)/180)[None, :, None]
ptagmap.tagmap[:, 4, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

ptagmap.tagmap[:, 5, :, :] = 1 - ptagmap.tagmap[:, 4, :, :]
ptagmap.tagmap[:, 5, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check scale of tagmap based on lon

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values

ptagmap.tagmap[:, 4, :, :] = (exp_org_o[expid[i]]['wiso'].tagmap.lon.values /360)[None, None, :]
ptagmap.tagmap[:, 4, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

ptagmap.tagmap[:, 5, :, :] = 1 - ptagmap.tagmap[:, 4, :, :]
ptagmap.tagmap[:, 5, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check scale of tagmap based on sst

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values

ptagmap.tagmap[:, 4, :, :] = (exp_org_o[expid[i]]['echam'].tsw.values - 260) / (310 - 260)
ptagmap.tagmap[:, 4, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

ptagmap.tagmap[:, 5, :, :] = 1 - ptagmap.tagmap[:, 4, :, :]
ptagmap.tagmap[:, 5, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)
# check bit identity
# (ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()
# (ptagmap.tagmap != exp_org_o[expid[i]]['wiso'].tagmap.values).sum()

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)
np.max(abs(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values))


'''
i1, i2, i3, i4 = np.where(abs(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values) == np.max(abs(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)))
abs(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values).values[i1[0], i2[0], i3[0], i4[0]]

ptagmap.tagmap[i1[0], i2[0], i3[0], i4[0]].values
exp_org_o[expid[i]]['wiso'].tagmap[i1[0], i2[0], i3[0], i4[0]].values

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
test.to_netcdf('/work/ollie/qigao001/0_backup/test1.nc')

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check scale of tagmap based on rh2m

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values

ptagmap.tagmap[:, 4, :, :] = (exp_org_o[expid[i]]['echam'].rh2m.values) / 2
ptagmap.tagmap[:, 4, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

ptagmap.tagmap[:, 5, :, :] = 1 - ptagmap.tagmap[:, 4, :, :]
ptagmap.tagmap[:, 5, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe(ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values, axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check scale of tagmap based on wind10

tagmap = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')

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

ptagmap.tagmap[:, 0:3, :, :] = 1
ptagmap.tagmap[:, 3, :, :] = tagmap.tagmap[3, :, :].values

ptagmap.tagmap[:, 4, :, :] = exp_org_o[expid[i]]['echam'].wind10.values / 50
ptagmap.tagmap[:, 4, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

ptagmap.tagmap[:, 5, :, :] = 1 - ptagmap.tagmap[:, 4, :, :]
ptagmap.tagmap[:, 5, :, :].values[np.where(ptagmap.tagmap[:, 3, :, :] == 1)] = 0

# check bit identity
(ptagmap.tagmap == exp_org_o[expid[i]]['wiso'].tagmap.values).all()

# check tagmap complementarity
stats.describe(ptagmap.tagmap[:, 3:, :, :].sum(axis=1), axis=None)

# check differences
stats.describe((ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values)[1:, :, :, :], axis=None)

# output and plot differences
test = ptagmap.tagmap - exp_org_o[expid[i]]['wiso'].tagmap.values
(test != 0).sum()
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


'''
'''
# endregion
# =============================================================================








