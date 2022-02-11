

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
# region import precipitation

mon_pre = {}
ann_pre = {}
am_pre = {}


#### pi_final_qg

pi_final_qg_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_echam.nc',)

pi_final_qg_echam = pi_final_qg_echam_org.assign_coords(
    time = pd.to_datetime(pi_final_qg_echam_org.time.astype(str),
                          format='%Y%m%d.%f'))

mon_pre['pi_final_qg_echam'] = \
    (pi_final_qg_echam.var142 + pi_final_qg_echam.var143) * \
        np.tile(month_days, int(pi_final_qg_echam.time.shape[0]/12))[
            :, None, None] * 24 * 3600

ann_pre['pi_final_qg_echam'] = (
    mon_pre['pi_final_qg_echam'].groupby('time.year').sum(dim='time'))
# ann_pre['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_pre.nc')

am_pre['pi_final_qg_echam'] = mon_pre['pi_final_qg_echam'].mean(axis=0) * 12
# am_pre['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_pre.nc')


#### pi_final_101

pi_final_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_300001_310012_echam.nc',
    decode_times=False)

pi_final_echam = pi_final_echam_org.assign_coords(
    time = xr.cftime_range(
        start=str(pi_final_echam_org.time[0].values)[0:8],
        end=str(pi_final_echam_org.time[-1].values)[0:8],
        freq="M",))

mon_pre['pi_final_echam'] = \
(pi_final_echam.var142 + pi_final_echam.var143) * \
    np.tile(month_days, int(pi_final_echam.time.shape[0]/12))[
        :, None, None] * 24 * 3600

ann_pre['pi_final_echam'] = (
    mon_pre['pi_final_echam'].groupby('time.year').sum(dim='time'))
# ann_pre['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_pre.nc')

am_pre['pi_final_echam'] = mon_pre['pi_final_echam'].mean(axis=0) * 12
# am_pre['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_pre.nc')



'''
time = pd.to_datetime(pi_final_echam.time.astype(str), format='%Y%m%d.%f')
(pi_final_qg_echam.var142.values == pi_final_qg_echam_org.var142.values).all()

# check
ddd = (pi_final_qg_echam.var143 + pi_final_qg_echam.var142)
ccc = np.tile(month_days, int(pi_final_qg_echam.time.shape[0]/12))
i = 200
j = 20
k = 30
ddd[i, j, k] * ccc[i] * 24 * 3600
mon_pre['pi_final_qg_echam'][i, j, k]

(pi_final_echam.var142.values == pi_final_echam_org.var142.values).all()

np.max(np.abs(ann_pre['pi_final_qg_echam'].mean(axis=0) - am_pre['pi_final_qg_echam']))

# check
ddd = (pi_final_echam.var143 + pi_final_echam.var142)
ccc = np.tile(month_days, int(pi_final_echam.time.shape[0]/12))
i = 200
j = 20
k = 30
ddd[i, j, k] * ccc[i] * 24 * 3600
mon_pre['pi_final_echam'][i, j, k]

np.max(np.abs(ann_pre['pi_final_echam'].mean(axis=0) - am_pre['pi_final_echam']))
'''
# endregion
# =============================================================================


# =============================================================================
# region check precipitation

ann_pre = {}
ann_pre['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_pre.nc')
ann_pre['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_pre.nc')

am_pre = {}
am_pre['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_pre.nc')
am_pre['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_pre.nc')


quick_var_plot(
    var=am_pre['pi_final_qg_echam'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso, pi_final_qg 30y',
    outputfile='figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.1_pre/6.0.0.1.0_global am_pre awi-esm-2.1-wiso pi_final_qg 30y.png',
)

quick_var_plot(
    var=am_pre['pi_final_echam'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso, pi_final 101y',
    outputfile='figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.1_pre/6.0.0.1.1_global am_pre awi-esm-2.1-wiso pi_final 101y.png',
)


# check relative differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pre['pi_final_qg_echam'].lon,
    am_pre['pi_final_qg_echam'].lat,
    (am_pre['pi_final_qg_echam'] - am_pre['pi_final_echam'])/ann_pre["pi_final_echam"].std(axis=0),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$-$]\nAWI-ESM-2-1-wiso: (pi_final_qg 30y - pi_final 101y)/std(pi_final 101y)',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.1_pre/6.0.0.1.3_global am_pre awi-esm-2.1-wiso (pi_final_qg 30y - pi_final 101y)_std(pi_final 101y).png',)


# check absolute differences
pltlevel = np.arange(-200, 200.01, 0.2)
pltticks = np.arange(-200, 200.01, 40)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pre['pi_final_qg_echam'].lon,
    am_pre['pi_final_qg_echam'].lat,
    am_pre['pi_final_qg_echam'] - am_pre['pi_final_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y - pi_final 101y',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.1_pre/6.0.0.1.2_global am_pre awi-esm-2.1-wiso pi_final_qg 30y - pi_final 101y.png',)

'''
np.max(np.abs(ann_pre['pi_final_qg_echam'].mean(axis=0) - am_pre['pi_final_qg_echam']))
np.max(np.abs(ann_pre['pi_final_echam'].mean(axis=0) - am_pre['pi_final_echam']))
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import evaporation

mon_evp = {}
ann_evp = {}
am_evp = {}

#### pi_final_qg

pi_final_qg_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_echam.nc',)

pi_final_qg_echam = pi_final_qg_echam_org.assign_coords(
    time = pd.to_datetime(pi_final_qg_echam_org.time.astype(str),
                          format='%Y%m%d.%f'))

mon_evp['pi_final_qg_echam'] = \
    pi_final_qg_echam.var182 * \
    np.tile(month_days, int(pi_final_qg_echam.time.shape[0]/12))[
                               :, None, None] * 24 * 3600 * (-1)

ann_evp['pi_final_qg_echam'] = (
    mon_evp['pi_final_qg_echam'].groupby('time.year').sum(dim='time'))
# ann_evp['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_evp.nc')

am_evp['pi_final_qg_echam'] = mon_evp['pi_final_qg_echam'].mean(axis=0) * 12
# am_evp['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_evp.nc')


#### pi_final_101

pi_final_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_300001_310012_echam.nc',)

pi_final_echam = pi_final_echam_org.assign_coords(
    time = xr.cftime_range(
        start=str(pi_final_echam_org.time[0].values)[0:8],
        end=str(pi_final_echam_org.time[-1].values)[0:8],
        freq="M",))


mon_evp['pi_final_echam'] = \
    pi_final_echam.var182 * \
    np.tile(month_days, int(pi_final_echam.time.shape[0]/12))[
                               :, None, None] * 24 * 3600 * (-1)

ann_evp['pi_final_echam'] = (
    mon_evp['pi_final_echam'].groupby('time.year').sum(dim='time'))
# ann_evp['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_evp.nc')

am_evp['pi_final_echam'] = mon_evp['pi_final_echam'].mean(axis=0) * 12
# am_evp['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_evp.nc')


'''
# check
ddd = (pi_final_qg_echam.var182)
ccc = np.tile(month_days, int(pi_final_qg_echam.time.shape[0]/12))
i = 200
j = 20
k = 30
ddd[i, j, k] * ccc[i] * 24 * 3600
mon_evp['pi_final_qg_echam'][i, j, k]

np.max(np.abs(ann_evp['pi_final_qg_echam'].mean(axis=0) - am_evp['pi_final_qg_echam']))

'''
# endregion
# =============================================================================


# =============================================================================
# region check evaporation

ann_evp = {}
ann_evp['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_evp.nc')
ann_evp['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_evp.nc')

am_evp = {}
am_evp['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_evp.nc')
am_evp['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_evp.nc')

# stats.describe(am_evp['pi_final_qg_echam'], axis=None)

quick_var_plot(
    var=am_evp['pi_final_qg_echam'], varname='evp',
    xlabel='Annual mean evaporation [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso, pi_final_qg 30y',
    outputfile='figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.2_evp/6.0.0.2.0_global am_evp awi-esm-2.1-wiso pi_final_qg 30y.png',
)

quick_var_plot(
    var=am_evp['pi_final_echam'], varname='evp',
    xlabel='Annual mean evaporation [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso, pi_final 101y',
    outputfile='figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.2_evp/6.0.0.2.1_global am_evp awi-esm-2.1-wiso pi_final 101y.png',
)


# check relative differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_evp['pi_final_qg_echam'].lon,
    am_evp['pi_final_qg_echam'].lat,
    (am_evp['pi_final_qg_echam'] - am_evp['pi_final_echam'])/ann_evp["pi_final_echam"].std(axis=0),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean evaporation relative difference [$-$]\nAWI-ESM-2-1-wiso: (pi_final_qg 30y - pi_final 101y)/std(pi_final 101y)',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.2_evp/6.0.0.2.3_global am_evp awi-esm-2.1-wiso (pi_final_qg 30y - pi_final 101y)_std(pi_final 101y).png',)


# check absolute differences
pltlevel = np.arange(-200, 200.01, 0.2)
pltticks = np.arange(-200, 200.01, 40)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_evp['pi_final_qg_echam'].lon,
    am_evp['pi_final_qg_echam'].lat,
    am_evp['pi_final_qg_echam'] - am_evp['pi_final_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean evaporation difference [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y - pi_final 101y',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.2_evp/6.0.0.2.2_global am_evp awi-esm-2.1-wiso pi_final_qg 30y - pi_final 101y.png',)


'''
np.max(np.abs(ann_evp['pi_final_qg_echam'].mean(axis=0) - am_evp['pi_final_qg_echam']))
np.max(np.abs(ann_evp['pi_final_echam'].mean(axis=0) - am_evp['pi_final_echam']))

quick_var_plot(
    var=am_evp['pi_final_echam'], varname='evp', whicharea='SH',
    xlabel='Annual mean evaporation [$mm\;yr^{-1}$]\nAWI-ESM-2-1-wiso, pi_final_qg 30y',
    outputfile='figures/0_test/trial.png',
)

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check ratio between precipitation and evaporation

am_pre = {}
am_pre['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_pre.nc')
am_pre['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_pre.nc')

am_evp = {}
am_evp['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_evp.nc')
am_evp['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_evp.nc')

# stats.describe(am_pre['pi_final_qg_echam'], axis=None)
# stats.describe(am_evp['pi_final_qg_echam'], axis=None)

pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_evp['pi_final_qg_echam'].lon,
    am_evp['pi_final_qg_echam'].lat,
    am_evp['pi_final_qg_echam']/am_pre['pi_final_qg_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean evaporation/precipitation [$-$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.2_evp/6.0.0.2.4_global am_evp_pre awi-esm-2.1-wiso pi_final_qg 30y.png',)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import 2m temperature

mon_temp2 = {}
ann_temp2 = {}
am_temp2 = {}


#### pi_final_qg

pi_final_qg_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_echam.nc',)

pi_final_qg_echam = pi_final_qg_echam_org.assign_coords(
    time = pd.to_datetime(pi_final_qg_echam_org.time.astype(str),
                          format='%Y%m%d.%f'))

mon_temp2['pi_final_qg_echam'] = pi_final_qg_echam.var167

ann_temp2['pi_final_qg_echam'] = mon_temp2['pi_final_qg_echam'].groupby('time.year').mean(dim='time')
# ann_temp2['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_temp2.nc')

am_temp2['pi_final_qg_echam'] = mon_temp2['pi_final_qg_echam'].mean(axis=0)
# am_temp2['pi_final_qg_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_temp2.nc')


#### pi_final_101

pi_final_echam_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_300001_310012_echam.nc',
    decode_times=False)

pi_final_echam = pi_final_echam_org.assign_coords(
    time = xr.cftime_range(
        start=str(pi_final_echam_org.time[0].values)[0:8],
        end=str(pi_final_echam_org.time[-1].values)[0:8],
        freq="M",))

mon_temp2['pi_final_echam'] = pi_final_echam.var167

ann_temp2['pi_final_echam'] = mon_temp2['pi_final_echam'].groupby('time.year').mean(dim='time')
# ann_temp2['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_temp2.nc')

am_temp2['pi_final_echam'] = mon_temp2['pi_final_echam'].mean(axis=0)
# am_temp2['pi_final_echam'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_temp2.nc')


'''
# check
ddd = (pi_final_qg_echam.var167)
i = 200
j = 20
k = 30
ddd[i, j, k]
mon_temp2['pi_final_qg_echam'][i, j, k]

np.max(np.abs(ann_temp2['pi_final_qg_echam'].mean(axis=0) - am_temp2['pi_final_qg_echam']))

# check
ddd = (pi_final_echam.var167)
i = 200
j = 20
k = 30
ddd[i, j, k]
mon_temp2['pi_final_echam'][i, j, k]

np.max(np.abs(ann_temp2['pi_final_echam'].mean(axis=0) - am_temp2['pi_final_echam']))
'''
# endregion
# =============================================================================


# =============================================================================
# region check 2m temperature

ann_temp2 = {}
ann_temp2['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_ann_temp2.nc')
ann_temp2['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_ann_temp2.nc')

am_temp2 = {}
am_temp2['pi_final_qg_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_echam_am_temp2.nc')
am_temp2['pi_final_echam'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_echam_am_temp2.nc')

# stats.describe(am_temp2['pi_final_qg_echam'], axis=None)


pltlevel = np.arange(220, 300.01, 0.2)
pltticks = np.arange(220, 300.01, 10)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_temp2['pi_final_qg_echam'].lon,
    am_temp2['pi_final_qg_echam'].lat,
    am_temp2['pi_final_qg_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean 2m temperature [$K$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.3_temp2/6.0.0.3.0_global am_temp2 awi-esm-2.1-wiso pi_final_qg 30y.png',)


pltlevel = np.arange(220, 300.01, 0.2)
pltticks = np.arange(220, 300.01, 10)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_temp2['pi_final_echam'].lon,
    am_temp2['pi_final_echam'].lat,
    am_temp2['pi_final_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean 2m temperature [$K$]\nAWI-ESM-2-1-wiso: pi_final 101y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.3_temp2/6.0.0.3.1_global am_temp2 awi-esm-2.1-wiso pi_final 101y.png',)


# check absolute differences
pltlevel = np.arange(-2, 2.01, 0.01)
pltticks = np.arange(-2, 2.01, 1)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_temp2['pi_final_qg_echam'].lon,
    am_temp2['pi_final_qg_echam'].lat,
    am_temp2['pi_final_qg_echam'] - am_temp2['pi_final_echam'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean 2m temperature difference [$K$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y - pi_final 101y',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.3_temp2/6.0.0.3.2_global am_temp2 awi-esm-2.1-wiso pi_final_qg 30y - pi_final 101y.png',)


# check relative differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_temp2['pi_final_qg_echam'].lon,
    am_temp2['pi_final_qg_echam'].lat,
    (am_temp2['pi_final_qg_echam'] - am_temp2['pi_final_echam'])/ann_temp2['pi_final_echam'].std(axis=0),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel('Annual mean 2m temperature relative difference [$-$]\nAWI-ESM-2-1-wiso: (pi_final_qg 30y - pi_final 101y)/std(pi_final 101y)',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.3_temp2/6.0.0.3.3_global am_temp2 awi-esm-2.1-wiso (pi_final_qg 30y - pi_final 101y)_std(pi_final 101y).png',)







'''
np.max(np.abs(ann_temp2['pi_final_qg_echam'].mean(axis=0) - am_temp2['pi_final_qg_echam']))
np.max(np.abs(ann_temp2['pi_final_echam'].mean(axis=0) - am_temp2['pi_final_echam']))
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import SST



# endregion
# =============================================================================



