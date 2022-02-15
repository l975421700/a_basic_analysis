

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

mon_sst = {}
ann_sst = {}
am_sst = {}

#### pi_final_qg

pi_final_qg_sst = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_200001_202912_sst.nc',)

mon_sst['pi_final_qg'] = pi_final_qg_sst.sst
ann_sst['pi_final_qg'] = mon_sst['pi_final_qg'].groupby('time.year').mean(dim='time')
# ann_sst['pi_final_qg'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_ann_sst.nc')

am_sst['pi_final_qg'] = mon_sst['pi_final_qg'].mean(axis=0)
# am_sst['pi_final_qg'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_am_sst.nc')


#### pi_final_qg, regridded

pi_final_qg_sst_res1 = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_200001_202912_sst_res1.nc',)

mon_sst['pi_final_qg_res1'] = pi_final_qg_sst_res1.sst
ann_sst['pi_final_qg_res1'] = mon_sst['pi_final_qg_res1'].groupby('time.year').mean(dim='time')
# ann_sst['pi_final_qg_res1'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_ann_sst_res1.nc')

am_sst['pi_final_qg_res1'] = mon_sst['pi_final_qg_res1'].mean(axis=0)
# am_sst['pi_final_qg_res1'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_am_sst_res1.nc')


#### pi_final

pi_final_sst = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_300001_310012_sst.nc',)

mon_sst['pi_final'] = pi_final_sst.sst
ann_sst['pi_final'] = mon_sst['pi_final'].groupby('time.year').mean(dim='time')
# ann_sst['pi_final'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_ann_sst.nc')

am_sst['pi_final'] = mon_sst['pi_final'].mean(axis=0)
# am_sst['pi_final'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_am_sst.nc')


'''
# check
ddd = (pi_final_qg_sst.sst)
i = 200
j = 20
ddd[i, j]
mon_sst['pi_final_qg'][i, j]

np.max(np.abs(ann_sst['pi_final_qg'].mean(axis=0) - am_sst['pi_final_qg']))

np.max(np.abs(ann_sst['pi_final_qg_res1'].mean(axis=0) - am_sst['pi_final_qg_res1']))

np.max(np.abs(ann_sst['pi_final'].mean(axis=0) - am_sst['pi_final']))
'''

# endregion
# =============================================================================


# =============================================================================
# region check SST


tri2plot = mesh2plot()

ann_sst = {}
ann_sst['pi_final_qg'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_ann_sst.nc')
ann_sst['pi_final_qg_res1'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_ann_sst_res1.nc')
ann_sst['pi_final'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_ann_sst.nc')

am_sst = {}
am_sst['pi_final_qg'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_am_sst.nc')
am_sst['pi_final_qg_res1'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_qg_am_sst_res1.nc')
am_sst['pi_final'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/pi_final_am_sst.nc')

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.tripcolor(
    tri2plot['x2'], tri2plot['y2'],
    tri2plot['elem2plot'], am_sst['pi_final_qg'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, transform=ccrs.PlateCarree(),
    edgecolors="k", lw=0.01,)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.4_sst/6.0.0.4.0_global am_sst awi-esm-2.1-wiso pi_final_qg 30y.png', dpi=1200)


pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_sst['pi_final_qg_res1'].lon,
    am_sst['pi_final_qg_res1'].lat,
    am_sst['pi_final_qg_res1'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean regridded SST [$°C$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.4_sst/6.0.0.4.1_global am_sst regridded awi-esm-2.1-wiso pi_final_qg 30y.png', dpi=1200)


pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.tripcolor(
    tri2plot['x2'], tri2plot['y2'],
    tri2plot['elem2plot'], am_sst['pi_final'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, transform=ccrs.PlateCarree(),
    edgecolors="k", lw=0.01,)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nAWI-ESM-2-1-wiso: pi_final 101y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.4_sst/6.0.0.4.2_global am_sst awi-esm-2.1-wiso pi_final 101y.png', dpi=1200)


# check absolute differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.tripcolor(
    tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
    am_sst['pi_final_qg'] - am_sst['pi_final'],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, transform=ccrs.PlateCarree(),
    edgecolors="k", lw=0.01,)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean SST difference [$°C$]\nAWI-ESM-2-1-wiso: pi_final_qg 30y - pi_final 101y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.4_sst/6.0.0.4.3_global am_sst awi-esm-2.1-wiso pi_final_qg 30y - pi_final 101y.png', dpi=1200)


# check relative differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.tripcolor(
    tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
    (am_sst['pi_final_qg']-am_sst['pi_final'])/ann_sst['pi_final'].std(axis=0),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, transform=ccrs.PlateCarree(),
    edgecolors="k", lw=0.01,)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean SST relative difference [$-$]\nAWI-ESM-2-1-wiso: (pi_final_qg 30y - pi_final 101y)/std(pi_final 101y)',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.4_sst/6.0.0.4.4_global am_sst awi-esm-2.1-wiso (pi_final_qg 30y - pi_final 101y)_std(pi_final 101y).png', dpi=1200)


'''
stats.describe(am_sst['pi_final_qg']+ 273.15)

np.max(np.abs(ann_sst['pi_final_qg'].mean(axis=0) - am_sst['pi_final_qg']))
np.max(np.abs(ann_sst['pi_final_qg_res1'].mean(axis=0) - am_sst['pi_final_qg_res1']))
np.max(np.abs(ann_sst['pi_final'].mean(axis=0) - am_sst['pi_final']))
# check
import pyfesom2 as pf
mesh = pf.load_mesh('/work/ollie/qigao001/startdump/core2/')
pf.tplot(
    mesh, am_sst['pi_final_qg'],
    cmap=cmp_cmap, levels=[0, 32.001, len(pltlevel)],
    ptype='tri', box=[-180, 180, -90, 90], mapproj='pc', lw=0.01)
plt.savefig('figures/0_test/trial.png', dpi=1200)

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import delta_wiso

mon_wisoaprt_d = {}
ann_wisoaprt_d = {}
am_wisoaprt_d = {}

#### pi_final_qg
pi_final_qg_wiso_d_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_wiso_d.nc',)
pi_final_qg_wiso_d = pi_final_qg_wiso_d_org.assign_coords(
    time = pd.to_datetime(pi_final_qg_wiso_d_org.time.astype(str),
                          format='%Y%m%d.%f'))

mon_wisoaprt_d['pi_final_qg'] = pi_final_qg_wiso_d.wisoaprt_d

ann_wisoaprt_d['pi_final_qg'] = mon_wisoaprt_d['pi_final_qg'].groupby(
    'time.year').mean(dim='time')
# ann_wisoaprt_d['pi_final_qg'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_ann_wisoaprt_d.nc')

am_wisoaprt_d['pi_final_qg'] = mon_wisoaprt_d['pi_final_qg'].mean(axis=0)
# am_wisoaprt_d['pi_final_qg'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_am_wisoaprt_d.nc')


#### pi_final
pi_final_wiso_d_org = xr.open_dataset(
    'output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_300001_310012_wiso_d.nc',)
pi_final_wiso_d = pi_final_wiso_d_org.assign_coords(
    time = xr.cftime_range(
        start=str(pi_final_wiso_d_org.time[0].values)[0:8],
        end=str(pi_final_wiso_d_org.time[-1].values)[0:8],
        freq="M",))

mon_wisoaprt_d['pi_final'] = pi_final_wiso_d.wisoaprt_d
ann_wisoaprt_d['pi_final'] = mon_wisoaprt_d['pi_final'].groupby(
    'time.year').mean(dim='time')
# ann_wisoaprt_d['pi_final'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_ann_wisoaprt_d.nc')

am_wisoaprt_d['pi_final'] = mon_wisoaprt_d['pi_final'].mean(axis=0)
# am_wisoaprt_d['pi_final'].to_netcdf('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_am_wisoaprt_d.nc')


'''
np.max(np.abs(ann_wisoaprt_d['pi_final_qg'].mean(axis=0) - am_wisoaprt_d['pi_final_qg']))
stats.describe(am_wisoaprt_d['pi_final_qg'], axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region check delta_wiso

ann_wisoaprt_d = {}
ann_wisoaprt_d['pi_final_qg'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_ann_wisoaprt_d.nc')
ann_wisoaprt_d['pi_final'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_ann_wisoaprt_d.nc')

am_wisoaprt_d = {}
am_wisoaprt_d['pi_final_qg'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_am_wisoaprt_d.nc')
am_wisoaprt_d['pi_final'] = xr.open_dataarray('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_am_wisoaprt_d.nc')

pltlevel = np.arange(-50, 0.01, 0.1)
pltticks = np.arange(-50, 0.01, 5)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_wisoaprt_d['pi_final_qg'].lon,
    am_wisoaprt_d['pi_final_qg'].lat,
    am_wisoaprt_d['pi_final_qg'].sel(lev = 2),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ [‰]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5,)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.5_wiso/6.0.0.5.0_global am_delta18O awi-esm-2.1-wiso pi_final_qg 30y.png',)


pltlevel = np.arange(-50, 0.01, 0.1)
pltticks = np.arange(-50, 0.01, 5)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_wisoaprt_d['pi_final'].lon,
    am_wisoaprt_d['pi_final'].lat,
    am_wisoaprt_d['pi_final'].sel(lev = 2),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ [‰]\nAWI-ESM-2-1-wiso: pi_final 101y',
    linespacing=1.5,)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.5_wiso/6.0.0.5.1_global am_delta18O awi-esm-2.1-wiso pi_final 101y.png',)


# check absolute differences
pltlevel = np.arange(-2, 2.01, 0.01)
pltticks = np.arange(-2, 2.01, 0.5)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_wisoaprt_d['pi_final'].lon,
    am_wisoaprt_d['pi_final'].lat,
    am_wisoaprt_d['pi_final_qg'].sel(lev = 2) - am_wisoaprt_d['pi_final'].sel(lev = 2),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True,
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ difference [‰]\nAWI-ESM-2-1-wiso: pi_final_qg 30y - pi_final 101y',
    linespacing=1.5)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.5_wiso/6.0.0.5.2_global am_delta18O awi-esm-2.1-wiso pi_final_qg 30y - pi_final 101y.png',)


# check relative differences
pltlevel = np.arange(-1, 1.01, 0.01)
pltticks = np.arange(-1, 1.01, 0.2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_wisoaprt_d['pi_final'].lon,
    am_wisoaprt_d['pi_final'].lat,
    (am_wisoaprt_d['pi_final_qg'].sel(lev = 2) - am_wisoaprt_d['pi_final'].sel(lev = 2))/ann_wisoaprt_d["pi_final"].sel(lev=2).std(axis=0),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True,
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ relative difference [$-$]\nAWI-ESM-2-1-wiso: (pi_final_qg 30y - pi_final 101y)/std(pi_final 101y)',
    linespacing=1.5)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.5_wiso/6.0.0.5.2_global am_delta18O awi-esm-2.1-wiso (pi_final_qg 30y - pi_final 101y)_std(pi_final 101y).png',)


'''
np.max(np.abs(ann_wisoaprt_d['pi_final_qg'].mean(axis=0) - am_wisoaprt_d['pi_final_qg']))

ann_wisoaprt_d['pi_final_qg_cdo'] = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_ann_wisoaprt_d.nc')
am_wisoaprt_d['pi_final_qg_cdo'] = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/echam/pi_final_qg_200001_202912_am_wisoaprt_d.nc')
np.max(np.abs(ann_wisoaprt_d['pi_final_qg_cdo'].wisoaprt_d.mean(axis=0) - am_wisoaprt_d['pi_final_qg_cdo'].wisoaprt_d))



pltlevel = np.arange(-50, 0.01, 0.1)
pltticks = np.arange(-50, 0.01, 5)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_wisoaprt_d['pi_final_qg'].lon,
    am_wisoaprt_d['pi_final_qg'].lat,
    ann_wisoaprt_d['pi_final_qg'].mean(axis=0).sel(lev = 2),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend='both')

cbar.ax.set_xlabel(
    'Annual mean $\delta^{18}O$ [‰]\nAWI-ESM-2-1-wiso: pi_final_qg 30y',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial.png',)

stats.describe(np.abs(ann_wisoaprt_d['pi_final_qg'].mean(axis=0).sel(lev = 3) - am_wisoaprt_d['pi_final_qg'].sel(lev = 3)), axis=None)

'''
# endregion
# =============================================================================

