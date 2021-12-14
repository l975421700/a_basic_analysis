

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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.colors import ListedColormap

# self defined
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    plot_maxmin_points,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    regrid,
)

from a00_basic_analysis.b_module.namelist import (
    month_days,
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean wind in HadGEM3-GC31-LL, historical, r1i1p1f3

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'Amon/'
variable = 'sfcWind/'
grid = 'gn/'
version = 'v20190624/'

variable1 = 'uas/'
variable2 = 'vas/'

si10_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))
si10_hg3_ll_hi_r1 = xr.open_mfdataset(
    si10_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
u10_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable1 + grid + version + '*.nc',
)))
u10_hg3_ll_hi_r1 = xr.open_mfdataset(
    u10_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
v10_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable2 + grid + version + '*.nc',
)))
v10_hg3_ll_hi_r1 = xr.open_mfdataset(
    v10_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_si10_hg3_ll_hi_r1 = si10_hg3_ll_hi_r1.sfcWind.sel(time=slice(
    '1979-01-01', '2014-12-30')).mean(axis=0)
am_si10_hg3_ll_hi_r1_80 = si10_hg3_ll_hi_r1.sfcWind.sel(time=slice(
    '1980-01-01', '2014-12-30')).mean(axis=0)

am_si10_hg3_ll_hi_r1_rg1 = regrid(am_si10_hg3_ll_hi_r1)

'''
# check
pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_hg3_ll_hi_r1_rg1.lon,
    am_si10_hg3_ll_hi_r1_rg1.lat,
    am_si10_hg3_ll_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nregridded HadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial.png',)

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean wind in HadGEM3-GC31-LL, historical, r1i1p1f3

pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_hg3_ll_hi_r1.lon,
    am_si10_hg3_ll_hi_r1.lat,
    am_si10_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nHadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.3_global annual mean wind speed HadGEM3-GC31-LL, historical, r1i1p1f3 1979_2014.png',)


pltlevel_sh = np.arange(0, 14.01, 0.05)
pltticks_sh = np.arange(0, 14.01, 2)
fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_si10_hg3_ll_hi_r1.lon,
    am_si10_hg3_ll_hi_r1.lat,
    am_si10_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nHadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.4_SH annual mean wind speed HadGEM3-GC31-LL, historical, r1i1p1f3 1979_2014.png',)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean wind in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'Amon/'
variable = 'sfcWind/'
grid = 'gn/'
version = 'v20200511/'

variable1 = 'uas/'
variable2 = 'vas/'

si10_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))
si10_awc_mr_hi_r1 = xr.open_mfdataset(
    si10_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
u10_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable1 + grid + version + '*.nc',
)))
u10_awc_mr_hi_r1 = xr.open_mfdataset(
    u10_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
v10_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable2 + grid + version + '*.nc',
)))
v10_awc_mr_hi_r1 = xr.open_mfdataset(
    v10_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_si10_awc_mr_hi_r1 = si10_awc_mr_hi_r1.sfcWind.sel(time=slice(
    '1979-01-01', '2014-12-30')).mean(axis=0)
am_si10_awc_mr_hi_r1_80 = si10_awc_mr_hi_r1.sfcWind.sel(time=slice(
    '1980-01-01', '2014-12-30')).mean(axis=0)

am_si10_awc_mr_hi_r1_rg1 = regrid(am_si10_awc_mr_hi_r1)

'''
# check
pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_awc_mr_hi_r1_rg1.lon,
    am_si10_awc_mr_hi_r1_rg1.lat,
    am_si10_awc_mr_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nregridded HadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial.png',)

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean wind in AWI-CM-1-1-MR, historical, r1i1p1f1

pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_awc_mr_hi_r1.lon,
    am_si10_awc_mr_hi_r1.lat,
    am_si10_awc_mr_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nAWI-CM-1-1-MR, historical, r1i1p1f1, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.7_global annual mean wind speed AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)


pltlevel_sh = np.arange(0, 14.01, 0.05)
pltticks_sh = np.arange(0, 14.01, 2)
fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_si10_awc_mr_hi_r1.lon,
    am_si10_awc_mr_hi_r1.lat,
    am_si10_awc_mr_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nAWI-CM-1-1-MR, historical, r1i1p1f1, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.8_SH annual mean wind speed AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean wind in ERA5

era5_mon_sl_79_21_10mwind = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_10mwind.nc')

mon_u10_era5 = xr.concat((
    era5_mon_sl_79_21_10mwind.u10[:-2, 0, :, :],
    era5_mon_sl_79_21_10mwind.u10[-2:, 1, :, :]), dim='time')
mon_v10_era5 = xr.concat((
    era5_mon_sl_79_21_10mwind.v10[:-2, 0, :, :],
    era5_mon_sl_79_21_10mwind.v10[-2:, 1, :, :]), dim='time')
mon_si10_era5 = xr.concat((
    era5_mon_sl_79_21_10mwind.si10[:-2, 0, :, :],
    era5_mon_sl_79_21_10mwind.si10[-2:, 1, :, :]), dim='time')

am_u10_era5 = mon_sea_ann_average(
    mon_u10_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0)
am_v10_era5 = mon_sea_ann_average(
    mon_v10_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0)
am_si10_era5 = mon_sea_ann_average(
    mon_si10_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0)

am_u10_era5_rg1 = regrid(am_u10_era5)
am_v10_era5_rg1 = regrid(am_v10_era5)
am_si10_era5_rg1 = regrid(am_si10_era5)

dif_am_hg3_ll_hi_r1_era5 = am_si10_hg3_ll_hi_r1_rg1 - am_si10_era5_rg1
dif_am_awc_mr_hi_r1_era5 = am_si10_awc_mr_hi_r1_rg1 - am_si10_era5_rg1

'''
stats.describe(dif_am_hg3_ll_hi_r1_era5, axis=None)
stats.describe(dif_am_awc_mr_hi_r1_era5, axis=None)

# check
(am_si10_hg3_ll_hi_r1_rg1.lat == am_si10_era5_rg1.lat).all()

pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_era5_rg1.lon,
    am_si10_era5_rg1.lat,
    am_si10_era5_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nregridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial.png',)


# deprecated
era5_mon_sl_79_21_10mwind.u10[200, 0, 200, 200]
era5_mon_sl_79_21_10mwind.v10[200, 0, 200, 200]
era5_mon_sl_79_21_10mwind.si10[200, 0, 200, 200]

wind_speed = ((mon_u10_era5.sel(time=slice('1979-01-01', '2014-12-30')))**2 + (mon_v10_era5.sel(time=slice('1979-01-01', '2014-12-30')))**2)**0.5
am_wind_era5 = mon_sea_ann_average(
    wind_speed, 'time.year', skipna=False
).mean(axis=0)
stats.describe(am_si10_era5 - am_wind_era5, axis=None, )
am_wind_era5_rg1 = regrid(am_wind_era5)
wind_speed[300, 200, 400]
(mon_u10_era5.sel(time=slice('1979-01-01', '2014-12-30')))[300, 200, 400]
(mon_v10_era5.sel(time=slice('1979-01-01', '2014-12-30')))[300, 200, 400]

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean wind in ERA5

pltlevel = np.arange(0, 14.01, 0.05)
pltticks = np.arange(0, 14.01, 2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_si10_era5.longitude,
    am_si10_era5.latitude,
    am_si10_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.1_global annual mean wind speed era5 1979_2014.png',)


pltlevel_sh = np.arange(0, 14.01, 0.05)
pltticks_sh = np.arange(0, 14.01, 2)
fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_si10_era5.longitude,
    am_si10_era5.latitude,
    am_si10_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m\;s^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.2_SH annual mean wind speed era5 1979_2014.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

pltlevel = np.arange(-4, 4.01, 0.05)
pltticks = np.arange(-4, 4.01, 1)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_era5.lon,
    dif_am_hg3_ll_hi_r1_era5.lat,
    dif_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed difference [$m\;s^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.5_global annual mean wind speed HadGEM3-GC31-LL, historical, r1i1p1f3 - ERA5 1979_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_era5.lon,
    dif_am_hg3_ll_hi_r1_era5.lat,
    dif_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed difference [$m\;s^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5\n1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.6_SH annual mean wind speed HadGEM3-GC31-LL, historical, r1i1p1f3 - ERA5 1979_2014.png',)


################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

pltlevel = np.arange(-4, 4.01, 0.05)
pltticks = np.arange(-4, 4.01, 1)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_era5.lon,
    dif_am_awc_mr_hi_r1_era5.lat,
    dif_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed difference [$m\;s^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.9_global annual mean wind speed AWI-CM-1-1-MR, historical, r1i1p1f1 - ERA5 1979_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_era5.lon,
    dif_am_awc_mr_hi_r1_era5.lat,
    dif_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed difference [$m\;s^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5\n1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.4_winds/4.0.4.10_SH annual mean wind speed AWI-CM-1-1-MR, historical, r1i1p1f1 - ERA5 1979_2014.png',)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean wind in MERRA2

speed_merra2_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/MERRA2/QLML_SPEED_TLML_TSH_ULML_VLML/MERRA2_*.tavgM_2d_flx_Nx.*.SUB.nc',
)))

speed_merra2 = xr.open_mfdataset(
    speed_merra2_fl, data_vars='minimal', coords='minimal', compat='override',
)

# endregion
# =============================================================================


