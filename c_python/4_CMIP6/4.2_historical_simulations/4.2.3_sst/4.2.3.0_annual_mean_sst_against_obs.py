

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
# region Annual mean SST in HadGEM3-GC31-LL, historical, r1i1p1f3

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'Omon/'
variable = 'tos/'
grid = 'gn/'
version = 'v20190624/'

sst_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))
sst_hg3_ll_hi_r1 = xr.open_mfdataset(
    sst_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_sst_hg3_ll_hi_r1 = sst_hg3_ll_hi_r1.tos.sel(time=slice(
    '1979-01-01', '2014-12-30')).mean(axis=0)
am_sst_hg3_ll_hi_r1_80 = sst_hg3_ll_hi_r1.tos.sel(time=slice(
    '1980-01-01', '2014-12-30')).mean(axis=0)

am_sst_hg3_ll_hi_r1_rg1 = regrid(am_sst_hg3_ll_hi_r1, method='nearest_s2d')
am_sst_hg3_ll_hi_r1_80_rg1 = regrid(
    am_sst_hg3_ll_hi_r1_80, method='nearest_s2d')


'''
stats.describe(sst_hg3_ll_hi_r1.latitude, axis =None)

am_sst_hg3_ll_hi_r1_rg1.to_netcdf('bas_palaeoclim_qino/scratch/0_trial/o.nc')

# stats.describe(sst_hg3_ll_hi_r1.tos, axis=None, nan_policy='omit')
# -2.3190186, 37.612717
# stats.describe(am_sst_hg3_ll_hi_r1, axis=None, nan_policy='omit')
# -1.8927556, 33.348568


# trial
am_sst_hg3_ll_hi_r1_trial = regrid(
    am_sst_hg3_ll_hi_r1, periodic=True)

stats.describe(am_sst_hg3_ll_hi_r1_rg1 - am_sst_hg3_ll_hi_r1_trial, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_hg3_ll_hi_r1_trial.lon,
    am_sst_hg3_ll_hi_r1_trial.lat,
    am_sst_hg3_ll_hi_r1_trial,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/0_test/trial.png',)


'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean SST in HadGEM3-GC31-LL, historical, r1i1p1f3

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_hg3_ll_hi_r1_rg1.lon,
    am_sst_hg3_ll_hi_r1_rg1.lat,
    am_sst_hg3_ll_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.0_global annual mean sst HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(-2, 14.01, 0.05)
pltticks_sh = np.arange(-2, 14.01, 2)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_sst_hg3_ll_hi_r1_rg1.lon,
    am_sst_hg3_ll_hi_r1_rg1.lat,
    am_sst_hg3_ll_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.1_SH annual mean sst HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png',)


'''
pltlevel = np.concatenate(
    (np.arange(-2, 0, 0.05), np.arange(0, 32.01, 0.2)))
pltticks = np.concatenate(
    (np.arange(-2, 0, 1), np.arange(0, 32.01, 4)))

cmp_cmap = ListedColormap(rb_colormap(np.zeros(321)).colors[120:, :])
# cmp_cmap = cm.get_cmap('seismic', 321)

# NH map
fig, ax = hemisphere_plot(
    southextent=60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_sst_hg3_ll_hi_r1_rg1.lon,
    am_sst_hg3_ll_hi_r1_rg1.lat,
    am_sst_hg3_ll_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean sst [$C$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/0_test/trial.png',)

'''
# endregion
# =============================================================================


# =============================================================================
# region Don't Annual mean SST in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'Omon/'
variable = 'tos/'
grid = 'gn/'
version = 'v20181218/'

sst_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

sst_awc_mr_hi_r1 = xr.open_mfdataset(
    sst_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_sst_awc_mr_hi_r1_org = sst_awc_mr_hi_r1.copy()
am_sst_awc_mr_hi_r1_org['tos'] = am_sst_awc_mr_hi_r1_org[
    'tos'][0, :]
am_sst_awc_mr_hi_r1_org['tos'][:] = \
    sst_awc_mr_hi_r1.tos.sel(time=slice(
        '1979-01-01', '2014-12-31')).mean(axis=0).values

am_sst_awc_mr_hi_r1_80_org = sst_awc_mr_hi_r1.copy()
am_sst_awc_mr_hi_r1_80_org['tos'] = am_sst_awc_mr_hi_r1_80_org[
    'tos'][0, :]
am_sst_awc_mr_hi_r1_80_org['tos'][:] = \
    sst_awc_mr_hi_r1.tos.sel(time=slice(
        '1980-01-01', '2014-12-31')).mean(axis=0).values


am_sst_awc_mr_hi_r1_org.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_org.nc')
am_sst_awc_mr_hi_r1_80_org.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_80_org.nc')

'''
#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_org.nc bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1.nc
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_80_org.nc bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_80.nc

stats.describe(am_sst_awc_mr_hi_r1_org.tos, axis=None, nan_policy='omit')
stats.describe(am_sst_awc_mr_hi_r1_80_org.tos, axis=None, nan_policy='omit')

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in AWI-CM-1-1-MR, historical, r1i1p1f1

am_sst_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1.nc'
)
am_sst_awc_mr_hi_r1_80 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1_80.nc'
)

am_sst_awc_mr_hi_r1_rg1 = regrid(am_sst_awc_mr_hi_r1.tos)
am_sst_awc_mr_hi_r1_80_rg1 = regrid(am_sst_awc_mr_hi_r1_80.tos)

'''
# check
stats.describe(am_sst_awc_mr_hi_r1_rg1, axis=None, nan_policy='omit')
(am_sst_awc_mr_hi_r1_rg1.values == am_sst_awc_mr_hi_r1.tos.values).sum()
(am_sst_awc_mr_hi_r1_80_rg1.values == am_sst_awc_mr_hi_r1_80.tos.values).sum()
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean SST in AWI-CM-1-1-MR, historical, r1i1p1f1

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_awc_mr_hi_r1_rg1.lon,
    am_sst_awc_mr_hi_r1_rg1.lat,
    am_sst_awc_mr_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nAWI-CM-1-1-MR, historical, r1i1p1f1, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.2_global annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(-2, 14.01, 0.05)
pltticks_sh = np.arange(-2, 14.01, 2)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_sst_awc_mr_hi_r1_rg1.lon,
    am_sst_awc_mr_hi_r1_rg1.lat,
    am_sst_awc_mr_hi_r1_rg1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nAWI-CM-1-1-MR, historical, r1i1p1f1, 1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.3_SH annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in ERA5

era5_mon_sl_79_21_sst = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sst.nc')

mon_sst_era5 = xr.concat((
    era5_mon_sl_79_21_sst.sst[:-2, 0, :, :],
    era5_mon_sl_79_21_sst.sst[-2:, 1, :, :]), dim='time')

am_sst_era5 = mon_sea_ann_average(
    mon_sst_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0) - zerok

am_sst_era5_rg1 = regrid(am_sst_era5)

dif_am_hg3_ll_hi_r1_era5 = am_sst_hg3_ll_hi_r1_rg1 - am_sst_era5_rg1
dif_am_awc_mr_hi_r1_era5 = am_sst_awc_mr_hi_r1_rg1 - am_sst_era5_rg1

'''
stats.describe(dif_am_awc_mr_hi_r1_era5, axis=None, nan_policy='omit')

# check
pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_era5_rg1.lon,
    am_sst_era5_rg1.lat,
    am_sst_era5_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nregridded ERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/0_test/trial.png',)


'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean SST in ERA5

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_era5.longitude,
    am_sst_era5.latitude,
    am_sst_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.4_global annual mean sst ERA5 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(-2, 14.01, 0.05)
pltticks_sh = np.arange(-2, 14.01, 2)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_sst_era5.longitude,
    am_sst_era5.latitude,
    am_sst_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nERA5, 1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.5_SH annual mean sst ERA5 1979_2014.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

pltlevel = np.arange(-6, 6.01, 0.05)
pltticks = np.arange(-6, 6.01, 2)
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
    'Annual mean SST difference [$K$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.6_global annual mean sst HadGEM3-GC31-LL, historical, r1i1p1f3 - ERA5 1979_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
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
    'Annual mean SST difference [$K$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5\n1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.7_SH annual mean sst HadGEM3-GC31-LL, historical, r1i1p1f3 - ERA5 1979_2014.png',)

################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

pltlevel = np.arange(-6, 6.01, 0.05)
pltticks = np.arange(-6, 6.01, 2)
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
    'Annual mean SST difference [$K$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.8_global annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 - ERA5 1979_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
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
    'Annual mean SST difference [$K$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5\n1979-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.9_SH annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 - ERA5 1979_2014.png',)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in MERRA2

sst_merra2_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/MERRA2/FRSEAICE_TSKINICE_TSKINWTR/MERRA2_*.tavgM_2d_ocn_Nx.*.SUB.nc',
)))

sst_merra2 = xr.open_mfdataset(
    sst_merra2_fl, data_vars='minimal', coords='minimal', compat='override',
)

am_sst_merra2 = mon_sea_ann_average(
    sst_merra2.TSKINWTR.sel(time=slice(
        '1980-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0) - zerok

am_sst_merra2_rg1 = regrid(am_sst_merra2)

dif_am_hg3_ll_hi_r1_merra2 = am_sst_hg3_ll_hi_r1_80_rg1 - am_sst_merra2_rg1
dif_am_awc_mr_hi_r1_merra2 = am_sst_awc_mr_hi_r1_80_rg1 - am_sst_merra2_rg1

'''
# check
(sst_merra2.TSKINWTR.values == 1000000000000000.0).sum()
stats.describe(dif_am_awc_mr_hi_r1_merra2, axis=None, nan_policy='omit')

# check
pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_merra2_rg1.lon,
    am_sst_merra2_rg1.lat,
    am_sst_merra2_rg1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nregridded MERRA2, 1979-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/0_test/trial.png',)


'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean SST in MERRA2

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
cmp_cmap = rb_colormap(pltlevel)

# Global map
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_sst_merra2.lon,
    am_sst_merra2.lat,
    am_sst_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nMERRA2, 1980-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.10_global annual mean sst MERRA2 1980_2014.png',)


# SH map
pltlevel_sh = np.arange(-2, 14.01, 0.05)
pltticks_sh = np.arange(-2, 14.01, 2)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_sst_merra2.lon,
    am_sst_merra2.lat,
    am_sst_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nMERRA2, 1980-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.11_SH annual mean sst MERRA2 1980_2014.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

pltlevel = np.arange(-6, 6.01, 0.05)
pltticks = np.arange(-6, 6.01, 2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_merra2.lon,
    dif_am_hg3_ll_hi_r1_merra2.lat,
    dif_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST difference [$K$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2, 1980-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.12_global annual mean sst HadGEM3-GC31-LL, historical, r1i1p1f3 - MERRA2 1980_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_merra2.lon,
    dif_am_hg3_ll_hi_r1_merra2.lat,
    dif_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST difference [$K$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2\n1980-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.13_SH annual mean sst HadGEM3-GC31-LL, historical, r1i1p1f3 - MERRA2 1980_2014.png',)

################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

pltlevel = np.arange(-6, 6.01, 0.05)
pltticks = np.arange(-6, 6.01, 2)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_merra2.lon,
    dif_am_awc_mr_hi_r1_merra2.lat,
    dif_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap,
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST difference [$K$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2, 1980-2014',
    linespacing=1.5,
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.14_global annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 1980_2014.png',)


pltlevel_sh = np.arange(-4, 4.01, 0.05)
pltticks_sh = np.arange(-4, 4.01, 1)
cmp_cmap_sh = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_merra2.lon,
    dif_am_awc_mr_hi_r1_merra2.lat,
    dif_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh, rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean SST difference [$K$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2\n1980-2014',
    linespacing=1.5,
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.15_SH annual mean sst AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 1980_2014.png',)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================

# =============================================================================


