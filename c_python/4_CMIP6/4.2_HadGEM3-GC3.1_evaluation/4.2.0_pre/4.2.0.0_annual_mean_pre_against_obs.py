

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
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual pre in HadGEM3-GC31-LL, historical, r1i1p1f3

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'Amon/'
variable = 'pr/'
grid = 'gn/'
version = 'v20190624/'

# pr, Precipitation, [kg m-2 s-1] / [mm/s]
pr_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

pr_hg3_ll_hi_r1 = xr.open_mfdataset(
    pr_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_pr_hg3_ll_hi_r1 = pr_hg3_ll_hi_r1.sel(time=slice(
    '1979-01-01', '2014-12-30')).pr.mean(axis=0) * 365 * 24 * 3600
am_pr_hg3_ll_hi_r1_80 = pr_hg3_ll_hi_r1.sel(time=slice(
    '1980-01-01', '2014-12-30')).pr.mean(axis=0) * 365 * 24 * 3600
# stats.describe(am_pr_hg3_ll_hi_r1, axis = None)

# endregion
# =============================================================================


# =============================================================================
# region plot Annual pre in HadGEM3-GC31-LL, historical, r1i1p1f3

# Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    pr_hg3_ll_hi_r1.lon,
    pr_hg3_ll_hi_r1.lat,
    am_pr_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png',
    dpi=1200)


# SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    pr_hg3_ll_hi_r1.lon,
    pr_hg3_ll_hi_r1.lat,
    am_pr_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.1_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png',
    dpi=1200)


# diverging colormap
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    pr_hg3_ll_hi_r1.lon,
    pr_hg3_ll_hi_r1.lat,
    am_pr_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.14_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 1979_2014 diverging colormap.png',)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual pre in AWI-ESM-1-1-LR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-ESM-1-1-LR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'Amon/'
variable = 'pr/'
grid = 'gn/'
version = 'v20200212/'

# pr, Precipitation, [kg m-2 s-1] / [mm/s]
pr_awe_lr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member + \
    table + variable + grid + version + '*.nc',
)))

pr_awe_lr_hi_r1 = xr.open_mfdataset(
    pr_awe_lr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_pr_awe_lr_hi_r1 = mon_sea_ann_average(pr_awe_lr_hi_r1.pr.sel(time=slice(
    '1979-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600
am_pr_awe_lr_hi_r1_80 = mon_sea_ann_average(pr_awe_lr_hi_r1.pr.sel(time=slice(
    '1980-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600

'''
testfile = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/day/pr/gn/v20200212/pr_day_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_20140101-20141231.nc')

'''
# stats.describe(am_pr_awe_lr_hi_r1, axis = None)

# endregion
# =============================================================================


# =============================================================================
# region plot Annual pre in AWI-ESM-1-1-LR, historical, r1i1p1f1

# Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_awe_lr_hi_r1.lon,
    am_pr_awe_lr_hi_r1.lat,
    am_pr_awe_lr_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.35_global annual pre AWI-ESM-1-1-LR, historical, r1i1p1f1 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_awe_lr_hi_r1.lon,
    am_pr_awe_lr_hi_r1.lat,
    am_pr_awe_lr_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.36_SH annual pre AWI-ESM-1-1-LR, historical, r1i1p1f1 1979_2014.png',)


# diverging colormap
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_awe_lr_hi_r1.lon,
    am_pr_awe_lr_hi_r1.lat,
    am_pr_awe_lr_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.37_SH annual pre AWI-ESM-1-1-LR, historical, r1i1p1f1 1979_2014 diverging colormap.png',)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual pre in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'Amon/'
variable = 'pr/'
grid = 'gn/'
version = 'v20200511/'

# pr, Precipitation, [kg m-2 s-1] / [mm/s]
pr_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

pr_awc_mr_hi_r1 = xr.open_mfdataset(
    pr_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_pr_awc_mr_hi_r1 = mon_sea_ann_average(pr_awc_mr_hi_r1.pr.sel(time=slice(
    '1979-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600
am_pr_awc_mr_hi_r1_80 = mon_sea_ann_average(pr_awc_mr_hi_r1.pr.sel(time=slice(
    '1980-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600

'''
# stats.describe(am_pr_awc_mr_hi_r1, axis = None)
'''

# endregion
# =============================================================================


# =============================================================================
# region plot Annual pre in AWI-CM-1-1-MR, historical, r1i1p1f1

# Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_awc_mr_hi_r1.lon,
    am_pr_awc_mr_hi_r1.lat,
    am_pr_awc_mr_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.38_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_awc_mr_hi_r1.lon,
    am_pr_awc_mr_hi_r1.lat,
    am_pr_awc_mr_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.39_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png',)


# diverging colormap
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_awc_mr_hi_r1.lon,
    am_pr_awc_mr_hi_r1.lat,
    am_pr_awc_mr_hi_r1,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.40_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014 diverging colormap.png',)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual pre in ERA5

# import data
era5_mon_sl_79_21_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')
pre = xr.concat((
    era5_mon_sl_79_21_pre.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre.tp[-2:, 1, :, :]), dim='time') * 1000

pre_ann_average = mon_sea_ann_average(
    pre.sel(time=slice('1979-01-01', '2014-12-30')), 'time.year')
pre_ann_sum = pre_ann_average * 365
am_pr_era5 = pre_ann_sum.mean(axis=0)


# compare with HadGEM3-GC31-LL, historical, r1i1p1f3
regridder = xe.Regridder(
    am_pr_era5, am_pr_hg3_ll_hi_r1, 'bilinear')
am_pr_era5_rg = regridder(am_pr_era5)

dif_am_hg3_ll_hi_r1_era5 = am_pr_hg3_ll_hi_r1 - am_pr_era5_rg
dif_pct_am_hg3_ll_hi_r1_era5 = dif_am_hg3_ll_hi_r1_era5/am_pr_era5_rg * 100


# compare with AWI-CM-1-1-MR, historical, r1i1p1f1
regridder_awc = xe.Regridder(
    am_pr_era5, am_pr_awc_mr_hi_r1, 'bilinear')
am_pr_era5_rg_awc = regridder_awc(am_pr_era5)

dif_am_awc_mr_hi_r1_era5 = am_pr_awc_mr_hi_r1 - am_pr_era5_rg_awc
dif_pct_am_awc_mr_hi_r1_era5 = dif_am_awc_mr_hi_r1_era5/am_pr_era5_rg_awc * 100


'''
# (am_pr_era5_rg.lat == am_pr_hg3_ll_hi_r1.lat).all()
# stats.describe(dif_am_hg3_ll_hi_r1_era5, axis =None)
# stats.describe(dif_pct_am_hg3_ll_hi_r1_era5, axis =None)
# np.where(dif_pct_am_hg3_ll_hi_r1_era5 == 46.82873635985134)

dif_pct_am_hg3_ll_hi_r1_era5[65, 149].values
dif_am_hg3_ll_hi_r1_era5[65, 149].values
am_pr_era5_rg[65, 149].values
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual pre in ERA5


# Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_pre.longitude,
    era5_mon_sl_79_21_pre.latitude,
    am_pr_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.2_global annual pre era5 1979_2014.png',
    dpi=1200)


# SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_pre.longitude,
    era5_mon_sl_79_21_pre.latitude,
    am_pr_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.3_SH annual pre era5 1979_2014.png',
    dpi=1200)


# diverging colormap
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_pre.longitude,
    era5_mon_sl_79_21_pre.latitude,
    am_pr_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.15_SH annual pre era5 1979_2014 diverging colormap.png',
    dpi=1200)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

# Global regridded map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_era5_rg.lon,
    am_pr_era5_rg.lat,
    am_pr_era5_rg,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.4_global regridded annual pre era5 1979_2014.png',
    dpi=1200)


# SH regridded map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_era5_rg.lon,
    am_pr_era5_rg.lat,
    am_pr_era5_rg,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.5_SH regridded annual pre era5 1979_2014.png',
    dpi=1200)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_era5.lon,
    dif_am_hg3_ll_hi_r1_era5.lat,
    dif_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.6_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 1979_2014.png',
    dpi=1200)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_era5.lon,
    dif_am_hg3_ll_hi_r1_era5.lat,
    dif_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.7_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 1979_2014.png',
    dpi=1200)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_era5.lon,
    dif_pct_am_hg3_ll_hi_r1_era5.lat,
    dif_pct_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5)/ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.22_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_era5.lon,
    dif_pct_am_hg3_ll_hi_r1_era5.lat,
    dif_pct_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5)/ERA5\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.23_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 percentage 1979_2014.png',)


################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

# Global regridded map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_era5_rg_awc.lon,
    am_pr_era5_rg_awc.lat,
    am_pr_era5_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.0_global regridded_awc annual pre era5 1979_2014.png',)


# SH regridded map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_era5_rg_awc.lon,
    am_pr_era5_rg_awc.lat,
    am_pr_era5_rg_awc,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc ERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.1_SH regridded annual pre era5 1979_2014.png',)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_era5.lon,
    dif_am_awc_mr_hi_r1_era5.lat,
    dif_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.2_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - era5 1979_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_era5.lon,
    dif_am_awc_mr_hi_r1_era5.lat,
    dif_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.3_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - era5 1979_2014.png',)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_era5.lon,
    dif_pct_am_awc_mr_hi_r1_era5.lat,
    dif_pct_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5)/ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.4_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - era5 percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_era5.lon,
    dif_pct_am_awc_mr_hi_r1_era5.lat,
    dif_pct_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5)/ERA5\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.5_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - era5 percentage 1979_2014.png',)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean pre in GPCP

# Import data
mon_pre_gpcp_79_21 = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/GPCP/precip.mon.mean.nc'
)
ann_pre_gpcp = mon_sea_ann_average(
    mon_pre_gpcp_79_21.precip.sel(time=slice('1979-01-01', '2014-12-30')),
    'time.year'
)
am_pre_gpcp = ann_pre_gpcp.mean(axis = 0) * 365

# compare with HadGEM3-GC31-LL, historical, r1i1p1f3
regridder = xe.Regridder(
    ann_pre_gpcp, am_pr_hg3_ll_hi_r1, 'bilinear', periodic=True)
am_pre_gpcp_rg = regridder(am_pre_gpcp)

dif_am_pr_hg3_ll_hi_r1_gpcp = am_pr_hg3_ll_hi_r1 - am_pre_gpcp_rg
dif_pct_am_hg3_ll_hi_r1_gpcp = dif_am_pr_hg3_ll_hi_r1_gpcp/am_pre_gpcp_rg * 100


# compare with AWI-CM-1-1-MR, historical, r1i1p1f1
regridder_awc = xe.Regridder(
    ann_pre_gpcp, am_pr_awc_mr_hi_r1, 'bilinear', periodic=True)
am_pr_gpcp_rg_awc = regridder_awc(am_pre_gpcp)

dif_am_awc_mr_hi_r1_gpcp = am_pr_awc_mr_hi_r1 - am_pr_gpcp_rg_awc
dif_pct_am_awc_mr_hi_r1_gpcp = dif_am_awc_mr_hi_r1_gpcp/am_pr_gpcp_rg_awc * 100


'''
# stats.describe(dif_am_pr_hg3_ll_hi_r1_gpcp, axis =None)
# stats.describe(dif_pct_am_hg3_ll_hi_r1_gpcp, axis =None)
# np.where(dif_pct_am_hg3_ll_hi_r1_gpcp == 1483.1467437098884)


# check
ltm_mon_pre_gpcp_79_21 = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/GPCP/precip.mon.ltm.nc'
)
am_pre_gpcp_check = ltm_mon_pre_gpcp_79_21.precip.mean(axis = 0) * 365
np.max(np.abs(am_pre_gpcp - am_pre_gpcp_check))
(am_pre_gpcp_rg.lat == am_pr_hg3_ll_hi_r1.lat).all()

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean pre in GPCP


# Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pre_gpcp.lon,
    am_pre_gpcp.lat,
    am_pre_gpcp,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nGPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.8_global annual pre GPCP 1979_2014.png',)


# SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pre_gpcp.lon,
    am_pre_gpcp.lat,
    am_pre_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nGPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.9_SH annual pre GPCP 1979_2014.png',)


# diverging color map
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pre_gpcp.lon,
    am_pre_gpcp.lat,
    am_pre_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nGPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.16_SH annual pre GPCP 1979_2014 diverging colormap.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

# regridded Global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pre_gpcp_rg.lon,
    am_pre_gpcp_rg.lat,
    am_pre_gpcp_rg,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.10_global regridded annual pre GPCP 1979_2014.png',)


# regridded SH map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pre_gpcp_rg.lon,
    am_pre_gpcp_rg.lat,
    am_pre_gpcp_rg,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded GPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.11_SH regridded annual pre GPCP 1979_2014.png',)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_pr_hg3_ll_hi_r1_gpcp.lon,
    dif_am_pr_hg3_ll_hi_r1_gpcp.lat,
    dif_am_pr_hg3_ll_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.12_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP 1979_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_pr_hg3_ll_hi_r1_gpcp.lon,
    dif_am_pr_hg3_ll_hi_r1_gpcp.lat,
    dif_am_pr_hg3_ll_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.13_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP 1979_2014.png',)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_gpcp.lon,
    dif_pct_am_hg3_ll_hi_r1_gpcp.lat,
    dif_pct_am_hg3_ll_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP)/GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.24_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_gpcp.lon,
    dif_pct_am_hg3_ll_hi_r1_gpcp.lat,
    dif_pct_am_hg3_ll_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP)/GPCP\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.25_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP percentage 1979_2014.png',)


################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

# Global regridded map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_gpcp_rg_awc.lon,
    am_pr_gpcp_rg_awc.lat,
    am_pr_gpcp_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.6_global regridded_awc annual pre GPCP 1979_2014.png',)


# SH regridded map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_gpcp_rg_awc.lon,
    am_pr_gpcp_rg_awc.lat,
    am_pr_gpcp_rg_awc,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc GPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.7_SH regridded annual pre GPCP 1979_2014.png',)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_gpcp.lon,
    dif_am_awc_mr_hi_r1_gpcp.lat,
    dif_am_awc_mr_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.8_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - GPCP 1979_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_gpcp.lon,
    dif_am_awc_mr_hi_r1_gpcp.lat,
    dif_am_awc_mr_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - GPCP\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.9_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - GPCP 1979_2014.png',)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_gpcp.lon,
    dif_pct_am_awc_mr_hi_r1_gpcp.lat,
    dif_pct_am_awc_mr_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - GPCP)/GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.10_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - GPCP percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_gpcp.lon,
    dif_pct_am_awc_mr_hi_r1_gpcp.lat,
    dif_pct_am_awc_mr_hi_r1_gpcp,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - GPCP)/GPCP\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.11_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - GPCP percentage 1979_2014.png',)



# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean pre in MERRA2

# import data
pr_merra2_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/MERRA2/PRECTOTCORR/*.nc',
)))
pr_merra2 = xr.open_mfdataset(
    pr_merra2_fl, data_vars='minimal', coords='minimal', compat='override',
)

am_pr_merra2 = mon_sea_ann_average(pr_merra2.PRECTOTCORR.sel(
    time=slice('1980-01-01', '2014-12-30')), 'time.year').mean(
        axis=0) * 365 * 24 * 3600

# compare with HadGEM3-GC31-LL, historical, r1i1p1f3
regridder = xe.Regridder(
    am_pr_merra2, am_pr_hg3_ll_hi_r1_80, 'bilinear')
am_pr_merra2_rg = regridder(am_pr_merra2)

dif_am_hg3_ll_hi_r1_merra2 = am_pr_hg3_ll_hi_r1_80 - am_pr_merra2_rg
dif_pct_am_hg3_ll_hi_r1_merra2 = dif_am_hg3_ll_hi_r1_merra2 /\
    am_pr_merra2_rg * 100

# compare with AWI-CM-1-1-MR, historical, r1i1p1f1
regridder_awc = xe.Regridder(
    am_pr_merra2, am_pr_awc_mr_hi_r1_80, 'bilinear')
am_pr_merra2_rg_awc = regridder_awc(am_pr_merra2)

dif_am_awc_mr_hi_r1_merra2 = am_pr_awc_mr_hi_r1_80 - am_pr_merra2_rg_awc
dif_pct_am_awc_mr_hi_r1_merra2 = \
    dif_am_awc_mr_hi_r1_merra2/am_pr_merra2_rg_awc * 100


'''
# stats.describe(dif_am_hg3_ll_hi_r1_merra2, axis =None)
# stats.describe(dif_pct_am_hg3_ll_hi_r1_merra2, axis =None)
# np.where(dif_pct_am_hg3_ll_hi_r1_merra2 == 5913.748067409738)

(pr_merra2.time.values == np.array(sorted(pr_merra2.time.values))).all()
(pr_merra2.PRECTOTCORR.values == 1000000000000000.0).sum() # 0
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean pre in MERRA2


# global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_merra2.lon,
    am_pr_merra2.lat,
    am_pr_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nMERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.17_global annual pre MERRA2 1980_2014.png',)


pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_merra2_rg.lon,
    am_pr_merra2_rg.lat,
    am_pr_merra2_rg,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.18_global regridded annual pre MERRA2 1980_2014.png',)


# diverging color map
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_merra2.lon,
    am_pr_merra2.lat,
    am_pr_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nMERRA2, 1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.20_SH annual pre MERRA2 1980_2014 diverging colormap.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

# absolute differences
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_merra2.lon,
    dif_am_hg3_ll_hi_r1_merra2.lat,
    dif_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.19_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - MERRA2 1980_2014.png',)



# SH absolute difference

pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_merra2.lon,
    dif_am_hg3_ll_hi_r1_merra2.lat,
    dif_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2\n1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.21_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - MERRA2 1980_2014.png',)


# relative differences
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_merra2.lon,
    dif_pct_am_hg3_ll_hi_r1_merra2.lat,
    dif_pct_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2)/MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.26_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - MERRA2 percentage 1980_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_merra2.lon,
    dif_pct_am_hg3_ll_hi_r1_merra2.lat,
    dif_pct_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2)/\nMERRA2, 1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.27_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - MERRA2 percentage 1980_2014.png',)


################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

# Global regridded map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_merra2_rg_awc.lon,
    am_pr_merra2_rg_awc.lat,
    am_pr_merra2_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.12_global regridded_awc annual pre MERRA2 1980_2014.png',)


# SH regridded map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_merra2_rg_awc.lon,
    am_pr_merra2_rg_awc.lat,
    am_pr_merra2_rg_awc,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc MERRA2, 1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.13_SH regridded_awc annual pre MERRA2 1980_2014.png',)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_merra2.lon,
    dif_am_awc_mr_hi_r1_merra2.lat,
    dif_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.14_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 1980_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_merra2.lon,
    dif_am_awc_mr_hi_r1_merra2.lat,
    dif_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2\n1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.15_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 1980_2014.png',)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_merra2.lon,
    dif_pct_am_awc_mr_hi_r1_merra2.lat,
    dif_pct_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2)/MERRA2, 1980-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.16_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 percentage 1980_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_merra2.lon,
    dif_pct_am_awc_mr_hi_r1_merra2.lat,
    dif_pct_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2)/\nMERRA2, 1980-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.17_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 percentage 1980_2014.png',)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean pre in JRA-55

pr_jra55_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/JRA-55/mon_tp/fcst_phy2m.061_tprat.reg_tl319.*[0-9]',
)))

pr_jra55 = xr.open_mfdataset(pr_jra55_fl, engine='cfgrib')

am_pr_jra55 = mon_sea_ann_average(pr_jra55.tpratsfc.sel(
    time=slice('1979-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365

# compare with HadGEM3-GC31-LL, historical, r1i1p1f3
regridder = xe.Regridder(
    am_pr_jra55, am_pr_hg3_ll_hi_r1, 'bilinear')
am_pr_jra55_rg = regridder(am_pr_jra55)

dif_am_hg3_ll_hi_r1_jra55 = am_pr_hg3_ll_hi_r1 - am_pr_jra55_rg
dif_pct_am_hg3_ll_hi_r1_jra55 = dif_am_hg3_ll_hi_r1_jra55 /\
    am_pr_jra55_rg * 100

# compare with AWI-CM-1-1-MR, historical, r1i1p1f1
regridder_awc = xe.Regridder(
    am_pr_jra55, am_pr_awc_mr_hi_r1, 'bilinear')
am_pr_jra55_rg_awc = regridder_awc(am_pr_jra55)

dif_am_awc_mr_hi_r1_jra55 = am_pr_awc_mr_hi_r1 - am_pr_jra55_rg_awc
dif_pct_am_awc_mr_hi_r1_jra55 = \
    dif_am_awc_mr_hi_r1_jra55/am_pr_jra55_rg_awc * 100


'''
# stats.describe(dif_am_hg3_ll_hi_r1_jra55, axis =None)
# stats.describe(dif_pct_am_hg3_ll_hi_r1_jra55, axis =None)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean pre in JRA-55


# global map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_jra55.longitude,
    am_pr_jra55.latitude,
    am_pr_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nJRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.28_global annual pre JRA-55 1979_2014.png',)


# diverging colormap
pltlevel_sh = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
pltticks_sh = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
cmp_cmap = rb_colormap(pltlevel_sh)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_jra55.longitude,
    am_pr_jra55.latitude,
    am_pr_jra55,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1.2, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nJRA-55, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.31_SH annual pre JRA-55 1979_2014 diverging colormap.png',)


################ compare with HadGEM3-GC31-LL, historical, r1i1p1f3

# regridded
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_jra55_rg.lon,
    am_pr_jra55_rg.lat,
    am_pr_jra55_rg,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.29_global regridded annual pre JRA-55 1979_2014.png',)


# absolute differences
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_jra55.lon,
    dif_am_hg3_ll_hi_r1_jra55.lat,
    dif_am_hg3_ll_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.30_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - JRA-55 1979_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_jra55.lon,
    dif_am_hg3_ll_hi_r1_jra55.lat,
    dif_am_hg3_ll_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - JRA-55\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.32_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - JRA-55 1979_2014.png',)


# relative differences
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_jra55.lon,
    dif_pct_am_hg3_ll_hi_r1_jra55.lat,
    dif_pct_am_hg3_ll_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - JRA-55)/JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.33_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - JRA-55 percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_hg3_ll_hi_r1_jra55.lon,
    dif_pct_am_hg3_ll_hi_r1_jra55.lat,
    dif_pct_am_hg3_ll_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((HadGEM3-GC31-LL, historical, r1i1p1f3) - JRA-55)/\nJRA-55, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.34_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - JRA-55 percentage 1979_2014.png',)


################ compare with AWI-CM-1-1-MR, historical, r1i1p1f1

# Global regridded map
pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_jra55_rg_awc.lon,
    am_pr_jra55_rg_awc.lat,
    am_pr_jra55_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.18_global regridded_awc annual pre JRA-55 1979_2014.png',)


# SH regridded map
pltlevel_sh = np.arange(0, 1200.01, 1)
pltticks_sh = np.arange(0, 1200.01, 200)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)

plt_cmp = ax.pcolormesh(
    am_pr_jra55_rg_awc.lon,
    am_pr_jra55_rg_awc.lat,
    am_pr_jra55_rg_awc,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\nregridded_awc JRA-55, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.19_SH regridded_awc annual pre JRA-55 1979_2014.png',)


# absolute difference
pltlevel = np.arange(-1500, 1500.01, 0.2)
pltticks = np.arange(-1500, 1500.01, 500)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_jra55.lon,
    dif_am_awc_mr_hi_r1_jra55.lat,
    dif_am_awc_mr_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.20_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - JRA-55 1979_2014.png',)


pltlevel_sh = np.arange(-400, 400.01, 0.2)
pltticks_sh = np.arange(-400, 400.01, 100)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_jra55.lon,
    dif_am_awc_mr_hi_r1_jra55.lat,
    dif_am_awc_mr_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual mean precipitation difference [$mm\;yr^{-1}$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - JRA-55\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.21_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - JRA-55 1979_2014.png',)


# relative difference
pltlevel = np.arange(-100, 100.01, 1)
pltticks = np.arange(-100, 100.01, 20)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_jra55.lon,
    dif_pct_am_awc_mr_hi_r1_jra55.lat,
    dif_pct_am_awc_mr_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - JRA-55)/JRA-55, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.22_global annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - JRA-55 percentage 1979_2014.png',)


pltlevel_sh = np.arange(-100, 100.01, 1)
pltticks_sh = np.arange(-100, 100.01, 20)
cmp_cmap_sh = rb_colormap(pltlevel_sh)
fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.96,
    figsize=np.array([8.8, 10.3]) / 2.54)

plt_cmp = ax.pcolormesh(
    dif_pct_am_awc_mr_hi_r1_jra55.lon,
    dif_pct_am_awc_mr_hi_r1_jra55.lat,
    dif_pct_am_awc_mr_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cmp_cmap_sh.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation relative difference [$\%$]\n((AWI-CM-1-1-MR, historical, r1i1p1f1) - JRA-55)/JRA-55\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.0_compare_with_awc_hi_r1/4.0.0.0.23_SH annual pre AWI-CM-1-1-MR, historical, r1i1p1f1 - JRA-55 percentage 1979_2014.png',)


# endregion
# =============================================================================


