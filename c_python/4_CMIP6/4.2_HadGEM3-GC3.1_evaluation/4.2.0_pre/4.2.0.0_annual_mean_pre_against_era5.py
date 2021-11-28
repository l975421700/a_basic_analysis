

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
    'Annual mean precipitation [$mm\;year^{-1}$]\n%s, %s, %s, 1979-2014' %
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
    'Annual mean precipitation [$mm\;year^{-1}$]\n%s, %s, %s, 1979-2014' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.1_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png',
    dpi=1200)


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
    'Annual mean precipitation [$mm\;year^{-1}$]\nERA5, 1979-2014',
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.3_SH annual pre era5 1979_2014.png',
    dpi=1200)


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Difference in Annual pre HadGEM3-GC31-LL, historical, r1i1p1f3 - ERA5

regridder = xe.Regridder(
    am_pr_era5, am_pr_hg3_ll_hi_r1, 'bilinear')
am_pr_era5_rg = regridder(am_pr_era5)

# (am_pr_era5_rg.lat == am_pr_hg3_ll_hi_r1.lat).all()
# endregion
# =============================================================================


# =============================================================================
# region plot regridded Annual pre in ERA5
# Global map
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nregridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.4_global regridded annual pre era5 1979_2014.png',
    dpi=1200)


# SH map
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nregridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.5_SH regridded annual pre era5 1979_2014.png',
    dpi=1200)


# endregion
# =============================================================================


# =============================================================================
# region plot the difference

dif_am_hg3_ll_hi_r1_era5 = am_pr_hg3_ll_hi_r1 - am_pr_era5_rg
# stats.describe(dif_am_hg3_ll_hi_r1_era5, axis =None)

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
    'Annual mean precipitation difference [$mm\;year^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.6_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 1979_2014.png',
    dpi=1200)


# SH map
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
    'Annual mean precipitation difference [$mm\;year^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.7_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - era5 1979_2014.png',
    dpi=1200)




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

regridder = xe.Regridder(
    ann_pre_gpcp, am_pr_hg3_ll_hi_r1, 'bilinear', periodic=True)
am_pre_gpcp_rg = regridder(am_pre_gpcp)

'''
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nGPCP, 1979-2014',
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nGPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.9_SH annual pre GPCP 1979_2014.png',)


# endregion
# =============================================================================


# =============================================================================
# region plot regridded Annual mean pre in GPCP
# Global map
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nregridded GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.10_global regridded annual pre GPCP 1979_2014.png',)


# SH map
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
    'Annual mean precipitation [$mm\;year^{-1}$]\nregridded GPCP, 1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.11_SH regridded annual pre GPCP 1979_2014.png',)

# endregion
# =============================================================================


# =============================================================================
# region plot the difference

dif_am_pr_hg3_ll_hi_r1_gpcp = am_pr_hg3_ll_hi_r1 - am_pre_gpcp_rg
# stats.describe(dif_am_pr_hg3_ll_hi_r1_gpcp, axis =None)

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
    'Annual mean precipitation difference [$mm\;year^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.12_global annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP 1979_2014.png',)


# SH map
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
    'Annual mean precipitation difference [$mm\;year^{-1}$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - GPCP\n1979-2014',
    linespacing=1.5
)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.13_SH annual pre HadGEM3-GC31-LL historical r1i1p1f3 - GPCP 1979_2014.png',)


# endregion
# =============================================================================


