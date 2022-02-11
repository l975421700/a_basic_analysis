

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
import metpy.calc as mpcalc
from metpy.units import units

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
# region topography in HadGEM3-GC31-LL, piControl, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'piControl/'
member = 'r1i1p1f1/'
table = 'fx/'
variable = 'orog/'
grid = 'gn/'
version = 'v20190709/'


orog_hg3_ll_pi_r1 = xr.open_mfdataset(
    top_dir + mip + institute + source + experiment + member + \
    table + variable + grid + version + '*.nc'
)
# endregion
# =============================================================================


# =============================================================================
# region plot topography in HadGEM3-GC31-LL, piControl, r1i1p1f1

# global
pltlevel = np.arange(0, 6001, 1)
pltticks = np.arange(0, 6001, 1000)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    orog_hg3_ll_pi_r1.orog.lon,
    orog_hg3_ll_pi_r1.orog.lat,
    orog_hg3_ll_pi_r1.orog,
    cmap=cm.get_cmap('Blues', len(pltlevel)),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    transform=ccrs.PlateCarree(), rasterized=True)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Topography [$m$]\n%s, %s, %s' % \
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.0_global topography HadGEM3-GC31-LL piControl r1i1p1f1.png')


# SH
pltlevel_sh = np.arange(0, 4001, 1)
pltticks_sh = np.arange(0, 4001, 1000)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)
plt_cmp = ax.pcolormesh(
    orog_hg3_ll_pi_r1.orog.lon,
    orog_hg3_ll_pi_r1.orog.lat,
    orog_hg3_ll_pi_r1.orog,
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel(
    'Topography [$m$]\n%s, %s, %s' %
    (source[:-1], experiment[:-1], member[:-1]), linespacing=1.5)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.1_SH topography HadGEM3-GC31-LL piControl r1i1p1f1.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region topography in ERA5

era5_mon_sl_20_gph = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_20_gph.nc'
)
era5_topograph = mpcalc.geopotential_to_height(
    era5_mon_sl_20_gph.z[0, :, :].squeeze().values * \
        units('meter ** 2 / second ** 2'))

# endregion
# =============================================================================


# =============================================================================
# region plot topography in ERA5

pltlevel = np.arange(0, 6001, 1)
pltticks = np.arange(0, 6001, 1000)
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_20_gph.longitude,
    era5_mon_sl_20_gph.latitude,
    era5_topograph.magnitude,
    cmap=cm.get_cmap('Blues', len(pltlevel)),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    transform=ccrs.PlateCarree(), rasterized=True)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel("Topography [$m$]\nERA5", linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.2_global topography ERA5.png')


# SH
pltlevel_sh = np.arange(0, 4001, 1)
pltticks_sh = np.arange(0, 4001, 1000)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)
plt_cmp = ax.pcolormesh(
    era5_mon_sl_20_gph.longitude,
    era5_mon_sl_20_gph.latitude,
    era5_topograph.magnitude,
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel('Topography [$m$]\nERA5', linespacing=1.5)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.3_SH topography ERA5.png')

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Difference in topography (HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5

regridder = xe.Regridder(
    era5_mon_sl_20_gph.z[0, :, :], orog_hg3_ll_pi_r1.orog, 'bilinear')
era5_topograph_rg = regridder(era5_topograph.magnitude)

# endregion
# =============================================================================


# =============================================================================
# region plot Difference in topography (HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5
pltlevel = np.arange(0, 6001, 1)
pltticks = np.arange(0, 6001, 1000)
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    orog_hg3_ll_pi_r1.orog.lon,
    orog_hg3_ll_pi_r1.orog.lat,
    era5_topograph_rg,
    cmap=cm.get_cmap('Blues', len(pltlevel)),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    transform=ccrs.PlateCarree(), rasterized=True)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel("Topography [$m$]\nregridded ERA5", linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.4_global regridded topography ERA5.png')


# SH
pltlevel_sh = np.arange(0, 4001, 1)
pltticks_sh = np.arange(0, 4001, 1000)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)
plt_cmp = ax.pcolormesh(
    orog_hg3_ll_pi_r1.orog.lon,
    orog_hg3_ll_pi_r1.orog.lat,
    era5_topograph_rg,
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel('Topography [$m$]\nregridded ERA5', linespacing=1.5)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.5_SH regridded topography ERA5.png')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot Antarctica surface height in Bedmap2

bedmap_tif = xr.open_rasterio(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface.tif')
surface_height_bedmap = bedmap_tif.values.copy()
surface_height_bedmap[surface_height_bedmap == 32767] = 0

bedmap_transform = ccrs.epsg(3031)

# SH
pltlevel_sh = np.arange(0, 4000.1, 1)
pltticks_sh = np.arange(0, 4000.1, 1000)

fig, ax = hemisphere_plot(
    northextent=-60, sb_length=1000, sb_barheight=100, fm_top=0.94,
    figsize=np.array([8.8, 9.8]) / 2.54)
plt_cmp = ax.pcolormesh(
    bedmap_tif.x, bedmap_tif.y,
    surface_height_bedmap[0, :, :],
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=bedmap_transform,
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.12, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='max')
cbar.ax.set_xlabel('Surface height [$m$]\nBedmap2', linespacing=1.5)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.6_SH topography Bedmap2.png')

'''
'''
# endregion
# =============================================================================
