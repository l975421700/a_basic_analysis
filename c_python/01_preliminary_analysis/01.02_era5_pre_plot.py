

# =============================================================================
# region import packages


# basic library
import numpy as np
import xarray as xr
import datetime
import glob
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# plot
import matplotlib.path as mpath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
mpl.rcParams['figure.dpi'] = 600
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'data_source/bg_cartopy'
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from geopy import distance
from scipy import linalg
from scipy import stats
from sklearn import mixture
import metpy.calc as mpcalc
from metpy.units import units

# self defined function and namelist
from a00_basic_analysis.b_module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot1,
    hemisphere_plot,
)

from a00_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

import warnings
warnings.filterwarnings('ignore')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot ERA5 topography

era5_mon_sl_20_gph = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_20_gph.nc'
    )

era5_topograph = mpcalc.geopotential_to_height(
    era5grid.Z.squeeze().values * units('meter ** 2 / second ** 2'))
# mpcalc.height_to_geopotential(era5_topograph.magnitude * units.m)
# era5_topograph.magnitude


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 10.5]) / 2.54)

pltlevel = np.arange(0, 5501, 50)
pltticks = np.arange(0, 5501, 500)

plt_cmp = ax.pcolormesh(
    era5grid.lon.values,
    era5grid.lat.values,
    era5_topograph,
    cmap=cm.get_cmap('Greys', len(pltlevel)),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    transform=ccrs.PlateCarree(), rasterized=True)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.09, shrink=0.8, aspect=30, anchor=(0.5, -2),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel("Topography [m] in the ERA5 reanalysis")

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)
fig.savefig('figures/02_era5/02_02_era5_constants/02.02.00 topography in era5.png', dpi=1200)


'''
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')

# check
coastline = ctp.feature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.5)
borders = ctp.feature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black',
    facecolor='none', lw=0.5)

era5grid = xr.open_dataset('ERA5CONST.nc')

era5_topograph = mpcalc.geopotential_to_height(
    era5grid.Z.squeeze().values * units('meter ** 2 / second ** 2'))

# mpcalc.height_to_geopotential(era5_topograph.magnitude * units.m)
# era5_topograph.magnitude

ticklabel = ticks_labels(-180, 180, -90, 90, 60, 30)
extent = [-180, 180, -90, 90]
transform = ctp.crs.PlateCarree()

fig, ax = plt.subplots(
    1, 1, figsize=np.array([15, 8.8]) / 2.54,
    subplot_kw={'projection': transform})
ax.set_extent(extent, crs=transform)
ax.set_xticks(ticklabel[0])
ax.set_xticklabels(ticklabel[1])
ax.set_yticks(ticklabel[2])
ax.set_yticklabels(ticklabel[3])


dem_level = np.arange(0, 5501, 50)
ticks = np.arange(0, 5501, 500)
plt_era5_dem = ax.pcolormesh(
    era5grid.lon.values, era5grid.lat.values, era5_topograph,
    cmap=cm.get_cmap('Greys', len(dem_level)),
    norm=BoundaryNorm(dem_level, ncolors=len(dem_level), clip=False),
    transform=transform, rasterized=True)

cbar = fig.colorbar(
    plt_era5_dem, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.09, shrink=0.8, aspect=30, anchor=(0.5, -2),
    ticks=ticks, extend='both')
cbar.ax.set_xlabel("Topograph in ERA5")

gl = ax.gridlines(crs=transform, linewidth=0.25,
                  color='gray', alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel[0])
gl.ylocator = mticker.FixedLocator(ticklabel[2])

ax.add_feature(coastline, lw=0.1)
ax.add_feature(borders, lw=0.1)

fig.subplots_adjust(left=0.075, right=0.96, bottom=0.08, top=0.98)
fig.savefig('figures/00_test/trial.png', dpi=1200)
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate monthly precipitation in era5

era5_mon_sl_79_21_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')

pre = xr.concat((
    era5_mon_sl_79_21_pre.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre.tp[-2:, 1, :, :]), dim='time') * 1000

pre_mon_average = mon_sea_ann_average(pre, 'time.month')
pre_mon_sum = pre_mon_average * month_days[:, None, None]

pltlevel = np.arange(0, 400.01, 1)
pltticks = np.arange(0, 400.01, 50)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot global monthly precipitation in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_pre.longitude.values,
    era5_mon_sl_79_21_pre.latitude.values,
    pre_mon_sum.sel(month=1),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in ERA5 (1979-2021)')
# ax.set_title(month[0], pad=5, size=10)
ax.text(
    0.5, 1.05, month[0], backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)
fig.savefig('figures/00_test/trial.png', dpi=1200)

# endregion
# =============================================================================


# =============================================================================
# region animate global monthly precipitation in era5

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
ims = []
for i in range(12):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_pre.longitude.values,
        era5_mon_sl_79_21_pre.latitude.values,
        pre_mon_sum.sel(month=(i+1)),
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),
        )
    textinfo = ax.text(
        0.5, 1.05, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Monthly precipitation [mm] in the ERA5 reanalysis (1979-2021)')
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_01_era5_pre/02.01.00 monthly precipitation in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate annual precipitation in era5

era5_mon_sl_79_21_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')

pre = xr.concat((
    era5_mon_sl_79_21_pre.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre.tp[-2:, 1, :, :]), dim='time') * 1000

pre_ann_average = mon_sea_ann_average(pre, 'time.year')
pre_ann_sum = pre_ann_average * 365

pltlevel = np.arange(0, 4000.01, 10)
pltticks = np.arange(0, 4000.01, 800)


'''
# check
stats.describe(pre_ann_sum.sel(year=1979), axis=None)

pre_1979 = (era5_mon_sl_79_21_pre.tp[0:12, 0, :, :].values * 1000 * \
    month_days[:, None, None]).sum(axis = 0)
stats.describe(pre_1979, axis=None)
np.max(abs(pre_ann_sum.sel(year=1979).values - pre_1979))

'''
# endregion
# =============================================================================


# =============================================================================
# region plot global annual precipitation in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_pre.longitude.values,
    era5_mon_sl_79_21_pre.latitude.values,
    pre_ann_sum.sel(year=1979),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in the ERA5 reanalysis')
ax.text(
    0.5, 1.05, '1979', backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)
fig.savefig('figures/00_test/trial.png', dpi=1200)

# endregion
# =============================================================================


# =============================================================================
# region animate global annual precipitation in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
ims = []

for i in range(len(pre_ann_sum.year) - 1):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_pre.longitude.values,
        era5_mon_sl_79_21_pre.latitude.values,
        pre_ann_sum.sel(year=(i+1979)),
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    
    textinfo = ax.text(
        0.5, 1.05, pre_ann_sum.year[i].values, backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(pre_ann_sum.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual precipitation [mm] in the ERA5 reanalysis')
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/02_era5/02_01_era5_pre/02.01.01 Annual precipitation in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''

'''
# endregion
# =============================================================================



