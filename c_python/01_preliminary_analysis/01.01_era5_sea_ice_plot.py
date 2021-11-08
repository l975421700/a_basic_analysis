

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

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from geopy import distance

# add ellipse
from scipy import linalg
from scipy import stats
from sklearn import mixture
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

import warnings
warnings.filterwarnings('ignore')

# endregion
# =============================================================================


# =============================================================================
# region plot NH seasonal sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

lon = era5_mon_sl_79_21_sic.longitude.values
lat = era5_mon_sl_79_21_sic.latitude.values
siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby("time.season") /
    month_length.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    "time.season").sum().values, np.ones(4))

# Calculate the weighted average
siconc_weighted = (siconc * weights).groupby("time.season").sum(dim="time")

# unweighted values for comparisons
siconc_unweighted = siconc.groupby("time.season").mean("time")
siconc_diff = siconc_weighted - siconc_unweighted

fig, ax = hemisphere_plot(southextent=30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    lon, lat, siconc_weighted.sel(season = 'DJF'),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("DJF sea ice cover [-] in the ERA5 reanalysis")

fig.savefig('figures/00_test/trial')


# endregion
# =============================================================================


# =============================================================================
# region plot NH monthly sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.month').sum().values, np.ones(12))

# Calculate the weighted average
siconc_weighted_mon = (siconc * weights).groupby("time.month").sum(dim="time")

fig, ax = hemisphere_plot(southextent=30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_sic.longitude.values,
    era5_mon_sl_79_21_sic.latitude.values[0:361],
    siconc_weighted_mon.sel(month=1)[0:361, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Jan sea ice cover [-] in the ERA5 reanalysis")
# ax.add_feature(cfeature.LAND, zorder=2)
fig.savefig('figures/00_test/trial')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate NH monthly sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

# Calculate the weighted average
siconc_weighted_mon = (siconc * weights).groupby("time.month").sum(dim="time")

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(12):  # range(12):
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_sic.longitude.values,
        era5_mon_sl_79_21_sic.latitude.values[0:241],
        siconc_weighted_mon.sel(month=(i+1))[0:241, :],
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),)
    textinfo = ax.text(
        -0.1, 1, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold')
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Sea ice cover [-] in the ERA5 reanalysis (1979-2021)")

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_00_era5_sea_ice/02.00.00 monthly sea ice cover in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot NH annual sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.year').sum().values, np.ones(43))

# Calculate the weighted average
siconc_weighted_ann = (siconc * weights).groupby("time.year").sum(dim="time")

fig, ax = hemisphere_plot(southextent=30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_sic.longitude.values,
    era5_mon_sl_79_21_sic.latitude.values[0:241],
    siconc_weighted_ann.sel(year=1979)[0:241, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("1979 sea ice cover [-] in the ERA5 reanalysis")
# ax.add_feature(cfeature.LAND, zorder=2)
fig.savefig('figures/00_test/trial.png')


'''
# Wrap it into a simple function
def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")
'''
# endregion
# =============================================================================


# =============================================================================
# region animate NH annual sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.year').sum().values, np.ones(43))

# Calculate the weighted average
siconc_weighted_ann = (siconc * weights).groupby("time.year").sum(dim="time")

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(len(siconc_weighted_ann.year) - 1):  # range(2): #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_sic.longitude.values,
        era5_mon_sl_79_21_sic.latitude.values[0:241],
        siconc_weighted_ann.sel(year=(i+1979))[0:241, :],
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),)
    textinfo = ax.text(
        -0.1, 1, siconc_weighted_ann.year[i].values,
        backgroundcolor='white', transform=ax.transAxes, fontweight='bold')
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(siconc_weighted_ann.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Annual sea ice cover [-] in the ERA5 reanalysis")

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)

ani.save(
    'figures/02_era5/02_00_era5_sea_ice/02.00.01 Annual sea ice cover in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot SH seasonal sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

lon = era5_mon_sl_79_21_sic.longitude.values
lat = era5_mon_sl_79_21_sic.latitude.values
siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby("time.season") /
    month_length.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    "time.season").sum().values, np.ones(4))

# Calculate the weighted average
siconc_weighted = (siconc * weights).groupby("time.season").sum(dim="time")

# unweighted values for comparisons
siconc_unweighted = siconc.groupby("time.season").mean("time")
siconc_diff = siconc_weighted - siconc_unweighted

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(northextent=-30, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    lon, lat[480:], siconc_weighted.sel(season='DJF')[480:, ],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("DJF sea ice cover [-] in the ERA5 reanalysis")

fig.savefig('figures/00_test/trial.png')


# endregion
# =============================================================================


# =============================================================================
# region plot SH monthly sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.month').sum().values, np.ones(12))

# Calculate the weighted average
siconc_weighted_mon = (siconc * weights).groupby("time.month").sum(dim="time")

fig, ax = hemisphere_plot(northextent=-30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_sic.longitude.values,
    era5_mon_sl_79_21_sic.latitude.values[480:],
    siconc_weighted_mon.sel(month=1)[480:, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Jan sea ice cover [-] in the ERA5 reanalysis")
# ax.add_feature(cfeature.LAND, zorder=2)
fig.savefig('figures/00_test/trial.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate SH monthly sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

# Calculate the weighted average
siconc_weighted_mon = (siconc * weights).groupby("time.month").sum(dim="time")

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(12):  # range(12):
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_sic.longitude.values,
        era5_mon_sl_79_21_sic.latitude.values[480:],
        siconc_weighted_mon.sel(month=(i+1))[480:, :],
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),)
    textinfo = ax.text(
        -0.1, 1, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold')
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Sea ice cover [-] in the ERA5 reanalysis (1979-2021)")

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_00_era5_sea_ice/02.00.02 monthly sea ice cover in ERA5_SH.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot SH annual sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.year').sum().values, np.ones(43))

# Calculate the weighted average
siconc_weighted_ann = (siconc * weights).groupby("time.year").sum(dim="time")

fig, ax = hemisphere_plot(northextent=-30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_sic.longitude.values,
    era5_mon_sl_79_21_sic.latitude.values[480:],
    siconc_weighted_ann.sel(year=1979)[480:, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("1979 sea ice cover [-] in the ERA5 reanalysis")
# ax.add_feature(cfeature.LAND, zorder=2)
fig.savefig('figures/00_test/trial.png')


'''
# Wrap it into a simple function
def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")
'''
# endregion
# =============================================================================


# =============================================================================
# region animate SH annual sea ice in era5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

siconc = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

month_length = era5_mon_sl_79_21_sic.time.dt.days_in_month

# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby(
    'time.year').sum().values, np.ones(43))

# Calculate the weighted average
siconc_weighted_ann = (siconc * weights).groupby("time.year").sum(dim="time")

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(len(siconc_weighted_ann.year) - 1):  # range(2): #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_sic.longitude.values,
        era5_mon_sl_79_21_sic.latitude.values[480:],
        siconc_weighted_ann.sel(year=(i+1979))[480:, :],
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),)
    textinfo = ax.text(
        -0.1, 1, siconc_weighted_ann.year[i].values,
        backgroundcolor='white', transform=ax.transAxes, fontweight='bold')
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(siconc_weighted_ann.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Annual sea ice cover [-] in the ERA5 reanalysis")

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)

ani.save(
    'figures/02_era5/02_00_era5_sea_ice/02.00.03 Annual sea ice cover in ERA5_SH.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


'''
'''
# endregion
# =============================================================================
