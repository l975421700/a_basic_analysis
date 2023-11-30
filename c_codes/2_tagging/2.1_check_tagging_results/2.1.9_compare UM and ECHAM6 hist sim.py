

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import cartopy.feature as cfeature

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    remove_trailing_zero_pos,
    remove_trailing_zero_pos_abs,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    time_weighted_mean,
    regrid,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region global plot

pre_weighted_lat = xr.open_dataset('/home/users/qino/scratch/share/for_alison/pre_weighted_lat.nc')
ocean_precipitation = xr.open_dataset('/home/users/qino/scratch/share/for_alison/ocean_precipitation.nc')

scaled_flux_Lat = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Lat.nc')

plt_data1 = pre_weighted_lat.pre_weighted_lat.weighted(
    ocean_precipitation['__xarray_dataarray_variable__']
).mean(dim='time')
plt_data2 = scaled_flux_Lat.mean_evap_source_latitude
plt_data3 = regrid(plt_data1) - regrid(plt_data2)
# stats.describe(plt_data3.values, axis=None)
output_png = 'figures/test/test.png'

#-------------------------------- plot
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PuOr',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3, cmap='PiYG',)

cbar_label1 = 'Source latitude [$°$]'
cbar_label2 = 'Differences [$°$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

# plot am values
plt_mesh1 = plot_t63_contourf(
    plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh2 = axs[1].contourf(
    plt_data2.longitude,
    plt_data2.latitude,
    plt_data2,
    levels=pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree()
)

plt_mesh3 = axs[2].contourf(
    plt_data3.lon, plt_data3.lat, plt_data3,
    levels=pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree()
)

plt.text(
    0.5, 1.05, '(a) ECHAM6',
    transform=axs[0].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b) UM',
    transform=axs[1].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c) Differences: (a) - (b)',
    transform=axs[2].transAxes, ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS plot

pre_weighted_lat = xr.open_dataset('/home/users/qino/scratch/share/for_alison/pre_weighted_lat.nc')
ocean_precipitation = xr.open_dataset('/home/users/qino/scratch/share/for_alison/ocean_precipitation.nc')

scaled_flux_Lat = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/alison/um_da072/clim/scaled_flux_Lat.nc')

plt_data1 = pre_weighted_lat.pre_weighted_lat.weighted(
    ocean_precipitation['__xarray_dataarray_variable__']
).mean(dim='time')
plt_data2 = scaled_flux_Lat.mean_evap_source_latitude
plt_data3 = regrid(plt_data1) - regrid(plt_data2)
# stats.describe(plt_data3.values, axis=None)
output_png = 'figures/test/test1.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-46, cm_max=-34, cm_interval1=1, cm_interval2=2, cmap='viridis',
    reversed=False)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-1, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='PiYG',
    asymmetric=True)

cbar_label1 = 'Source latitude [$°\;S$]'
cbar_label2 = 'Differences [$°$]'

nrow = 1
ncol = 3
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

plt_mesh1 = plot_t63_contourf(
    plt_data1.lon,
    plt_data1.lat.sel(lat=slice(-60 + 2, -90)),
    plt_data1.sel(lat=slice(-60 + 2, -90)),
    axs[0], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh2 = plot_t63_contourf(
    plt_data2.longitude,
    plt_data2.latitude.sel(latitude=slice(-90, -60 + 2)),
    plt_data2.sel(latitude=slice(-90, -60 + 2)),
    axs[1], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
# plt_mesh2 = axs[1].contourf(
#     plt_data2.longitude,
#     plt_data2.latitude,
#     plt_data2,
#     levels=pltlevel, extend='both',
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree()
# )

plt_mesh3 = axs[2].contourf(
    plt_data3.lon, plt_data3.lat, plt_data3,
    levels=pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree()
)

axs[0].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
axs[1].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
axs[2].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

plt.text(
    0.5, 1.05, '(a): ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b): UM', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c): (a) - (b)', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos_abs, )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(1.1,-2.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


