

# =============================================================================
# region import packages

# management
import glob
from pickletools import float8
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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month_days,
    month,
    seasons,
)
# endregion
# =============================================================================


# =============================================================================
# region era5 wind10

era5_wind10_am = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/wind10/wind10m_ERA5_mon_sl_197901_201412_am.nc')

era5_gridarea = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/constants/ERA5_gridarea.nc')
era5_lsm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_lsmask.nc')


era5_wind10_am_strength = era5_wind10_am.si10.squeeze()
era5_gridarea_values = era5_gridarea.cell_area
# 7 bins, 8 tracers

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54, dpi=600)

plt_hist = plt.hist(
    x=(era5_wind10_am_strength.values[era5_lsm.lsm.squeeze() == 0],),
    weights=(era5_gridarea_values.values[era5_lsm.lsm.squeeze() == 0],),
    color=['lightgray', ],
    bins=np.arange(3, 13, 1),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(3, 13, 1))
ax.set_xticklabels(np.arange(3, 13, 1), size=8)
ax.set_xlabel('Annual mean 10 metre wind speed [$m \; s^{-1}$] over ocean\nERA5, 1979-2014', linespacing=1.5)

ax.set_yticks(np.arange(0, 0.31, 0.05))
ax.set_yticklabels(np.arange(0, 0.3001, 0.05, dtype=np.float32), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.97)

fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.5_wind10/2.0.5.0.0_global annual mean 10 metre winds histogram_ERA5.png',)


pltlevel = np.arange(0, 12.001, 1)
pltticks = np.arange(0, 12.001, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
# pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()
pltcmp = cm.get_cmap('viridis', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_wind10_am_strength.longitude,
    era5_wind10_am_strength.latitude,
    era5_wind10_am_strength,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="max",)

cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m \; s^{-1}$]\nERA5, 1979-2014',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.5_wind10/2.0.5.0.1_global annual mean 10 metre winds_ERA5.png')


era5_wind10_am_strength_noland = era5_wind10_am_strength.copy()
era5_wind10_am_strength_noland.values[era5_lsm.lsm.squeeze() != 0] = np.nan

pltlevel = np.arange(3, 12.001, 1)
pltticks = np.arange(3, 12.001, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()
# pltcmp = cm.get_cmap('viridis', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_wind10_am_strength.longitude,
    era5_wind10_am_strength.latitude,
    era5_wind10_am_strength_noland,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean 10 metre wind speed [$m \; s^{-1}$] over ocean\nERA5, 1979-2014',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.5_wind10/2.0.5.0.2_global annual mean 10 metre winds_ERA5_noland.png')

'''
'''
# endregion
# =============================================================================


