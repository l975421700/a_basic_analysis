

# -----------------------------------------------------------------------------
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
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
)


from a_basic_analysis.b_module.namelist import (
    month_days,
    zerok,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import hist am sst

hist_sst_dir = 'scratch/cmip6/hist/sst/'
hist_sst_ds = ['ESACCI', ]

hist_am_sst = {}
hist_am_sst['ESACCI_org'] = xr.open_dataset(hist_sst_dir +
    'sst_mon_ESACCI-2.1_198201_201612_am.nc')
hist_am_sst['ESACCI_rg_echam6_t63'] = xr.open_dataset(hist_sst_dir + 'sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm.nc')
hist_am_sst['ESACCI_rg_echam6_t63_trim'] = xr.open_dataset(hist_sst_dir + 'sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')


cell_area = {}

cell_area['cdo_rg1'] = xr.open_dataset('others/one_degree_grids_cdo_area.nc')
cell_area['echam6_t63'] = xr.open_dataset('others/land_sea_masks/ECHAM6_T63_slm_area.nc')


'''
cell_area['cdo_rg1'].cell_area.sum(axis=None)
cell_area['echam6_t63'].cell_area.sum(axis=None)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist am sst

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()



fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    hist_am_sst['ESACCI_org'].lon,
    hist_am_sst['ESACCI_org'].lat,
    hist_am_sst['ESACCI_org'].analysed_sst.squeeze() - zerok,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nESACCI_org, 1982-2016',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.0_ESACCI_org_am_SST.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    hist_am_sst['ESACCI_rg_echam6_t63'].lon,
    hist_am_sst['ESACCI_rg_echam6_t63'].lat,
    hist_am_sst['ESACCI_rg_echam6_t63'].analysed_sst.squeeze() - zerok,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nESACCI_rg_echam6_t63, 1982-2016',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.1_ESACCI_rg_echam6_t63_am_SST.png')



'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check trimed land sea mask
# check
slm = xr.open_dataset(
    'others/land_sea_masks/ECHAM6_T63_slm.nc')
trim_sst = xr.open_dataset(
    'scratch/cmip6/hist/sst/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')


lakes = 1 - slm.slm.values
lakes[np.isfinite(trim_sst.analysed_sst.values)] = 0
np.sum(slm.slm.values == 0)
np.sum(np.isfinite(trim_sst.analysed_sst.values))
np.sum(lakes)

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    trim_sst.lon,
    trim_sst.lat,
    trim_sst.analysed_sst - zerok,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nESACCI_rg_echam6_t63_slm_trim, 1982-2016',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.2_ESACCI_rg_echam6_t63_am_SST_slm_trim.png')


pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    slm.lon,
    slm.lat,
    slm.slm,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Sea land mask [$-$]\n1=land, 0=sea/lakes',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial0.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    slm.lon,
    slm.lat,
    lakes,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Lakes [$-$]\n',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/0_test/trial1.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check SST distribution

esacci_sst = hist_am_sst['ESACCI_rg_echam6_t63_trim'].analysed_sst.values.flatten()[~np.isnan(hist_am_sst['ESACCI_rg_echam6_t63_trim'].analysed_sst.values.flatten())] - zerok
echam6_t63_cellarea = cell_area['echam6_t63'].cell_area.values.flatten()[~np.isnan(hist_am_sst['ESACCI_rg_echam6_t63_trim'].analysed_sst.values.flatten())]

# 16 bins, 16 tracers

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54, dpi=600)

plt_hist = plt.hist(
    x=(esacci_sst,),
    weights=(echam6_t63_cellarea,),
    color=['lightgray', ],
    bins=np.arange(-2, 32, 2),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(-2, 32, 2))
ax.set_xticklabels(np.arange(-2, 32, 2), size=8)
ax.set_xlabel('Annual mean ESACCI SST [$°C$] (1982-2016)', size=10)

ax.set_yticks(np.arange(0, 0.081, 0.02))
ax.set_yticklabels(np.arange(0, 0.081, 0.02), size=8)
ax.set_ylabel('Grid cells area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.97)

fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.3_global annual mean sst histogram_ESACCI_rg_echam6_t63_trim.png',)

stats.describe(esacci_sst, axis=None, nan_policy='omit')


# pltlevel = np.arange(-2, 30.01, 2)
# pltticks = np.arange(-2, 30.01, 2)

pltlevel = np.arange(-2, 30.01, 4)
pltticks = np.arange(-2, 30.01, 4)

pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    hist_am_sst['ESACCI_rg_echam6_t63_trim'].lon,
    hist_am_sst['ESACCI_rg_echam6_t63_trim'].lat,
    hist_am_sst['ESACCI_rg_echam6_t63_trim'].analysed_sst.squeeze() - zerok,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean SST [$°C$]\nESACCI_rg_echam6_t63_trim, 1982-2016',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
# fig.savefig(
#     'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.4_ESACCI_rg_echam6_t63_am_SST_trim_bins.png')
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.4_sst/2.0.4.0_esacci/2.0.4.0.5_ESACCI_rg_echam6_t63_am_SST_trim_8bins.png')

# endregion
# -----------------------------------------------------------------------------


