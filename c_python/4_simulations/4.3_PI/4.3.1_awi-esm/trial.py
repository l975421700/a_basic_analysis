


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
import pandas as pd

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
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    awi_esm_odir,
)

# endregion
# =============================================================================


# =============================================================================
# region import output

expid = 'pi_final_qg_tag3'
yrstart = 2000
yrend = 2001


awi_esm_o = {}

#### pi_final_qg_tag3

awi_esm_o[expid] = {}

## echam
awi_esm_o[expid]['echam'] = {}

awi_esm_o[expid]['echam']['echam'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.nc')

awi_esm_o[expid]['echam']['echam_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.am.nc')

awi_esm_o[expid]['echam']['echam_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.ann.nc')

## wiso
awi_esm_o[expid]['wiso'] = {}

awi_esm_o[expid]['wiso']['wiso'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.nc')

awi_esm_o[expid]['wiso']['wiso_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.am.nc')

awi_esm_o[expid]['wiso']['wiso_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.ann.nc')

## wiso_d
awi_esm_o[expid]['wiso_d'] = {}

awi_esm_o[expid]['wiso_d']['wiso_d'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.nc')

awi_esm_o[expid]['wiso_d']['wiso_d_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.am.nc')

awi_esm_o[expid]['wiso_d']['wiso_d_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.ann.nc')


# endregion
# =============================================================================


# =============================================================================
# region check water tagging

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap.nc')
tag4_frc = (awi_esm_o[expid]['wiso']['wiso_am'].wisoaprl[0, 4, :, :] + awi_esm_o[expid]['wiso']['wiso_am'].wisoaprc[0, 4, :, :]) / (awi_esm_o[expid]['echam']['echam_am'].aprl[0, :, :] + awi_esm_o[expid]['echam']['echam_am'].aprc[0, :, :])
tag5_frc = (awi_esm_o[expid]['wiso']['wiso_am'].wisoaprl[0, 5, :, :] + awi_esm_o[expid]['wiso']['wiso_am'].wisoaprc[0, 5, :, :]) / (awi_esm_o[expid]['echam']['echam_am'].aprl[0, :, :] + awi_esm_o[expid]['echam']['echam_am'].aprc[0, :, :])
# tag5_frc.to_netcdf('output/scratch/trial.nc', mode='w')


pltlevel = np.arange(0, 1.001, 0.01)
pltticks = np.arange(0, 1.001, 0.2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag4_frc.lon,
    tag4_frc.lat,
    tag4_frc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)).reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[4, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither")

cbar.ax.set_xlabel(
    'Fraction of pre from tag region 5 [$-$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag3',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.1_pi_final_qg_tag3/6.0.1.0_tag_frac/6.0.1.0.0_global am_tag4_pre_frc awi-esm-2.1-wiso pi_final_qg_tag3.png',)


pltlevel = np.arange(0, 1.001, 0.01)
pltticks = np.arange(0, 1.001, 0.2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag5_frc.lon,
    tag5_frc.lat,
    tag5_frc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)).reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[5, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither")

cbar.ax.set_xlabel(
    'Fraction of pre from tag region 6 [$-$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag3',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.1_pi_final_qg_tag3/6.0.1.0_tag_frac/6.0.1.0.1_global am_tag5_pre_frc awi-esm-2.1-wiso pi_final_qg_tag3.png',)


pltlevel = np.arange(0, 1.001, 0.01)
pltticks = np.arange(0, 1.001, 0.2)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_o[expid]['wiso']['wiso'].lon,
    awi_esm_o[expid]['wiso']['wiso'].lat,
    (awi_esm_o[expid]['wiso']['wiso'].wisoaprl[12, 4, :, :] / awi_esm_o[expid]['echam']['echam'].aprl[12, :, :]),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)).reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),
)
# ax.contour(
#     tagmap.lon, tagmap.lat, tagmap.tagmap[5, :, :], colors='black',
#     levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither")

cbar.ax.set_xlabel(
    'Fraction of pre from tag empty region 4 in 200101 [$-$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag3',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.1_pi_final_qg_tag3/6.0.1.0_tag_frac/6.0.1.0.2_global 200101_tag4_pre_frc awi-esm-2.1-wiso pi_final_qg_tag3.png',)



# endregion
# =============================================================================


# =============================================================================
# region


# endregion
# =============================================================================

