

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
import metpy.calc as mpcalc
from metpy.units import units

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
# region calculate 2m RH from tem and prs

era5_2m_tem_prs = xr.open_dataset('scratch/cmip6/hist/rh/2m_tem_prs_ERA5_mon_sl_197901_201412.nc')

# era5_2m_tem_prs.d2m
# era5_2m_tem_prs.t2m

era5_2m_rh = mpcalc.relative_humidity_from_dewpoint(
    era5_2m_tem_prs.t2m * units('K'),
    era5_2m_tem_prs.d2m * units('K'),
)

era5_2m_rh = era5_2m_rh.rename('rh2m')

era5_2m_rh.to_netcdf('scratch/cmip6/hist/rh/2m_rh_ERA5_mon_sl_197901_201412.nc')

'''
i = 10
j = 20
k = 30
era5_2m_tem_prs.t2m[i, j, k] * units('K')
era5_2m_tem_prs.d2m[i, j, k] * units('K')
mpcalc.relative_humidity_from_dewpoint(
    era5_2m_tem_prs.t2m[i, j, k] * units('K'),
    era5_2m_tem_prs.d2m[i, j, k] * units('K'),
)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot 2m RH

era5_2m_rh_am = xr.open_dataset('scratch/cmip6/hist/rh/2m_rh_ERA5_mon_sl_197901_201412_am.nc')

era5_gridarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')
era5_gridarea_values = era5_gridarea.cell_area
era5_lsm = xr.open_dataset('observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_lsmask.nc')


stats.describe(era5_2m_rh_am.rel_hum.squeeze().values[era5_lsm.lsm.squeeze() == 0] * 100)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54, dpi=600)

plt_hist = plt.hist(
    x=(era5_2m_rh_am.rel_hum.squeeze().values[era5_lsm.lsm.squeeze() == 0] * 100,),
    weights=(era5_gridarea_values.values[era5_lsm.lsm.squeeze() == 0],),
    color=['lightgray', ],
    bins=np.arange(68, 88.1, 2, dtype=np.int32),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(68, 88.1, 2, dtype=np.int32))
ax.set_xticklabels(np.arange(68, 88.1, 2, dtype=np.int32), size=8)
ax.set_xlabel('Annual mean 2 metre relative humidity [$\%$] over ocean\nERA5, 1979-2014', linespacing=1.5)

ax.set_yticks(np.arange(0, 0.1501, 0.03))
ax.set_yticklabels(np.around(np.arange(0, 0.1501, 0.03), 2), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.97)

fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.6_rh/2.0.6.0.0_global annual mean 2 metre relative humidity histogram_ERA5.png',)


pltlevel = np.arange(68, 88.1, 2, dtype=np.int32)
pltticks = np.arange(68, 88.1, 2, dtype=np.int32)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
# pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()
pltcmp = cm.get_cmap('Blues', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_2m_rh_am.rel_hum.squeeze().longitude,
    era5_2m_rh_am.rel_hum.squeeze().latitude,
    era5_2m_rh_am.rel_hum.squeeze() * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean 2 metre relative humidity [$\%$]\nERA5, 1979-2014',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.6_rh/2.0.6.0.1_global annual mean 2 metre relative humidity_ERA5.png')


era5_2m_rh_am_noland = era5_2m_rh_am.rel_hum.squeeze().copy()
era5_2m_rh_am_noland.values[era5_lsm.lsm.squeeze() != 0] = np.nan

pltlevel = np.arange(68, 88.1, 2, dtype=np.int32)
pltticks = np.arange(68, 88.1, 2, dtype=np.int32)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()
# pltcmp = cm.get_cmap('Blues', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_2m_rh_am.rel_hum.squeeze().longitude,
    era5_2m_rh_am.rel_hum.squeeze().latitude,
    era5_2m_rh_am_noland * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Annual mean 2 metre relative humidity [$\%$] over ocean\nERA5, 1979-2014',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.6_rh/2.0.6.0.2_global annual mean 2 metre relative humidity_ERA5_noland.png')



# endregion
# =============================================================================


