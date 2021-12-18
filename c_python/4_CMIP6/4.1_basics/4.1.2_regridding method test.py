

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
# region Annual pre in ERA5

# import data
era5_mon_sl_79_21_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')
pre = xr.concat((
    era5_mon_sl_79_21_pre.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre.tp[-2:, 1, :, :]), dim='time') * 1000


am_pr_era5 = (mon_sea_ann_average(
    pre.sel(time=slice('1979-01-01', '2014-12-30')), 'time.year') * 365
              ).mean(axis=0)

am_pr_era5.to_netcdf('bas_palaeoclim_qino/scratch/0_trial/o.nc')
# cdo -remapcon2,global_1 bas_palaeoclim_qino/scratch/0_trial/o.nc bas_palaeoclim_qino/scratch/0_trial/o_cdoremapcon2.nc
regridder_bilinear = xe.Regridder(
    era5_mon_sl_79_21_pre.tp, xe.util.grid_global(1, 1), 'bilinear')

am_pr_era5_cdoremapcon2 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/0_trial/o_cdoremapcon2.nc'
)

pltlevel = np.arange(0, 4000.01, 2)
pltticks = np.arange(0, 4000.01, 500)

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    am_pr_era5_cdoremapcon2.lon,
    am_pr_era5_cdoremapcon2.lat,
    am_pr_era5_cdoremapcon2.__xarray_dataarray_variable__,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='max')
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded ERA5, 1979-2014',
    linespacing=1.5
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/0_test/trial.png',)


'''
'''
# endregion
# =============================================================================




