

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
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    regrid,
)

from a00_basic_analysis.b_module.namelist import (
    month_days,
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in HadGEM3-GC31-LL, historical, r1i1p1f3

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'Omon/'
variable = 'tos/'
grid = 'gn/'
version = 'v20190624/'

sst_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))
sst_hg3_ll_hi_r1 = xr.open_mfdataset(
    sst_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_sst_hg3_ll_hi_r1 = sst_hg3_ll_hi_r1.tos.sel(time=slice(
    '1979-01-01', '2014-12-30')).mean(axis=0)

am_sst_hg3_ll_hi_r1_rg1 = regrid(
    am_sst_hg3_ll_hi_r1, method='nearest_s2d').values



'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in AWI-CM-1-1-MR, historical, r1i1p1f1

am_sst_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/sst/am_sst_awc_mr_hi_r1.nc'
)

am_sst_awc_mr_hi_r1_rg1 = regrid(am_sst_awc_mr_hi_r1.tos).values


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean SST in ERA5

era5_mon_sl_79_21_sst = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sst.nc')

mon_sst_era5 = xr.concat((
    era5_mon_sl_79_21_sst.sst[:-2, 0, :, :],
    era5_mon_sl_79_21_sst.sst[-2:, 1, :, :]), dim='time')

am_sst_era5 = mon_sea_ann_average(
    mon_sst_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year', skipna=False
).mean(axis=0) - zerok

am_sst_era5_rg1 = regrid(am_sst_era5).values

# endregion
# =============================================================================


# =============================================================================
# region histogram plot

one_degree_grids = xe.util.grid_global(1, 1)
one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

plt_hist = plt.hist(
    x=(
        am_sst_era5_rg1.flatten()[~np.isnan(
            am_sst_era5_rg1.flatten())],
        am_sst_hg3_ll_hi_r1_rg1.flatten()[~np.isnan(
            am_sst_hg3_ll_hi_r1_rg1.flatten())],
        am_sst_awc_mr_hi_r1_rg1.flatten()[~np.isnan(
            am_sst_awc_mr_hi_r1_rg1.flatten())],
        ),
    weights=(
        one_degree_grids_cdo_area.cell_area.values.flatten()[~np.isnan(
            am_sst_era5_rg1.flatten())],
        one_degree_grids_cdo_area.cell_area.values.flatten()[~np.isnan(
            am_sst_hg3_ll_hi_r1_rg1.flatten())],
        one_degree_grids_cdo_area.cell_area.values.flatten()[~np.isnan(
            am_sst_awc_mr_hi_r1_rg1.flatten())],
    ),
    color=['black', 'gray', 'lightgray'],
    bins=np.arange(-2, 38, 4),
    density=True,
    rwidth=0.8,
)

ax.legend(
    plt_hist[2], ['ERA5',
                  'HadGEM3-GC31-LL, historical',
                  'AWI-CM-1-1-MR, historical'],
    loc='upper left', fontsize=8, handletextpad=0.2,
    )

ax.set_xticks(np.arange(-2, 38, 4))
ax.set_xticklabels(np.arange(-2, 38, 4), size=8)
ax.set_xlabel('Annual mean SST [$°C$] (1979-2014)', size=10)

ax.set_yticks(np.arange(0, 0.081, 0.02))
ax.set_yticklabels(np.arange(0, 0.081, 0.02), size=8)
ax.set_ylabel('Grid cells area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.17, right=0.99, bottom=0.14, top=0.97)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.3_sst/4.0.3.16_global annual mean sst histogram.png',)


'''
stats.describe(am_sst_hg3_ll_hi_r1_rg1.flatten()[~np.isnan(am_sst_hg3_ll_hi_r1_rg1.flatten())])
stats.describe(am_sst_awc_mr_hi_r1_rg1.flatten()[~np.isnan(
                   am_sst_awc_mr_hi_r1_rg1.flatten())])
stats.describe(am_sst_era5_rg1.flatten()[~np.isnan(
                   am_sst_era5_rg1.flatten())])
'''
# endregion
# =============================================================================

