

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

from a00_basic_analysis.b_module.namelist import (
    month_days,
    month,
)
# endregion
# =============================================================================


# =============================================================================
# region monthly area-mean pre in HadGEM3-GC31-LL, historical, r1i1p1f3

# import area of each cell
areacella_hg3_ll = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/areacella/gn/v20190709/areacella_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc'
    )
# stats.describe(areacella_hg3_ll.areacella.values, axis = None)

# import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'Amon/'
variable = 'pr/'
grid = 'gn/'
version = 'v20190624/'

pr_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member + \
    table + variable + grid + version + '*.nc',
    )))
pr_hg3_ll_hi_r1 = xr.open_mfdataset(
    pr_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
    )
mon_pr_hg3_ll_hi_r1 = pr_hg3_ll_hi_r1.sel(
    time=slice('1979-01-01', '2014-12-30')).pr.groupby(
        'time.month').mean(dim='time')

global_mon_pr_hg3_ll_hi_r1 = ((
    mon_pr_hg3_ll_hi_r1 * areacella_hg3_ll.areacella /
    np.sum(areacella_hg3_ll.areacella)).sum(
        axis=(1, 2))).values * 24 * 3600 * month_days

sh_mon_pr_hg3_ll_hi_r1 = ((
    mon_pr_hg3_ll_hi_r1.sel(lat=slice(-90, 0)) *
    areacella_hg3_ll.areacella.sel(lat=slice(-90, 0)) /
    np.sum(areacella_hg3_ll.areacella.sel(lat=slice(-90, 0)))).sum(
        axis=(1, 2))).values * 24 * 3600 * month_days

sh60_mon_pr_hg3_ll_hi_r1 = ((
    mon_pr_hg3_ll_hi_r1.sel(lat=slice(-90, -60)) *
    areacella_hg3_ll.areacella.sel(lat=slice(-90, -60)) /
    np.sum(areacella_hg3_ll.areacella.sel(lat=slice(-90, -60)))).sum(
        axis=(1, 2))).values * 24 * 3600 * month_days

'''
nh_mon_pr_hg3_ll_hi_r1 = ((
    mon_pr_hg3_ll_hi_r1.sel(lat=slice(0, 90)) *
    areacella_hg3_ll.areacella.sel(lat=slice(0, 90)) /
    np.sum(areacella_hg3_ll.areacella.sel(lat=slice(0, 90)))).sum(
        axis=(1, 2))).values * 24 * 3600 * month_days

nh60_mon_pr_hg3_ll_hi_r1 = ((
    mon_pr_hg3_ll_hi_r1.sel(lat=slice(60, 90)) *
    areacella_hg3_ll.areacella.sel(lat=slice(60, 90)) /
    np.sum(areacella_hg3_ll.areacella.sel(lat=slice(60, 90)))).sum(
        axis=(1, 2))).values * 24 * 3600 * month_days
'''
# endregion
# =============================================================================


# =============================================================================
# region monthly area-mean pre in ERA5

# import data
era5_mon_sl_79_21_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')

pre = xr.concat((
    era5_mon_sl_79_21_pre.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre.tp[-2:, 1, :, :]), dim='time') * 1000


# endregion
# =============================================================================


# =============================================================================
# region plot the monthly pre

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi = 600)

plt1, = ax.plot(
    month, global_mon_pr_hg3_ll_hi_r1,
    '.-', markersize=2.5, linewidth=0.5, color='black',)
plt2, = ax.plot(
    month, sh_mon_pr_hg3_ll_hi_r1,
    '.--', markersize=2.5, linewidth=0.5, color='black',)
plt3, = ax.plot(
    month, sh60_mon_pr_hg3_ll_hi_r1,
    '.:', markersize=2.5, linewidth=0.5, color='black',)

ax_legend = ax.legend(
    [plt1, plt2, plt3],
    ['Global', 'Southern hemisphere', '60° S to 90° S', ],
    loc='lower center', frameon=False, ncol=3, fontsize=8,
    bbox_to_anchor=(0.5, -0.28), handlelength=1,
    columnspacing=1)

ax.set_xticks(month)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 121, 20))
ax.set_yticklabels(np.arange(0, 121, 20))
# ax.set_xlabel('', size=10)
ax.set_ylabel("Monthly precipitation [$mm\;mon^{-1}$]", size=10)
ax.set_ylim(0, 130)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.2, top=0.96)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.0_precipitation/4.0.0.1_monthly_pre/4.0.0.1.0_monthly precipitation.png',)

'''
(mon_pr_hg3_ll_hi_r1 * areacella_hg3_ll.areacella)[10, 20, 30].values
(mon_pr_hg3_ll_hi_r1[10, 20, 30] * areacella_hg3_ll.areacella[20, 30]).values

'''
# endregion
# =============================================================================




