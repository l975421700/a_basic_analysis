

# =============================================================================
# region import packages

# management
import statsmodels.api as sm
from calendar import monthcalendar
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
import pickle
import pandas as pd
import pymannkendall as mk

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
    quick_var_plot,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    create_ais_mask,
)

from a00_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion

# region input AIS masks, 1d grid area

with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)
with open('bas_palaeoclim_qino/others/ais_area.pickle', 'rb') as handle:
    ais_area = pickle.load(handle)

grid_area_1d = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import cdo regridded data and calculate spatial average

mon_pre_cdorg1 = {}
mon_pre_var = {}
mon_pre_spa = {}
ann_pre_spa = {}

#### ERA5
mon_pre_cdorg1['era5'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/era5_mon_sl_79_21_pre_cdorg1.nc')
mon_pre_var['era5'] = mon_pre_cdorg1['era5'].tp[:, 0, :, :].sel(time=slice(
    '1979-01-01', '2014-12-30')) * 1000

mon_pre_spa['era5'] = {}
mon_pre_spa['era5']['eais'] = \
    (mon_pre_var['era5'] * grid_area_1d.cell_area * ais_masks['eais_mask01']
     ).sum(axis=(1, 2)) / ais_area['eais'] * \
    np.tile(month_days, int(mon_pre_var['era5'].shape[0]/12))
mon_pre_spa['era5']['wais'] = \
    (mon_pre_var['era5'] * grid_area_1d.cell_area * ais_masks['wais_mask01']
     ).sum(axis=(1, 2)) / ais_area['wais'] * \
    np.tile(month_days, int(mon_pre_var['era5'].shape[0]/12))
mon_pre_spa['era5']['ap'] = \
    (mon_pre_var['era5'] * grid_area_1d.cell_area * ais_masks['ap_mask01']
     ).sum(axis=(1, 2)) / ais_area['ap'] * \
    np.tile(month_days, int(mon_pre_var['era5'].shape[0]/12))
mon_pre_spa['era5']['ais'] = (
    mon_pre_spa['era5']['eais'] * ais_area['eais'] + \
        mon_pre_spa['era5']['wais'] * ais_area['wais'] + \
    mon_pre_spa['era5']['ap'] * ais_area['ap']) / ais_area['ais']

ann_pre_spa['era5'] = {}
ann_pre_spa['era5']['eais'] = mon_pre_spa['era5']['eais'].groupby(
    'time.year').sum(dim='time')
ann_pre_spa['era5']['wais'] = mon_pre_spa['era5']['wais'].groupby(
    'time.year').sum(dim='time')
ann_pre_spa['era5']['ap'] = mon_pre_spa['era5']['ap'].groupby(
    'time.year').sum(dim='time')
ann_pre_spa['era5']['ais'] = mon_pre_spa['era5']['ais'].groupby(
    'time.year').sum(dim='time')


#### HadGEM3-GC31-LL, historical, r1i1p1f3
mon_pre_cdorg1['hg3_ll_hi_r1'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/hg3_ll_hi_r1_mon_pre_cdorg1.nc')
mon_pre_var['hg3_ll_hi_r1'] = mon_pre_cdorg1['hg3_ll_hi_r1'].pr * 24 * 3600

mon_pre_spa['hg3_ll_hi_r1'] = {}
mon_pre_spa['hg3_ll_hi_r1']['eais'] = \
    (mon_pre_var['hg3_ll_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['eais_mask01']).sum(axis=(1, 2)) / ais_area['eais'] * \
    np.tile(month_days, int(mon_pre_var['hg3_ll_hi_r1'].shape[0]/12))
mon_pre_spa['hg3_ll_hi_r1']['wais'] = \
    (mon_pre_var['hg3_ll_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['wais_mask01']).sum(axis=(1, 2)) / ais_area['wais'] * \
    np.tile(month_days, int(mon_pre_var['hg3_ll_hi_r1'].shape[0]/12))
mon_pre_spa['hg3_ll_hi_r1']['ap'] = \
    (mon_pre_var['hg3_ll_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['ap_mask01']).sum(axis=(1, 2)) / ais_area['ap'] * \
    np.tile(month_days, int(mon_pre_var['hg3_ll_hi_r1'].shape[0]/12))
mon_pre_spa['hg3_ll_hi_r1']['ais'] = \
    (mon_pre_var['hg3_ll_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['ais_mask01']).sum(axis=(1, 2)) / ais_area['ais'] * \
    np.tile(month_days, int(mon_pre_var['hg3_ll_hi_r1'].shape[0]/12))

ann_pre_spa['hg3_ll_hi_r1'] = {}
ann_pre_spa['hg3_ll_hi_r1']['eais'] = mon_pre_spa['hg3_ll_hi_r1'][
    'eais'].groupby('time.year').sum(dim='time')
ann_pre_spa['hg3_ll_hi_r1']['wais'] = mon_pre_spa['hg3_ll_hi_r1'][
    'wais'].groupby('time.year').sum(dim='time')
ann_pre_spa['hg3_ll_hi_r1']['ap'] = mon_pre_spa['hg3_ll_hi_r1'][
    'ap'].groupby('time.year').sum(dim='time')
ann_pre_spa['hg3_ll_hi_r1']['ais'] = mon_pre_spa['hg3_ll_hi_r1'][
    'ais'].groupby('time.year').sum(dim='time')


#### AWI-ESM-1-1-LR, historical, r1i1p1f1
mon_pre_cdorg1['awe_lr_hi_r1'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/awe_lr_hi_r1_mon_pre_cdorg1.nc')
mon_pre_var['awe_lr_hi_r1'] = mon_pre_cdorg1['awe_lr_hi_r1'].pr * 24 * 3600

mon_pre_spa['awe_lr_hi_r1'] = {}
mon_pre_spa['awe_lr_hi_r1']['eais'] = \
    (mon_pre_var['awe_lr_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['eais_mask01']).sum(axis=(1, 2)) / ais_area['eais'] * \
    np.tile(month_days, int(mon_pre_var['awe_lr_hi_r1'].shape[0]/12))
mon_pre_spa['awe_lr_hi_r1']['wais'] = \
    (mon_pre_var['awe_lr_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['wais_mask01']).sum(axis=(1, 2)) / ais_area['wais'] * \
    np.tile(month_days, int(mon_pre_var['awe_lr_hi_r1'].shape[0]/12))
mon_pre_spa['awe_lr_hi_r1']['ap'] = \
    (mon_pre_var['awe_lr_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['ap_mask01']).sum(axis=(1, 2)) / ais_area['ap'] * \
    np.tile(month_days, int(mon_pre_var['awe_lr_hi_r1'].shape[0]/12))
mon_pre_spa['awe_lr_hi_r1']['ais'] = \
    (mon_pre_var['awe_lr_hi_r1'] * grid_area_1d.cell_area *
     ais_masks['ais_mask01']).sum(axis=(1, 2)) / ais_area['ais'] * \
    np.tile(month_days, int(mon_pre_var['awe_lr_hi_r1'].shape[0]/12))

ann_pre_spa['awe_lr_hi_r1'] = {}
ann_pre_spa['awe_lr_hi_r1']['eais'] = mon_pre_spa['awe_lr_hi_r1'][
    'eais'].groupby('time.year').sum(dim='time')
ann_pre_spa['awe_lr_hi_r1']['wais'] = mon_pre_spa['awe_lr_hi_r1'][
    'wais'].groupby('time.year').sum(dim='time')
ann_pre_spa['awe_lr_hi_r1']['ap'] = mon_pre_spa['awe_lr_hi_r1'][
    'ap'].groupby('time.year').sum(dim='time')
ann_pre_spa['awe_lr_hi_r1']['ais'] = mon_pre_spa['awe_lr_hi_r1'][
    'ais'].groupby('time.year').sum(dim='time')

'''
np.max(np.abs((mon_pre_spa['hg3_ll_hi_r1']['ais'] - (
    mon_pre_spa['hg3_ll_hi_r1']['eais'] * ais_area['eais'] +
    mon_pre_spa['hg3_ll_hi_r1']['wais'] * ais_area['wais'] +
    mon_pre_spa['hg3_ll_hi_r1']['ap'] * ais_area['ap']) / ais_area['ais'])))
'''

# endregion
# =============================================================================


# =============================================================================
# region plot the annual spatial average

mk.original_test(ann_pre_spa['era5']['ais'])
mk.original_test(ann_pre_spa['hg3_ll_hi_r1']['ais'].sel(
    year=slice('1979', '2014')))
mk.original_test(ann_pre_spa['awe_lr_hi_r1']['ais'].sel(
    year=slice('1979', '2014')))

# 1979-2014
# stats.describe(ann_pre_spa['era5']['ais'])
fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

plt_e, = ax.plot(
    ann_pre_spa['era5']['ais'].sel(year=slice('1979', '2014')).year,
    ann_pre_spa['era5']['ais'].sel(year=slice('1979', '2014')),
    '.-', markersize=2.5, linewidth=0.5, color='black',)

plt_h, = ax.plot(
    ann_pre_spa['hg3_ll_hi_r1']['ais'].sel(year=slice('1979', '2014')).year,
    ann_pre_spa['hg3_ll_hi_r1']['ais'].sel(year=slice('1979', '2014')),
    '.-', markersize=2.5, linewidth=0.5, color='blue',)

plt_a, = ax.plot(
    ann_pre_spa['awe_lr_hi_r1']['ais'].sel(year=slice('1979', '2014')).year,
    ann_pre_spa['awe_lr_hi_r1']['ais'].sel(year=slice('1979', '2014')),
    '.-', markersize=2.5, linewidth=0.5, color='red',)

ax_legend = ax.legend(
    [plt_e, plt_h, plt_a],
    ['ERA5', 'HadGEM3-GC3.1-LL hist1', 'AWI-ESM-1-1-LR hist1', ],
    loc='lower center', frameon=False, ncol=3, fontsize=8,
    bbox_to_anchor=(0.45, -0.22), handlelength=1,
    columnspacing=0.5)

ax.set_xticks(ann_pre_spa['era5']['ap'].year[::5])
ax.set_xticklabels(ann_pre_spa['era5']['ap'].year[::5].values, size=8)
ax.set_yticks(np.arange(150, 201, 10))
ax.set_yticklabels(np.arange(150, 201, 10))
# ax.set_xlabel('', size=10)
ax.set_ylabel("Annual precipitation [$mm\;yr^{-1}$]", size=10)
ax.set_ylim(145, 200)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.2, top=0.96)

fig.savefig(
    'figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.2_ann_pre_spa/4.1.0.2.0_ann_pre_spa_ais_79_14.png',)


mk.original_test(ann_pre_spa['era5']['ais'])
mk.original_test(ann_pre_spa['hg3_ll_hi_r1']['ais'])
mk.original_test(ann_pre_spa['awe_lr_hi_r1']['ais'])

# 1850-2014
fig, ax = plt.subplots(1, 1, figsize=np.array([17.6, 8]) / 2.54, dpi=600)

plt_e, = ax.plot(
    ann_pre_spa['era5']['ais'].year,
    ann_pre_spa['era5']['ais'],
    '.-', markersize=2.5, linewidth=0.5, color='black',)

plt_h, = ax.plot(
    ann_pre_spa['hg3_ll_hi_r1']['ais'].year,
    ann_pre_spa['hg3_ll_hi_r1']['ais'],
    '.-', markersize=2.5, linewidth=0.5, color='blue',)

plt_a, = ax.plot(
    ann_pre_spa['awe_lr_hi_r1']['ais'].year,
    ann_pre_spa['awe_lr_hi_r1']['ais'],
    '.-', markersize=2.5, linewidth=0.5, color='red',)

ax_legend = ax.legend(
    [plt_e, plt_h, plt_a],
    ['ERA5', 'HadGEM3-GC3.1-LL hist1', 'AWI-ESM-1-1-LR hist1', ],
    loc='lower center', frameon=False, ncol=3, fontsize=8,
    bbox_to_anchor=(0.45, -0.2), handlelength=1,
    columnspacing=0.5)

ax.set_xticks(np.arange(1850, 2016, 15))
ax.set_xticklabels(np.arange(1850, 2016, 15), size=8)
ax.set_yticks(np.arange(140, 200, 10))
ax.set_yticklabels(np.arange(140, 200, 10))
# ax.set_xlabel('', size=10)
ax.set_ylabel("Annual precipitation [$mm\;yr^{-1}$]", size=10)
ax.set_ylim(135, 195)
# ax.set_xlim(1845, 2014)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.96)

fig.savefig(
    'figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.2_ann_pre_spa/4.1.0.2.1_ann_pre_spa_ais_1850_2014.png',)





'''
fig, ax = plt.subplots(figsize=(12, 8))
sm.graphics.tsa.plot_acf(ann_pre_spa['era5']['ais'], lags=20, ax=ax)
sm.graphics.tsa.plot_acf(ann_pre_spa['hg3_ll_hi_r1']['ais'], lags=20, ax=ax)
sm.graphics.tsa.plot_acf(ann_pre_spa['awe_lr_hi_r1']['ais'], lags=20, ax=ax)
fig.savefig('figures/0_test/trial.png')
'''
# endregion
# =============================================================================


# =============================================================================
# region plot the monthly spatial average

stats.describe(mon_pre_spa['era5']['ais'])

fig, ax = plt.subplots(1, 1, figsize=np.array([17.6, 16]) / 2.54, dpi=600)

plt_ais, = ax.plot(
    mon_pre_spa['era5']['ais'].time, mon_pre_spa['era5']['ais'],
    '.-', markersize=2.5, linewidth=0.5, color='black',)

# plt1, = ax.plot(
#     mon_pre_spa['era5']['ap'].time, mon_pre_spa['era5']['eais'],
#     '.-', markersize=2.5, linewidth=0.5, color='black',)
# plt2, = ax.plot(
#     mon_pre_spa['era5']['ap'].time, mon_pre_spa['era5']['wais'],
#     '.--', markersize=2.5, linewidth=0.5, color='black',)
# plt3, = ax.plot(
#     mon_pre_spa['era5']['ap'].time, mon_pre_spa['era5']['ap'],
#     '.:', markersize=2.5, linewidth=0.5, color='black',)

# ax_legend = ax.legend(
#     [plt1, plt2, plt3],
#     ['EAIS', 'WAIS', 'AP', ],
#     loc='lower center', frameon=False, ncol=3, fontsize=8,
#     bbox_to_anchor=(0.5, -0.28), handlelength=1,
#     columnspacing=1)

ax.set_xticks(mon_pre_spa['era5']['ap'].time[::60])
ax.set_xticklabels(
    pd.DatetimeIndex(mon_pre_spa['era5']['ap'].time[::60]).year, size=8)
ax.set_yticks(np.arange(5, 31, 5))
ax.set_yticklabels(np.arange(5, 31, 5))
# ax.set_xlabel('', size=10)
ax.set_ylabel("Monthly precipitation [$mm\;mon^{-1}$]", size=10)
ax.set_ylim(5, 30)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.98)

fig.savefig(
    'figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.1_mon_pre_spa/4.1.0.1.0_mon_pre_spa_ais.png',)


# endregion
# =============================================================================





