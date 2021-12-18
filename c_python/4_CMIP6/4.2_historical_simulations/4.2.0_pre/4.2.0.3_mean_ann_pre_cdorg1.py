

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
    quick_var_plot,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

# endregion
# =============================================================================


# =============================================================================
# region import data


#### ERA5
era5_mon_sl_79_21_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/era5_mon_sl_79_21_pre_cdorg1.nc')
mon_pre_era5 = xr.concat((
    era5_mon_sl_79_21_pre_cdorg1.tp[:-2, 0, :, :],
    era5_mon_sl_79_21_pre_cdorg1.tp[-2:, 1, :, :]), dim='time') * 1000
am_pre_era5 = (mon_sea_ann_average(
    mon_pre_era5.sel(time=slice('1979-01-01', '2014-12-30')), 'time.year'
)).mean(axis=0) * 365


#### HadGEM3-GC31-LL, historical, r1i1p1f3
hg3_ll_hi_r1_mon_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/hg3_ll_hi_r1_mon_pre_cdorg1.nc')
am_pre_hg3_ll_hi_r1 = hg3_ll_hi_r1_mon_pre_cdorg1.pr.sel(time=slice(
    '1979-01-01', '2014-12-30')).mean(axis=0) * 365 * 24 * 3600
am_pre_hg3_ll_hi_r1_80 = hg3_ll_hi_r1_mon_pre_cdorg1.pr.sel(time=slice(
    '1980-01-01', '2014-12-30')).mean(axis=0) * 365 * 24 * 3600


#### AWI-ESM-1-1-LR, historical, r1i1p1f1
awe_lr_hi_r1_mon_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/awe_lr_hi_r1_mon_pre_cdorg1.nc')
am_pre_awe_lr_hi_r1 = mon_sea_ann_average(
    awe_lr_hi_r1_mon_pre_cdorg1.pr.sel(time=slice(
    '1979-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600
am_pre_awe_lr_hi_r1_80 = mon_sea_ann_average(
    awe_lr_hi_r1_mon_pre_cdorg1.pr.sel(time=slice(
    '1980-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365 * 24 * 3600


'''
# deprecated
#### Merra2
merra2_mon_sl_79_21_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/merra2_mon_sl_80_21_pre_cdorg1.nc')
am_pre_merra2 = mon_sea_ann_average(
    merra2_mon_sl_79_21_pre_cdorg1.PRECTOTCORR.sel(
    time=slice('1980-01-01', '2014-12-30')), 'time.year').mean(
        axis=0) * 365 * 24 * 3600
#### GPCP
gpcp_mon_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/gpcp_mon_pre_cdorg1.nc')
am_pre_gpcp = mon_sea_ann_average(
    gpcp_mon_pre_cdorg1.precip.sel(time=slice('1979-01-01', '2014-12-30')),
    'time.year'
).mean(axis=0) * 365
#### JRA-55
jra55_mon_pre_cdorg1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/jra55_mon_pre_cdorg1.nc')
am_pre_jra55 = mon_sea_ann_average(jra55_mon_pre_cdorg1.tpratsfc.sel(
    time=slice('1979-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 365


# check
# Global map

quick_var_plot(
    var=am_pre_era5, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded ERA5, 1979-2014',
    )

quick_var_plot(
    var=am_pre_merra2, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded MERRA2, 1980-2014',
    )

quick_var_plot(
    var=am_pre_jra55, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded JRA-55, 1979-2014',
    )

quick_var_plot(
    var=am_pre_gpcp, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded GPCP, 1979-2014',
    )

quick_var_plot(
    var=am_pr_hg3_ll_hi_r1, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded HadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    )

quick_var_plot(
    var=am_pre_awe_lr_hi_r1, varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded AWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
    )

quick_var_plot(
    var=am_pre_awe_lr_hi_r1, varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded AWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
    )


# check mass conservation
one_degree_grids_cdo = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo.nc')
areacella_hg3_ll = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/areacella/gn/v20190709/areacella_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc'
    )

pr_hg3_ll_hi_r1 = xr.open_mfdataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/pr/gn/v20190624/*.nc',
    data_vars='minimal', coords='minimal', compat='override',
)
am_pr_hg3_ll_hi_r1 = pr_hg3_ll_hi_r1.sel(time=slice(
    '1979-01-01', '2014-12-30')).pr.mean(axis=0) * 365 * 24 * 3600


((am_pr_hg3_ll_hi_r1 * areacella_hg3_ll.areacella).sum(axis=None)/areacella_hg3_ll.areacella.sum(axis=None)).values

((am_pre_hg3_ll_hi_r1 * one_degree_grids_cdo.areacella).sum(axis=None)/one_degree_grids_cdo.areacella.sum(axis=None)).values

'''


# endregion
# =============================================================================




