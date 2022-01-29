

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
# =============================================================================
# region import original data


mon_pre_org = {}
am_pre_org = {}


#### HadGEM3-GC31-LL, historical, r1i1p1f3
mon_pre_org['hg3_ll_hi_r1'] = xr.open_mfdataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/pr/gn/v20190624/*.nc',
    data_vars='minimal', coords='minimal', compat='override',)
am_pre_org['hg3_ll_hi_r1'] = mon_pre_org['hg3_ll_hi_r1'].pr.sel(
    time=slice('1979-01-01', '2014-12-30')).mean(axis=0) * 365 * 24 * 3600


#### AWI-ESM-1-1-LR, historical, r1i1p1f1
mon_pre_org['awe_lr_hi_r1'] = xr.open_mfdataset(
    '/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/Amon/pr/gn/v20200212/*.nc',
    data_vars='minimal', coords='minimal', compat='override',)
am_pre_org['awe_lr_hi_r1'] = mon_sea_ann_average(
    mon_pre_org['awe_lr_hi_r1'].pr.sel(time=slice('1979-01-01', '2014-12-30')),
    'time.year').mean(axis=0) * 365 * 24 * 3600


#### ERA5
mon_pre_org['era5'] = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')
am_pre_org['era5'] = mon_sea_ann_average(
    mon_pre_org['era5'].tp[:-2, 0, :, :].sel(
        time=slice('1979-01-01', '2014-12-30')), 'time.year'
    ).mean(axis=0) * 1000 * 365


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import cdo regridded data


mon_pre_cdorg1 = {}
am_pre_cdorg1 = {}


#### ERA5
mon_pre_cdorg1['era5'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/era5_mon_sl_79_21_pre_cdorg1.nc')
am_pre_cdorg1['era5'] = (mon_sea_ann_average(
    mon_pre_cdorg1['era5'].tp[:-2, 0, :, :].sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year')).mean(axis=0) * 365 * 1000


#### HadGEM3-GC31-LL, historical, r1i1p1f3
mon_pre_cdorg1['hg3_ll_hi_r1'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/hg3_ll_hi_r1_mon_pre_cdorg1.nc')
am_pre_cdorg1['hg3_ll_hi_r1'] = mon_pre_cdorg1['hg3_ll_hi_r1'].pr.sel(
    time=slice(
        '1979-01-01', '2014-12-30')).mean(axis=0) * 365 * 24 * 3600


#### AWI-ESM-1-1-LR, historical, r1i1p1f1
mon_pre_cdorg1['awe_lr_hi_r1'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/awe_lr_hi_r1_mon_pre_cdorg1.nc')
am_pre_cdorg1['awe_lr_hi_r1'] = mon_sea_ann_average(
    mon_pre_cdorg1['awe_lr_hi_r1'].pr.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year'
    ).mean(axis=0) * 365 * 24 * 3600


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


# check mass conservation (three are similar)
one_degree_grids_cdo = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo.nc')
areacella_hg3_ll = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/areacella/gn/v20190709/areacella_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc'
)

am_pre_hg3_ll_hi_r1_rg1 = regrid(am_pre_org['hg3_ll_hi_r1'])

((am_pre_org['hg3_ll_hi_r1'] * areacella_hg3_ll.areacella).sum(axis=None) /
 areacella_hg3_ll.areacella.sum(axis=None)).values

((am_pre_cdorg1['hg3_ll_hi_r1'] * one_degree_grids_cdo.areacella).sum(axis=None) /
 one_degree_grids_cdo.areacella.sum(axis=None)).values

((am_pre_hg3_ll_hi_r1_rg1.values * one_degree_grids_cdo.areacella.values).sum(axis=None) /
 one_degree_grids_cdo.areacella.sum(axis=None)).values

'''
# endregion
# =============================================================================


# =============================================================================
# region plot the imported data

quick_var_plot(
    var=am_pre_org['hg3_ll_hi_r1'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nHadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.0.0_global am_pre hg3_ll_hi_r1 1979_2014.png',
)

quick_var_plot(
    var=am_pre_org['awe_lr_hi_r1'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nAWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.0.1_global am_pre awe_lr_hi_r1 1979_2014.png',
)

quick_var_plot(
    var=am_pre_org['era5'], varname='pre',
    lon=am_pre_org['era5'].longitude, lat=am_pre_org['era5'].latitude,
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.0.2_global am_pre era5 1979_2014.png',
)


quick_var_plot(
    var=am_pre_org['hg3_ll_hi_r1'], varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nHadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.1.0_SH am_pre hg3_ll_hi_r1 1979_2014.png',
)

quick_var_plot(
    var=am_pre_org['awe_lr_hi_r1'], varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nAWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.1.1_SH am_pre awe_lr_hi_r1 1979_2014.png',
)

quick_var_plot(
    var=am_pre_org['era5'], varname='pre', whicharea='SH',
    lon=am_pre_org['era5'].longitude, lat=am_pre_org['era5'].latitude,
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014',
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0.1.2_SH am_pre era5 1979_2014.png',
)


'''
# check regridded products
quick_var_plot(
    var=am_pre_cdorg1['hg3_ll_hi_r1'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded HadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
)

quick_var_plot(
    var=am_pre_cdorg1['awe_lr_hi_r1'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded AWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
)

quick_var_plot(
    var=am_pre_cdorg1['era5'], varname='pre',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo regridded ERA5, 1979-2014',
)

quick_var_plot(
    var=am_pre_cdorg1['hg3_ll_hi_r1'], varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo HadGEM3-GC31-LL, historical, r1i1p1f3, 1979-2014',
)

quick_var_plot(
    var=am_pre_cdorg1['awe_lr_hi_r1'], varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo AWI-ESM-1-1-LR, historical, r1i1p1f1, 1979-2014',
)

quick_var_plot(
    var=am_pre_cdorg1['era5'], varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\ncdo ERA5, 1979-2014',
)


'''

# endregion
# =============================================================================


