

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
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in HadGEM3-GC31-LL, historical, r1i1p1f3

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'MOHC/'
source = 'HadGEM3-GC31-LL/'
experiment = 'historical/'
member = 'r1i1p1f3/'
table = 'SImon/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20200330/'

siconc_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member + \
    table + variable + grid + version + '*.nc',
)))
siconc_hg3_ll_hi_r1 = xr.open_mfdataset(
    siconc_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_hg3_ll_hi_r1 = siconc_hg3_ll_hi_r1.sel(time=slice(
    '1979-01-01', '2014-12-30')).siconc.mean(axis=0)
am_siconc_hg3_ll_hi_r1_80 = siconc_hg3_ll_hi_r1.sel(time=slice(
    '1980-01-01', '2014-12-30')).siconc.mean(axis=0)
# stats.describe(am_siconc_hg3_ll_hi_r1, axis=None, nan_policy='omit')

# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in HadGEM3-GC31-LL, historical, r1i1p1f3


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_hg3_ll_hi_r1.longitude,
    am_siconc_hg3_ll_hi_r1.latitude,
    am_siconc_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nHadGEM3-GC31-LL historical r1i1p1f3, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.1_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 1979_2014.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in HadISST

siconc_hadisst = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/HadISST/HadISST_ice.nc'
)

ann_siconc_hadisst = mon_sea_ann_average(
    siconc_hadisst.sic.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year'
)

am_siconc_hadisst = ann_siconc_hadisst.mean(axis=0) * 100

# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in HadISST

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_hadisst.longitude,
    am_siconc_hadisst.latitude,
    am_siconc_hadisst,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nHadISST1, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.0_SH annual siconc HadISST1 1979_2014.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Don't run Annual mean siconc in NSIDC -- convert tif to nc

siconc_nsidc_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/S_*_concentration_v3.0.tif',
)))

siconc_nsidc_197811 = xr.open_rasterio(siconc_nsidc_fl[0])

time = pd.date_range('1978-11', '2021-11', freq = '1M',)

siconc_nsidc_nc = xr.Dataset(
    {"siconc": (
        ("time", "y", "x"), np.zeros(
            (len(time), len(siconc_nsidc_197811.y),
             len(siconc_nsidc_197811.x)))),
     },
    coords={
        "time": time,
        "y": siconc_nsidc_197811.y,
        "x": siconc_nsidc_197811.x,
    },
    attrs=siconc_nsidc_197811.attrs,
)

for i in range(len(siconc_nsidc_fl)):
    # i = 30
    siconc_nsidc_tif = xr.open_rasterio(siconc_nsidc_fl[i])
    # stats.describe(siconc_nsidc_tif.values, axis =None)
    
    # clean the data
    siconc_nsidc_tif_values = siconc_nsidc_tif.values.copy()
    siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2510] = 0
    siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2530] = 0
    siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2540] = 0
    siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2550] = 0
    siconc_nsidc_tif_values = siconc_nsidc_tif_values/10
    
    # time: siconc_nsidc_fl[i][83:89]
    
    tif_file_date = datetime.strptime(siconc_nsidc_fl[i][83:89], '%Y%m')
    
    siconc_nsidc_nc.siconc[
        np.where(
            (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == tif_file_date.month) &
            (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == tif_file_date.year))[0], :, :
    ] = siconc_nsidc_tif_values
    # stats.describe(siconc_nsidc_tif_values, axis =None)
    # stats.describe(siconc_nsidc_nc.siconc[
    #     np.where(
    #         (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == tif_file_date.month) &
    #         (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == tif_file_date.year))[0], :, :
    # ], axis=None)
    
    print(str(i) + '/' + str(len(siconc_nsidc_fl)))

# Set the value of two month with missing values as nan
siconc_nsidc_nc.siconc[
    np.where(
        (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == 12) &
        (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == 1987))[0], :, :
] = np.nan
siconc_nsidc_nc.siconc[
    np.where(
        (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == 1) &
        (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == 1988))[0], :, :
] = np.nan

siconc_nsidc_nc.to_netcdf(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc'
)


'''
# No data on 1987.12 and 1988.1

# check
siconc_nsidc_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/S_*_concentration_v3.0.tif',
)))

i = 300
siconc_nsidc = xr.open_rasterio(siconc_nsidc_fl[i])
siconc_nsidc_tif_values = siconc_nsidc.values.copy()
siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2510] = 0
siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2530] = 0
siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2540] = 0
siconc_nsidc_tif_values[siconc_nsidc_tif_values == 2550] = 0
siconc_nsidc_tif_values = siconc_nsidc_tif_values/10

tif_file_date = datetime.strptime(siconc_nsidc_fl[i][83:89], '%Y%m')

siconc_nsidc_nc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc'
)
(siconc_nsidc_nc.siconc[np.where(
                            (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == tif_file_date.month) &
                            (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == tif_file_date.year))[0], :, :].values == siconc_nsidc_tif_values[0]).all()

np.isnan(siconc_nsidc_nc.siconc[np.where(
    (pd.DatetimeIndex(siconc_nsidc_nc.time.values).month == 1) &
    (pd.DatetimeIndex(siconc_nsidc_nc.time.values).year == 1988))[0], :, :].values).all()

# import rasterio
# rasterio.open(siconc_nsidc_fl[0])
# siconc_nsidc = xr.open_dataset(siconc_nsidc_fl[0], engine='rasterio',)
# xr.open_dataset(
#     'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/S_197811_concentration_v3.0.tif', engine='rasterio',)
'''
# endregion
# =============================================================================


# =============================================================================
# region Annual mean siconc in NSIDC

siconc_nsidc_nc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc'
)

mon_siconc_nsidc = mon_sea_ann_average(
    siconc_nsidc_nc.siconc.sel(time=slice(
        '1979-01-01', '2015-01-01')), 'time.month'
)

am_siconc_nsidc = (mon_siconc_nsidc * month_days[:, None, None]
                   ).sum(axis=0)/month_days.sum()
# stats.describe(am_siconc_nsidc, axis =None)

'''
ann_siconc_hadisst = mon_sea_ann_average(
    siconc_hadisst.sic.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year'
)

am_siconc_hadisst = ann_siconc_hadisst.mean(axis=0) * 100

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in NSIDC

nsidc_transform = ccrs.epsg(3976)

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_nsidc.x,
    am_siconc_nsidc.y,
    am_siconc_nsidc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=nsidc_transform,)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nNSIDC, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.2_SH annual siconc NSIDC 1979_2014.png')


'''
i = 70
nsidc_transform = ccrs.epsg(3976)

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    siconc_nsidc_nc.x,
    siconc_nsidc_nc.y,
    siconc_nsidc_nc.siconc[i, :, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=nsidc_transform,)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Monthly sea ice area fraction [$\%$]\nNSIDC, ' + \
        str(siconc_nsidc_nc.time[i].values)[0:10],
    linespacing=1.5,)

fig.savefig(
    'figures/0_test/trial.png')

'''
# endregion
# =============================================================================



