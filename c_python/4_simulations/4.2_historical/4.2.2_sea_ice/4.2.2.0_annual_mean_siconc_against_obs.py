

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
# region Don't run Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'SIday/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20181218/'

siconc_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

siconc_awc_mr_hi_r1 = xr.open_mfdataset(
    siconc_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_awc_mr_hi_r1_org = siconc_awc_mr_hi_r1.copy()
am_siconc_awc_mr_hi_r1_org['siconc'] = am_siconc_awc_mr_hi_r1_org[
    'siconc'][0, :]
am_siconc_awc_mr_hi_r1_org['siconc'][:] = \
    siconc_awc_mr_hi_r1.siconc.sel(time=slice(
        '1979-01-01', '2014-12-31')).mean(axis=0).values

am_siconc_awc_mr_hi_r1_80_org = siconc_awc_mr_hi_r1.copy()
am_siconc_awc_mr_hi_r1_80_org['siconc'] = am_siconc_awc_mr_hi_r1_80_org[
    'siconc'][0, :]
am_siconc_awc_mr_hi_r1_80_org['siconc'][:] = \
    siconc_awc_mr_hi_r1.siconc.sel(time=slice(
        '1980-01-01', '2014-12-31')).mean(axis=0).values


am_siconc_awc_mr_hi_r1_org.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_org.nc')
am_siconc_awc_mr_hi_r1_80_org.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_org.nc')


'''
#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_org.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_org.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc

# global_2: lonlat (180x90) grid
# global_1: lonlat (360x180) grid


#### slow
from cdo import Cdo
cdo = Cdo()
cdo.remapcon(
    'r360x180',
    input='/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc',
    output='bas_palaeoclim_qino/scratch/cmip6/historical/siconc/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231_cdo_regrid.nc')

stats.describe(am_siconc_awc_mr_hi_r1.siconc, axis=None, nan_policy='omit')
stats.describe(am_siconc_awc_mr_hi_r1_80.siconc, axis=None, nan_policy='omit')
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

am_siconc_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc'
)
am_siconc_awc_mr_hi_r1_80 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc'
)

# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_awc_mr_hi_r1.lon,
    am_siconc_awc_mr_hi_r1.lat,
    am_siconc_awc_mr_hi_r1.siconc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nAWI-CM-1-1-MR, historical, r1i1p1f1, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.3_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 1979_2014.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in HadISST

siconc_hadisst = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/HadISST/HadISST.2.2.0.0_sea_ice_concentration.nc'
)

ann_siconc_hadisst = mon_sea_ann_average(
    siconc_hadisst.sic.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year'
)

am_siconc_hadisst = ann_siconc_hadisst.mean(axis=0) * 100

am_siconc_hadisst_rg_hg3 = regrid(am_siconc_hadisst, am_siconc_hg3_ll_hi_r1)
am_siconc_hadisst_rg_awc = regrid(am_siconc_hadisst, am_siconc_awc_mr_hi_r1)

dif_am_hg3_ll_hi_r1_hadisst = am_siconc_hg3_ll_hi_r1 - am_siconc_hadisst_rg_hg3
dif_am_awc_mr_hi_r1_hadisst = am_siconc_awc_mr_hi_r1.siconc - am_siconc_hadisst_rg_awc

'''
stats.describe(am_siconc_hg3_ll_hi_r1, axis=None, nan_policy='omit')

# check
pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_hadisst_rg_hg3.longitude,
    am_siconc_hadisst_rg_hg3.latitude,
    am_siconc_hadisst_rg_hg3,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded HadISST1, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_hadisst_rg_awc.lon,
    am_siconc_hadisst_rg_awc.lat,
    am_siconc_hadisst_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded HadISST1, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')



stats.describe(am_siconc_hadisst_rg_hg3, axis =None)
(am_siconc_hadisst_rg_hg3.longitude.values == am_siconc_hg3_ll_hi_r1.longitude.values).all()
(am_siconc_hadisst_rg_hg3.latitude.values == am_siconc_hg3_ll_hi_r1.latitude.values).all()

regridder = xe.Regridder(
    am_siconc_hadisst, am_siconc_hg3_ll_hi_r1, 'bilinear', periodic=True)
am_siconc_hadisst_rg_hg3_1 = regridder(am_siconc_hadisst)

(am_siconc_hadisst_rg_hg3.values == am_siconc_hadisst_rg_hg3_1.values).all()

stats.describe(am_siconc_hadisst_rg_awc, axis =None)
(am_siconc_hadisst_rg_awc.lon.values == am_siconc_awc_mr_hi_r1.lon.values).all()
(am_siconc_hadisst_rg_awc.lat.values == am_siconc_awc_mr_hi_r1.lat.values).all()

# siconc_hadisst1 = xr.open_dataset(
    #     'bas_palaeoclim_qino/observations/products/HadISST/HadISST_ice.nc'
    # )
'''
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
    'Annual mean sea ice area fraction [$\%$]\nHadISST, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.0_SH annual siconc HadISST 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_hadisst.longitude,
    dif_am_hg3_ll_hi_r1_hadisst.latitude,
    dif_am_hg3_ll_hi_r1_hadisst,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - HadISST\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.7_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 - HadISST 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_hadisst.lon,
    dif_am_awc_mr_hi_r1_hadisst.lat,
    dif_am_awc_mr_hi_r1_hadisst,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - HadISST\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.8_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 - HadISST 1979_2014.png')


'''
stats.describe(dif_am_hg3_ll_hi_r1_hadisst, axis=None, nan_policy='omit')
'''
# endregion
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
https://stackoverflow.com/questions/56851980/how-to-project-x-y-coordinates-to-lat-lon-in-netcdf-file
#### gdalwarp commands does not work
#  -to SRC_METHOD=NO_GEOTRANSFORM  -r bilinear
# gdalwarp -overwrite -to SRC_METHOD=NO_GEOTRANSFORM -t_srs EPSG:4326 bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110_gdal_reformat.nc

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
# region Don't run Annual mean siconc in NSIDC -- transform xy to lonlat

from pyproj import Proj, transform

siconc_nsidc_nc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc'
)

x2d, y2d = np.meshgrid(siconc_nsidc_nc.x.values, siconc_nsidc_nc.y.values)
lon, lat = transform(Proj(init='epsg:3976'), Proj(init='epsg:4326'), x2d, y2d)


siconc_nsidc = xr.Dataset(
    {"siconc": (("time", "i", "j"), siconc_nsidc_nc.siconc.values),
     'lon': (("i", "j"), lon),
     'lat': (("i", "j"), lat),
     },
    coords={
        "time": siconc_nsidc_nc.time.values,
        "i": np.arange(0, len(siconc_nsidc_nc.y)),
        "j": np.arange(0, len(siconc_nsidc_nc.x)),
    },
)

siconc_nsidc.to_netcdf(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_197811_202110.nc'
)


'''
# check
siconc_nsidc_nc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_nc_197811_202110.nc'
)
siconc_nsidc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_197811_202110.nc'
)
stats.describe(siconc_nsidc.siconc.values, axis =None, nan_policy='omit')
(siconc_nsidc_nc.siconc.values == siconc_nsidc.siconc.values).sum()


# inProj = Proj(init='epsg:3976')
# outProj = Proj(init='epsg:4326')
# x1,y1 = -11705274.6374,4826473.6922
# x2,y2 = transform(inProj,outProj,x1,y1)
# print (x2,y2)

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in NSIDC

siconc_nsidc = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/NSIDC/Sea_Ice_Index/sh_monthly_geotiff/siconc_nsidc_197811_202110.nc'
)

mon_siconc_nsidc = mon_sea_ann_average(
    siconc_nsidc.siconc.sel(time=slice(
        '1979-01-01', '2015-01-01')), 'time.month'
)

am_siconc_nsidc = (mon_siconc_nsidc * month_days[:, None, None]
                   ).sum(axis=0)/month_days.sum()
# stats.describe(am_siconc_nsidc, axis =None)

regridder = xe.Regridder(
    siconc_nsidc, am_siconc_hg3_ll_hi_r1, 'bilinear')
am_siconc_nsidc_rg_hg3 = regridder(am_siconc_nsidc)
regridder = xe.Regridder(
    siconc_nsidc, am_siconc_awc_mr_hi_r1, 'bilinear')
am_siconc_nsidc_rg_awc = regridder(am_siconc_nsidc)

# am_siconc_nsidc_rg_hg3 = regrid(am_siconc_nsidc, am_siconc_hg3_ll_hi_r1)
# am_siconc_nsidc_rg_awc = regrid(am_siconc_nsidc, am_siconc_awc_mr_hi_r1)

dif_am_hg3_ll_hi_r1_nsidc = am_siconc_hg3_ll_hi_r1 - am_siconc_nsidc_rg_hg3
dif_am_awc_mr_hi_r1_nsidc = am_siconc_awc_mr_hi_r1.siconc - am_siconc_nsidc_rg_awc


'''
# check
(am_siconc_nsidc_rg_hg3.latitude.values == am_siconc_hg3_ll_hi_r1.latitude.values).all()
(am_siconc_nsidc_rg_awc.lat.values == am_siconc_awc_mr_hi_r1.lat.values).all()

stats.describe(am_siconc_hg3_ll_hi_r1, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_nsidc_rg_hg3.longitude,
    am_siconc_nsidc_rg_hg3.latitude,
    am_siconc_nsidc_rg_hg3,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded NSIDC, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_nsidc_rg_awc.lon,
    am_siconc_nsidc_rg_awc.lat,
    am_siconc_nsidc_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded NSIDC, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

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
    # am_siconc_nsidc.x, am_siconc_nsidc.y,
    siconc_nsidc.lon, siconc_nsidc.lat,
    am_siconc_nsidc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    # transform=nsidc_transform,
    transform=ccrs.PlateCarree(),
    )
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nNSIDC, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.2_SH annual siconc NSIDC 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_nsidc.longitude,
    dif_am_hg3_ll_hi_r1_nsidc.latitude,
    dif_am_hg3_ll_hi_r1_nsidc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - NSIDC\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.9_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 - NSIDC 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_nsidc.lon,
    dif_am_awc_mr_hi_r1_nsidc.lat,
    dif_am_awc_mr_hi_r1_nsidc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - NSIDC\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.10_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 - NSIDC 1979_2014.png')


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


# =============================================================================
# =============================================================================
# region Annual mean siconc in ERA5

era5_mon_sl_79_21_sic = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')

mon_siconc_era5 = xr.concat((
    era5_mon_sl_79_21_sic.siconc[:-2, 0, :, :],
    era5_mon_sl_79_21_sic.siconc[-2:, 1, :, :]), dim='time')

am_siconc_era5 = mon_sea_ann_average(
    mon_siconc_era5.sel(time=slice(
        '1979-01-01', '2014-12-30')), 'time.year').mean(axis = 0) * 100


regridder = xe.Regridder(
    am_siconc_era5, am_siconc_hg3_ll_hi_r1, 'bilinear')
am_siconc_era5_rg_hg3 = regridder(am_siconc_era5)
regridder = xe.Regridder(
    am_siconc_era5, am_siconc_awc_mr_hi_r1, 'bilinear')
am_siconc_era5_rg_awc = regridder(am_siconc_era5)

dif_am_hg3_ll_hi_r1_era5 = am_siconc_hg3_ll_hi_r1 - am_siconc_era5_rg_hg3
dif_am_awc_mr_hi_r1_era5 = am_siconc_awc_mr_hi_r1.siconc - am_siconc_era5_rg_awc


'''
# check
(am_siconc_era5_rg_hg3.longitude.values == am_siconc_hg3_ll_hi_r1.longitude.values).all()
(am_siconc_era5_rg_awc.lon.values == am_siconc_awc_mr_hi_r1.lon.values).all()

stats.describe(am_siconc_era5_rg_hg3, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_era5_rg_hg3.longitude,
    am_siconc_era5_rg_hg3.latitude,
    am_siconc_era5_rg_hg3,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded ERA5, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_era5_rg_awc.lon,
    am_siconc_era5_rg_awc.lat,
    am_siconc_era5_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded ERA5, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in ERA5


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_era5.longitude,
    am_siconc_era5.latitude,
    am_siconc_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nERA5, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.4_SH annual siconc ERA5 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_era5.longitude,
    dif_am_hg3_ll_hi_r1_era5.latitude,
    dif_am_hg3_ll_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - ERA5\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.11_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 - ERA5 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_era5.lon,
    dif_am_awc_mr_hi_r1_era5.lat,
    dif_am_awc_mr_hi_r1_era5,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - ERA5\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.12_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 - ERA5 1979_2014.png')

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in MERRA2


# import data
siconc_merra2_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/MERRA2/FRSEAICE_TSKINICE_TSKINWTR/*.nc'
)))

siconc_merra2 = xr.open_mfdataset(
    siconc_merra2_fl, data_vars='minimal', coords='minimal', compat='override',
)

am_siconc_merra2 = mon_sea_ann_average(siconc_merra2.FRSEAICE.sel(
    time=slice('1980-01-01', '2014-12-30')), 'time.year').mean(axis=0) * 100


regridder = xe.Regridder(
    am_siconc_merra2, am_siconc_hg3_ll_hi_r1, 'bilinear', periodic=True)
am_siconc_merra2_rg_hg3 = regridder(am_siconc_merra2)
regridder = xe.Regridder(
    am_siconc_merra2, am_siconc_awc_mr_hi_r1, 'bilinear', periodic=True)
am_siconc_merra2_rg_awc = regridder(am_siconc_merra2)

dif_am_hg3_ll_hi_r1_merra2 = am_siconc_hg3_ll_hi_r1_80 - am_siconc_merra2_rg_hg3
dif_am_awc_mr_hi_r1_merra2 = am_siconc_awc_mr_hi_r1_80.siconc - am_siconc_merra2_rg_awc


'''
# check
(am_siconc_merra2_rg_hg3.latitude.values == am_siconc_hg3_ll_hi_r1.latitude.values).all()
(am_siconc_merra2_rg_awc.lat.values == am_siconc_awc_mr_hi_r1.lat.values).all()

stats.describe(am_siconc_merra2_rg_hg3, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_merra2_rg_hg3.longitude,
    am_siconc_merra2_rg_hg3.latitude,
    am_siconc_merra2_rg_hg3,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded MERRA2, 1980-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_merra2_rg_awc.lon,
    am_siconc_merra2_rg_awc.lat,
    am_siconc_merra2_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded MERRA2, 1980-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

'''
'''
(siconc_merra2.FRSEAICE.values == 1000000000000000.0).sum()
stats.describe(am_siconc_merra2, axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in MERRA2


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_merra2.lon,
    am_siconc_merra2.lat,
    am_siconc_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nMERRA2, 1980-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.5_SH annual siconc MERRA2 1980_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_merra2.longitude,
    dif_am_hg3_ll_hi_r1_merra2.latitude,
    dif_am_hg3_ll_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - MERRA2\n1980-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.13_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 - MERRA2 1980_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_merra2.lon,
    dif_am_awc_mr_hi_r1_merra2.lat,
    dif_am_awc_mr_hi_r1_merra2,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - MERRA2\n1980-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.14_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 - MERRA2 1980_2014.png')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Annual mean siconc in JRA-55

siconc_jra55_fl = np.array(sorted(glob.glob(
    'bas_palaeoclim_qino/observations/reanalysis/JRA-55/mon_ice_cover/ice.091_icec.reg_tl319.*[0-9]'
)))
siconc_jra55 = xr.open_mfdataset(siconc_jra55_fl, engine='cfgrib')


am_siconc_jra55 = (mon_sea_ann_average(siconc_jra55.ci.sel(
    time=slice('1978-12-30', '2014-12-30')), 'time.month'
) * month_days[:, None, None]).sum(axis=0)/month_days.sum() * 100


regridder = xe.Regridder(
    am_siconc_jra55, am_siconc_hg3_ll_hi_r1, 'bilinear', periodic=True)
am_siconc_jra55_rg_hg3 = regridder(am_siconc_jra55)
regridder = xe.Regridder(
    am_siconc_jra55, am_siconc_awc_mr_hi_r1, 'bilinear', periodic=True)
am_siconc_jra55_rg_awc = regridder(am_siconc_jra55)

dif_am_hg3_ll_hi_r1_jra55 = am_siconc_hg3_ll_hi_r1 - am_siconc_jra55_rg_hg3
dif_am_awc_mr_hi_r1_jra55 = am_siconc_awc_mr_hi_r1.siconc - am_siconc_jra55_rg_awc


'''
# check
(am_siconc_jra55_rg_hg3.longitude.values == am_siconc_hg3_ll_hi_r1.longitude.values).all()
(am_siconc_jra55_rg_awc.lon.values == am_siconc_awc_mr_hi_r1.lon.values).all()

stats.describe(am_siconc_jra55_rg_hg3, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_jra55_rg_hg3.longitude,
    am_siconc_jra55_rg_hg3.latitude,
    am_siconc_jra55_rg_hg3,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded JRA-55, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_jra55_rg_awc.lon,
    am_siconc_jra55_rg_awc.lat,
    am_siconc_jra55_rg_awc,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nregridded JRA-55, 1979-2014',
    linespacing=1.5,)

fig.savefig('figures/0_test/trial.png')

'''
'''
(siconc_jra55.ci.values == 9999).sum()
stats.describe(am_siconc_jra55, axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region plot Annual mean siconc in JRA-55


pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_top=0.94, )

plt_cmp = ax.pcolormesh(
    am_siconc_jra55.longitude,
    am_siconc_jra55.latitude,
    am_siconc_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.12,
    shrink=1, aspect=40, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction [$\%$]\nJRA-55, 1979-2014',
    linespacing=1.5,)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.6_SH annual siconc JRA-55 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_hg3_ll_hi_r1_jra55.longitude,
    dif_am_hg3_ll_hi_r1_jra55.latitude,
    dif_am_hg3_ll_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(HadGEM3-GC31-LL, historical, r1i1p1f3) - JRA-55\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.15_SH annual siconc HadGEM3-GC31-LL historical r1i1p1f3 - JRA-55 1979_2014.png')


pltlevel = np.arange(-40, 40.01, 0.1)
pltticks = np.arange(-40, 40.01, 10)
cmp_cmap = rb_colormap(pltlevel)

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 10.3]) / 2.54, fm_top=0.96, )

plt_cmp = ax.pcolormesh(
    dif_am_awc_mr_hi_r1_jra55.lon,
    dif_am_awc_mr_hi_r1_jra55.lat,
    dif_am_awc_mr_hi_r1_jra55,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(),
    rasterized=True, transform=ccrs.PlateCarree(),
)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.15, shrink=1, aspect=40, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual mean sea ice area fraction difference [$\%$]\n(AWI-CM-1-1-MR, historical, r1i1p1f1) - JRA-55\n1979-2014',
    linespacing=1.5
)

fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.2_siconc/4.0.2.16_SH annual siconc AWI-CM-1-1-MR, historical, r1i1p1f1 - JRA-55 1979_2014.png')

'''
'''
# endregion
# =============================================================================
