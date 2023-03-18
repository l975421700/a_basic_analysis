

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region resample hourly era5 tp to daily tp

input_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_12_21_Antarctica.nc',
]

output_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_12_21_Antarctica.nc',
]


for ifile in range(len(input_files)):
    # ifile = 0
    print('#-------- ' + str(ifile))
    print(input_files[ifile])
    print(output_files[ifile])
    
    tp_era5_hourly = xr.open_dataset(input_files[ifile])
    
    tp_era5_daily = (tp_era5_hourly.tp.resample({'time': '1D'}).sum() * 1000).compute()
    
    tp_era5_daily.to_netcdf(output_files[ifile])
    
    del tp_era5_hourly, tp_era5_daily



'''
# 20 min to run
#SBATCH --time=10:00:00
#SBATCH --partition=fat

#-------------------------------- check
input_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_12_21_Antarctica.nc',
]

output_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_12_21_Antarctica.nc',
]

ifile = 3
tp_era5_hourly = xr.open_dataset(input_files[ifile])
tp_era5_daily = xr.open_dataset(output_files[ifile])

ilat = 100
ilon = 100

data1 = tp_era5_hourly.tp[:, ilat, ilon].resample({'time': '1D'}).sum() * 1000
data2 = tp_era5_daily.tp[:, ilat, ilon]
(data1 == data2).all().values
np.max(abs(data1 - data2) / data2)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region get fraction of precipitation amount below 0.02 mm/day

#-------------------------------- 0.02 mm/day

tp_era5_daily = xr.open_mfdataset(
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_??_??_Antarctica.nc',
    data_vars='minimal', coords='minimal', parallel=True,
)

tp_era5_daily_td = tp_era5_daily.copy().compute()

tp_era5_daily_td.tp.values[tp_era5_daily_td.tp.values < 0.02] = 0

tp_era5_frc_td = (tp_era5_daily_td.tp.mean(dim='time') / \
    tp_era5_daily.tp.mean(dim='time')).compute()

np.min(tp_era5_frc_td)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 era5 aprt_frc am below threshold 0.02.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    0, 14, 2, 2, cmap='Blues', reversed=False)

fig, ax = hemisphere_plot(northextent=-60)
cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
plt_cmp = ax.pcolormesh(
    tp_era5_frc_td.longitude,
    tp_era5_frc_td.latitude,
    100 - tp_era5_frc_td * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of daily precipitation amount\nbelow the threshold 0.02 $mm \; day^{-1}$  [$\%$]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)


#-------------------------------- 0.002 mm/day

tp_era5_daily = xr.open_mfdataset(
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_??_??_Antarctica.nc',
    data_vars='minimal', coords='minimal', parallel=True,
)

tp_era5_daily_st = tp_era5_daily.copy().compute()

tp_era5_daily_st.tp.values[tp_era5_daily_st.tp.values < 0.002] = 0

tp_era5_frc_st = (tp_era5_daily_st.tp.mean(dim='time') / \
    tp_era5_daily.tp.mean(dim='time')).compute()

print(np.min(tp_era5_frc_st))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region fractioin of EPE, with a threshold of 0.002 mm/day

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')
tp_era5_daily = xr.open_mfdataset(
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_??_??_Antarctica.nc',
    ).chunk({'time': 15706, 'longitude': 20, 'latitude': 1})

lon = tp_era5_daily.longitude
lat = tp_era5_daily.latitude

tp_era5 = {}
tp_era5['original'] = tp_era5_daily.tp.copy().where(
    tp_era5_daily.tp >= 0.002,
    other=np.nan,).compute()

tp_era5['quantiles_90'] = \
    tp_era5['original'].quantile(0.9, dim='time', skipna=True).compute()

tp_era5['mask_90'] = (tp_era5_daily.tp.copy() >= tp_era5['quantiles_90']).compute()

tp_era5['masked_90'] = tp_era5_daily.tp.copy().where(
    tp_era5['mask_90'],
    other=0,).compute()

tp_era5['frc'] = (tp_era5['masked_90'].mean(dim='time') / tp_era5_daily.tp.copy().mean(dim='time'))


output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ear5 st_daily precipitation percentile_90_frc Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=30, cm_max=70, cm_interval1=2.5, cm_interval2=5, cmap='Oranges',
    reversed=False)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 6.8]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt_data = tp_era5['frc'] * 100

plt1 = ax.contourf(
    lon,
    lat,
    plt_data,
    levels=pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

plt_ctr = ax.contour(
    lon,
    lat.sel(latitude=slice(-60, -90)),
    plt_data.sel(latitude=slice(-60, -90)),
    [50],
    colors = 'b', linewidths=0.3, transform=ccrs.PlateCarree(),)
ax.clabel(
    plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=[50], inline_spacing=10, fontsize=8,)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='both',
    pad=0.04, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Contribution of HP to total precipitation [$\%$]', linespacing=1.5,
    fontsize=8)
fig.savefig(output_png, dpi=600)


'''
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

# plt1 = plot_t63_contourf(
#     lon, lat, plt_data, ax,
#     pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
'''
# endregion
# -----------------------------------------------------------------------------
