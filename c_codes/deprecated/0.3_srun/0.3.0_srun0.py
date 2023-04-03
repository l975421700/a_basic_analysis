

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
os.chdir('/work/ollie/qigao001')

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
