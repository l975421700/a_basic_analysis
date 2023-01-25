

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
i = 0


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
import xskillscore as xs

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
    find_nearest_1d,
    get_mon_sam,
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


#---- import source properties
pre_weighted_var = {}
pre_weighted_var[expid[i]] = {}

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']

prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.pre_weighted_lat.pkl',
    prefix + '.pre_weighted_lon.pkl',
    prefix + '.pre_weighted_sst.pkl',
    prefix + '.pre_weighted_rh2m.pkl',
    prefix + '.pre_weighted_wind10.pkl',
    prefix + '.transport_distance.pkl',
]

for ivar, ifile in zip(source_var, source_var_files):
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        pre_weighted_var[expid[i]][ivar] = pickle.load(f)

lon = pre_weighted_var[expid[i]]['lat']['am'].lon
lat = pre_weighted_var[expid[i]]['lat']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

#---- import ice core sites
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between source lat and sst

ivar = 'sst'

cor_lat_var = xr.corr(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').compute()

cor_lat_var_p = xs.pearson_r_eff_p_value(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').values

#---------------- plot
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 ' + expid[i] + ' correlation lat_' + ivar + ' ann.png'

# np.min(cor_lat_var.sel(lat=slice(-60, -90)))
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0.75, cm_max=1, cm_interval1=0.025, cm_interval2=0.05,
    cmap='PuOr', reversed=False)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_lat_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_lat_var_p <= 0.05],
    y=lat_2d[cor_lat_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between source lat\nand source SST [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)




'''
cor_ann = xr.corr(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').compute()

cor_mon = xr.corr(
    pre_weighted_var[expid[i]][ivar]['mon'],
    pre_weighted_var[expid[i]]['lat']['mon'],
    dim='time').compute()

cor_ann.to_netcdf('scratch/test/test.nc')
cor_mon.to_netcdf('scratch/test/test1.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between source lat and rh2m

ivar = 'rh2m'

cor_lat_var = xr.corr(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').compute()

cor_lat_var_p = xs.pearson_r_eff_p_value(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').values

#---------------- plot
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 ' + expid[i] + ' correlation lat_' + ivar + ' ann.png'

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=-0.9, cm_max=-0.5, cm_interval1=0.025, cm_interval2=0.05,
#     cmap='Greens')
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.9, cm_max=-0.5, cm_interval1=0.05, cm_interval2=0.05,
    cmap='PuOr')

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_lat_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_lat_var_p <= 0.05],
    y=lat_2d[cor_lat_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between source lat\nand source ' + ivar + ' [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between source lat and wind10

ivar = 'wind10'

cor_lat_var = xr.corr(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').compute()

cor_lat_var_p = xs.pearson_r_eff_p_value(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').values

# np.min(cor_lat_var.sel(lat=slice(-60, -90)))
# np.max(cor_lat_var.sel(lat=slice(-60, -90)))
#---------------- plot
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 ' + expid[i] + ' correlation lat_' + ivar + ' ann.png'

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=-0.8, cm_max=-0.4, cm_interval1=0.025, cm_interval2=0.05,
#     cmap='Greens')
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.2,
    cmap='PuOr', asymmetric=True)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_lat_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_lat_var_p <= 0.05],
    y=lat_2d[cor_lat_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between source lat\nand source ' + ivar + ' [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between source lat and distance

ivar = 'distance'

cor_lat_var = xr.corr(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').compute()

cor_lat_var_p = xs.pearson_r_eff_p_value(
    pre_weighted_var[expid[i]][ivar]['ann'],
    pre_weighted_var[expid[i]]['lat']['ann'],
    dim='time').values

# np.min(cor_lat_var.sel(lat=slice(-60, -90)))
# np.max(cor_lat_var.sel(lat=slice(-60, -90)))
#---------------- plot
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 ' + expid[i] + ' correlation lat_' + ivar + ' ann.png'

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=0.5, cm_max=1, cm_interval1=0.025, cm_interval2=0.05,
#     cmap='Greens', reversed=False)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.1, cm_max=1, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False, asymmetric=True)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_lat_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_lat_var_p <= 0.05],
    y=lat_2d[cor_lat_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between source lat\nand source-sink distance [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------



