

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
from scipy.stats import circmean

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
    calc_lon_diff_np,
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

westerlies_40_65_zm_mm = xr.open_dataarray(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.westerlies_40_65_zm_mm.nc')

westerlies_40_65_zm_mm.time.values[:] = \
    pre_weighted_var[expid[i]]['lat']['mon'].time.values.copy()

#---- broadcast westerlies
b_westerlies, _ = xr.broadcast(
    westerlies_40_65_zm_mm,
    pre_weighted_var[expid[i]]['lat']['mon'])

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies vs. source lat - with seasonality

ivar = 'lat'

cor_westerlies_var = xr.corr(
    b_westerlies, pre_weighted_var[expid[i]][ivar]['mon'],
    dim='time',).compute()

cor_westerlies_var_p = xs.pearson_r_eff_p_value(
    b_westerlies, pre_weighted_var[expid[i]][ivar]['mon'],
    dim='time',).values


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.3, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=True, reversed=False)
pltticks[-4] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' correlation westerlies_' + ivar + ' mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_westerlies_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_westerlies_var_p <= 0.05],
    y=lat_2d[cor_westerlies_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between westerlies\nand source latitude [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)


np.min(cor_westerlies_var.values[echam6_t63_ais_mask['mask']['AIS']])

area1 = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']].sum()
area2 = echam6_t63_cellarea.cell_area.values[
    echam6_t63_ais_mask['mask']['AIS'] & \
        (cor_westerlies_var_p <= 0.05)
    ].sum()
area2 / area1

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies vs. source lat - without seasonality

ivar = 'lat'

b_westerlies_mm = b_westerlies.groupby('time.month').mean().compute()
b_westerlies_anom = (b_westerlies.groupby('time.month') - \
    b_westerlies_mm).compute()

pre_weighted_var_mm = pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month').mean().compute()
pre_weighted_var_anom = (pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month') - pre_weighted_var_mm).compute()

cor_westerlies_var_anom = xr.corr(
    b_westerlies_anom, pre_weighted_var_anom,
    dim='time',).compute()

cor_westerlies_var_anom_p = xs.pearson_r_eff_p_value(
    b_westerlies_anom, pre_weighted_var_anom,
    dim='time',).values


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.3, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=True, reversed=False)
pltticks[-4] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' correlation westerlies_' + ivar + '_anom mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_westerlies_var_anom,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_westerlies_var_anom_p <= 0.05],
    y=lat_2d[cor_westerlies_var_anom_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between westerlies\nand source latitude anomalies [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)


np.min(cor_westerlies_var_anom.values[echam6_t63_ais_mask['mask']['AIS']])

area1 = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']].sum()
area2 = echam6_t63_cellarea.cell_area.values[
    echam6_t63_ais_mask['mask']['AIS'] & \
        (cor_westerlies_var_anom_p <= 0.05)
    ].sum()
area2 / area1

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies vs. relative source lon - with seasonality

ivar = 'lon'

source_lon_mon = pre_weighted_var[expid[i]][ivar]['mon']
rel_source_lon_mon = calc_lon_diff(source_lon_mon, lon_2d)

cor_westerlies_var = xr.corr(
    b_westerlies, rel_source_lon_mon,
    dim='time',).compute()
cor_westerlies_var_p = xs.pearson_r_eff_p_value(
    b_westerlies, rel_source_lon_mon,
    dim='time',).values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.6, cm_max=0.3, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=True, reversed=False)
pltticks[-4] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' correlation westerlies_' + ivar + ' mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_westerlies_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_westerlies_var_p <= 0.05],
    y=lat_2d[cor_westerlies_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between westerlies\nand relative source longitude [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)


np.min(cor_westerlies_var.values[echam6_t63_ais_mask['mask']['AIS']])

area1 = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']].sum()
area2 = echam6_t63_cellarea.cell_area.values[
    echam6_t63_ais_mask['mask']['AIS'] & \
        (cor_westerlies_var_p <= 0.05)
    ].sum()
area2 / area1

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies vs. relative source lon - without seasonality

ivar = 'lon'

b_westerlies_mm = b_westerlies.groupby('time.month').mean().compute()
b_westerlies_anom = (b_westerlies.groupby('time.month') - \
    b_westerlies_mm).compute()

pre_weighted_var_mm = pre_weighted_var[expid[i]][ivar]['mm']
pre_weighted_var_anom = calc_lon_diff(
    pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month'),
    pre_weighted_var_mm,
)


# pre_weighted_var_mm = calc_lon_diff(
#     pre_weighted_var[expid[i]][ivar]['mm'], lon_2d)
# pre_weighted_var_anom = calc_lon_diff(
#     calc_lon_diff(
#         pre_weighted_var[expid[i]][ivar]['mon'],
#         lon_2d).groupby('time.month'),
#     pre_weighted_var_mm,
# )

cor_westerlies_var = xr.corr(
    b_westerlies_anom, pre_weighted_var_anom,
    dim='time',).compute()

cor_westerlies_var_p = xs.pearson_r_eff_p_value(
    b_westerlies_anom, pre_weighted_var_anom,
    dim='time',).values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.6, cm_max=0.2, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=True, reversed=False)
pltticks[-3] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' correlation westerlies_' + ivar + '_anom mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_westerlies_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_westerlies_var_p <= 0.05],
    y=lat_2d[cor_westerlies_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between westerlies\nand relative source longitude anomalies [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)


np.min(cor_westerlies_var.values[echam6_t63_ais_mask['mask']['AIS']])
np.max(cor_westerlies_var.values[echam6_t63_ais_mask['mask']['AIS']])

area1 = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']].sum()
area2 = echam6_t63_cellarea.cell_area.values[
    echam6_t63_ais_mask['mask']['AIS'] & \
        (cor_westerlies_var_p <= 0.05)
    ].sum()
area2 / area1

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SAM and westerlies

westerlies_mm = westerlies_40_65_zm_mm.groupby('time.month').mean().compute()
westerlies_anom = (westerlies_40_65_zm_mm.groupby('time.month') - \
    westerlies_mm).compute()

sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

from scipy.stats import pearsonr
pearsonr(sam_mon.sam, westerlies_anom)
# 0.96



# endregion
# -----------------------------------------------------------------------------


