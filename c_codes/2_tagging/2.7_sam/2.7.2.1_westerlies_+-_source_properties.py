

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

# #---- broadcast westerlies
# b_westerlies, _ = xr.broadcast(
#     westerlies_40_65_zm_mm,
#     pre_weighted_var[expid[i]]['lat']['mon'])

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

sam_marshall = pd.read_csv(
    'data_sources/climate_indices/SAM/SAM.txt', sep = '\s+',
    index_col=0)

#---------------- get positive negative months
westerlies_mean = westerlies_40_65_zm_mm.mean().values
westerlies_std = westerlies_40_65_zm_mm.std(ddof = 1).values

westerlies_posneg_ind = {}
westerlies_posneg_ind['pos'] = \
    (westerlies_40_65_zm_mm > (westerlies_mean + westerlies_std))
westerlies_posneg_ind['neg'] = \
    (westerlies_40_65_zm_mm < (westerlies_mean - westerlies_std))

westerlies_mm = westerlies_40_65_zm_mm.groupby('time.month').mean().compute()
westerlies_anom = (westerlies_40_65_zm_mm.groupby('time.month') - \
    westerlies_mm).compute()

#---------------- get anom positive negative months
westerlies_anom_mean = westerlies_anom.mean().values
westerlies_anom_std = westerlies_anom.std(ddof = 1).values

westerlies_anom_posneg_ind = {}
westerlies_anom_posneg_ind['pos'] = \
    (westerlies_anom > (westerlies_anom_mean + westerlies_anom_std))
westerlies_anom_posneg_ind['neg'] = \
    (westerlies_anom < (westerlies_anom_mean - westerlies_anom_std))


'''
sns.histplot(westerlies_40_65_zm_mm.values)
plt.axvline(westerlies_mean + westerlies_std, 0, 10**4)
plt.axvline(westerlies_mean - westerlies_std, 0, 10**4)
plt.savefig('figures/test/trial.png')
plt.close()

westerlies_posneg_ind['neg'].sum()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check normality

#---------------- check normality

stats.shapiro(sam_marshall.iloc[14:44].to_numpy().flatten())
stats.anderson(sam_marshall.iloc[14:44].to_numpy().flatten())

stats.shapiro(westerlies_40_65_zm_mm)
stats.anderson(westerlies_40_65_zm_mm.values)

stats.shapiro(sam_mon.sam.values)
stats.anderson(sam_mon.sam.values)

sns.histplot(westerlies_40_65_zm_mm.values)
sns.histplot(sam_mon.sam.values)
plt.savefig('figures/test/trial.png')
plt.close()
np.mean(sam_mon.sam.values)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies_posneg source lat

ivar = 'lat'

westerlies_posneg_var = {}
westerlies_posneg_var[ivar] = {}
westerlies_posneg_var[ivar]['pos'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][westerlies_posneg_ind['pos']]
westerlies_posneg_var[ivar]['pos_mean'] = westerlies_posneg_var[ivar]['pos'].mean(dim='time')

westerlies_posneg_var[ivar]['neg'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][westerlies_posneg_ind['neg']]
westerlies_posneg_var[ivar]['neg_mean'] = westerlies_posneg_var[ivar]['neg'].mean(dim='time')


output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' westerlies_posneg_' + ivar + ' Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-6, cm_max=3, cm_interval1=1, cm_interval2=1, cmap='PiYG',
    asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    westerlies_posneg_var[ivar]['pos_mean'] - \
        westerlies_posneg_var[ivar]['neg_mean'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    westerlies_posneg_var[ivar]['pos'],
    westerlies_posneg_var[ivar]['neg'],)
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Source latitude differences between\nwesterlies+ and westerlies- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies_posneg source lon

ivar = 'lon'

westerlies_posneg_var = {}
westerlies_posneg_var[ivar] = {}
westerlies_posneg_var[ivar]['pos'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][westerlies_posneg_ind['pos']]
westerlies_posneg_var[ivar]['pos_mean'] = \
    circmean(westerlies_posneg_var[ivar]['pos'].values, high=360, low=0, axis=0,
             nan_policy='omit')

westerlies_posneg_var[ivar]['neg'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][westerlies_posneg_ind['neg']]
westerlies_posneg_var[ivar]['neg_mean'] = \
    circmean(westerlies_posneg_var[ivar]['neg'].values, high=360, low=0, axis=0,
             nan_policy='omit')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' westerlies_posneg_' + ivar + ' Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn',)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff_np(
        westerlies_posneg_var[ivar]['pos_mean'],
        westerlies_posneg_var[ivar]['neg_mean'],),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
wwtest_res = circ.watson_williams(
    westerlies_posneg_var[ivar]['pos'].values * np.pi / 180,
    westerlies_posneg_var[ivar]['neg'].values * np.pi / 180,
    axis=0,
    )[0] < 0.05
ax.scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Source longitude differences between\nwesterlies+ and westerlies- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies_posneg source lat - removed mm


ivar = 'lat'

pre_weighted_var_mm = pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month').mean().compute()
pre_weighted_var_anom = (pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month') - pre_weighted_var_mm).compute()


westerlies_anom_posneg_var = {}
westerlies_anom_posneg_var[ivar] = {}
westerlies_anom_posneg_var[ivar]['pos'] = \
    pre_weighted_var_anom[westerlies_anom_posneg_ind['pos']]
westerlies_anom_posneg_var[ivar]['pos_mean'] = \
    westerlies_anom_posneg_var[ivar]['pos'].mean(dim='time')

westerlies_anom_posneg_var[ivar]['neg'] = \
    pre_weighted_var_anom[westerlies_anom_posneg_ind['neg']]
westerlies_anom_posneg_var[ivar]['neg_mean'] = \
    westerlies_anom_posneg_var[ivar]['neg'].mean(dim='time')


output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' westerlies_posneg_' + ivar + '_rmm Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-6, cm_max=3, cm_interval1=1, cm_interval2=1, cmap='PiYG',
    asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    westerlies_anom_posneg_var[ivar]['pos_mean'] - \
        westerlies_anom_posneg_var[ivar]['neg_mean'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    westerlies_anom_posneg_var[ivar]['pos'],
    westerlies_anom_posneg_var[ivar]['neg'],)
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Source latitude anomaly differences between\nwesterlies+ and westerlies- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)





'''
westerlies_anom_posneg_ind['pos'].sum()
westerlies_anom_posneg_ind['neg'].sum()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region westerlies_posneg source lon - removed mm

ivar = 'lon'

pre_weighted_var_mm = pre_weighted_var[expid[i]][ivar]['mm']
pre_weighted_var_anom = calc_lon_diff(
    pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month'),
    pre_weighted_var_mm,)

westerlies_anom_posneg_var = {}
westerlies_anom_posneg_var[ivar] = {}
westerlies_anom_posneg_var[ivar]['pos'] = \
    pre_weighted_var_anom[westerlies_anom_posneg_ind['pos']]
westerlies_anom_posneg_var[ivar]['pos_mean'] = \
    circmean(westerlies_anom_posneg_var[ivar]['pos'].values,
             high=360, low=0, axis=0, nan_policy='omit')

westerlies_anom_posneg_var[ivar]['neg'] = \
    pre_weighted_var_anom[westerlies_anom_posneg_ind['neg']]
westerlies_anom_posneg_var[ivar]['neg_mean'] = \
    circmean(westerlies_anom_posneg_var[ivar]['neg'].values,
             high=360, low=0, axis=0, nan_policy='omit')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' westerlies_posneg_' + ivar + '_rmm Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn',)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff_np(
        westerlies_anom_posneg_var[ivar]['pos_mean'],
        westerlies_anom_posneg_var[ivar]['neg_mean'],),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
wwtest_res = circ.watson_williams(
    westerlies_anom_posneg_var[ivar]['pos'].values * np.pi / 180,
    westerlies_anom_posneg_var[ivar]['neg'].values * np.pi / 180,
    axis=0,
    )[0] < 0.05
ax.scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Source longitude anomaly differences between\nwesterlies+ and westerlies- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


