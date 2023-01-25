

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

#---- import sam
sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

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


sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_mon.sam > sam_mon.sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_mon.sam < (-1 * sam_mon.sam.std(ddof = 1))



'''
np.min(sam_mon.sam[sam_posneg_ind['pos']])
np.max(sam_mon.sam[sam_posneg_ind['neg']])

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sam_posneg source lat

ivar = 'lat'

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = sam_posneg_var[ivar]['pos'].mean(dim='time')

sam_posneg_var[ivar]['neg'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = sam_posneg_var[ivar]['neg'].mean(dim='time')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' sam_posneg_' + ivar + ' Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-6, cm_max=3, cm_interval1=1, cm_interval2=1, cmap='PiYG',
    asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    sam_posneg_var[ivar]['pos_mean'] - sam_posneg_var[ivar]['neg_mean'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    sam_posneg_var[ivar]['pos'],
    sam_posneg_var[ivar]['neg'],)
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
    'Source latitude differences between\nSAM+ and SAM- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)





'''
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']].time == \
    sam_posneg_ind['pos'].time).all()
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']].time == \
    sam_posneg_ind['neg'].time).all()

from scipy.stats import normaltest
normaltest(sam_mon.sam.values)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sam_posneg source lon

ivar = 'lon'

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = \
    circmean(sam_posneg_var[ivar]['pos'].values, high=360, low=0, axis=0,
             nan_policy='omit')

sam_posneg_var[ivar]['neg'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = \
    circmean(sam_posneg_var[ivar]['neg'].values, high=360, low=0, axis=0,
             nan_policy='omit')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' sam_posneg_' + ivar + ' Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn',)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff_np(
        sam_posneg_var[ivar]['pos_mean'],
        sam_posneg_var[ivar]['neg_mean'],),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    sam_posneg_var[ivar]['pos'].values * np.pi / 180,
    sam_posneg_var[ivar]['neg'].values * np.pi / 180,
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
    'Source longitude differences between\nSAM+ and SAM- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)





'''
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']].time == \
    sam_posneg_ind['pos'].time).all()
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']].time == \
    sam_posneg_ind['neg'].time).all()

from scipy.stats import normaltest
normaltest(sam_mon.sam.values)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sam_posneg source lat - removed mm

ivar = 'lat'

clim = pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month').mean().compute()
anom = (pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month') - clim).compute()

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = anom[sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = sam_posneg_var[ivar]['pos'].mean(dim='time')

sam_posneg_var[ivar]['neg'] = anom[sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = sam_posneg_var[ivar]['neg'].mean(dim='time')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' sam_posneg_' + ivar + '_rmm Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-6, cm_max=3, cm_interval1=1, cm_interval2=1, cmap='PiYG',
    asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    sam_posneg_var[ivar]['pos_mean'] - sam_posneg_var[ivar]['neg_mean'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    sam_posneg_var[ivar]['pos'],
    sam_posneg_var[ivar]['neg'],)
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
    'Source latitude differences between\nSAM+ and SAM- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)





'''
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']].time == \
    sam_posneg_ind['pos'].time).all()
(pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']].time == \
    sam_posneg_ind['neg'].time).all()

from scipy.stats import normaltest
normaltest(sam_mon.sam.values)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sam_posneg source lon - removed mm

ivar = 'lon'

clim = pre_weighted_var[expid[i]][ivar]['mm']
anom = calc_lon_diff(
    pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month'),
    clim,
)

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = anom[sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = \
    circmean(sam_posneg_var[ivar]['pos'].values, high=360, low=0, axis=0,
             nan_policy='omit')

sam_posneg_var[ivar]['neg'] = anom[sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = \
    circmean(sam_posneg_var[ivar]['neg'].values, high=360, low=0, axis=0,
             nan_policy='omit')

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_' + ivar + '/6.1.9.0 ' + expid[i] + ' sam_posneg_' + ivar + '_rmm Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn',)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff_np(
        sam_posneg_var[ivar]['pos_mean'],
        sam_posneg_var[ivar]['neg_mean'],),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    sam_posneg_var[ivar]['pos'].values * np.pi / 180,
    sam_posneg_var[ivar]['neg'].values * np.pi / 180,
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
    'Source longitude differences between\nSAM+ and SAM- months [$°$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region statistics

ivar = 'lat'

clim = pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month').mean().compute()
anom = (pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month') - clim).compute()

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = anom[sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = sam_posneg_var[ivar]['pos'].mean(dim='time')

sam_posneg_var[ivar]['neg'] = anom[sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = sam_posneg_var[ivar]['neg'].mean(dim='time')

var_diff = sam_posneg_var[ivar]['pos_mean'] - sam_posneg_var[ivar]['neg_mean']


'''
ivar = 'lon'

clim = pre_weighted_var[expid[i]][ivar]['mm']
anom = calc_lon_diff(
    pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month'),
    clim,
)

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = anom[sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = \
    circmean(sam_posneg_var[ivar]['pos'].values, high=360, low=0, axis=0,
             nan_policy='omit')

sam_posneg_var[ivar]['neg'] = anom[sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = \
    circmean(sam_posneg_var[ivar]['neg'].values, high=360, low=0, axis=0,
             nan_policy='omit')

var_diff = calc_lon_diff_np(
    sam_posneg_var[ivar]['pos_mean'], sam_posneg_var[ivar]['neg_mean'],)
'''

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

for imask in ['AIS', 'EAIS', 'WAIS', 'AP']:
    # imask = 'AIS'
    print('#-------- ' + imask)
    
    mask = echam6_t63_ais_mask['mask'][imask]
    
    ave_var_diff = np.average(
        var_diff[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],
        )
    
    print(str(np.round(ave_var_diff, 1)))


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

for isite in ['EDC', 'Halley']:
    # isite = 'EDC'
    print(isite)
    
    res = var_diff[
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']]
    
    print(np.round(res, 1))


np.max(var_diff[echam6_t63_ais_mask['mask']['AIS']])
np.min(var_diff[echam6_t63_ais_mask['mask']['AIS']])



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sensitivity to seasonality

# with seasonality
ivar = 'lat'

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = sam_posneg_var[ivar]['pos'].mean(dim='time')

sam_posneg_var[ivar]['neg'] = \
    pre_weighted_var[expid[i]][ivar]['mon'][sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = sam_posneg_var[ivar]['neg'].mean(dim='time')

diff_with_sea = sam_posneg_var[ivar]['pos_mean'] - sam_posneg_var[ivar]['neg_mean']


# without seasonality
ivar = 'lat'
clim = pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month').mean().compute()
anom = (pre_weighted_var[expid[i]][ivar]['mon'].groupby(
    'time.month') - clim).compute()

sam_posneg_var = {}
sam_posneg_var[ivar] = {}
sam_posneg_var[ivar]['pos'] = anom[sam_posneg_ind['pos']]
sam_posneg_var[ivar]['pos_mean'] = sam_posneg_var[ivar]['pos'].mean(dim='time')

sam_posneg_var[ivar]['neg'] = anom[sam_posneg_ind['neg']]
sam_posneg_var[ivar]['neg_mean'] = sam_posneg_var[ivar]['neg'].mean(dim='time')

diff_without_sea = sam_posneg_var[ivar]['pos_mean'] - sam_posneg_var[ivar]['neg_mean']

sensitivity_sea = diff_with_sea - diff_without_sea

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')


np.max(sensitivity_sea.values[echam6_t63_ais_mask['mask']['AIS']])
np.max(abs(sensitivity_sea.values[echam6_t63_ais_mask['mask']['AIS']]))

np.average(
    sensitivity_sea.values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']]
)

np.average(
    abs(sensitivity_sea.values[echam6_t63_ais_mask['mask']['AIS']]),
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']]
)
# 0.05


# endregion
# -----------------------------------------------------------------------------

