

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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

#---- import sam
sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

#---- import ice core sites
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


'''
wisoaprt_epe_st[expid[i]]['mask']['90%'].keys()


'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate mon EPE days

wisoaprt_epe_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_st.pkl',
    'rb') as f:
    wisoaprt_epe_st[expid[i]] = pickle.load(f)

lon = wisoaprt_epe_st[expid[i]]['mask']['90%'].lon.values
lat = wisoaprt_epe_st[expid[i]]['mask']['90%'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)


mon_epe_days = {}

mon_epe_days['original'] = wisoaprt_epe_st[expid[i]]['mask']['90%'].resample(
    {'time': '1M'}).sum()

mon_epe_days['mm'] = mon_epe_days['original'].groupby(
    'time.month').mean().compute()

mon_epe_days['mon_anom'] = (mon_epe_days['original'].groupby(
    'time.month') - mon_epe_days['mm']).compute()

#---- broadcast sam_mon
b_sam_mon, _ = xr.broadcast(
    sam_mon.sam,
    mon_epe_days['original'])

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between sam and mon_epe_days anomalies

cor_sam_var_anom = xr.corr(
    b_sam_mon, mon_epe_days['mon_anom'], dim='time').compute()

cor_sam_var_anom_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mon_epe_days['mon_anom'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe mon_anom.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_var_anom,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_var_anom_p <= 0.05],
    y=lat_2d[cor_sam_var_anom_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand EPE frequency anomaly [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between sam and mon_epe_days

cor_sam_var = xr.corr(
    b_sam_mon, mon_epe_days['original'], dim='time').compute()

cor_sam_var_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mon_epe_days['original'], dim='time').values

cor_sam_var.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
cor_sam_var_p[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.4, cm_max=0.4, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False, asymmetric=False,)
pltticks[-5] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_var,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_var_p <= 0.05],
    y=lat_2d[cor_sam_var_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation: SAM & EPE frequency',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Differences in EPE days

sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_mon.sam > sam_mon.sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_mon.sam < (-1 * sam_mon.sam.std(ddof = 1))


sam_posneg_epe = {}
sam_posneg_epe['pos'] = \
    mon_epe_days['original'][sam_posneg_ind['pos']]
sam_posneg_epe['pos_mean'] = sam_posneg_epe['pos'].mean(dim='time')

sam_posneg_epe['neg'] = \
    mon_epe_days['original'][sam_posneg_ind['neg']]
sam_posneg_epe['neg_mean'] = sam_posneg_epe['neg'].mean(dim='time')

posneg_epe_diff = sam_posneg_epe['pos_mean'] - sam_posneg_epe['neg_mean']
# posneg_epe_diff.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' sam_posneg_epe mon.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-3, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='PiYG',
    asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = plot_t63_contourf(
    lon, lat, posneg_epe_diff, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
ax.add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

# plt1 = ax.pcolormesh(
#     lon,
#     lat,
#     posneg_epe_diff,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    sam_posneg_epe['pos'],
    sam_posneg_epe['neg'],)
ax.scatter(
    x=lon_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    y=lat_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    s=1.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'HP frequency differences [$days \; month^{-1}$]\nSAM+ vs. SAM-',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate mon pre

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoaprt_alltime[expid[i]]['am'].lon.values
lat = wisoaprt_alltime[expid[i]]['am'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)


mon_pre = {}
mon_pre['original'] = wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1)

mon_pre['mm'] = mon_pre['original'].groupby(
    'time.month').mean().compute()
mon_pre['mon_anom'] = (mon_pre['original'].groupby(
    'time.month') - mon_pre['mm']).compute()

#---- broadcast sam_mon
b_sam_mon, _ = xr.broadcast(
    sam_mon.sam,
    mon_pre['original'])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and precipitation anomaly

cor_sam_pre_anom = xr.corr(
    b_sam_mon, mon_pre['mon_anom'], dim='time').compute()

cor_sam_pre_anom_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mon_pre['mon_anom'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_pre mon_anom.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_pre_anom,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_pre_anom_p <= 0.05],
    y=lat_2d[cor_sam_pre_anom_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand precipitation anomaly [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and precipitation

cor_sam_pre = xr.corr(
    b_sam_mon, mon_pre['original'], dim='time').compute()

cor_sam_pre_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mon_pre['original'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_pre mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_pre,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_pre_p <= 0.05],
    y=lat_2d[cor_sam_pre_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand precipitation [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate epe mean intensity over epe days

wisoaprt_masked_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_st.pkl',
    'rb') as f:
    wisoaprt_masked_st[expid[i]] = pickle.load(f)

lon = wisoaprt_masked_st[expid[i]]['meannan']['90%']['mon'].lon.values
lat = wisoaprt_masked_st[expid[i]]['meannan']['90%']['mon'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)

meannan_intensity = {}
meannan_intensity['original'] = \
    wisoaprt_masked_st[expid[i]]['meannan']['90%']['mon']

meannan_intensity['mm'] = meannan_intensity['original'].groupby(
    'time.month').mean().compute()

meannan_intensity['mon_anom'] = (meannan_intensity['original'].groupby(
    'time.month') - meannan_intensity['mm']).compute()

#---- broadcast sam_mon
b_sam_mon, _ = xr.broadcast(
    sam_mon.sam,
    meannan_intensity['original'])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and epe mean intensity over epe days anomaly

cor_sam_meannan_anom = xr.corr(
    b_sam_mon, meannan_intensity['mon_anom'], dim='time').compute()

cor_sam_meannan_anom_p = xs.pearson_r_eff_p_value(
    b_sam_mon, meannan_intensity['mon_anom'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe_meannan mon_anom.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_meannan_anom,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_meannan_anom_p <= 0.05],
    y=lat_2d[cor_sam_meannan_anom_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand EPE intensity anomaly [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and epe mean intensity over epe days

cor_sam_meannan = xr.corr(
    b_sam_mon, meannan_intensity['original'], dim='time').compute()

cor_sam_meannan_p = xs.pearson_r_eff_p_value(
    b_sam_mon, meannan_intensity['original'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe_meannan mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_meannan,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_meannan_p <= 0.05],
    y=lat_2d[cor_sam_meannan_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand EPE intensity [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate epe mean intensity over epe days

wisoaprt_masked_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_st.pkl',
    'rb') as f:
    wisoaprt_masked_st[expid[i]] = pickle.load(f)

lon = wisoaprt_masked_st[expid[i]]['meannan']['90%']['mon'].lon.values
lat = wisoaprt_masked_st[expid[i]]['meannan']['90%']['mon'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)

mean_intensity = {}
mean_intensity['original'] = \
    wisoaprt_masked_st[expid[i]]['mean']['90%']['mon']

mean_intensity['mm'] = mean_intensity['original'].groupby(
    'time.month').mean().compute()

mean_intensity['mon_anom'] = (mean_intensity['original'].groupby(
    'time.month') - mean_intensity['mm']).compute()

#---- broadcast sam_mon
b_sam_mon, _ = xr.broadcast(
    sam_mon.sam,
    mean_intensity['original'])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and epe mean intensity over epe days anomaly

cor_sam_mean_anom = xr.corr(
    b_sam_mon, mean_intensity['mon_anom'], dim='time').compute()

cor_sam_mean_anom_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mean_intensity['mon_anom'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe_mean mon_anom.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_mean_anom,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_mean_anom_p <= 0.05],
    y=lat_2d[cor_sam_mean_anom_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand EPE precipitation anomaly [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation between SAM and epe mean intensity over epe days

cor_sam_mean = xr.corr(
    b_sam_mon, mean_intensity['original'], dim='time').compute()

cor_sam_mean_p = xs.pearson_r_eff_p_value(
    b_sam_mon, mean_intensity['original'], dim='time').values

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', reversed=False)
pltticks[-6] = 0

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_epe/6.1.9.0 ' + expid[i] + ' correlation sam_epe_mean mon.png'

fig, ax = hemisphere_plot(northextent=-60,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    cor_sam_mean,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[cor_sam_mean_p <= 0.05],
    y=lat_2d[cor_sam_mean_p <= 0.05],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation coefficient between SAM\nand EPE precipitation [$-$]',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------





