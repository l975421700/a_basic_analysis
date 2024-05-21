

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress
import pingouin as pg

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
import cartopy.feature as cfeature
from scipy.stats import pearsonr
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
    remove_trailing_zero_pos_abs,
    hemisphere_conic_plot,
    ticks_labels,
    plot_maxmin_points,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
    mean_over_ais,
    time_weighted_mean,
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
    plot_labels_no_unit,
    plot_labels,
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
    cplot_ttest,
    xr_par_cor,
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


source_var = ['lat', 'sst',]
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        prefix + '.q_sfc_weighted_sst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check std of daily source lat

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=10, cm_interval1=1, cm_interval2=2,
    cmap='viridis',)

# q_sfc_weighted_var[expid[i]]['lat']['daily'].std(dim='time', ddof=1).to_netcdf('scratch/test/test0.nc')

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.7 ' + expid[i] + ' std of daily q_sfc_weighted_lat.png'

cbar_label = 'Standard deviation of source latitude of daily surface vapour [°]'

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lat,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].std(dim='time', ddof=1),
    ax, pltlevel, 'max', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.12,)

cbar.ax.tick_params(length=0.5, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2, size=9)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check correlation between moisture source SST and latitude

daily_corr_src_sst_lat = xr.corr(
    q_sfc_weighted_var[expid[i]]['sst']['daily'],
    q_sfc_weighted_var[expid[i]]['lat']['daily'],
    dim='time',
    ).compute()

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.7 ' + expid[i] + ' corr. daily q_sfc src_sst and src_lat.png'

cbar_label = 'Correlation: source SST and latitude of daily surface vapour [-]'

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lat,
    daily_corr_src_sst_lat,
    ax, pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.12,)

cbar.ax.tick_params(length=0.5, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2, size=9)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am source SST - local SST

stats.describe(q_sfc_weighted_var[expid[i]]['sst']['am'] - tsw_alltime[expid[i]]['am'], axis=None, nan_policy='omit')

# plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=10, cm_interval1=1, cm_interval2=2,
    cmap='PuOr', asymmetric=True, reversed=True)

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.7 ' + expid[i] + ' am q_sfc source_local sst.png'

cbar_label = 'Differences between source and local SST of\nannual mean surface vapour [$°C$]'

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6.5]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.14, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lat,
    (q_sfc_weighted_var[expid[i]]['sst']['daily'] - tsw_alltime[expid[i]]['daily'].values).mean(dim='time'),
    # q_sfc_weighted_var[expid[i]]['sst']['am'] - tsw_alltime[expid[i]]['am'],
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.14,)

cbar.ax.tick_params(length=1.5, width=0.6)
cbar.ax.set_xlabel(cbar_label, linespacing=1.3)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot std of daily source SST - local SST

stats.describe((q_sfc_weighted_var[expid[i]]['sst']['daily'] - tsw_alltime[expid[i]]['daily'].values).std(dim='time', ddof=1), axis=None, nan_policy='omit')

# plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=8, cm_interval1=1, cm_interval2=1,
    cmap='viridis',)

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.7 ' + expid[i] + ' daily std q_sfc source_local sst.png'

cbar_label = 'Standard deviation of differences between source\nand local SST of daily surface vapour [$°C$]'

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6.5]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.14, fm_top=0.99,)

plt1 = plot_t63_contourf(
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lon,
    q_sfc_weighted_var[expid[i]]['lat']['daily'].lat,
    (q_sfc_weighted_var[expid[i]]['sst']['daily'] - tsw_alltime[expid[i]]['daily'].values).std(dim='time', ddof=1),
    ax, pltlevel, 'max', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.14,)

cbar.ax.tick_params(length=1.5, width=0.6)
cbar.ax.set_xlabel(cbar_label, linespacing=1.3)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------

