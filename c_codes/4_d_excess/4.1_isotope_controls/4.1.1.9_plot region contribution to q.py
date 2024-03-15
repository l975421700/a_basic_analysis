

# salloc --account=paleodyn.paleodyn --partition=fat --qos=12h --time=12:00:00 --nodes=1 --mem=1024GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
    time_labels,
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

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

lon = q_geo7_sfc_alltiime[expid[i]]['am'].lon
lat = q_geo7_sfc_alltiime[expid[i]]['am'].lat
plev = q_geo7_alltiime[expid[i]]['am'].plev

# plev = corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']['r'].plev

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot open ocean contribution to surface q

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=50, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='viridis',)

output_png = 'figures/8_d-excess/8.1_controls/8.1.8_region_contribution/8.1.8.0 ' + expid[i] + ' annual mean open ocean contribution to surface q.png'

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.5]) / 2.54,)

plt1 = plot_t63_contourf(
    lon, lat,
    q_geo7_sfc_alltiime[expid[i]]['am'].sel(geo_regions = 'Open Ocean') / q_geo7_sfc_alltiime[expid[i]]['am'].sel(geo_regions = 'Sum') * 100,
    ax, pltlevel, 'min', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks,
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Open ocean contribution\nto surface humidity [$\%$]',
                   linespacing=1.5)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot open ocean contribution to zonal mean q

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=50, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='viridis',)

output_png = 'figures/8_d-excess/8.1_controls/8.1.8_region_contribution/8.1.8.0 ' + expid[i] + ' annual mean open ocean contribution to zonal mean q.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)),
    plev.sel(plev=slice(1e+5, 2e+4)) / 100,
    (q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='Open Ocean') / q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='Sum') * 100).sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='min',)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -88.57)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.1, fraction=0.04, anchor=(0.4, -1),)

cbar.ax.set_xlabel('Open ocean contribution to zonal mean humidity [$\%$]',
                   linespacing=1.5)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sea ice contribution to zonal mean q

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='Blues',
    reversed=False,)

output_png = 'figures/8_d-excess/8.1_controls/8.1.8_region_contribution/8.1.8.0 ' + expid[i] + ' annual mean sea ice contribution to zonal mean q.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)),
    plev.sel(plev=slice(1e+5, 2e+4)) / 100,
    (q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='SH seaice') / q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='Sum') * 100).sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='max',)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -88.57)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.1, fraction=0.04, anchor=(0.4, -1),)

cbar.ax.set_xlabel('Sea ice contribution to zonal mean humidity [$\%$]',
                   linespacing=1.5)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot land contribution to zonal mean q

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=30, cm_interval1=3, cm_interval2=6, cmap='Greens',
    reversed=False,)

output_png = 'figures/8_d-excess/8.1_controls/8.1.8_region_contribution/8.1.8.0 ' + expid[i] + ' annual mean Land excl. AIS contribution to zonal mean q.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)),
    plev.sel(plev=slice(1e+5, 2e+4)) / 100,
    (q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='Land excl. AIS') / q_geo7_alltiime[expid[i]]['am'].mean(dim='lon').sel(geo_regions='Sum') * 100).sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='max',)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -88.57)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.1, fraction=0.04, anchor=(0.4, -1),)

cbar.ax.set_xlabel('Land excl. AIS contribution to zonal mean humidity [$\%$]',
                   linespacing=1.5)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------
