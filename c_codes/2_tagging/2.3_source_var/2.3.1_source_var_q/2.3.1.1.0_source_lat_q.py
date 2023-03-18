

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
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
from metpy.calc import pressure_to_height_std
from metpy.units import units

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
    cplot_lon180,
    cplot_wind_vectors,
    cplot_lon180_quiver,
    cplot_lon180_ctr,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

q_weighted_lat = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lat.pkl',
          'rb') as f:
    q_weighted_lat[expid[i]] = pickle.load(f)

q_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_plev.pkl',
    'rb') as f:
    q_plev[expid[i]] = pickle.load(f)

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

lon = q_weighted_lat[expid[i]]['am'].lon
lat = q_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)
plevs = q_weighted_lat[expid[i]]['am'].plev

tpot_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tpot_plev.pkl',
    'rb') as f:
    tpot_plev[expid[i]] = pickle.load(f)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e_alltime.pkl',
    'rb') as f:
    theta_e_alltime = pickle.load(f)


'''
plevs/100

uv_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl',
    'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot q_weighted_lat am zm

am_zm_source_lat = q_weighted_lat[expid[i]]['am'].weighted(
        ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
        ).mean(dim='lon')

am_zm_q = q_plev[expid[i]]['am'].mean(dim='lon') * 1000

am_zm_tpot = tpot_plev[expid[i]]['am'].mean(dim='lon') - zerok


output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am zm.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=-10, cm_interval1=2.5, cm_interval2=5, cmap='PuOr',)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
plt_mesh = ax.pcolormesh(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_source_lat.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp,
)

# q contours
q_intervals = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10])
plt_ctr = ax.contour(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_q.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    colors='b', levels=q_intervals, linewidths=0.4,
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=q_intervals, inline_spacing=1, fontsize=6,)

# tpot contours
tpot_intervals = np.arange(-20, 80 + 1e-4, 5)
plt_ctr_tpot = ax.contour(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_tpot.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    colors='k', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr_tpot, inline=1, colors='k', fmt=remove_trailing_zero,
    levels=tpot_intervals, inline_spacing=1, fontsize=6,)

# x-axis
ax.set_xticks(np.arange(-20, -90 - 1e-4, -10))
ax.set_xlim(-20, -90)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    anchor=(1.3, -1),
    )
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in abs(pltticks)])
cbar.ax.set_xlabel('Source latitude [$°\;S$]',)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# contours legend
h1, _ = plt_ctr.legend_elements()
h2, _ = plt_ctr_tpot.legend_elements()
ax_legend = ax.legend(
    [h1[0], h2[0]],
    ['Specific humidity [$g \; kg^{-1}$]',
     'Potential temperature [$°C$]'],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.13, -0.36), labelspacing=1.2,
    handlelength=1, columnspacing=1)

for line in ax_legend.get_lines():
    line.set_linewidth(1)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.14, top=0.98)
fig.savefig(output_png)





'''
# q_weighted_lat_am_zm = q_weighted_lat[expid[i]]['am'].weighted(
#     ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
#     ).mean(dim='lon')

# q_plev[expid[i]]['am'].mean(dim='lon')


#-------- check weighted average
iplev = 20
ilat = 48
np.average(
    q_weighted_lat[expid[i]]['am'][iplev, ilat],
    weights=ocean_q_alltime[expid[i]]['am'].sel(var_names='lat')[iplev, ilat],
    )
q_weighted_lat_am_zm[iplev, ilat].values

np.average(
     q_weighted_lat[expid[i]]['am'],
     weights=ocean_q_alltime[expid[i]]['am'].sel(var_names='lat')
)

q_weighted_lat_am_zm = q_weighted_lat[expid[i]]['am'].weighted(
    ocean_q_alltime[expid[i]]['am'].sel(var_names='lat')
    ).mean(dim='lon')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot relative q_weighted_lat am zm

am_zm_rel_source_lat = q_weighted_lat[expid[i]]['am'].weighted(
    ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
    ).mean(dim='lon') - lat.values[None, :]

am_zm_q = q_plev[expid[i]]['am'].mean(dim='lon') * 1000

am_zm_tpot = tpot_plev[expid[i]]['am'].mean(dim='lon') - zerok


output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' relative q_weighted_lat am zm.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=65, cm_interval1=5, cm_interval2=5, cmap='PiYG',)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
plt_mesh = ax.pcolormesh(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_rel_source_lat.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp,
)

# q contours
q_intervals = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10])
plt_ctr = ax.contour(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_q.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    colors='b', levels=q_intervals, linewidths=0.4,
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=q_intervals, inline_spacing=1, fontsize=6,)

# tpot contours
tpot_intervals = np.arange(-20, 80 + 1e-4, 5)
plt_ctr_tpot = ax.contour(
    lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_tpot.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
    colors='k', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr_tpot, inline=1, colors='k', fmt=remove_trailing_zero,
    levels=tpot_intervals, inline_spacing=1, fontsize=6,)

# x-axis
ax.set_xticks(np.arange(-20, -90 - 1e-4, -10))
ax.set_xlim(-20, -90)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    anchor=(1.3, -1),
    )
cbar.ax.set_xlabel('Relative source latitude [$°$]',)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# contours legend
h1, _ = plt_ctr.legend_elements()
h2, _ = plt_ctr_tpot.legend_elements()
ax_legend = ax.legend(
    [h1[0], h2[0]],
    ['Specific humidity [$g \; kg^{-1}$]',
     'Potential temperature [$°C$]'],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.13, -0.36), labelspacing=1.2,
    handlelength=1, columnspacing=1)

for line in ax_legend.get_lines():
    line.set_linewidth(1)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.14, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot q_weighted_lat am zm SH

am_zm_source_lat = q_weighted_lat[expid[i]]['am'].weighted(
        ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
        ).mean(dim='lon')

am_zm_q = q_plev[expid[i]]['am'].mean(dim='lon') * 1000

# am_zm_tpot = tpot_plev[expid[i]]['am'].mean(dim='lon') - zerok
am_zm_tpot = theta_e_alltime['am'].mean(dim='lon') - zerok

output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am zm_SH.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=0, cm_interval1=2.5, cm_interval2=5, cmap='viridis',
    reversed=False)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
# plt_mesh = ax.pcolormesh(
#     lat.sel(lat=slice(0, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
#     am_zm_source_lat.sel(lat=slice(0, -90), plev=slice(1e+5, 2e+4)),
#     norm=pltnorm, cmap=pltcmp,
# )
plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_source_lat.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both'
)

# # q contours
# q_intervals = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10])
# plt_ctr = ax.contour(
#     lat.sel(lat=slice(0, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
#     am_zm_q.sel(lat=slice(0, -90), plev=slice(1e+5, 2e+4)),
#     colors='b', levels=q_intervals, linewidths=0.4,
#     clip_on=True, zorder=2)
# ax_clabel = ax.clabel(
#     plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
#     levels=q_intervals, inline_spacing=1, fontsize=10,)

# tpot contours
tpot_intervals = np.arange(-20, 80 + 1e-4, 5)
plt_ctr_tpot = ax.contour(
    lat.sel(lat=slice(3, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_tpot.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    colors='k', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr_tpot, inline=1, colors='k', fmt=remove_trailing_zero,
    levels=tpot_intervals, inline_spacing=1, fontsize=10,)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -88.57)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos_abs,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    anchor=(1.1, -1),
    )
# cbar.ax.set_xticklabels([remove_trailing_zero(x) for x in abs(pltticks)])
cbar.ax.set_xlabel('Source latitude [$°\;S$]',)
cbar.ax.invert_xaxis()

# # 2nd y-axis
# height = np.round(
#     pressure_to_height_std(
#         pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
# ax2 = ax.twinx()
# ax2.invert_yaxis()
# ax2.set_ylim(1000, 200)
# ax2.set_yticks(np.arange(1000, 200 - 1e-4, -100))
# ax2.set_yticklabels(height.magnitude, c = 'gray')
# ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# contours legend
# h1, _ = plt_ctr.legend_elements()
h2, _ = plt_ctr_tpot.legend_elements()
ax_legend = ax.legend(
    [
        # h1[0],
        h2[0],
        ],
    [
        # 'Specific humidity [$g \; kg^{-1}$]',
        'Equivalent\npotential temperature [$°C$]',
        ],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.13, -0.3), labelspacing=1.2,
    handlelength=1, columnspacing=1)

for line in ax_legend.get_lines():
    line.set_linewidth(1)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot relative q_weighted_lat am zm SH

am_zm_rel_source_lat = q_weighted_lat[expid[i]]['am'].weighted(
    ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').fillna(0)
    ).mean(dim='lon') - lat.values[None, :]

am_zm_q = q_plev[expid[i]]['am'].mean(dim='lon') * 1000

am_zm_tpot = tpot_plev[expid[i]]['am'].mean(dim='lon') - zerok


output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' relative q_weighted_lat am zm_SH.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=60, cm_interval1=5, cm_interval2=5, cmap='BrBG',
    asymmetric=True)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
# plt_mesh = ax.pcolormesh(
#     lat.sel(lat=slice(0, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
#     am_zm_rel_source_lat.sel(lat=slice(0, -90), plev=slice(1e+5, 2e+4)),
#     norm=pltnorm, cmap=pltcmp,
# )
plt_mesh = ax.contourf(
    lat.sel(lat=slice(2, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_rel_source_lat.sel(lat=slice(2, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both'
)

# q contours
q_intervals = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
plt_ctr = ax.contour(
    lat.sel(lat=slice(2, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_q.sel(lat=slice(2, -90), plev=slice(1e+5, 2e+4)),
    colors='b', levels=q_intervals, linewidths=0.4,
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=q_intervals, inline_spacing=1, fontsize=10,)

# # tpot contours
# tpot_intervals = np.arange(-20, 80 + 1e-4, 5)
# plt_ctr_tpot = ax.contour(
#     lat.sel(lat=slice(-20, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
#     am_zm_tpot.sel(lat=slice(-20, -90), plev=slice(1e+5, 2e+4)),
#     colors='k', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
#     clip_on=True, zorder=2)
# ax_clabel = ax.clabel(
#     plt_ctr_tpot, inline=1, colors='k', fmt=remove_trailing_zero,
#     levels=tpot_intervals, inline_spacing=1, fontsize=10,)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -88.57)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    anchor=(1.1, -1),
    )
cbar.ax.set_xlabel('Relative source latitude [$°$]',)
cbar.ax.invert_xaxis()

# # 2nd y-axis
# height = np.round(
#     pressure_to_height_std(
#         pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
# ax2 = ax.twinx()
# ax2.invert_yaxis()
# ax2.set_ylim(1000, 200)
# ax2.set_yticks(np.arange(1000, 200 - 1e-4, -100))
# ax2.set_yticklabels(height.magnitude, c = 'gray')
# ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# contours legend
h1, _ = plt_ctr.legend_elements()
# h2, _ = plt_ctr_tpot.legend_elements()
ax_legend = ax.legend(
    [h1[0],
    #  h2[0],
     ],
    ['Specific humidity [$g \; kg^{-1}$]',
    #  'Potential temperature [$°C$]',
     ],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.13, -0.3), labelspacing=1.2,
    handlelength=1, columnspacing=1)

for line in ax_legend.get_lines():
    line.set_linewidth(1)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot q_weighted_lat am zm SH

djf_zm_tpot = theta_e_alltime['sm'].sel(season='DJF').mean(dim='lon') - zerok
jja_zm_tpot = theta_e_alltime['sm'].sel(season='JJA').mean(dim='lon') - zerok


output_png = 'figures/test/trial.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# tpot contours
tpot_intervals = np.arange(-30, 100 + 1e-4, 5)
plt_ctr_tpot = ax.contour(
    lat.sel(lat=slice(3, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    djf_zm_tpot.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    colors='k', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr_tpot, inline=1, colors='k', fmt=remove_trailing_zero,
    levels=tpot_intervals, inline_spacing=1, fontsize=10,)

# tpot contours
plt_ctr_tpot = ax.contour(
    lat.sel(lat=slice(3, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    jja_zm_tpot.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    colors='r', levels=tpot_intervals, linewidths=0.4, linestyles='solid',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr_tpot, inline=1, colors='r', fmt=remove_trailing_zero,
    levels=tpot_intervals, inline_spacing=1, fontsize=10,)

# x-axis
ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
ax.set_xlim(0, -90)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# region plot q_weighted_lat am

# ilat = 80
ilat = 64
output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am ' + str(np.round(lat[ilat].values, 1)) + '.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-55, cm_max=-10, cm_interval1=2.5, cm_interval2=5, cmap='PuOr',)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = cplot_lon180(
    lon, plevs / 100, ax, pltnorm, pltcmp,
    q_weighted_lat[expid[i]]['am'].sel(
        lat = lat[ilat].values, plev=slice(1e+5, 2e+4)),)

plt_quiver = cplot_lon180_quiver(
    lon, plevs / 100, ax,
    plt_data = uv_plev[expid[i]]['u']['am'].sel(
        lat = uv_plev[expid[i]]['u']['am'].lat[ilat].values,
        plev=slice(1e+5, 2e+4)), color='k')

plt_ctr, ax_clabel = cplot_lon180_ctr(
    lon, plevs / 100, ax,
    plt_data = q_plev[expid[i]]['am'].sel(
        lat_2 = lat[ilat].values, plev=slice(1e+5, 2e+4)) * 1000
    )

# x-axis
ax.set_xticks(np.arange(-180, 180 + 1e-4, 60))
ax.set_xlim(-180, 180)
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', which='both',)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticklabels(np.flip(height.magnitude), c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# label lat cross-section
plt.text(
    0.02, 0.98, str(np.negative(np.round(lat[ilat].values, 1))) + '$°\;S$',
    transform=ax.transAxes, weight='bold',
    ha='left', va='top', rotation='horizontal',
    bbox=dict(boxstyle='round', fc='white', ec='gray', lw=1, alpha=0.7),)

# cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.25, anchor=(1.1, -0.6),)
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.set_xlabel('Humidity-weighted open-oceanic source latitude [$°\;S$]',)

# quiver key
ax.quiverkey(plt_quiver, X=-0.05, Y=-0.15, U=10,
             label='10 $m \; s^{-1}$ zonal wind',
             labelpos='E', labelsep=0.05,)

# contour label
h1, _ = plt_ctr.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Specific humidity [$g \; kg^{-1}$]'],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.13, -0.32),
    handlelength=1, columnspacing=1)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.23, top=0.98)
fig.savefig(output_png)





# lon_180 = np.concatenate([
#     lon[int(len(lon) / 2):] - 360, lon[:int(len(lon) / 2)], ])

# plt_data = q_plev[expid[i]]['am'].sel(
#     lat_2 = lat[ilat].values, plev=slice(1e+5, 2e+4)) * 1000
# plt_data_180 = xr.concat([
#     plt_data.sel(lon=slice(180, 360)),
#     plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')

# q_intervals = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 10])

# plt_ctr = ax.contour(
#     lon_180, plevs / 100, plt_data_180,
#     colors='b', levels=q_intervals, linewidths=0.2, clip_on=True)
# ax_clabel = ax.clabel(
#     plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
#     levels=q_intervals, inline_spacing=10, fontsize=6)

# quiver key
ax.quiverkey(plt_quiver, X=-0.05, Y=-0.2, U=10,
             label='10 $m \; s^{-1}$ zonal wind',
             labelpos='E', labelsep=0.05,)

'''
stats.describe(q_plev[expid[i]]['am'].sel(
    lat_2 = lat[ilat].values, plev=slice(1e+5, 2e+4)), axis=None,
               nan_policy='omit')

#-------------------------------- plot [-180, 180] quiver manually

lon_180 = np.concatenate([lon[int(len(lon) / 2):] - 360,
                          lon[:int(len(lon) / 2)], ])
plt_data = uv_plev[expid[i]]['u']['am'].sel(
    lat = uv_plev[expid[i]]['u']['am'].lat[ilat].values,
    plev=slice(1e+5, 2e+4))
plt_data_180 = xr.concat([
    plt_data.sel(lon=slice(180, 360)),
    plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')

iarrow = 5
ax.quiver(
    lon_180[::iarrow], plevs / 100,
    plt_data_180[:, ::iarrow],
    np.zeros(plt_data_180.shape)[:, ::iarrow],
    color='magenta', units='height', scale=600, zorder=2,
    width=0.002, headwidth=3, headlength=5, alpha=1,)

#-------------------------------- plot [-180, 180] mesh manually
plt_data = q_weighted_lat[expid[i]]['am'].sel(
    lat = lat[ilat].values, plev=slice(1e+5, 2e+4))

lon_180 = np.concatenate([lon[96:] - 360, lon[:96], ])
plt_data_180 = xr.concat([plt_data.sel(lon=slice(180, 360)),
                          plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')

plt_mesh = ax.pcolormesh(
    lon_180,
    plevs / 100,
    plt_data_180,
    norm=pltnorm, cmap=pltcmp,)

#-------------------------------- plot 0-360 as check
ilat = 80
output_png = 'figures/trial.png'
plt_data = q_weighted_lat[expid[i]]['am'].sel(
    lat = lat[ilat].values, plev=slice(1e+5, 2e+4))

pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
pltticks = np.arange(-55, -20 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

ax.pcolormesh(
    lon,
    plevs / 100,
    plt_data,
    norm=pltnorm, cmap=pltcmp,)

ax.set_xticks(np.arange(0, 360 + 1e-4, 60))
ax.set_xlim(0, 360)

ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))

ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')

from matplotlib.ticker import AutoMinorLocator
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
        which='both',)

fig.subplots_adjust(left=0.11, right=0.89, bottom=0.22, top=0.98)
fig.savefig(output_png)


q_weighted_lat[expid[i]]['am'].sel(lat = lat[ilat].values).to_netcdf(
    'scratch/test/test.nc')
q_weighted_lat[expid[i]]['am'].to_netcdf('scratch/test/test.nc')

# ax.set_yticks(np.arange(100000, 20000 - 1e-4, -10000))
# ax.set_yticklabels(
#     [remove_trailing_zero(x) for x in np.arange(1000, 200 - 1e-4, -100)],
#     )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate q_weighted_lat am


#-------- settings

output_mp4 = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am SH.mp4'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-55, cm_max=-10, cm_interval1=2.5, cm_interval2=5, cmap='PuOr',)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# x-axis
ax.set_xticks(np.arange(-180, 180 + 1e-4, 60))
ax.set_xlim(-180, 180)
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', which='both',
        zorder=2)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticklabels(np.flip(height.magnitude), c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# cbar
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.25, anchor=(1.1, -0.6),)
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.set_xlabel('Humidity-weighted open-oceanic source latitude [$°\;S$]',)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.23, top=0.98)


plt_objs = []

def update_frames(ilat):
    # ilat = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_mesh = cplot_lon180(
        lon, plevs / 100, ax, pltnorm, pltcmp,
        q_weighted_lat[expid[i]]['am'].sel(
            lat = lat[ilat].values, plev=slice(1e+5, 2e+4)),)
    
    # which lat cross-section
    plt_text = plt.text(
        0.02, 0.98, str(np.negative(np.round(lat[ilat].values, 1))) + '$°\;S$',
        transform=ax.transAxes, weight='bold',
        ha='left', va='top', rotation='horizontal',
        bbox=dict(boxstyle='round', fc='white', ec='gray', lw=1, alpha=0.7),)
    
    plt_quiver = cplot_lon180_quiver(
        lon, plevs / 100, ax,
        plt_data = uv_plev[expid[i]]['u']['am'].sel(
            lat = uv_plev[expid[i]]['u']['am'].lat[ilat].values,
            plev=slice(1e+5, 2e+4)), color='k')
    
    plt_quiver_key = ax.quiverkey(plt_quiver, X=-0.05, Y=-0.15, U=10,
             label='10 $m \; s^{-1}$ zonal wind',
             labelpos='E', labelsep=0.05,)
    
    plt_ctr, ax_clabel = cplot_lon180_ctr(
        lon, plevs / 100, ax,
        plt_data = q_plev[expid[i]]['am'].sel(
            lat_2 = lat[ilat].values, plev=slice(1e+5, 2e+4)) * 1000
        )
    plt_ctr.__class__ = mpl.contour.QuadContourSet
    
    # contour label
    h1, _ = plt_ctr.legend_elements()
    ax_legend = ax.legend(
        [h1[0]], ['Specific humidity [$g \; kg^{-1}$]'],
        loc='lower left', frameon=False,
        bbox_to_anchor=(-0.13, -0.32),
        handlelength=1, columnspacing=1)
    
    plt_objs = [plt_mesh, plt_text, plt_quiver, plt_quiver_key,
                ax_legend,] + plt_ctr.collections + ax_clabel
    
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=np.arange(59, 96, 1), interval=1000, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot q_weighted_lat DJF - JJA zm


output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat DJF-JJA zm.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2, cmap='PiYG',)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
plt_mesh = ax.pcolormesh(
    lat, plevs / 100,
    q_weighted_lat[expid[i]]['sm'].sel(season='DJF').weighted(
        ocean_q_alltime[expid[i]]['sm'].sel(
            season='DJF', var_names='lat').fillna(0)
        ).mean(dim='lon').sel(plev=slice(1e+5, 2e+4)) - \
            q_weighted_lat[expid[i]]['sm'].sel(season='JJA').weighted(
        ocean_q_alltime[expid[i]]['sm'].sel(
            season='JJA', var_names='lat').fillna(0)
        ).mean(dim='lon').sel(plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp,
)

# contours
q_intervals = np.array([
    -2, -1, -0.5, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.5, 1, 2])
plt_ctr = ax.contour(
    lat.sel(lat=slice(-30, -90)), plevs / 100,
    (q_plev[expid[i]]['sm'].sel(season='DJF') - \
        q_plev[expid[i]]['sm'].sel(season='JJA')).mean(dim='lon').sel(
        lat_2=slice(-30, -90), plev=slice(1e+5, 2e+4)) * 1000,
    colors='b', levels=q_intervals, linewidths=0.3, linestyle='-',
    clip_on=True, zorder=2)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=q_intervals, inline_spacing=1, fontsize=6,)

# x-axis
ax.set_xticks(np.arange(-30, -90 - 1e-4, -10))
ax.set_xlim(-30, -90)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', which='both',)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticklabels(np.flip(height.magnitude), c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.25, anchor=(1.3, -0.6),)
cbar.ax.set_xlabel('DJF - JJA source latitude [$°$]',)

# contours legend
h1, _ = plt_ctr.legend_elements()
ax_legend = ax.legend(
    [h1[-1]], ['DJF - JJA specific humidity [$g \; kg^{-1}$]'],
    loc='lower left', frameon=False,
    bbox_to_anchor=(-0.15, -0.25),
    handlelength=1, columnspacing=1)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.23, top=0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


