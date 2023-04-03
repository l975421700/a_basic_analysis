

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

q_weighted_sst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_sst.pkl',
          'rb') as f:
    q_weighted_sst[expid[i]] = pickle.load(f)

q_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_plev.pkl',
    'rb') as f:
    q_plev[expid[i]] = pickle.load(f)

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

lon = q_weighted_sst[expid[i]]['am'].lon
lat = q_weighted_sst[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)
plevs = q_weighted_sst[expid[i]]['am'].plev

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
# region plot q_weighted_sst am zm SH

am_zm_source_sst = q_weighted_sst[expid[i]]['am'].weighted(
        ocean_q_alltime[expid[i]]['am'].sel(var_names='sst').fillna(0)
        ).mean(dim='lon')

am_zm_q = q_plev[expid[i]]['am'].mean(dim='lon') * 1000

am_zm_tpot = theta_e_alltime['am'].mean(dim='lon') - zerok

# output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_sst/6.1.5.0 ' + expid[i] + ' q_weighted_sst am zm_SH.png'
output_png = 'figures/test/trial1.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=28, cm_interval1=1, cm_interval2=2, cmap='viridis',
    reversed=False)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)), plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_source_sst.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both'
)

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
cbar.ax.set_xlabel('Source SST [$°C$]',)
cbar.ax.invert_xaxis()

h2, _ = plt_ctr_tpot.legend_elements()
ax_legend = ax.legend(
    [
        h2[0],
        ],
    [
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
