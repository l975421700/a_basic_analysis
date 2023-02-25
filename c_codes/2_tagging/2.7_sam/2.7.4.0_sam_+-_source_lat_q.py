

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

q_weighted_lat = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lat.pkl',
          'rb') as f:
    q_weighted_lat[expid[i]] = pickle.load(f)

lon = q_weighted_lat[expid[i]]['am'].lon
lat = q_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)
plevs = q_weighted_lat[expid[i]]['am'].plev

#---- import sam
sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_mon.sam > sam_mon.sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_mon.sam < (-1 * sam_mon.sam.std(ddof = 1))

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get sam+- source_lat_q

zm_mon_source_lat_q = q_weighted_lat[expid[i]]['mon'].weighted(
    ocean_q_alltime[expid[i]]['mon'].sel(var_names='lat').fillna(0)
    ).mean(dim='lon')

clim = zm_mon_source_lat_q.groupby('time.month').mean().compute()
anom = (zm_mon_source_lat_q.groupby('time.month') - clim).compute()

sam_posneg_lat_q = {}
sam_posneg_lat_q['pos'] = anom[sam_posneg_ind['pos'].values]
sam_posneg_lat_q['pos_mean'] = sam_posneg_lat_q['pos'].mean(
    dim='time', skipna=True)

sam_posneg_lat_q['neg'] = anom[sam_posneg_ind['neg'].values]
sam_posneg_lat_q['neg_mean'] = sam_posneg_lat_q['neg'].mean(
    dim='time', skipna=True)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sam+- source_lat_q

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.0_cor_lat/6.1.9.0 ' + expid[i] + ' sam_posneg_lat_q SH.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-6, cm_max=6, cm_interval1=1, cm_interval2=1, cmap='BrBG',
    asymmetric=True)
plat_2d, plevs_2d = np.meshgrid(
    lat.sel(lat=slice(0, -90)),
    plevs.sel(plev=slice(1e+5, 2e+4)) / 100,)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# mesh
plt_mesh = ax.contourf(
    lat.sel(lat=slice(0, -90)),
    plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    (sam_posneg_lat_q['pos_mean'] - sam_posneg_lat_q['neg_mean']
     ).sel(lat=slice(0, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both',)
ttest_fdr_res = ttest_fdr_control(
    sam_posneg_lat_q['pos'].sel(
        lat=slice(0, -90), plev=slice(1e+5, 2e+4)).values,
    sam_posneg_lat_q['neg'].sel(
        lat=slice(0, -90), plev=slice(1e+5, 2e+4)).values,)
ax.scatter(
    x=plat_2d[ttest_fdr_res],
    y=plevs_2d[ttest_fdr_res],
    s=1, c='k', marker='.', edgecolors='none',
    )

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

# mesh cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    # anchor=(1.1, -1),
    )
cbar.ax.set_xlabel('Source latitude differences [$°$] SAM+ vs. SAM-',)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------




