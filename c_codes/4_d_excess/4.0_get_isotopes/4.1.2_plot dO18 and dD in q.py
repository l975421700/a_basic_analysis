

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
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
# sys.path.append('/work/ollie/qigao001')

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
from statsmodels.stats import multitest
import pycircstat as circ
import math

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

dO18_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl', 'rb') as f:
    dO18_q_alltime[expid[i]] = pickle.load(f)

dD_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
    dD_q_alltime[expid[i]] = pickle.load(f)

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)

lon = dO18_q_alltime[expid[i]]['am'].lon
lat = dO18_q_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)
plevs = dO18_q_alltime[expid[i]]['am'].plev

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot dD_q_alltime

am_zm_dD = dD_q_alltime[expid[i]]['am'].weighted(
    wiso_q_plev_alltime[expid[i]]['q16o']['am'].fillna(0)
    ).mean(dim='lon').compute()

output_png = 'figures/test/test.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-550, cm_max=-50, cm_interval1=25, cm_interval2=50, cmap='viridis',
    reversed=False)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    lat.sel(lat=slice(3, -90)),
    plevs.sel(plev=slice(1e+5, 2e+4)) / 100,
    am_zm_dD.sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='both'
)

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
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.04,
    anchor=(0.5, -1),
    )
cbar.ax.set_xlabel(plot_labels['dD'],)
cbar.ax.invert_xaxis()

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------

