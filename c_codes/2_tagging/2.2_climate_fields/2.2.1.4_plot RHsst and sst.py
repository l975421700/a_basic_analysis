

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    ]
i=0


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
    plot_labels_no_unit,
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

RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

lon = RHsst_alltime[expid[i]]['am'].lon
lat = RHsst_alltime[expid[i]]['am'].lat


'''

tsw_alltime[expid[i]]['am']


pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am RHsst

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=50, cm_max=90, cm_interval1=5, cm_interval2=10,
    cmap='PuOr', asymmetric=False, reversed=True)

output_png = 'figures/8_d-excess/8.2_climate/8.2.0 ' + expid[i] + ' am RHsst.png'

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.5]) / 2.54,)

plt1 = plot_t63_contourf(
    lon, lat, RHsst_alltime[expid[i]]['am'],
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks,
    pad=0.02, fraction=0.2,
    )

cbar.ax.set_xlabel('Annual mean RHsst [$\%$]', linespacing=1.5)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am sst

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=24, cm_interval1=2, cm_interval2=4,
    cmap='viridis', asymmetric=False, reversed=True)

output_png = 'figures/8_d-excess/8.2_climate/8.2.0 ' + expid[i] + ' am SST.png'

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.5]) / 2.54,)

plt1 = plot_t63_contourf(
    lon, lat, tsw_alltime[expid[i]]['am'],
    ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks,
    pad=0.02, fraction=0.2,
    )

cbar.ax.set_xlabel('Annual mean SST [$°C$]', linespacing=1.5)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sm RHsst

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=50, cm_max=90, cm_interval1=5, cm_interval2=10,
    cmap='PuOr', asymmetric=False, reversed=True)

for iseason in RHsst_alltime[expid[i]]['sm'].season.values:
    # iseason = 'DJF'
    print(iseason)
    
    output_png = 'figures/8_d-excess/8.2_climate/8.2.0 ' + expid[i] + ' ' + iseason + ' RHsst.png'
    
    fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.5]) / 2.54,)
    
    plt1 = plot_t63_contourf(
        lon, lat, RHsst_alltime[expid[i]]['sm'].sel(season=iseason),
        ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks,
        pad=0.02, fraction=0.2,
        )
    
    cbar.ax.set_xlabel(iseason + ' RHsst [$\%$]', linespacing=1.5)
    
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------



