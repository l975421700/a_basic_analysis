

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_502_5.0',]
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

sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

d_excess_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    d_excess_alltime[expid[i]] = pickle.load(f)

lon = d_excess_alltime[expid[i]]['am'].lon
lat = d_excess_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

b_sam_mon, _ = xr.broadcast(
    sam_mon.sam,
    d_excess_alltime[expid[i]]['mon'])

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d-excess & SAM

cor_sam_d_excess = xr.corr(
    b_sam_mon,
    d_excess_alltime[expid[i]]['mon'],
    dim='time').compute()

cor_sam_d_excess_p = xs.pearson_r_eff_p_value(
    b_sam_mon,
    d_excess_alltime[expid[i]]['mon'],
    dim='time').values


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.4, cm_max=0.4, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)

pltticks[-5] = 0

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.0 ' + expid[i] + ' corr. sam_d-excess mon.png'

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 6.8]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

cor_sam_d_excess.values[cor_sam_d_excess_p > 0.05] = np.nan

plt1 = plot_t63_contourf(
    lon, lat, cor_sam_d_excess, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

# plt1 = ax.pcolormesh(
#     lon,
#     lat,
#     cor_sam_d_excess,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ax.add_feature(
    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Correlation: SAM & d-excess',
    linespacing=1.5, fontsize=8)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


