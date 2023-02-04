

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

epe_st_weighted_rh2m = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_rh2m.pkl', 'rb') as f:
    epe_st_weighted_rh2m[expid[i]] = pickle.load(f)

dc_st_weighted_rh2m = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_st_weighted_rh2m.pkl', 'rb') as f:
    dc_st_weighted_rh2m[expid[i]] = pickle.load(f)

lon = epe_st_weighted_rh2m[expid[i]]['90%']['am'].lon
lat = epe_st_weighted_rh2m[expid[i]]['90%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_st_weighted_rh2m - dc_st_weighted_rh2m) am Antarctica

iqtl = '90%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.3_source_rh2m/6.1.7.0.3 ' + expid[i] + ' epe_st_weighted_rh2m - dc_st_weighted_rh2m am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-8, cm_max=0, cm_interval1=1, cm_interval2=1, cmap='BrBG',
    reversed=True)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    epe_st_weighted_rh2m[expid[i]][iqtl]['am'] - \
        dc_st_weighted_rh2m[expid[i]][iqtl]['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    epe_st_weighted_rh2m[expid[i]][iqtl]['ann'],
    dc_st_weighted_rh2m[expid[i]][iqtl]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('EPE source rh2m anomalies [$\%$]', linespacing=2)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_st_weighted_rh2m '90%'-dc_st_weighted_rh2m '10%') am Antarctica

# iqtl = '90%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.3_source_rh2m/6.1.7.0.3 ' + expid[i] + ' epe_st_weighted_rh2m_90 - dc_st_weighted_rh2m_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-10, cm_max=2, cm_interval1=1, cm_interval2=1, cmap='BrBG',
    reversed=True, asymmetric=True,)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    epe_st_weighted_rh2m[expid[i]]['90%']['am'] - \
        dc_st_weighted_rh2m[expid[i]]['10%']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    epe_st_weighted_rh2m[expid[i]]['90%']['ann'],
    dc_st_weighted_rh2m[expid[i]]['10%']['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('EPE-DD source rh2m [$\%$]', linespacing=2)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (dc_st_weighted_rh2m_10 - epe_st_weighted_rh2m_10) am Antarctica

iqtl = '10%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.3_source_rh2m/6.1.7.0.3 ' + expid[i] + ' dc_st_weighted_rh2m_10 - epe_st_weighted_rh2m_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=16, cm_interval1=1, cm_interval2=2, cmap='BrBG',
    reversed=True, asymmetric=True)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    dc_st_weighted_rh2m[expid[i]][iqtl]['am'] - \
        epe_st_weighted_rh2m[expid[i]][iqtl]['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    dc_st_weighted_rh2m[expid[i]][iqtl]['ann'],
    epe_st_weighted_rh2m[expid[i]][iqtl]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('LP source rh2m anomalies [$\%$]', linespacing=2)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


