

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
os.chdir('/work/ollie/qigao001')
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
from sklearn.metrics import mean_squared_error

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

with open('scratch/cmip6/hist/psl/psl_era5_79_21_alltime.pkl', 'rb') as f:
    psl_era5_79_21_alltime = pickle.load(f)


mon_psl = {}
mon_psl['sim_-40'] = (psl_zh[expid[i]]['psl']['mon'].sel(
    lat=-40, method='nearest').mean(dim='lon') / 100).compute()
mon_psl['sim_-65'] = (psl_zh[expid[i]]['psl']['mon'].sel(
    lat=-65, method='nearest').mean(dim='lon') / 100).compute()

mon_psl['era5_-40'] = (psl_era5_79_21_alltime['mon'].sel(
    latitude=-40, method='nearest').mean(dim='longitude') / 100).compute()
mon_psl['era5_-65'] = (psl_era5_79_21_alltime['mon'].sel(
    latitude=-65, method='nearest').mean(dim='longitude') / 100).compute()


mm_psl = {}
mm_psl['sim_-40'] = (psl_zh[expid[i]]['psl']['mm'].sel(
    lat=-40, method='nearest').mean(dim='lon') / 100).compute()
mm_psl['sim_-65'] = (psl_zh[expid[i]]['psl']['mm'].sel(
    lat=-65, method='nearest').mean(dim='lon') / 100).compute()

mm_psl['era5_-40'] = (psl_era5_79_21_alltime['mm'].sel(
    latitude=-40, method='nearest').mean(dim='longitude') / 100).compute()
mm_psl['era5_-65'] = (psl_era5_79_21_alltime['mm'].sel(
    latitude=-65, method='nearest').mean(dim='longitude') / 100).compute()


mon_psl_diff_std = {}
mon_psl_diff_std['sim'] = \
    (mon_psl['sim_-40'].groupby('time.month').std(ddof=1) ** 2 + \
        mon_psl['sim_-65'].groupby('time.month').std(ddof=1) ** 2) ** 0.5

mon_psl_diff_std['era5'] = \
    (mon_psl['era5_-40'].groupby('time.month').std(ddof=1) ** 2 + \
        mon_psl['era5_-65'].groupby('time.month').std(ddof=1) ** 2) ** 0.5




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mslp

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.1_mslp/6.1.9.1 ' + expid[i] + ' mslp zm mm 40_65.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

#-------- 40 degree south

ax.plot(
    month, mm_psl['sim_-40'], '.-',
    color='b', lw=1, markersize=6,)

ax.plot(
    month, mm_psl['era5_-40'], '.-',
    color='r', lw=1, markersize=6,)


ax.fill_between(
    x = month,
    y1 = mm_psl['sim_-40'] - mon_psl['sim_-40'].groupby('time.month').std(ddof=1),
    y2 = mm_psl['sim_-40'] + mon_psl['sim_-40'].groupby('time.month').std(ddof=1),
    color='b', alpha=0.2,)

ax.fill_between(
    x = month,
    y1 = mm_psl['era5_-40'] - mon_psl['era5_-40'].groupby('time.month').std(ddof=1),
    y2 = mm_psl['era5_-40'] + mon_psl['era5_-40'].groupby('time.month').std(ddof=1),
    color='r', alpha=0.2,)

#-------- 65 degree south

ax.plot(
    month, mm_psl['sim_-65'], '.--',
    color='b', lw=1, markersize=6,)

ax.plot(
    month, mm_psl['era5_-65'], '.--',
    color='r', lw=1, markersize=6,)

ax.fill_between(
    x = month,
    y1 = mm_psl['sim_-65'] - mon_psl['sim_-65'].groupby('time.month').std(ddof=1),
    y2 = mm_psl['sim_-65'] + mon_psl['sim_-65'].groupby('time.month').std(ddof=1),
    color='b', alpha=0.2,)

ax.fill_between(
    x = month,
    y1 = mm_psl['era5_-65'] - mon_psl['era5_-65'].groupby('time.month').std(ddof=1),
    y2 = mm_psl['era5_-65'] + mon_psl['era5_-65'].groupby('time.month').std(ddof=1),
    color='r', alpha=0.2,)


l1 = plt.plot([],[], '.-', color='r', lw=1, markersize=6, label='ERA5')
l2 = plt.plot([],[], '.-', color='b', lw=1, markersize=6, label='Simulation')
l3 = plt.plot([],[], '.-', color='black', lw=1, markersize=6, label='40$°\;S$')
l4 = plt.plot([],[], '.--', color='black', lw=1, markersize=6, label='65$°\;S$')


ax.legend(
    ncol=2, frameon=True,
    loc = 'center', handletextpad=0.5,)

ax.set_xlabel('MSLP [$hPa$]')

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mslp diff

output_png = 'figures/6_awi/6.1_echam6/6.1.9_sam/6.1.9.1_mslp/6.1.9.1 ' + expid[i] + ' mslp_diff zm mm 40_65.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.plot(
    month, mm_psl['sim_-40'] - mm_psl['sim_-65'], '.-',
    color='b', lw=1, markersize=6,)

ax.plot(
    month, mm_psl['era5_-40'] - mm_psl['era5_-65'], '.-',
    color='r', lw=1, markersize=6,)

ax.fill_between(
    x = month,
    y1 = mm_psl['sim_-40'] - mm_psl['sim_-65'] - mon_psl_diff_std['sim'],
    y2 = mm_psl['sim_-40'] - mm_psl['sim_-65'] + mon_psl_diff_std['sim'],
    color='b', alpha=0.2,)

ax.fill_between(
    x = month,
    y1 = mm_psl['era5_-40'] - mm_psl['era5_-65'] - mon_psl_diff_std['era5'],
    y2 = mm_psl['era5_-40'] - mm_psl['era5_-65'] + mon_psl_diff_std['era5'],
    color='r', alpha=0.2,)

l1 = plt.plot([],[], '.-', color='r', lw=1, markersize=6, label='ERA5')
l2 = plt.plot([],[], '.-', color='b', lw=1, markersize=6, label='Simulation')
ax.legend(
    ncol=1, frameon=True,
    loc = 'upper left', handletextpad=0.5,)

ax.set_xlabel('Differences in MSLP: 40$°\;S$ vs. 65$°\;S$ [$hPa$]')

ax.set_ylim(22, 41)
ax.set_yticks(np.arange(22, 41+1e-4, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check statistics (RMSE)


print('#---- RMSE: mm MSLP at 40° S')
print(np.round(mean_squared_error(
    mm_psl['sim_-40'],
    mm_psl['era5_-40'],
    squared=False), 1))

print('#---- RMSE: mm MSLP at 65° S')
print(np.round(mean_squared_error(
    mm_psl['sim_-65'],
    mm_psl['era5_-65'],
    squared=False), 1))


print('#---- RMSE: mm MSLP at 65° S')
print(np.round(mean_squared_error(
    mm_psl['sim_-40'] - mm_psl['sim_-65'],
    mm_psl['era5_-40'] - mm_psl['era5_-65'],
    squared=False), 1))



# endregion
# -----------------------------------------------------------------------------


