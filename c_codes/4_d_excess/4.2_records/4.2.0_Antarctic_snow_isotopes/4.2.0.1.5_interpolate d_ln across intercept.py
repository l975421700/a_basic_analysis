

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_605_5.5',
    ]


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
from scipy import interpolate

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
    find_ilat_ilon_general,
    find_multi_gridvalue_at_site,
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
    expid_colours,
    expid_labels,
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


Antarctic_snow_isotopes_sim_grouped = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped.pkl', 'rb') as f:
        Antarctic_snow_isotopes_sim_grouped[expid[i]] = pickle.load(f)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region interpolate d_ln to different intercept of Si

Si_intercept_values = np.arange(1, 1.04 + 1e-6, 0.001)

interpolated_dln = {}

for ivalue in range(len(Si_intercept_values)):
    print('#-------- ' + str(ivalue) + ': ' + str(np.round(Si_intercept_values[ivalue], 3)))
    
    No_record = len(Antarctic_snow_isotopes_sim_grouped['pi_600_5.0'].index)
    interpolated_dln[str(ivalue)] = np.zeros(No_record)
    
    for irecord in range(No_record):
        # print(irecord)
        
        interpolated_dln[str(ivalue)][irecord] = interpolate.interp1d(
            np.array([1, 1.02]),
            np.array([
                Antarctic_snow_isotopes_sim_grouped['pi_605_5.5']['d_ln_sim'][irecord],
                Antarctic_snow_isotopes_sim_grouped['pi_600_5.0']['d_ln_sim'][irecord],
                ]),
            fill_value="extrapolate",
        )(Si_intercept_values[ivalue])




'''
#-------------------------------- check that it simulates the original one right

(interpolated_dln['0'] == Antarctic_snow_isotopes_sim_grouped['pi_605_5.5']['d_ln_sim']).all()
np.max(abs(interpolated_dln['20'] - Antarctic_snow_isotopes_sim_grouped['pi_600_5.0']['d_ln_sim']))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot interpolated d_ln

output_png = 'figures/8_d-excess/8.1_controls/8.1.7_parameterisation/8.1.7.0.1 pi_60_05 interpolated d_ln across intercepts.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

marker='o'
linewidth=1
markersize=3

# 1, Si_intercept_values[0]
l1, = ax.plot(
    interpolated_dln['0'], color='tab:red', marker=marker,
    linewidth=linewidth, markersize=markersize,)

# 1.01, Si_intercept_values[10]
l2, = ax.plot(
    interpolated_dln['10'], color='tab:green', marker=marker,
    linewidth=linewidth, markersize=markersize)

# 1.02, Si_intercept_values[20]
l3, = ax.plot(
    interpolated_dln['20'], color='black', marker=marker,
    linewidth=linewidth, markersize=markersize)

# 1.03, Si_intercept_values[30]
l4, = ax.plot(
    interpolated_dln['30'], color='tab:orange', marker=marker,
    linewidth=linewidth, markersize=markersize)

# 1.04, Si_intercept_values[40]
l5, = ax.plot(
    interpolated_dln['40'], color='tab:purple', marker=marker,
    linewidth=linewidth, markersize=markersize)

ax.legend(
    [l1, l2, l3, l4, l5,],
    [str(np.round(Si_intercept_values[0], 2)),
     str(np.round(Si_intercept_values[10], 2)),
     str(np.round(Si_intercept_values[20], 2)),
     str(np.round(Si_intercept_values[30], 2)),
     str(np.round(Si_intercept_values[40], 2)),],
    title = '$Si$ intercept', title_fontsize = 10, handlelength=1,
    ncol=2, frameon=False, loc = 'lower left', handletextpad=0.2,
    columnspacing=0.5,
    )

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Random order of grid cells', labelpad=6)
ax.set_ylabel('Interpolated ' + plot_labels['d_ln'], labelpad=6)

ax.grid(
    True, which='both', linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot RMSE

RMSE = np.zeros(len(Si_intercept_values))

for ivalue in range(len(Si_intercept_values)):
    print('#-------- ' + str(ivalue) + ': ' + str(np.round(Si_intercept_values[ivalue], 3)))
    
    RMSE[ivalue] = np.sqrt(np.average(np.square(
        interpolated_dln[str(ivalue)] - \
            Antarctic_snow_isotopes_sim_grouped['pi_600_5.0']['d_ln']
    )))

output_png = 'figures/8_d-excess/8.1_controls/8.1.7_parameterisation/8.1.7.0.1 pi_60_05 interpolated d_ln across intercepts RMSE.png'

marker='o'
linewidth=1
markersize=3

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.plot(
    Si_intercept_values, RMSE, color='black',
    marker=marker, linewidth=linewidth, markersize=markersize,)
ax.plot(
    Si_intercept_values[np.argmin(RMSE)], RMSE[np.argmin(RMSE)], color='tab:red',
    marker=marker, linewidth=linewidth, markersize=markersize,)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.axvline(x=1.02)

ax.set_xlabel('$Si$ intercept', labelpad=6)
ax.set_ylabel('RMSE in $d_{ln}$ [$‰$] compared to MD08', labelpad=6)

ax.grid(
    True, which='both', linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
fig.savefig(output_png)





'''
np.min(RMSE)
np.argmin(RMSE)


Si_intercept_values[0]
np.round(RMSE[0], 1)

Si_intercept_values[20]
np.round(RMSE[20], 1)


'''
# endregion
# -----------------------------------------------------------------------------



