

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import pearsonr
from scipy.stats import linregress

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    ticks_labels,
    hemisphere_conic_plot,
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
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

MC16_Dome_C_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.MC16_Dome_C_1d_sim.pkl', 'rb') as f:
    MC16_Dome_C_1d_sim[expid[i]] = pickle.load(f)

with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

'''
#---------------- check correlation
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_3m', 'q']:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    subset = np.isfinite(MC16_Dome_C_1d_sim[expid[i]][var_name]) & np.isfinite(MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'])
    
    print(np.round(pearsonr(MC16_Dome_C_1d_sim[expid[i]][var_name][subset], MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'][subset], ).statistic ** 2, 3))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_3m', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.0 ' + expid[i] + ' MC16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = MC16_Dome_C_1d_sim[expid[i]][var_name]
    ydata = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata,
        s=12,
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline(
        (0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot time series

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_3m', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.1 ' + expid[i] + ' MC16 time series of observed and simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 6.6]) / 2.54)
    
    xdata = MC16_Dome_C_1d_sim[expid[i]]['time'].values
    ydata = MC16_Dome_C_1d_sim[expid[i]][var_name].values
    ydata_sim = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].values
    
    if (var_name == 'q'):
        ydata = ydata * 1000
        ydata_sim = ydata_sim * 1000
    
    ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5, label='Sim.',)
    ax.plot(xdata, ydata, 'o', ls='-', ms=2, lw=0.5, label='Obs.',)
    
    hourly_y = MC16_Dome_C['1h'][var_name].values
    if (var_name == 'q'):
        hourly_y = hourly_y * 1000
    ax.plot(
        MC16_Dome_C['1h']['time'].values,
        hourly_y,
        ls='-', lw=0.2, label='Hourly Obs.',)
    
    ax.set_xticks(xdata[::4])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Date', labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[var_name], labelpad=6)
    plt.xticks(rotation=30, ha='right')
    
    ax.legend(handlelength=1, loc='upper right', framealpha=0.25,)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.3, top=0.98)
    fig.savefig(output_png)



# MC16_Dome_C_1d_sim[expid[i]]['time']

# endregion
# -----------------------------------------------------------------------------


