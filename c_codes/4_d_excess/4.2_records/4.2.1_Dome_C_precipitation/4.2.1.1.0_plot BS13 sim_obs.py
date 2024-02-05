

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    
    'nudged_705_6.0',
    
    # 'nudged_703_6.0_k52',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress
from scipy.stats import pearsonr

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
import matplotlib.dates as mdates

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
    monthini,
    plot_labels,
    expid_colours,
    expid_labels,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('data_sources/Dome_C_records/BS13_Dome_C.pkl', 'rb') as f:
    BS13_Dome_C = pickle.load(f)

isotopes_alltime_icores = {}
temp2_alltime_icores = {}
wisoaprt_alltime_icores = {}

for i in range(len(expid)):
    print('#-------------------------------- ' + expid[i])
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
        temp2_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
        wisoaprt_alltime_icores[expid[i]] = pickle.load(f)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot

for ivar in ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]:
    # ivar = 'temp2'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]
    print('#-------------------------------- ' + ivar)
    
    if (ivar == 'dD'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        ydata_am = np.nanmean(ydata)
    elif (ivar == 'd18O'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = isotopes_alltime_icores[expid[i]]['dO18']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        ydata_am = np.nanmean(ydata)
    elif (ivar == 'd_xs'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = isotopes_alltime_icores[expid[i]]['d_excess']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        ydata_am = np.nanmean(ydata)
    elif (ivar == 'd_ln'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        ydata_am = np.nanmean(ydata)
    elif (ivar == 'pre'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')) * seconds_per_d
        ydata_am = np.nanmean(ydata)
    elif (ivar == 'temp2'):
        xdata = BS13_Dome_C['mon'][ivar].values
        xdata_am = BS13_Dome_C['am'][ivar]
        ydata = temp2_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        ydata_am = np.nanmean(ydata)
    
    subset = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    print(pearsonr(xdata, ydata).statistic ** 2)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.4_dome_c/8.0.4.0_sim_obs/8.0.4.0.0 ' + expid[i] + ' BS13 observed vs. simulated mon ' + ivar + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    ax.scatter(x=xdata, y=ydata, s=12, marker = 'x')
    ax.scatter(x=xdata_am, y=ydata_am, s=12, marker='*')
    
    linearfit = linregress(x = xdata, y = ydata,)
    # ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (ivar == 'pre'):
        round_digit = 2
    else:
        round_digit = 1
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, round_digit)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, round_digit))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, round_digit)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, round_digit))
    
    plt.text(0.5, 0.05, eq_text, transform=ax.transAxes, fontsize=10,
             ha='left', va='bottom', linespacing=2)
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    if (ivar == 'pre'):
        ax.set_xticks(np.arange(0, 0.14 + 1e-4, 0.02))
        ax.set_yticks(np.arange(0, 0.14 + 1e-4, 0.02))
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[ivar], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[ivar], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region time series plot - multiple models


for ivar in ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]:
    # ivar = 'temp2'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]
    print('#-------------------------------- ' + ivar)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.4_dome_c/8.0.4.0_sim_obs/8.0.4.0.1 nudged703_11 BS13 time series of observed vs. simulated mon ' + ivar + '.png'
    
    xdata = BS13_Dome_C['mon']['date'].values
    obs_var = BS13_Dome_C['mon'][ivar].values
    obs_var_am = BS13_Dome_C['am'][ivar]
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    ax.plot(xdata,
            obs_var, 'o', ls='-', ms=2, lw=0.5, c='k', label='Observation',
            )
    
    for i in range(len(expid)):
        # len(expid)
        print('#---------------- ' + expid[i])
        
        if (ivar == 'dD'):
            sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd18O'):
            sim_var = isotopes_alltime_icores[expid[i]]['dO18']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd_xs'):
            sim_var = isotopes_alltime_icores[expid[i]]['d_excess']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd_ln'):
            sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'pre'):
            sim_var = wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')) * seconds_per_d
        elif (ivar == 'temp2'):
            sim_var = temp2_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        
        sim_var_am = np.nanmean(sim_var)
        
        subset = np.isfinite(obs_var) & np.isfinite(sim_var)
        
        RMSE = np.sqrt(np.average(np.square(obs_var[subset] - sim_var[subset])))
        rsquared = pearsonr(obs_var[subset], sim_var[subset]).statistic ** 2
        
        if (ivar == 'pre'):
            round_digit = 2
        else:
            round_digit = 1
        
        ax.plot(xdata,
                sim_var, 'o', ls='-', ms=2, lw=0.5,
                c=expid_colours[expid[i]],
                label=expid_labels[expid[i]] + \
                    ': $RMSE = $' + str(np.round(RMSE, round_digit)),
                )
    
    ax.set_xticks(xdata[::4])
    
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.set_xlabel('Date', labelpad=3)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[ivar], labelpad=6)
    plt.xticks(rotation=30, ha='right')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    
    ax.legend(
        handlelength=1, loc=(-0.2, -0.65),
        framealpha=0.25, ncol=2, columnspacing=1, fontsize=9)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.4, top=0.98)
    fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region time series plot - only one model


for ivar in ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]:
    # ivar = 'temp2'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]
    print('#-------------------------------- ' + ivar)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.4_dome_c/8.0.4.0_sim_obs/8.0.4.0.1 ' + expid[i] + ' BS13 time series of observed vs. simulated mon ' + ivar + '.png'
    
    xdata = BS13_Dome_C['mon']['date'].values
    obs_var = BS13_Dome_C['mon'][ivar].values
    # obs_var_am = BS13_Dome_C['am'][ivar]
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    ax.plot(xdata,
            obs_var, 'o', ls='-', ms=2, lw=0.5, c='k', label='Observation',
            )
    
    for i in range(1):
        # len(expid)
        print('#---------------- ' + expid[i])
        
        if (ivar == 'dD'):
            sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd18O'):
            sim_var = isotopes_alltime_icores[expid[i]]['dO18']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd_xs'):
            sim_var = isotopes_alltime_icores[expid[i]]['d_excess']['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'd_ln'):
            sim_var = isotopes_alltime_icores[expid[i]][ivar]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        elif (ivar == 'pre'):
            sim_var = wisoaprt_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31')) * seconds_per_d
        elif (ivar == 'temp2'):
            sim_var = temp2_alltime_icores[expid[i]]['EDC']['mon'].sel(time=slice('2008-01-31', '2010-12-31'))
        
        # sim_var_am = np.nanmean(sim_var)
        
        subset = np.isfinite(obs_var) & np.isfinite(sim_var)
        
        RMSE = np.sqrt(np.average(np.square(obs_var[subset] - sim_var[subset])))
        rsquared = pearsonr(obs_var[subset], sim_var[subset]).statistic ** 2
        
        if (ivar == 'pre'):
            round_digit = 2
        else:
            round_digit = 1
        
        ax.plot(xdata,
                sim_var, 'o', ls='-', ms=2, lw=0.5,
                c=expid_colours[expid[i]],
                label=expid_labels[expid[i]] + \
                    ': $R^2 = $' + str(np.round(rsquared, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, round_digit))
                )
    
    ax.set_xticks(xdata[::4])
    
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.set_xlabel('Date', labelpad=3)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[ivar], labelpad=6)
    plt.xticks(rotation=30, ha='right')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    
    ax.legend(
        handlelength=1, loc=(-0.2, -0.3),
        framealpha=0.25, ncol=2, columnspacing=1, fontsize=9)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.25, top=0.98)
    fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


