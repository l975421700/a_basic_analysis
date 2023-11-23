

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

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
    monthini,
    plot_labels,
    expid_labels,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('data_sources/Dome_C_records/BS13_Dome_C.pkl', 'rb') as f:
    BS13_Dome_C = pickle.load(f)

isotopes_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
    isotopes_alltime_icores[expid[i]] = pickle.load(f)

temp2_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
    temp2_alltime_icores[expid[i]] = pickle.load(f)

wisoaprt_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    wisoaprt_alltime_icores[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot

for ivar in ['dD', 'd18O', 'd_xs', 'd_ln', 'pre', 'temp2', ]:
    # ivar = 'temp2'
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
    
    plt.text(0.32, 0.15, eq_text, transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
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


