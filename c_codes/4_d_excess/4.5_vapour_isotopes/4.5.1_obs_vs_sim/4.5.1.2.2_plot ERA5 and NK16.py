

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

with open('scratch/ERA5/temp2/NK16_Australia_Syowa_1d_era5.pkl', 'rb') as f:
    NK16_Australia_Syowa_1d_era5 = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot

output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 NK16 vs. ERA5 daily 2m temperature.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

xdata = NK16_Australia_Syowa_1d_era5['t_air']
ydata = NK16_Australia_Syowa_1d_era5['t2m_era5'] - zerok
subset = (np.isfinite(xdata) & np.isfinite(ydata))
xdata = xdata[subset]
ydata = ydata[subset]

RMSE = np.sqrt(np.average(np.square(xdata - ydata)))

sns.scatterplot( x=xdata, y=ydata, s=12,)
linearfit = linregress(x = xdata, y = ydata,)
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)

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
ax.set_xlabel('Observed '  + plot_labels['t_air'], labelpad=6)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_ylabel('ERA5 ' + plot_labels['t_air'], labelpad=6)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------



