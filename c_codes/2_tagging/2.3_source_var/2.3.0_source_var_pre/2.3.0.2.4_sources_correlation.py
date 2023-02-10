

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
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
from scipy.stats import linregress
from scipy.stats import pearsonr

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
    remove_trailing_zero_pos_abs,
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
    calc_lon_diff_np,
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

pre_weighted_var = {}
pre_weighted_var[expid[i]] = {}

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']

prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.pre_weighted_lat.pkl',
    prefix + '.pre_weighted_lon.pkl',
    prefix + '.pre_weighted_sst.pkl',
    prefix + '.pre_weighted_rh2m.pkl',
    prefix + '.pre_weighted_wind10.pkl',
    prefix + '.transport_distance.pkl',
]

for ivar, ifile in zip(source_var, source_var_files):
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        pre_weighted_var[expid[i]][ivar] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source lat and var

imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]

pearsonr(
    pre_weighted_var[expid[i]]['lat']['am'].values[mask],
    pre_weighted_var[expid[i]]['sst']['am'].values[mask],
)
# PearsonRResult(statistic=0.9635111986045921, pvalue=0.0)

pearsonr(
    pre_weighted_var[expid[i]]['lat']['am'].values[mask],
    pre_weighted_var[expid[i]]['rh2m']['am'].values[mask],
)
# PearsonRResult(statistic=-0.9299568995122842, pvalue=0.0)

pearsonr(
    pre_weighted_var[expid[i]]['lat']['am'].values[mask],
    pre_weighted_var[expid[i]]['wind10']['am'].values[mask],
)
# PearsonRResult(statistic=-0.8164258799993254, pvalue=0.0)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual source lat vs. wind10

imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_wind10/6.1.3.6.0 ' + expid[i] + ' correlation lat_wind10 am_AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

sns.scatterplot(
    pre_weighted_var[expid[i]]['lat']['am'].values[mask],
    pre_weighted_var[expid[i]]['wind10']['am'].values[mask],
)

linearfit = linregress(
    x = pre_weighted_var[expid[i]]['lat']['am'].values[mask],
    y = pre_weighted_var[expid[i]]['wind10']['am'].values[mask],
    )
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5,
          c='r')
plt.text(
    0.05, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, linespacing=1.5)

ax.set_xlim(-46, -34)
ax.set_xlabel('Source latitude [$°\;S$]')
ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylim(10, 11.4)
ax.set_ylabel('Source wind10 [$m\;s^{-1}$]')

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.98)
plt.savefig(output_png)
plt.close()


# endregion
# -----------------------------------------------------------------------------


