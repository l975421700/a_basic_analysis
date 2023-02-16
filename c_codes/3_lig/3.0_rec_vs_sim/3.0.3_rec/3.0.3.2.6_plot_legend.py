

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
sys.path.append('/home/users/qino')
os.chdir('/home/users/qino')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp

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
    regional_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
    find_ilat_ilon,
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
    marker_recs,
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
# region plot legend of symbols - blue


fig, ax = plt.subplots(1, 1, figsize=np.array([4, 2.4]) / 2.54)

symbol_size = 48
linewidth = 1
alpha = 1

l1 = plt.scatter(
    [],[], marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='b',
    lw=linewidth, alpha=alpha,)
l2 = plt.scatter(
    [],[], marker=marker_recs['JH'],
    s=symbol_size, c='white', edgecolors='b',
    lw=linewidth, alpha=alpha,)
l3 = plt.scatter(
    [],[], marker=marker_recs['DC'],
    s=symbol_size, c='white', edgecolors='b',
    lw=linewidth, alpha=alpha,)
l4 = plt.scatter(
    [],[], marker=marker_recs['MC'],
    s=symbol_size, c='white', edgecolors='b',
    lw=linewidth, alpha=alpha,)
plt.legend(
    [l1, l2, l3, l4,],
    ['Capron et al. (2017)',
     'Hoffman et al. (2017)',
     'Chandler et al. (2021)',
     'Chadwick et al. (2021)',],
    title = 'Datasets',
    title_fontsize = 10,
    ncol=1, frameon=False,
    loc = 'center', handletextpad=0.05,)

plt.axis('off')

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 legend of symbols_blue.png'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot legend of symbols - black


fig, ax = plt.subplots(1, 1, figsize=np.array([4, 2.4]) / 2.54)

symbol_size = 48
linewidth = 1
alpha = 1

l1 = plt.scatter(
    [],[], marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='black',
    lw=linewidth, alpha=alpha,)
l2 = plt.scatter(
    [],[], marker=marker_recs['JH'],
    s=symbol_size, c='white', edgecolors='black',
    lw=linewidth, alpha=alpha,)
l3 = plt.scatter(
    [],[], marker=marker_recs['DC'],
    s=symbol_size, c='white', edgecolors='black',
    lw=linewidth, alpha=alpha,)
l4 = plt.scatter(
    [],[], marker=marker_recs['MC'],
    s=symbol_size, c='white', edgecolors='black',
    lw=linewidth, alpha=alpha,)
plt.legend(
    [l1, l2, l3, l4,],
    ['Capron et al. (2017)',
     'Hoffman et al. (2017)',
     'Chandler et al. (2021)',
     'Chadwick et al. (2021)',],
    title = 'Datasets',
    title_fontsize = 10,
    ncol=1, frameon=False,
    loc = 'center', handletextpad=0.05,)

plt.axis('off')

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 legend of symbols_black.png'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot legend of uncertainties - size

fig, ax = plt.subplots(1, 1, figsize=np.array([4, 1.6]) / 2.54)

max_size = 80
scale_size = 16
linewidth = 0.5

l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=linewidth, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=linewidth, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=linewidth, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=linewidth, edgecolors = 'black',)

l5 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=linewidth, edgecolors = 'black',)
l6 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=linewidth, edgecolors = 'black',)
l7 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=linewidth, edgecolors = 'black',)
l8 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=linewidth, edgecolors = 'black',)
plt.legend(
    [l1, l5,
     l2, l6,
     l3, l7,
     l4, l8,],
    ['1', '1',
     '2', '2',
     '3', '3',
     '4', '4'],
    title = 'Two-sigma errors [$°C$]',
    ncol=4, frameon=False,
    loc = 'center', handletextpad=0.05, columnspacing=0.3,)

plt.axis('off')

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 legend of uncertainties.png'
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------

