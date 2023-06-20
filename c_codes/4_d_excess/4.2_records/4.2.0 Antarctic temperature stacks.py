

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


ATS_800kyr = pd.read_csv(
    'data_sources/Parrenin_2013/datasets/ATS.tab',
    sep='\t', header=0, skiprows=11,)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ATS of past 800,000 years


output_png = 'figures/8_d-excess/8.0_records/8.0.1_ice cores/8.0.1.0 ATS of past 800,000 years.pdf'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    ATS_800kyr['Age [ka BP]'],
    ATS_800kyr['T cal [°C]'],
    c='k', lw=0.3, ls='-')

ax.set_ylabel('Temperature anomalies [$°C$]')
ax.set_ylim(-11, 5)
ax.set_yticks(np.arange(-10, 4 + 1e-4, 2))

ax.set_xlabel('Age before 1950 [thousand years]')
ax.set_xlim(0, 800)
ax.set_xticks(np.arange(0, 800 + 1e-4, 100))

# ax.spines[['right', 'top']].set_visible(False)

fig.subplots_adjust(left=0.1, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------


