

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')

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
from statsmodels.stats import multitest
import pycircstat as circ
import math

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

AC21_simulations = {}
AC21_simulations['HR'] = {}
AC21_simulations['LR'] = {}


AC21_simulations['HR']['d18O'] = xr.open_dataset('data_sources/AC21_JAMES/E6_HR_ERA5.d18Op_timmean.nc').wisoaprt_d
AC21_simulations['HR']['d_xs'] = xr.open_dataset('data_sources/AC21_JAMES/E6_HR_ERA5.dexp_timmean.nc').wisoaprt_d
AC21_simulations['HR']['dD']   = (8 * AC21_simulations['HR']['d18O'] + AC21_simulations['HR']['d_xs'].values).compute()


AC21_simulations['LR']['d18O'] = xr.open_dataset('data_sources/AC21_JAMES/E6_LR_ERA5.d18Op_timmean.nc').wisoaprt_d
AC21_simulations['LR']['d_xs'] = xr.open_dataset('data_sources/AC21_JAMES/E6_LR_ERA5.dexp_timmean.nc').wisoaprt_d
AC21_simulations['LR']['dD']   = (8 * AC21_simulations['LR']['d18O'] + AC21_simulations['LR']['d_xs'].values).compute()


with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region

AC21_simulations['LR']['dD'].squeeze().sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')

AC21_simulations['HR']['dD'].squeeze().sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')


# endregion
# -----------------------------------------------------------------------------

