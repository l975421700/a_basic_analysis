

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_502_5.0',]
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
from scipy.stats import circstd

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
    remove_trailing_zero_pos_abs
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
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

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

evapiac_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.evapiac_alltime.pkl', 'rb') as f:
    evapiac_alltime[expid[i]] = pickle.load(f)

echam_t63_area = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region evaporation from sea ice and leads

# mm/year * global area
aprt_SHseaice = np.average(
    aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=21),
    weights=echam_t63_area.cell_area,) * \
        seconds_per_d * 365 * \
        echam_t63_area.cell_area.sum()

evapiac_SHseaice = np.average(
    evapiac_alltime[expid[i]]['am'].sel(lat=slice(-40, -90)),
    weights=echam_t63_area.cell_area.sel(lat=slice(-40, -90)),) * \
        seconds_per_d * 365 * \
            echam_t63_area.cell_area.sel(lat=slice(-40, -90)).sum()

evapiac_SHseaice / aprt_SHseaice
# only 10% of aprt from SHseaice is evaporated from sea ice
# endregion
# -----------------------------------------------------------------------------


