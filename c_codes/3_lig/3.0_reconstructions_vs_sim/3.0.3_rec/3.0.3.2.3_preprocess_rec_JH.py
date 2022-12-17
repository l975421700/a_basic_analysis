

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
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
    find_ilat_ilon_general,
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

lig_recs = {}
lig_recs['JH'] = {}

lig_recs['JH']['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name=' Hoffman et al. 2017', header=0, skiprows=14, nrows=37,)

lig_recs['JH']['SO_ann'] = lig_recs['JH']['original'].loc[
    (lig_recs['JH']['original']['Region']=='Southern Ocean') & \
        ['Annual SST' in string for string in lig_recs['JH']['original']['Type']],
        ]

lig_recs['JH']['SO_jfm'] = lig_recs['JH']['original'].loc[
    (lig_recs['JH']['original']['Region']=='Southern Ocean') & \
        ['Summer SST' in string for string in lig_recs['JH']['original']['Type']],
        ]

lig_recs['JH']['NH_ann'] = lig_recs['JH']['original'].loc[
    (lig_recs['JH']['original']['Region']=='North Atlantic') & \
        ['Annual SST' in string for string in lig_recs['JH']['original']['Type']], ]

lig_recs['JH']['NH_jas'] = lig_recs['JH']['original'].loc[
    (lig_recs['JH']['original']['Region']=='North Atlantic') & \
        ['Summer SST' in string for string in lig_recs['JH']['original']['Type']], ]

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'wb') as f:
    pickle.dump(lig_recs['JH'], f)


'''
#-------------------------------- check
lig_recs = {}
with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

# 37
lig_recs['JH']['original']
# 12
lig_recs['JH']['SO_ann']
# 7
lig_recs['JH']['SO_jfm']
# 9
lig_recs['JH']['NH_ann']
# 9
lig_recs['JH']['NH_jas']

12 + 7 + 9 + 9

'''
# endregion
# -----------------------------------------------------------------------------

