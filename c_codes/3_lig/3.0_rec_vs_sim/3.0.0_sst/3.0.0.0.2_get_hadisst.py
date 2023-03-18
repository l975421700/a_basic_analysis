

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
# region get HadISST mon_sea_ann sst

HadISST = {}
HadISST['sst'] = {}
HadISST['sst']['original'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_sst.nc')

HadISST['sst']['original'].sst.values[
    HadISST['sst']['original'].sst.values == -1000] = np.nan

HadISST['sst']['1deg'] = regrid(HadISST['sst']['original'].sst)

HadISST['sst']['1deg_alltime'] = mon_sea_ann(
    var_monthly=HadISST['sst']['1deg'].isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)

with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'wb') as f:
    pickle.dump(HadISST['sst'], f)


'''
#-------- check
HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)

(HadISST['sst']['original'].sst.values < -5).sum()

HadISST['sst']['original']
data1 = HadISST['sst']['1deg'][:360].values
data2 = HadISST['sst']['1deg_alltime']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


HadISST['sst_alltime'] = mon_sea_ann(
    var_monthly=HadISST['sst'].sst.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)

HadISST['sst_alltime']['am'].to_netcdf('scratch/test/run/test1.nc')
HadISST['sst_1deg_alltime']['am'].to_netcdf('scratch/test/run/test2.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get HadISST mon_sea_ann sic

HadISST = {}
HadISST['sic'] = {}
HadISST['sic']['original'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_ice.nc')

HadISST['sic']['original'].sic.values[:] = \
    HadISST['sic']['original'].sic.values * 100

HadISST['sic']['1deg'] = regrid(HadISST['sic']['original'].sic)

HadISST['sic']['1deg_alltime'] = mon_sea_ann(
    var_monthly=HadISST['sic']['1deg'].isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)

with open('data_sources/LIG/HadISST1.1/HadISST_sic.pkl', 'wb') as f:
    pickle.dump(HadISST['sic'], f)



'''

#-------- check
HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sic.pkl', 'rb') as f:
    HadISST['sic'] = pickle.load(f)

data1 = HadISST['sic']['1deg'][:360].values
data2 = HadISST['sic']['1deg_alltime']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


data2 = mon_sea_ann(
    var_monthly=HadISST['sic']['original'].sic.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)

data2['am'].to_netcdf('scratch/test/run/test1.nc')
HadISST['sic']['1deg_alltime']['am'].to_netcdf('scratch/test/run/test2.nc')


'''
# endregion
# -----------------------------------------------------------------------------


