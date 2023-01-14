

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
# region resample hourly era5 tp to daily tp

input_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_hourly_sl_12_21_Antarctica.nc',
]

output_files = [
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_79_89_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_90_00_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_01_11_Antarctica.nc',
    'scratch/cmip6/hist/pre/tp_ERA5_daily_sl_12_21_Antarctica.nc',
]


for ifile in range(len(input_files)):
    # ifile = 0
    print('#-------- ' + str(ifile))
    print(input_files[ifile])
    print(output_files[ifile])
    
    tp_era5_hourly = xr.open_dataset(input_files[ifile])
    
    tp_era5_daily = (tp_era5_hourly.tp.resample({'time': '1D'}).sum() * 1000).compute()
    
    tp_era5_daily.to_netcdf(output_files[ifile])
    
    del tp_era5_hourly, tp_era5_daily







'''
'''
# endregion
# -----------------------------------------------------------------------------


