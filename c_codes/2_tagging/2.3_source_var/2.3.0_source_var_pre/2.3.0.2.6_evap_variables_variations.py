

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
mpl.rc('font', family='Times New Roman', size=8)
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

era5_evap_var2201 = xr.open_dataset('scratch/products/era5/era5_evap_hourly_variables_202201.nc')

era5_area = xr.open_dataset('scratch/products/era5/era5_gridarea.nc')
'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region variations in variables

sst_std = era5_evap_var2201.sst.std(dim='time', ddof=1, skipna = True)
np.nanmean(sst_std.sel(latitude=slice(-20, -90)).values)
# 0.4 degree * 0.3 = 0.12

era5_wind10 = ((era5_evap_var2201.u10)**2 + (era5_evap_var2201.v10)**2)**0.5

wind10_std = era5_wind10.std(dim='time', ddof=1)
np.mean(wind10_std.sel(latitude=slice(-20, -90)).values)
np.mean(wind10_std.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)])
# 3.1 m/s * 0.1 = 0.31

np.average(
    wind10_std.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)],
    weights=era5_area.cell_area.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)],
)
# 3.0 m/s * 0.1 = 0.30

from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

era5_rh2m = relative_humidity_from_dewpoint(
    era5_evap_var2201.t2m * units.degC,
    era5_evap_var2201.d2m * units.degC, )

rh2m_std = era5_rh2m.std(dim='time', ddof=1) * 100
np.mean(rh2m_std.sel(latitude=slice(-20, -90)).values)
np.mean(rh2m_std.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)])
# 2.40 m/s * 0.1 = 0.24

np.average(
    rh2m_std.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)],
    weights=era5_area.cell_area.sel(latitude=slice(-20, -90)).values[np.isfinite(sst_std.sel(latitude=slice(-20, -90)).values)],
)
# 2.4 % * 0.1 = 0.24

'''
sst_std.to_netcdf('scratch/test/run/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------

