

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
from datetime import datetime

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
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import linregress
from siphon.simplewebservice.igra2 import IGRAUpperAir

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
    inversion_top,
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
# region download EDC radiosonde data

daterange = [datetime(2006, 1, 1, 0), datetime(2022, 12, 31, 23)]
station = 'AYM00089625'

EDC_df_drvd, EDC_header_drvd = IGRAUpperAir.request_data(
    daterange, station, derived=True)
EDC_df_drvd.to_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')

'''
EDC_df, EDC_header = IGRAUpperAir.request_data(daterange, station)
EDC_df.to_pickle('scratch/radiosonde/igra2/EDC_df.pkl')
EDC_header.to_pickle('scratch/radiosonde/igra2/EDC_header.pkl')

EDC_header_drvd.to_pickle(
    'scratch/radiosonde/igra2/EDC_header_drvd.pkl')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get inversion top info

EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

EDC_radiosonde_inversion_height = np.zeros((len(date)))

for i in range(len(date)):
    # i=0
    altitude = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'calculated_height'].values / 1000
    temperature = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'temperature'].values
    
    t_it, h_it = inversion_top(temperature, altitude)
    
    EDC_radiosonde_inversion_height[i] = h_it

EDC_radiosonde_inversion_height = EDC_radiosonde_inversion_height[np.isfinite(EDC_radiosonde_inversion_height)]

np.mean(EDC_radiosonde_inversion_height * 1000 - 3233.0)
np.std(EDC_radiosonde_inversion_height * 1000 - 3233.0, ddof = 1)

# endregion
# -----------------------------------------------------------------------------

