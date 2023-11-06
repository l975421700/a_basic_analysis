

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)

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
# region general settings

sim_periods = ['PI', 'LIG', 'LIG0.25']

sim_folder = {
    'PI': 'xpjaa',
    'LIG': 'xpkba',
    'LIG0.25': 'xppfa',
}

var_names = {
    'SAT': 'temp_mm_1_5m',
    'SIC': 'iceconc_mm_uo',
    'SST': 'temp_mm_uo',
}

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get original data


hadcm3_output = {}

for iperiod in ['PI', 'LIG',]:
    # iperiod = 'LIG'
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output[iperiod] = {}
    
    for ivar in ['SAT', 'SIC', 'SST']:
        # ivar = 'SAT'
        print('#---------------- ' + ivar)
        
        filelists = sorted(glob.glob('scratch/share/from_rahul/data_qingang/' + sim_folder[iperiod] + '/' + ivar + '/*'))
        
        hadcm3_output[iperiod][ivar] = xr.open_mfdataset(filelists)[
            var_names[ivar]]

var_names = {
    'SAT': 'temp_mm_1_5m',
    'SIC': 'iceconc_mm_srf',
    'SST': 'temp_mm_uo',
}

iperiod = 'LIG0.25'
hadcm3_output[iperiod] = {}
for ivar in ['SAT', 'SIC', 'SST']:
    # ivar = 'SAT'
    print('#---------------- ' + ivar)
    
    hadcm3_output[iperiod][ivar] = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/xppfa_' + ivar + '_bristol_BAS_13sep23.nc')[var_names[ivar]]

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'wb') as f:
    pickle.dump(hadcm3_output, f)



'''
#-------------------------------- check

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

for iperiod in ['LIG0.25']:
    # iperiod = 'LIG'
    # ['PI', 'LIG', 'LIG0.25']
    print('#-------------------------------- ' + iperiod)
    
    for ivar in ['SAT', 'SIC', 'SST']:
        # ivar = 'SAT'
        print('#---------------- ' + ivar)
        
        #---------------- length
        print(hadcm3_output[iperiod][ivar])

# hadcm3_output['LIG0.25']['SST'].time
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get cleaned data

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

hadcm3_output_cleaned = {}

for iperiod in ['PI', 'LIG', 'LIG0.25',]:
    # iperiod = 'PI'
    print('#-------------------------------- ' + iperiod)
    
    hadcm3_output_cleaned[iperiod] = {}
    
    for ivar in ['SAT', 'SIC', 'SST']:
        # ivar = 'SAT'
        print('#---------------- ' + ivar)
        
        hadcm3_output_cleaned[iperiod][ivar] = hadcm3_output[iperiod][ivar].squeeze()
        
        hadcm3_output_cleaned[iperiod][ivar].coords
        


'''
# squeeze out the non useful dimension
# change name of time dimension to 'time'
# change name of lat/lat dimention to 'lat'/'lon'
# select data from the last 100 years of PI and LIG, while the whole of LIG0.25

# assign time attribute to 'SST' in LIG0.25
'''
# endregion
# -----------------------------------------------------------------------------


