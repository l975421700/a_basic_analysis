

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
from iteration_utilities import duplicates

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

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

lig_datasets = pd.read_excel('data_sources/LIG/lig_datasets.xlsx', header=0,)

'''
lig_recs['DC'].keys()
lig_recs['DC']['annual_128'].to_csv('scratch/test/test0.csv')
lig_recs['DC']['JFM_128'].to_csv('scratch/test/test1.csv')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region number of sites in common

# 1
itype = 'Annual SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Hoffman et al. (2017)'

# 2
itype = 'Annual SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Chandler et al. (2021)'

# 3
itype = 'Annual SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Chadwick et al. (2021)'

# 4
itype = 'Annual SST'
irec1 = 'Hoffman et al. (2017)'
irec2 = 'Chandler et al. (2021)'

# 5
itype = 'Annual SST'
irec1 = 'Hoffman et al. (2017)'
irec2 = 'Chadwick et al. (2021)'

# 6
itype = 'Annual SST'
irec1 = 'Chandler et al. (2021)'
irec2 = 'Chadwick et al. (2021)'

# 7
itype = 'Summer SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Hoffman et al. (2017)'

# 8
itype = 'Summer SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Chandler et al. (2021)'

# 9
itype = 'Summer SST'
irec1 = 'Capron et al. (2017)'
irec2 = 'Chadwick et al. (2021)'

# 10
itype = 'Summer SST'
irec1 = 'Hoffman et al. (2017)'
irec2 = 'Chandler et al. (2021)'

# 11
itype = 'Summer SST'
irec1 = 'Hoffman et al. (2017)'
irec2 = 'Chadwick et al. (2021)'

# 12
itype = 'Summer SST'
irec1 = 'Chandler et al. (2021)'
irec2 = 'Chadwick et al. (2021)'




data_subsets = lig_datasets.loc[
    (lig_datasets.Type == itype) & \
        ((lig_datasets.Dataset == irec1) | \
            (lig_datasets.Dataset == irec2))
    ]

duplicate_stations = list(duplicates(data_subsets.station_unified))

print(duplicate_stations)












'''

#-------------------------------- check differences

for isite in duplicate_stations:
    # isite = duplicate_stations[0]
    print('#---------------- ' + isite)
    
    duplicate_subsets = data_subsets.loc[data_subsets.station_unified == isite]
    # duplicate_subsets.columns
    
    value1 = np.round(duplicate_subsets['127 ka SST anomalies [°C]'].values[0], 2)
    value2 = np.round(duplicate_subsets['127 ka SST anomalies [°C]'].values[1], 2)
    twosigma1 = np.round(duplicate_subsets['two-sigma errors [°C]'].values[0], 2)
    twosigma2 = np.round(duplicate_subsets['two-sigma errors [°C]'].values[1], 2)
    
    one_in_two = (value1 < value2 + twosigma2) & (value1 > value2 - twosigma2)
    two_in_one = (value2 < value1 + twosigma1) & (value2 > value1 - twosigma1)
    
    if not (one_in_two | two_in_one):
        print('two datasets differ significantly')


(lig_datasets.Dataset == 'Capron et al. (2017)').sum()
(lig_datasets.Dataset == 'Hoffman et al. (2017)').sum()
((lig_datasets.Dataset == 'Capron et al. (2017)') | \
    (lig_datasets.Dataset == 'Hoffman et al. (2017)')).sum()

(lig_datasets.Type == 'Annual SST').sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region values of common sites

itypes = [
    'Annual SST', 'Annual SST', 'Annual SST',
    'Summer SST', 'Summer SST', 'Summer SST',]

irec1s = [
    'Capron et al. (2017)', 'Capron et al. (2017)', 'Hoffman et al. (2017)',
    'Capron et al. (2017)', 'Capron et al. (2017)', 'Hoffman et al. (2017)',]

irec2s = [
    'Hoffman et al. (2017)', 'Chandler et al. (2021)', 'Chandler et al. (2021)',
    'Hoffman et al. (2017)', 'Chandler et al. (2021)', 'Chandler et al. (2021)',
]

for iind in range(len(itypes)):
    # iind = 0
    print('#-------------------------------- ' + itypes[iind])
    
    print('#---------------- ' + irec1s[iind] + ' vs. ' + irec2s[iind])
    
    data_subsets = lig_datasets.loc[
        (lig_datasets.Type == itypes[iind]) & \
            ((lig_datasets.Dataset == irec1s[iind]) | \
                (lig_datasets.Dataset == irec2s[iind]))
            ]
    
    duplicate_stations = list(duplicates(data_subsets.station_unified))
    
    # print(duplicate_stations)
    
    if (len(duplicate_stations) > 0):
        for istation in duplicate_stations:
            # istation = duplicate_stations[1]
            print('#-------- ' + istation)
            
            station_subsets = lig_datasets.loc[
                (lig_datasets.station_unified == istation) & \
                    (lig_datasets.Type == itypes[iind])
            ]
            
            for idataset in station_subsets.Dataset:
                # idataset = station_subsets.Dataset.iloc[0]
                print('#---- ' + idataset)
                anomaly = np.round(station_subsets.loc[
                    station_subsets.Dataset == idataset
                    ]['127 ka SST anomalies [°C]'].values[0], 1)
                
                try:
                    std = np.round(station_subsets.loc[
                        station_subsets.Dataset == idataset
                        ]['two-sigma errors [°C]'].values[0], 1)
                except:
                    std = np.nan
                
                if(np.isnan(std)):
                    print(anomaly)
                else:
                    print(str(anomaly) + ' ± ' + str(std))





'''

'''
# endregion
# -----------------------------------------------------------------------------


