

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

# import data
lig_recs = {}
lig_recs['DC'] = {}

lig_recs['DC']['original'] = pd.read_csv(
    'data_sources/LIG/Chandler-Langebroek_2021_SST-anom.tab',
    sep='\t', header=0, skiprows=76,)
lig_recs['DC']['original'] = lig_recs['DC']['original'].rename(columns={'Site': 'Station'})

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'rb') as f:
    hadisst_site_values = pickle.load(f)

'''
print(hadisst_site_values['sst']['DC']['annual'])
print(hadisst_site_values['sst']['DC']['JFM'])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region preprocess

# split annual and summer
lig_recs['DC']['annual'] = lig_recs['DC']['original'][
    lig_recs['DC']['original']['Season (A annual, S summer (months JFM))'] == 'A']
lig_recs['DC']['JFM'] = lig_recs['DC']['original'][
    lig_recs['DC']['original']['Season (A annual, S summer (months JFM))'] == 'S']


# combine annual SST
lig_recs['DC']['annual'] = pd.merge(
    lig_recs['DC']['annual'],
    hadisst_site_values['sst']['DC']['annual'])

lig_recs['DC']['annual']['sst_anom_hadisst_ann'] = \
    lig_recs['DC']['annual']['SST [°C] (Reconstructed)'] - \
        lig_recs['DC']['annual']['HadISST_annual']

lig_recs['DC']['annual']['sst_anom_hadisst_ann'][
    np.isnan(lig_recs['DC']['annual']['SST anomaly [°C] (Anomaly relative to the World...)'])] = np.nan


lig_recs['DC']['JFM'] = pd.merge(
    lig_recs['DC']['JFM'],
    hadisst_site_values['sst']['DC']['JFM'],)

lig_recs['DC']['JFM']['sst_anom_hadisst_jfm'] = \
    lig_recs['DC']['JFM']['SST [°C] (Reconstructed)'] - \
        lig_recs['DC']['JFM']['HadISST_JFM']

lig_recs['DC']['JFM']['sst_anom_hadisst_jfm'][
    np.isnan(lig_recs['DC']['JFM']['SST anomaly [°C] (Anomaly relative to the World...)'])] = np.nan

annual_slices = ['annual_130', 'annual_128', 'annual_126', 'annual_124',]
jfm_slices = ['JFM_130', 'JFM_128', 'JFM_126', 'JFM_124',]
time_slices = [130, 128, 126, 124,]

for islice in range(len(annual_slices)):
    print('#-------- ' + str(islice))
    print(annual_slices[islice])
    print(jfm_slices[islice])
    print(time_slices[islice])
    
    # annual
    lig_recs['DC'][annual_slices[islice]] = lig_recs['DC']['annual'][
        lig_recs['DC']['annual'][
            'Age [ka BP] (Chronology follows Lisiecki a...)'] \
                == time_slices[islice]]
    
    lig_recs['DC'][annual_slices[islice]] = \
        lig_recs['DC'][annual_slices[islice]].dropna(
            subset='sst_anom_hadisst_ann')
    
    lig_recs['DC'][annual_slices[islice]] = \
        lig_recs['DC'][annual_slices[islice]].groupby(
            'Station').first().reset_index()
    
    # jfm
    lig_recs['DC'][jfm_slices[islice]] = lig_recs['DC']['JFM'][
        lig_recs['DC']['JFM'][
            'Age [ka BP] (Chronology follows Lisiecki a...)'] \
                == time_slices[islice]]
    
    lig_recs['DC'][jfm_slices[islice]] = \
        lig_recs['DC'][jfm_slices[islice]].dropna(
            subset='sst_anom_hadisst_jfm')
    
    lig_recs['DC'][jfm_slices[islice]] = \
        lig_recs['DC'][jfm_slices[islice]].groupby(
            'Station').first().reset_index()

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'wb') as f:
    pickle.dump(lig_recs['DC'], f)




'''
#-------------------------------- check

lig_recs = {}
with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'rb') as f:
    hadisst_site_values = pickle.load(f)


lig_recs['DC']['annual_128']




# previous manual calculation
lig_recs['DC']['annual_130'] = lig_recs['DC']['annual'][
    lig_recs['DC']['annual']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 130]
lig_recs['DC']['annual_128'] = lig_recs['DC']['annual'][
    lig_recs['DC']['annual']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 128]
lig_recs['DC']['annual_126'] = lig_recs['DC']['annual'][
    lig_recs['DC']['annual']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 126]
lig_recs['DC']['annual_124'] = lig_recs['DC']['annual'][
    lig_recs['DC']['annual']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 124]

lig_recs['DC']['JFM_130'] = lig_recs['DC']['JFM'][
    lig_recs['DC']['JFM']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 130]
lig_recs['DC']['JFM_128'] = lig_recs['DC']['JFM'][
    lig_recs['DC']['JFM']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 128]
lig_recs['DC']['JFM_126'] = lig_recs['DC']['JFM'][
    lig_recs['DC']['JFM']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 126]
lig_recs['DC']['JFM_124'] = lig_recs['DC']['JFM'][
    lig_recs['DC']['JFM']['Age [ka BP] (Chronology follows Lisiecki a...)']\
        == 124]

lig_recs['DC']['annual_130'] = \
    lig_recs['DC']['annual_130'].groupby('Station').first().reset_index()
lig_recs['DC']['annual_128'] = \
    lig_recs['DC']['annual_128'].groupby('Station').first().reset_index()
lig_recs['DC']['annual_126'] = \
    lig_recs['DC']['annual_126'].groupby('Station').first().reset_index()
lig_recs['DC']['annual_124'] = \
    lig_recs['DC']['annual_124'].groupby('Station').first().reset_index()

lig_recs['DC']['JFM_130'] = \
    lig_recs['DC']['JFM_130'].groupby('Station').first().reset_index()
lig_recs['DC']['JFM_128'] = \
    lig_recs['DC']['JFM_128'].groupby('Station').first().reset_index()
lig_recs['DC']['JFM_126'] = \
    lig_recs['DC']['JFM_126'].groupby('Station').first().reset_index()
lig_recs['DC']['JFM_124'] = \
    lig_recs['DC']['JFM_124'].groupby('Station').first().reset_index()


'''
# endregion
# -----------------------------------------------------------------------------



