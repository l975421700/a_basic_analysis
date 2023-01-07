

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
lig_recs['MC'] = {}
lig_recs['MC']['original'] = pd.read_excel(
    'data_sources/LIG/Chadwick_et_al_2021/AICC2012 ages.xlsx',
    header=0,)

lig_recs['MC']['original'] = lig_recs['MC']['original'].rename(columns={
    'Core name': 'Station',
    'Latitude (degrees South)': 'Latitude',
    'Longitude (degrees East)': 'Longitude',})

lig_recs['MC']['original'].Latitude = -1 * lig_recs['MC']['original'].Latitude

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'rb') as f:
    hadisst_site_values = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region preprocess

lig_recs['MC']['interpolated'] = pd.DataFrame(
    data = {
        'Station': lig_recs['MC']['original'].Station.unique(),
        'Latitude': 0.,
        'Longitude': 0.,
        'AICC age [ka]': 127,
        'rec_sic_sep': 0.,
        'rec_sst_jfm': 0.,
        'hadisst_sic_sep': 0.,
        'hadisst_sst_jfm': 0.,
        'sic_anom_hadisst_sep': 0.,
        'sst_anom_hadisst_jfm': 0.,
    }
)

for istation in lig_recs['MC']['original'].Station.unique():
    # istation = lig_recs['MC']['original'].Station.unique()[0]
    print('#-------- ' + istation)
    
    AICC_ages = lig_recs['MC']['original'].loc[
        lig_recs['MC']['original'].Station == istation,
        'Age on AICC2012 chronology (ka)',]
    
    SIC_Sep = lig_recs['MC']['original'].loc[
        lig_recs['MC']['original'].Station == istation,
        'MAT September sea-ice concentration (%)',]
    
    SST_JFM = lig_recs['MC']['original'].loc[
        lig_recs['MC']['original'].Station == istation,
        'MAT summer SST (degrees celsius)',]
    
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation, 'Latitude'] = \
            lig_recs['MC']['original'].loc[
                lig_recs['MC']['original'].Station == istation,
                'Latitude'].iloc[0]
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation, 'Longitude'] = \
            lig_recs['MC']['original'].loc[
                lig_recs['MC']['original'].Station == istation,
                'Longitude'].iloc[0]
    
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation, 'rec_sic_sep'] = \
            np.interp(127, AICC_ages, SIC_Sep)
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation, 'rec_sst_jfm'] = \
            np.interp(127, AICC_ages, SST_JFM)
    
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation,
        'hadisst_sic_sep'] = \
            hadisst_site_values['sic']['MC']['Sep'].loc[
                hadisst_site_values['sic']['MC']['Sep'].Station == istation,
                'HadISST_Sep',
            ]
    
    lig_recs['MC']['interpolated'].loc[
        lig_recs['MC']['interpolated'].Station == istation,
        'hadisst_sst_jfm'] = \
            hadisst_site_values['sst']['MC']['JFM'].loc[
                hadisst_site_values['sst']['MC']['JFM'].Station == istation,
                'HadISST_JFM'
            ]

lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'] = \
    lig_recs['MC']['interpolated']['rec_sic_sep'] - \
        lig_recs['MC']['interpolated']['hadisst_sic_sep']

lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'] = \
    lig_recs['MC']['interpolated']['rec_sst_jfm'] - \
        lig_recs['MC']['interpolated']['hadisst_sst_jfm']

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'wb') as f:
    pickle.dump(lig_recs['MC'], f)




'''
#-------------------------------- check

lig_recs = {}
with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'rb') as f:
    hadisst_site_values = pickle.load(f)

lig_recs['MC']['interpolated']
hadisst_site_values['sst']['MC']['JFM']

hadisst_site_values['sic']['MC']['Sep']

lig_recs['MC']['interpolated'].columns

lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'] == \
    lig_recs['MC']['interpolated']['rec_sic_sep'] - \
        lig_recs['MC']['interpolated']['hadisst_sic_sep']

lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'] == \
    lig_recs['MC']['interpolated']['rec_sst_jfm'] - \
        lig_recs['MC']['interpolated']['hadisst_sst_jfm']



lig_recs['MC']['original'].loc[
        lig_recs['MC']['original'].Station == istation
    ]
len(lig_recs['MC']['original'].Station.unique())
'''
# endregion
# -----------------------------------------------------------------------------
