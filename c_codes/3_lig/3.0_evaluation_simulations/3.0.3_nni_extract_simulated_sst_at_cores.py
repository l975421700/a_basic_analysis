

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
import proplot as pplt
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


with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

models=sorted(lig_sst_alltime.keys())

#-------- import EC reconstruction
ec_sst_rec = {}
# 47 cores
ec_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name='Capron et al. 2017', header=0, skiprows=12, nrows=47,
    usecols=['Station', 'Latitude', 'Longitude', 'Area', 'Type',
             '127 ka Median PIAn [°C]', '127 ka 2s PIAn [°C]'])

# 2 cores
ec_sst_rec['SO_ann'] = ec_sst_rec['original'].loc[
    (ec_sst_rec['original']['Area']=='Southern Ocean') & \
        (ec_sst_rec['original']['Type']=='Annual SST'),]
# 15 cores
ec_sst_rec['SO_djf'] = ec_sst_rec['original'].loc[
    (ec_sst_rec['original']['Area']=='Southern Ocean') & \
        (ec_sst_rec['original']['Type']=='Summer SST'),]


#-------- import JH reconstruction
jh_sst_rec = {}
# 37 cores
jh_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name=' Hoffman et al. 2017', header=0, skiprows=14, nrows=37,)
# 12 cores
jh_sst_rec['SO_ann'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']], ]
# 7 cores
jh_sst_rec['SO_djf'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']], ]


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices

loc_indices_rec_ec = {}
loc_indices_rec_ec['EC'] = {}
loc_indices_rec_ec['JH'] = {}

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------- ' + model)
    
    loc_indices_rec_ec['EC'][model] = {}
    loc_indices_rec_ec['JH'][model] = {}
    
    lon = pi_sst[model].lon.values
    lat = pi_sst[model].lat.values
    
    for istation in ec_sst_rec['original'].index:
        # istation = 10
        station = ec_sst_rec['original'].Station[istation]
        print('#---- ' + str(istation) + ': ' + station)
        
        slon = ec_sst_rec['original'].Longitude[istation]
        slat = ec_sst_rec['original'].Latitude[istation]
        
        loc_indices_rec_ec['EC'][model][station] = \
            find_ilat_ilon_general(slat, slon, lat, lon)
    
    for istation in jh_sst_rec['original'].index:
        # istation = 10
        station = jh_sst_rec['original'].Station[istation]
        print('#---- ' + str(istation) + ': ' + station)
        
        slon = jh_sst_rec['original'].Longitude[istation]
        slat = jh_sst_rec['original'].Latitude[istation]
        
        loc_indices_rec_ec['JH'][model][station] = \
            find_ilat_ilon_general(slat, slon, lat, lon)


with open('scratch/cmip6/lig/loc_indices_rec_ec.pkl', 'wb') as f:
    pickle.dump(loc_indices_rec_ec, f)





'''

from haversine import haversine

        # # check
        # iind0, iind1 = find_ilat_ilon_general(slat, slon, lat, lon)
        # if (lon.ndim == 2):
        #     print(haversine(
        #         [slat, slon], [lat[iind0, iind1], lon[iind0, iind1]],
        #         normalize=True,))
        # elif (lon.ndim == 1):
        #     print(haversine([slat, slon], [lat[iind0], lon[iind0]],
        #                     normalize=True,))


jh_sst_rec['original'].Longitude[0]

'''
# endregion
# -----------------------------------------------------------------------------

