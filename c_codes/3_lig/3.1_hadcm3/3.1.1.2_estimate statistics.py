

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

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
from sklearn.metrics import mean_squared_error

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
    marker_recs,
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

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime = pickle.load(f)

lon = hadcm3_output_regridded_alltime['PI']['SST']['am'].lon.values
lat = hadcm3_output_regridded_alltime['PI']['SST']['am'].lat.values

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values.pkl', 'rb') as f:
    hadcm3_output_site_values = pickle.load(f)

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)
with open('data_sources/LIG/HadISST1.1/HadISST_sic.pkl', 'rb') as f:
    HadISST['sic'] = pickle.load(f)

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

mask_SO = ((lat <= -40) & np.isfinite(hadcm3_output_regridded_alltime['PI']['SST']['am'].squeeze().values))
cellarea_SO = cdo_area1deg.cell_area.values[mask_SO]

with open('scratch/others/land_sea_masks/cdo_1deg_ais_mask.pkl', 'rb') as f:
    cdo_1deg_ais_mask = pickle.load(f)
mask_AIS = cdo_1deg_ais_mask['mask']['AIS']
cellarea_AIS = cdo_area1deg.cell_area.values[mask_AIS]

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region piControl vs. HadISST

# annual sst
data_diff = hadcm3_output_regridded_alltime['PI']['SST']['am'].squeeze().values - HadISST['sst']['1deg_alltime']['am'].values
data_diff_SO = data_diff[mask_SO]
RMSE = np.sqrt(np.ma.average(
    np.ma.MaskedArray(np.square(data_diff_SO), mask=np.isnan(data_diff_SO)),
    weights=cellarea_SO))
print(np.round(RMSE, 1))

# summer sst
data_diff = hadcm3_output_regridded_alltime['PI']['SST']['sm'].sel(time=3).values - HadISST['sst']['1deg_alltime']['sm'].sel(month=3).values
data_diff_SO = data_diff[mask_SO]
RMSE = np.sqrt(np.ma.average(
    np.ma.MaskedArray(np.square(data_diff_SO), mask=np.isnan(data_diff_SO)),
    weights=cellarea_SO))
print(np.round(RMSE, 1))

# september SIA
data1 = hadcm3_output_regridded_alltime['PI']['SIC']['mm'].sel(time=9).values * 100
data2 = HadISST['sic']['1deg_alltime']['mm'].sel(month=9).values
diff = (np.nansum(data1[mask_SO] * cellarea_SO) - np.nansum(data2[mask_SO] * cellarea_SO)) / np.nansum(data2[mask_SO] * cellarea_SO) * 100
print(np.round(diff, 1))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region lig127k vs. piControl


# iperiod = 'LIG_PI'
iperiod = 'LIG0.25_PI'


# annual sst
data_diff = hadcm3_output_regridded_alltime[iperiod]['SST']['am'].squeeze().values
data_diff_SO = data_diff[mask_SO]
mean_diff = np.ma.average(
    np.ma.MaskedArray(data_diff_SO, mask=np.isnan(data_diff_SO)),
    weights=cellarea_SO,)
std_diff = np.nanstd(data_diff_SO)
print(str(np.round(mean_diff, 1)) + '±' + str(np.round(std_diff, 1)))


# summer sst
data_diff = hadcm3_output_regridded_alltime[iperiod]['SST']['sm'].sel(time=3).values
data_diff_SO = data_diff[mask_SO]
mean_diff = np.ma.average(
    np.ma.MaskedArray(data_diff_SO, mask=np.isnan(data_diff_SO)),
    weights=cellarea_SO,)
std_diff = np.nanstd(data_diff_SO)
print(str(np.round(mean_diff, 1)) + '±' + str(np.round(std_diff, 1)))


# annual sat
data_diff = hadcm3_output_regridded_alltime[iperiod]['SAT']['am'].squeeze().values
data_diff_AIS = data_diff[mask_AIS]
mean_diff = np.ma.average(
    np.ma.MaskedArray(data_diff_AIS, mask=np.isnan(data_diff_AIS)),
    weights=cellarea_AIS,)
std_diff = np.nanstd(data_diff_AIS)
print(str(np.round(mean_diff, 1)) + '±' + str(np.round(std_diff, 1)))


# iperiod = 'LIG'
iperiod = 'LIG0.25'

# sep sic
data1 = hadcm3_output_regridded_alltime[iperiod]['SIC']['mm'].sel(time=9).values
data2 = hadcm3_output_regridded_alltime['PI']['SIC']['mm'].sel(time=9).values
data1_SO = data1[mask_SO]
data2_SO = data2[mask_SO]

diff = (np.nansum(data1_SO * cellarea_SO) - np.nansum(data2_SO * cellarea_SO)) / np.nansum(data2_SO * cellarea_SO) * 100
print(np.round(diff, 0))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sim vs. rec

for iproxy in hadcm3_output_site_values.keys():
    print('#-------------------------------- ' + iproxy)
    
    for irec in hadcm3_output_site_values[iproxy].keys():
        print('#---------------- ' + irec)
        
        # diff = hadcm3_output_site_values[iproxy][irec]['sim_rec_lig_pi'].values
        diff = hadcm3_output_site_values[iproxy][irec]['sim_rec_lig0.25Sv_pi'].values
        
        RMSE = np.sqrt(np.nanmean(np.square(diff)))
        
        print(np.round(RMSE, 1))


# endregion
# -----------------------------------------------------------------------------

