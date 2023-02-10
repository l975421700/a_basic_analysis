

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
# region import data

with open('scratch/others/land_sea_masks/cdo_1deg_ais_mask.pkl', 'rb') as f:
    cdo_1deg_ais_mask = pickle.load(f)

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

lat = cdo_area1deg.lat.values
mask_so = (lat < -40)

mask_ais = cdo_1deg_ais_mask['mask']['AIS']

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',
    ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO annual SST

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

#-------------------------------- model ensembles

data = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_so]

# data = lig_pi_sst_regrid_alltime_ens['am'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_so]


mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


#-------------------------------- each model
for imodel in models:
    # imodel = 'ACCESS-ESM1-5'
    print('#-------- ' + imodel)
    
    data = lig_pi_sst_regrid_alltime_ens['am'].sel(ensemble=imodel)[0].values[mask_so]
    
    mean_value = np.ma.average(
        np.ma.MaskedArray(data, mask = np.isnan(data)),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    std_values = np.ma.std(
        np.ma.MaskedArray(data, mask = np.isnan(data)),)
    
    print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))





'''
data1 = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_so]

data2 = lig_pi_sst_regrid_alltime_ens['am'].mean(
    dim='ensemble', skipna = True)[0].values[mask_so]
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO summer SST

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)


#-------------------------------- model ensembles

data = sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0].values[mask_so]

# data = lig_pi_sst_regrid_alltime_ens['sm'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_so]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#-------------------------------- each model
for imodel in models:
    # imodel = 'ACCESS-ESM1-5'
    print('#-------- ' + imodel)
    
    data = lig_pi_sst_regrid_alltime_ens['sm'].sel(ensemble=imodel)[0].values[mask_so]
    
    mean_value = np.ma.average(
        np.ma.MaskedArray(data, mask = np.isnan(data)),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    std_values = np.ma.std(
        np.ma.MaskedArray(data, mask = np.isnan(data)),)
    
    print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO Sep SIC

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

#-------------------------------- model ensembles

data = sic_regrid_alltime_ens_stats['lig_pi']['mm']['mean'][8].values[mask_so]

# data = lig_pi_sic_regrid_alltime_ens['mm'].mean(
#     dim='ensemble', skipna = True)[8].values[mask_so]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


#-------------------------------- each model
for imodel in models:
    # imodel = 'ACCESS-ESM1-5'
    print('#-------- ' + imodel)
    
    data = lig_pi_sic_regrid_alltime_ens['mm'].sel(ensemble=imodel)[8].values[mask_so]
    
    mean_value = np.ma.average(
        np.ma.MaskedArray(data, mask = np.isnan(data)),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    std_values = np.ma.std(
        np.ma.MaskedArray(data, mask = np.isnan(data)),)
    
    print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS annual SAT

with open('scratch/cmip6/lig/tas/lig_pi_tas_regrid_alltime.pkl', 'rb') as f:
    lig_pi_tas_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/tas/lig_pi_tas_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_tas_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/tas/tas_regrid_alltime_ens_stats.pkl', 'rb') as f:
    tas_regrid_alltime_ens_stats = pickle.load(f)

#-------------------------------- model ensembles

data = tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_ais]

# data = lig_pi_tas_regrid_alltime_ens['am'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_ais]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_ais])

std_values = np.ma.std(np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#-------------------------------- each model
for imodel in models:
    # imodel = 'ACCESS-ESM1-5'
    print('#-------- ' + imodel)
    
    data = lig_pi_tas_regrid_alltime_ens['am'].sel(ensemble=imodel)[0].values[mask_ais]
    
    mean_value = np.ma.average(
        np.ma.MaskedArray(data, mask = np.isnan(data)),
        weights=cdo_area1deg.cell_area.values[mask_ais])
    
    std_values = np.ma.std(
        np.ma.MaskedArray(data, mask = np.isnan(data)),)
    
    print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))



# endregion
# -----------------------------------------------------------------------------

