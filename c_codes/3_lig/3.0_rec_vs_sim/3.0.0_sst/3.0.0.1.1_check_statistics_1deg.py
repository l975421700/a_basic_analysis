

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

mask = {}
mask['SO'] = lat <= -40

mask_ais = cdo_1deg_ais_mask['mask']['AIS']

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',
    ]

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'rb') as f:
    AIS_ann_tas_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------




# lig vs. pi: mean±std
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region SO annual SST

#-------------------------------- PMIP4 model ensemble
with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

data = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_so]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


#-------------------------------- each model
with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime_ens = pickle.load(f)

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


#-------------------------------- PMIP3

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc')

pmip3_gridarea = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/pmip3_gridarea.nc')

latitude = pmip3_lig_sim['annual_sst'].latitude.values

data = pmip3_lig_sim['annual_sst'].sst.values[latitude < -40]
mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=pmip3_gridarea.cell_area.values[latitude < -40])
std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))




'''
data1 = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_so]

data2 = lig_pi_sst_regrid_alltime_ens['am'].mean(
    dim='ensemble', skipna = True)[0].values[mask_so]
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

# data = lig_pi_sst_regrid_alltime_ens['am'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_so]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO summer SST

#-------------------------------- model ensembles
with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

data = sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0].values[mask_so]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#-------------------------------- each model
with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime_ens = pickle.load(f)

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


#-------------------------------- PMIP3

pmip3_lig_sim = {}
pmip3_lig_sim['summer_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sstdjf_c.nc')


pmip3_gridarea = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/pmip3_gridarea.nc')

latitude = pmip3_lig_sim['summer_sst'].latitude.values

data = pmip3_lig_sim['summer_sst'].sstdjf.values[latitude < -40]
mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=pmip3_gridarea.cell_area.values[latitude < -40])
std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))



'''
with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

# data = lig_pi_sst_regrid_alltime_ens['sm'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_so]


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO Sep SIC

#-------------------------------- model ensembles
with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

data = sic_regrid_alltime_ens_stats['lig_pi']['mm']['mean'][8].values[mask_so]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_so])

std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#-------------------------------- each model
with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime_ens = pickle.load(f)

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

'''
with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

# data = lig_pi_sic_regrid_alltime_ens['mm'].mean(
#     dim='ensemble', skipna = True)[8].values[mask_so]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS annual SAT

#-------------------------------- model ensembles
with open('scratch/cmip6/lig/tas/tas_regrid_alltime_ens_stats.pkl', 'rb') as f:
    tas_regrid_alltime_ens_stats = pickle.load(f)

data = tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0].values[mask_ais]

mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=cdo_area1deg.cell_area.values[mask_ais])

std_values = np.ma.std(np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#-------------------------------- each model
with open('scratch/cmip6/lig/tas/lig_pi_tas_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_tas_regrid_alltime_ens = pickle.load(f)

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


#-------------------------------- PMIP3

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfc_c.nc')

pmip3_gridarea = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/pmip3_gridarea.nc')
with open('scratch/others/land_sea_masks/pmip3_ais_mask.pkl', 'rb') as f:
    pmip3_ais_mask = pickle.load(f)

latitude = pmip3_lig_sim['annual_sat'].latitude.values
mask_ais = pmip3_ais_mask['mask']['AIS']

data = pmip3_lig_sim['annual_sat'].sfc.values[mask_ais]
mean_value = np.ma.average(
    np.ma.MaskedArray(data, mask = np.isnan(data)),
    weights=pmip3_gridarea.cell_area.values[mask_ais])
std_values = np.ma.std(
    np.ma.MaskedArray(data, mask = np.isnan(data)),)

print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


'''
with open('scratch/cmip6/lig/tas/lig_pi_tas_regrid_alltime.pkl', 'rb') as f:
    lig_pi_tas_regrid_alltime = pickle.load(f)

# data = lig_pi_tas_regrid_alltime_ens['am'].mean(
#     dim='ensemble', skipna = True)[0].values[mask_ais]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SO Sep SIA

#-------------------------------- model ensembles
with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

lig_data = sic_regrid_alltime_ens_stats['lig']['mm']['mean'][8].values[mask_so]
pi_data  = sic_regrid_alltime_ens_stats['pi']['mm']['mean'][8].values[mask_so]
sia_lig = np.nansum(lig_data / 100 * cdo_area1deg.cell_area.values[mask_so]) / 1e+12
sia_pi = np.nansum(pi_data / 100 * cdo_area1deg.cell_area.values[mask_so]) / 1e+12
sia_lig_pi = int(np.round((sia_lig - sia_pi) / sia_pi * 100, 0))

print(sia_lig_pi)

#-------------------------------- each model

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

for imodel in models:
    # imodel = 'ACCESS-ESM1-5'
    print('#-------- ' + imodel)
    
    lig_data = lig_sic_regrid_alltime_ens['mm'].sel(ensemble=imodel)[8].values[mask_so]
    pi_data  = pi_sic_regrid_alltime_ens['mm'].sel(ensemble=imodel)[8].values[mask_so]
    sia_lig = np.nansum(lig_data / 100 * cdo_area1deg.cell_area.values[mask_so]) / 1e+12
    sia_pi = np.nansum(pi_data / 100 * cdo_area1deg.cell_area.values[mask_so]) / 1e+12
    sia_lig_pi = int(np.round((sia_lig - sia_pi) / sia_pi * 100, 0))
    
    print(sia_lig_pi)




'''
with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

# data = lig_pi_sic_regrid_alltime_ens['mm'].mean(
#     dim='ensemble', skipna = True)[8].values[mask_so]

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime_ens = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------




# pi vs hadisst: RMSE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region annual sst

with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)

for model in models:
    print(model)
    if True:
        # calculate RMSE
        am_data = pi_sst_regrid_alltime[model]['am'].values[0] - \
            HadISST['sst']['1deg_alltime']['am'].values
        diff = {}
        diff['SO'] = am_data[mask_so]
        
        area = {}
        area['SO'] = cdo_area1deg.cell_area.values[mask_so]
        
        rmse = {}
        rmse['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['SO']), mask=np.isnan(diff['SO'])),
            weights=area['SO']))
        
        print(np.round(rmse['SO'], 1))


with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

am_data = sst_regrid_alltime_ens_stats['pi']['am']['mean'][0].values - \
    HadISST['sst']['1deg_alltime']['am'].values
diff = {}
diff['SO'] = am_data[mask_so]

area = {}
area['SO'] = cdo_area1deg.cell_area.values[mask_so]

rmse = {}
rmse['SO'] = np.sqrt(np.ma.average(
    np.ma.MaskedArray(
        np.square(diff['SO']), mask=np.isnan(diff['SO'])),
    weights=area['SO']))

print(np.round(rmse['SO'], 1))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region summer sst

with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)

for model in models:
    print(model)
    if True:
        # calculate RMSE
        sm_data = pi_sst_regrid_alltime[model]['sm'][0].values - \
            HadISST['sst']['1deg_alltime']['sm'][0].values
        diff = {}
        diff['SO'] = sm_data[mask_so]
        
        area = {}
        area['SO'] = cdo_area1deg.cell_area.values[mask_so]
        
        rmse = {}
        rmse['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['SO']), mask=np.isnan(diff['SO'])),
            weights=area['SO']))
        
        print(np.round(rmse['SO'], 1))


with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

sm_data = sst_regrid_alltime_ens_stats['pi']['sm']['mean'][0].values - \
    HadISST['sst']['1deg_alltime']['sm'][0].values
diff = {}
diff['SO'] = sm_data[mask_so]

area = {}
area['SO'] = cdo_area1deg.cell_area.values[mask_so]

rmse = {}
rmse['SO'] = np.sqrt(np.ma.average(
    np.ma.MaskedArray(
        np.square(diff['SO']), mask=np.isnan(diff['SO'])),
    weights=area['SO']))

print(np.round(rmse['SO'], 1))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region sep sic

with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sic.pkl', 'rb') as f:
    HadISST['sic'] = pickle.load(f)

for model in models:
    print(model)
    if True:
        
        # calculate SIA changes
        sepm_pi = pi_sic_regrid_alltime[model]['mm'][8].values.copy()
        sepm_hadisst = HadISST['sic']['1deg_alltime']['mm'][8].values.copy()
        
        sia_pi = {}
        sia_hadisst = {}
        sia_pi_hadisst = {}
        
        for iregion in ['SO']: # , 'Atlantic', 'Indian', 'Pacific'
            # iregion = 'SO'
            sia_pi[iregion] = np.nansum(sepm_pi[mask[iregion]] / 100 * \
                cdo_area1deg.cell_area.values[mask[iregion]]) / 1e+12
            sia_hadisst[iregion] = np.nansum(
                sepm_hadisst[mask[iregion]] / 100 * \
                cdo_area1deg.cell_area.values[mask[iregion]]) / 1e+12
            sia_pi_hadisst[iregion] = int(np.round((sia_pi[iregion] - sia_hadisst[iregion]) / sia_hadisst[iregion] * 100, 0))
        
        print(sia_pi_hadisst['SO'])



with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

# calculate SIA changes
sepm_pi = sic_regrid_alltime_ens_stats['pi']['mm']['mean'][8].values.copy()
sepm_hadisst = HadISST['sic']['1deg_alltime']['mm'][8].values.copy()

sia_pi = {}
sia_hadisst = {}
sia_pi_hadisst = {}

for iregion in ['SO']: # , 'Atlantic', 'Indian', 'Pacific'
    # iregion = 'SO'
    sia_pi[iregion] = np.nansum(sepm_pi[mask[iregion]] / 100 * \
        cdo_area1deg.cell_area.values[mask[iregion]]) / 1e+12
    sia_hadisst[iregion] = np.nansum(
        sepm_hadisst[mask[iregion]] / 100 * \
        cdo_area1deg.cell_area.values[mask[iregion]]) / 1e+12
    sia_pi_hadisst[iregion] = int(np.round((sia_pi[iregion] - sia_hadisst[iregion]) / sia_hadisst[iregion] * 100, 0))

print(sia_pi_hadisst['SO'])


# endregion
# -----------------------------------------------------------------------------




# lig_pi sim vs. rec
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region annual sst

for model in models:
    print('#-------- ' + model)
    data_to_plot = {}
    data_to_plot['EC'] = SO_ann_sst_site_values['EC'].loc[SO_ann_sst_site_values['EC']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
    data_to_plot['JH'] = SO_ann_sst_site_values['JH'].loc[SO_ann_sst_site_values['JH']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
    data_to_plot['DC'] = SO_ann_sst_site_values['DC'].loc[SO_ann_sst_site_values['DC']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
    
    rms_err  = {}
    for irec in ['EC', 'JH', 'DC']:
        print('#---- ' + irec)
        # irec = 'EC'
        rms_err[irec] = np.round(mean_squared_error(
            data_to_plot[irec]['rec_ann_sst_lig_pi'],
            data_to_plot[irec]['sim_ann_sst_lig_pi'],
            squared=False), 1)
        
        print(rms_err[irec])




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region summer sst

for model in models:
    print('#-------- ' + model)
    
    data_to_plot = {}
    data_to_plot['EC'] = SO_jfm_sst_site_values['EC'].loc[SO_jfm_sst_site_values['EC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
    data_to_plot['JH'] = SO_jfm_sst_site_values['JH'].loc[SO_jfm_sst_site_values['JH']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
    data_to_plot['DC'] = SO_jfm_sst_site_values['DC'].loc[SO_jfm_sst_site_values['DC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
    data_to_plot['MC'] = SO_jfm_sst_site_values['MC'].loc[SO_jfm_sst_site_values['MC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
    
    rms_err  = {}
    for irec in ['EC', 'JH', 'DC', 'MC']:
        print('#---- ' + irec)
        # irec = 'EC'
        rms_err[irec] = np.round(mean_squared_error(
            data_to_plot[irec]['rec_jfm_sst_lig_pi'],
            data_to_plot[irec]['sim_jfm_sst_lig_pi'],
            squared=False), 1)
        
        print(rms_err[irec])





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region annual sat

for model in models:
    print('#-------- ' + model)
    data_to_plot = {}
    
    data_to_plot['EC_tas'] = AIS_ann_tas_site_values['EC'].loc[AIS_ann_tas_site_values['EC']['Model'] == model][['rec_ann_tas_lig_pi', 'sim_ann_tas_lig_pi']]
    
    rms_err['EC_tas'] = np.round(
        mean_squared_error(
            data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
            data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
            squared=False), 1)
    
    print(rms_err['EC_tas'])



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region september sic

for model in models:
    print('#-------- ' + model)
    data_to_plot = {}
    
    data_to_plot['MC'] = SO_sep_sic_site_values['MC'].loc[SO_sep_sic_site_values['MC']['Model'] == model][['rec_sep_sic_lig_pi', 'sim_sep_sic_lig_pi']]
    
    rms_err  = {}
    
    rms_err['MC'] = np.int(mean_squared_error(
        data_to_plot['MC']['rec_sep_sic_lig_pi'],
        data_to_plot['MC']['sim_sep_sic_lig_pi'],
        squared=False))
    
    print(rms_err['MC'])



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region no responses

model = 'NorESM2-LM'

rec_data = SO_ann_sst_site_values['EC'].loc[
    SO_ann_sst_site_values['EC']['Model'] == model][
        ['rec_ann_sst_lig_pi']]
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)


rec_data = SO_ann_sst_site_values['JH'].loc[
    SO_ann_sst_site_values['JH']['Model'] == model][
        ['rec_ann_sst_lig_pi']]
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)

rec_data = SO_ann_sst_site_values['DC'].loc[
    SO_ann_sst_site_values['DC']['Model'] == model][
        ['rec_ann_sst_lig_pi']]
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)


rec_data = SO_jfm_sst_site_values['EC'].loc[SO_jfm_sst_site_values['EC']['Model'] == model]['rec_jfm_sst_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)

rec_data = SO_jfm_sst_site_values['JH'].loc[SO_jfm_sst_site_values['JH']['Model'] == model]['rec_jfm_sst_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)

rec_data = SO_jfm_sst_site_values['DC'].loc[SO_jfm_sst_site_values['DC']['Model'] == model]['rec_jfm_sst_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)

rec_data = SO_jfm_sst_site_values['MC'].loc[SO_jfm_sst_site_values['MC']['Model'] == model]['rec_jfm_sst_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)


rec_data = AIS_ann_tas_site_values['EC'].loc[AIS_ann_tas_site_values['EC']['Model'] == model]['rec_ann_tas_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 1)


rec_data = SO_sep_sic_site_values['MC'].loc[SO_sep_sic_site_values['MC']['Model'] == model]['rec_sep_sic_lig_pi']
np.round(mean_squared_error(rec_data,np.zeros(len(rec_data)), squared=False), 0)


# endregion
# -----------------------------------------------------------------------------
