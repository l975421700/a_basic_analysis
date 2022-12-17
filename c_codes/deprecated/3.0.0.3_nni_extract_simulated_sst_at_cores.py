

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
# -----------------------------------------------------------------------------
# region import EC&JH reconstructions: am/sum sst


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
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']],]
jh_sst_rec['SO_djf'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']],]


'''
# 7 cores
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
#---------------- check
from haversine import haversine
with open('scratch/cmip6/lig/loc_indices_rec_ec.pkl', 'rb') as f:
    loc_indices_rec_ec = pickle.load(f)

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------------------------------- ' + model)
    
    lon = pi_sst[model].lon.values
    lat = pi_sst[model].lat.values
    
    for istation in ec_sst_rec['original'].index:
        # istation = 10
        station = ec_sst_rec['original'].Station[istation]
        print('#---------------- ' + str(istation) + ': ' + station)
        
        slon = ec_sst_rec['original'].Longitude[istation]
        slat = ec_sst_rec['original'].Latitude[istation]
        
        if (lon.ndim == 2):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_ec['EC'][model][station][0],
                     loc_indices_rec_ec['EC'][model][station][1]],
                 lon[loc_indices_rec_ec['EC'][model][station][0],
                     loc_indices_rec_ec['EC'][model][station][1]]],
                normalize=True,)
        elif (lon.ndim == 1):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_ec['EC'][model][station][0]],
                 lon[loc_indices_rec_ec['EC'][model][station][0]]],
                normalize=True,)
        
        if (distance > 100):
            print(np.round(distance, 0))
    
    for istation in jh_sst_rec['original'].index:
        # istation = 10
        station = jh_sst_rec['original'].Station[istation]
        print('#---------------- ' + str(istation) + ': ' + station)
        
        slon = jh_sst_rec['original'].Longitude[istation]
        slat = jh_sst_rec['original'].Latitude[istation]
        
        if (lon.ndim == 2):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_ec['JH'][model][station][0],
                     loc_indices_rec_ec['JH'][model][station][1]],
                 lon[loc_indices_rec_ec['JH'][model][station][0],
                     loc_indices_rec_ec['JH'][model][station][1]]],
                normalize=True,)
        elif (lon.ndim == 1):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_ec['JH'][model][station][0]],
                 lon[loc_indices_rec_ec['JH'][model][station][0]]],
                normalize=True,)
        
        if (distance > 100):
            print(np.round(distance, 0))


from haversine import haversine

        # check
        iind0, iind1 = find_ilat_ilon_general(slat, slon, lat, lon)
        if (lon.ndim == 2):
            print(haversine(
                [slat, slon], [lat[iind0, iind1], lon[iind0, iind1]],
                normalize=True,))
        elif (lon.ndim == 1):
            print(haversine([slat, slon], [lat[iind0], lon[iind0]],
                            normalize=True,))


jh_sst_rec['original'].Longitude[0]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract LIG-PI am and summer SST : EC&JH

with open('scratch/cmip6/lig/loc_indices_rec_ec.pkl', 'rb') as f:
    loc_indices_rec_ec = pickle.load(f)

obs_sim_lig_pi_so_sst = pd.DataFrame(columns=(
    'datasets', 'types', 'stations', 'models',
    'slat', 'slon', 'glat', 'glon',
    'obs_lig_pi', 'obs_lig_pi_2s', 'sim_lig_pi', 'sim_lig_pi_2s',
    'sim_obs_lig_pi', 'sim_obs_lig_pi_2s', ))


#-------------------------------- EC
dataset = 'EC'

#---------------- Annual SST
type = 'Annual SST'

for istation in ec_sst_rec['SO_ann'].index:
    # istation = ec_sst_rec['SO_ann'].index[0]
    station = ec_sst_rec['SO_ann'].Station[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = ec_sst_rec['SO_ann'].Latitude[istation]
    slon = ec_sst_rec['SO_ann'].Longitude[istation]
    obs_lig_pi = ec_sst_rec['SO_ann']['127 ka Median PIAn [°C]'][istation]
    obs_lig_pi_2s = ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]'][istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        print(model)
        lon = pi_sst[model].lon.values
        lat = pi_sst[model].lat.values
        
        if (lon.shape == pi_sst_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_ec[dataset][model][station][0]
            iind1 = loc_indices_rec_ec[dataset][model][station][1]
        else:
            print('shape differs')
            iind0 = loc_indices_rec_ec[dataset][model][station][1]
            iind1 = loc_indices_rec_ec[dataset][model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['am'][iind0, iind1].values - \
                    pi_sst_alltime[model]['am'][iind0, iind1].values
                sigma1 = lig_sst_alltime[model]['ann'][
                    :, iind0, iind1].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['am'], ds_out = pi_sst[model],
                    )[iind0, iind1+1].values - \
                        pi_sst_alltime[model]['am'][iind0, iind1].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['ann'], ds_out = pi_sst[model])[
                    :, iind0, iind1+1].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['ann'][
                :, iind0, iind1].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['am'][iind0].values - \
                    pi_sst_alltime[model]['am'][iind0].values
                sigma1 = lig_sst_alltime[model]['ann'][
                    :, iind0].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['am'], ds_out = pi_sst[model],
                    )[iind0].values - \
                        pi_sst_alltime[model]['am'][iind0].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['ann'], ds_out = pi_sst[model])[
                    :, iind0].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['ann'][
                :, iind0].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        sim_obs_lig_pi_2s=((obs_lig_pi_2s/2)**2 + (sim_lig_pi_2s/2)**2)**0.5 * 2
        
        obs_sim_lig_pi_so_sst = pd.concat([
            obs_sim_lig_pi_so_sst,
            pd.DataFrame(data={
                'datasets': dataset,
                'types': type,
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'obs_lig_pi_2s': obs_lig_pi_2s,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                'sim_obs_lig_pi_2s': sim_obs_lig_pi_2s,
                }, index=[0])], ignore_index=True,)

#---------------- Summer SST
type = 'Summer SST'

for istation in ec_sst_rec['SO_djf'].index:
    # istation = ec_sst_rec['SO_djf'].index[0]
    station = ec_sst_rec['SO_djf'].Station[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = ec_sst_rec['SO_djf'].Latitude[istation]
    slon = ec_sst_rec['SO_djf'].Longitude[istation]
    obs_lig_pi = ec_sst_rec['SO_djf']['127 ka Median PIAn [°C]'][istation]
    obs_lig_pi_2s = ec_sst_rec['SO_djf']['127 ka 2s PIAn [°C]'][istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        # model = 'GISS-E2-1-G'
        print(model)
        lon = pi_sst[model].lon.values
        lat = pi_sst[model].lat.values
        
        if (lon.shape == pi_sst_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_ec[dataset][model][station][0]
            iind1 = loc_indices_rec_ec[dataset][model][station][1]
        else:
            print('shape differs')
            iind0 = loc_indices_rec_ec[dataset][model][station][1]
            iind1 = loc_indices_rec_ec[dataset][model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0, iind1].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0, iind1+1].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0, iind1].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0, iind1+1].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0, iind1].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        sim_obs_lig_pi_2s=((obs_lig_pi_2s/2)**2 + (sim_lig_pi_2s/2)**2)**0.5 * 2
        
        obs_sim_lig_pi_so_sst = pd.concat([
            obs_sim_lig_pi_so_sst,
            pd.DataFrame(data={
                'datasets': dataset,
                'types': type,
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'obs_lig_pi_2s': obs_lig_pi_2s,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                'sim_obs_lig_pi_2s': sim_obs_lig_pi_2s,
                }, index=[0])], ignore_index=True,)


#-------------------------------- JH
dataset = 'JH'

#---------------- Annual SST
type = 'Annual SST'

for istation in jh_sst_rec['SO_ann'].index:
    # istation = jh_sst_rec['SO_ann'].index[0]
    station = jh_sst_rec['SO_ann'].Station[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = jh_sst_rec['SO_ann'].Latitude[istation]
    slon = jh_sst_rec['SO_ann'].Longitude[istation]
    obs_lig_pi = jh_sst_rec['SO_ann']['127 ka SST anomaly (°C)'][istation]
    obs_lig_pi_2s = jh_sst_rec['SO_ann']['127 ka 2σ (°C)'][istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        print(model)
        lon = pi_sst[model].lon.values
        lat = pi_sst[model].lat.values
        
        if (lon.shape == pi_sst_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_ec[dataset][model][station][0]
            iind1 = loc_indices_rec_ec[dataset][model][station][1]
        else:
            print('shape differs')
            iind0 = loc_indices_rec_ec[dataset][model][station][1]
            iind1 = loc_indices_rec_ec[dataset][model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['am'][iind0, iind1].values - \
                    pi_sst_alltime[model]['am'][iind0, iind1].values
                sigma1 = lig_sst_alltime[model]['ann'][
                    :, iind0, iind1].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['am'], ds_out = pi_sst[model],
                    )[iind0, iind1+1].values - \
                        pi_sst_alltime[model]['am'][iind0, iind1].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['ann'], ds_out = pi_sst[model])[
                    :, iind0, iind1+1].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['ann'][
                :, iind0, iind1].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['am'][iind0].values - \
                    pi_sst_alltime[model]['am'][iind0].values
                sigma1 = lig_sst_alltime[model]['ann'][
                    :, iind0].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['am'], ds_out = pi_sst[model],
                    )[iind0].values - \
                        pi_sst_alltime[model]['am'][iind0].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['ann'], ds_out = pi_sst[model])[
                    :, iind0].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['ann'][
                :, iind0].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        sim_obs_lig_pi_2s=((obs_lig_pi_2s/2)**2 + (sim_lig_pi_2s/2)**2)**0.5 * 2
        
        obs_sim_lig_pi_so_sst = pd.concat([
            obs_sim_lig_pi_so_sst,
            pd.DataFrame(data={
                'datasets': dataset,
                'types': type,
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'obs_lig_pi_2s': obs_lig_pi_2s,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                'sim_obs_lig_pi_2s': sim_obs_lig_pi_2s,
                }, index=[0])], ignore_index=True,)

#---------------- Summer SST
type = 'Summer SST'

for istation in jh_sst_rec['SO_djf'].index:
    # istation = jh_sst_rec['SO_djf'].index[0]
    station = jh_sst_rec['SO_djf'].Station[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = jh_sst_rec['SO_djf'].Latitude[istation]
    slon = jh_sst_rec['SO_djf'].Longitude[istation]
    obs_lig_pi = jh_sst_rec['SO_djf']['127 ka SST anomaly (°C)'][istation]
    obs_lig_pi_2s = jh_sst_rec['SO_djf']['127 ka 2σ (°C)'][istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        print(model)
        lon = pi_sst[model].lon.values
        lat = pi_sst[model].lat.values
        
        if (lon.shape == pi_sst_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_ec[dataset][model][station][0]
            iind1 = loc_indices_rec_ec[dataset][model][station][1]
        else:
            print('shape differs')
            iind0 = loc_indices_rec_ec[dataset][model][station][1]
            iind1 = loc_indices_rec_ec[dataset][model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0],
                       loc_indices_rec_ec[dataset][model][station][1]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0, iind1].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0, iind1+1].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0, iind1].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0, iind1+1].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0, iind1].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_ec[dataset][model][station][0]]
            glon = lon[loc_indices_rec_ec[dataset][model][station][0]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        sim_obs_lig_pi_2s=((obs_lig_pi_2s/2)**2 + (sim_lig_pi_2s/2)**2)**0.5 * 2
        
        obs_sim_lig_pi_so_sst = pd.concat([
            obs_sim_lig_pi_so_sst,
            pd.DataFrame(data={
                'datasets': dataset,
                'types': type,
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'obs_lig_pi_2s': obs_lig_pi_2s,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                'sim_obs_lig_pi_2s': sim_obs_lig_pi_2s,
                }, index=[0])], ignore_index=True,)


with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst.pkl', 'wb') as f:
    pickle.dump(obs_sim_lig_pi_so_sst, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst.pkl', 'rb') as f:
    obs_sim_lig_pi_so_sst = pickle.load(f)

site_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_so_sst.slat, obs_sim_lig_pi_so_sst.slon)]
grid_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_so_sst.glat, obs_sim_lig_pi_so_sst.glon)]

from haversine import haversine_vector
distances = haversine_vector(
    site_pairs, grid_pairs, normalize=True,
    )
np.max(distances)




            # sim_lig_pi = lig_pi_am[iind0, iind1]
            # sim_lig_pi_2s = lig_pi_ann[:, iind0, iind1].std(ddof=1) * 2
            # sim_lig_pi = lig_pi_am[iind0]
            # sim_lig_pi_2s = lig_pi_ann[:, iind0].std(ddof=1) * 2

# Propogation of uncertainties
https://en.wikipedia.org/wiki/Propagation_of_uncertainty

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import EC reconstruction of SAT

#-------- import simulations
with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'rb') as f:
    pi_tas_alltime = pickle.load(f)

models=sorted(lig_tas_alltime.keys())

#-------- import EC reconstruction
ec_sst_rec = {}
ec_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name='Capron et al. 2017', header=0, skiprows=12, nrows=47,
    usecols=['Station', 'Latitude', 'Longitude', 'Area', 'Type',
             '127 ka Median PIAn [°C]', '127 ka 2s PIAn [°C]'])

ec_sst_rec['AIS_am'] = ec_sst_rec['original'].loc[
    ec_sst_rec['original']['Area']=='Antarctica',]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices

loc_indices_rec_ec_atmos = {}

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------- ' + model)
    
    loc_indices_rec_ec_atmos[model] = {}
    
    lon = lig_tas_alltime[model]['am'].lon.values
    lat = lig_tas_alltime[model]['am'].lat.values
    
    for istation in ec_sst_rec['AIS_am'].index:
        # istation = ec_sst_rec['AIS_am'].index[0]
        station = ec_sst_rec['AIS_am'].Station[istation]
        print('#---- ' + str(istation) + ': ' + station)
        
        slon = ec_sst_rec['AIS_am'].Longitude[istation]
        slat = ec_sst_rec['AIS_am'].Latitude[istation]
        
        loc_indices_rec_ec_atmos[model][station] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/loc_indices_rec_ec_atmos.pkl', 'wb') as f:
    pickle.dump(loc_indices_rec_ec_atmos, f)


'''
#---------------- check
from haversine import haversine
with open('scratch/cmip6/lig/loc_indices_rec_ec_atmos.pkl', 'rb') as f:
    loc_indices_rec_ec_atmos = pickle.load(f)

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------------------------------- ' + model)
    
    lon = lig_tas_alltime[model]['am'].lon.values
    lat = lig_tas_alltime[model]['am'].lat.values
    
    for istation in ec_sst_rec['AIS_am'].index:
        # istation = 10
        station = ec_sst_rec['AIS_am'].Station[istation]
        print('#---------------- ' + str(istation) + ': ' + station)
        
        slon = ec_sst_rec['AIS_am'].Longitude[istation]
        slat = ec_sst_rec['AIS_am'].Latitude[istation]
        
        distance = haversine(
            [slat, slon],
            [lat[loc_indices_rec_ec_atmos[model][station][0],
                 loc_indices_rec_ec_atmos[model][station][1]],
            lon[loc_indices_rec_ec_atmos[model][station][0],
                 loc_indices_rec_ec_atmos[model][station][1]]],
            normalize=True,)
        
        if (distance > 100):
            print(np.round(distance, 0))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract LIG-PI am SAT : EC

with open('scratch/cmip6/lig/loc_indices_rec_ec_atmos.pkl', 'rb') as f:
    loc_indices_rec_ec_atmos = pickle.load(f)

obs_sim_lig_pi_ais_tas = pd.DataFrame(columns=(
    'stations', 'models',
    'slat', 'slon', 'glat', 'glon',
    'obs_lig_pi', 'obs_lig_pi_2s', 'sim_lig_pi', 'sim_lig_pi_2s',
    'sim_obs_lig_pi', 'sim_obs_lig_pi_2s', ))

for istation in ec_sst_rec['AIS_am'].index:
    # istation = ec_sst_rec['AIS_am'].index[0]
    station = ec_sst_rec['AIS_am'].Station[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = ec_sst_rec['AIS_am'].Latitude[istation]
    slon = ec_sst_rec['AIS_am'].Longitude[istation]
    obs_lig_pi = ec_sst_rec['AIS_am']['127 ka Median PIAn [°C]'][istation]
    obs_lig_pi_2s = ec_sst_rec['AIS_am']['127 ka 2s PIAn [°C]'][istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        print(model)
        lon = lig_tas_alltime[model]['am'].lon.values
        lat = lig_tas_alltime[model]['am'].lat.values
        
        if (lon.shape == lig_tas_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_ec_atmos[model][station][0]
            iind1 = loc_indices_rec_ec_atmos[model][station][1]
        else:
            print('shape differs')
            iind0 = loc_indices_rec_ec_atmos[model][station][1]
            iind1 = loc_indices_rec_ec_atmos[model][station][0]
        
        glat = lat[loc_indices_rec_ec_atmos[model][station][0],
                   loc_indices_rec_ec_atmos[model][station][1]]
        glon = lon[loc_indices_rec_ec_atmos[model][station][0],
                   loc_indices_rec_ec_atmos[model][station][1]]
        sim_lig_pi = \
            lig_tas_alltime[model]['am'][iind0, iind1].values - \
                pi_tas_alltime[model]['am'][iind0, iind1].values
        sigma1 = lig_tas_alltime[model]['ann'][
            :, iind0, iind1].std(ddof=1).values
        sigma2 = pi_tas_alltime[model]['ann'][
            :, iind0, iind1].std(ddof=1).values
        sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        sim_obs_lig_pi_2s=((obs_lig_pi_2s/2)**2 + (sim_lig_pi_2s/2)**2)**0.5 * 2
        
        obs_sim_lig_pi_ais_tas = pd.concat([
            obs_sim_lig_pi_ais_tas,
            pd.DataFrame(data={
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'obs_lig_pi_2s': obs_lig_pi_2s,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                'sim_obs_lig_pi_2s': sim_obs_lig_pi_2s,
                }, index=[0])], ignore_index=True,)


with open('scratch/cmip6/lig/obs_sim_lig_pi_ais_tas.pkl', 'wb') as f:
    pickle.dump(obs_sim_lig_pi_ais_tas, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/obs_sim_lig_pi_ais_tas.pkl', 'rb') as f:
    obs_sim_lig_pi_ais_tas = pickle.load(f)

site_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_ais_tas.slat, obs_sim_lig_pi_ais_tas.slon)]
grid_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_ais_tas.glat, obs_sim_lig_pi_ais_tas.glon)]

from haversine import haversine_vector
distances = haversine_vector(
    site_pairs, grid_pairs, normalize=True,
    )
np.max(distances)



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import MC reconstructions: SST


with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

models=sorted(pi_sst.keys())

#-------- import MC reconstruction
with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices

loc_indices_rec_mc = {}

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------- ' + model)
    
    loc_indices_rec_mc[model] = {}
    
    lon = pi_sst[model].lon.values
    lat = pi_sst[model].lat.values
    
    for istation in chadwick_interp.index:
        # istation = 2
        station = chadwick_interp.sites[istation]
        print('#---- ' + str(istation) + ': ' + station)
        
        slon = chadwick_interp.lon[istation]
        slat = chadwick_interp.lat[istation]
        
        loc_indices_rec_mc[model][station] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/loc_indices_rec_mc.pkl', 'wb') as f:
    pickle.dump(loc_indices_rec_mc, f)



'''
#---------------- check
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)
models=sorted(pi_sst.keys())
with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

from haversine import haversine
with open('scratch/cmip6/lig/loc_indices_rec_mc.pkl', 'rb') as f:
    loc_indices_rec_mc = pickle.load(f)

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------------------------------- ' + model)
    
    lon = pi_sst[model].lon.values
    lat = pi_sst[model].lat.values
    
    for istation in chadwick_interp.index:
        # istation = 10
        station = chadwick_interp.sites[istation]
        print('#---------------- ' + str(istation) + ': ' + station)
        
        slon = chadwick_interp.lon[istation]
        slat = chadwick_interp.lat[istation]
        
        if (lon.ndim == 2):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_mc[model][station][0],
                     loc_indices_rec_mc[model][station][1]],
                 lon[loc_indices_rec_mc[model][station][0],
                     loc_indices_rec_mc[model][station][1]]],
                normalize=True,)
        elif (lon.ndim == 1):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_mc[model][station][0]],
                 lon[loc_indices_rec_mc[model][station][0]]],
                normalize=True,)
        
        if (distance > 100):
            print(np.round(distance, 0))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract LIG-PI summer SST: MC

with open('scratch/cmip6/lig/loc_indices_rec_mc.pkl', 'rb') as f:
    loc_indices_rec_mc = pickle.load(f)

obs_sim_lig_pi_so_sst_mc = pd.DataFrame(columns=(
    'stations', 'models',
    'slat', 'slon', 'glat', 'glon',
    'obs_lig_pi', 'sim_lig_pi', 'sim_lig_pi_2s',
    'sim_obs_lig_pi', ))


#---------------- Summer SST

for istation in chadwick_interp.index:
    # istation = chadwick_interp.index[0]
    station = chadwick_interp.sites[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = chadwick_interp.lat[istation]
    slon = chadwick_interp.lon[istation]
    obs_lig_pi = chadwick_interp.sst_sum[istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        # model = 'GISS-E2-1-G'
        print(model)
        lon = pi_sst[model].lon.values
        lat = pi_sst[model].lat.values
        
        if (lon.shape == pi_sst_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_mc[model][station][0]
            iind1 = loc_indices_rec_mc[model][station][1]
        else:
            iind0 = loc_indices_rec_mc[model][station][1]
            iind1 = loc_indices_rec_mc[model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_mc[model][station][0],
                       loc_indices_rec_mc[model][station][1]]
            glon = lon[loc_indices_rec_mc[model][station][0],
                       loc_indices_rec_mc[model][station][1]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0, iind1].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0, iind1].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0, iind1+1].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0, iind1].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0, iind1+1].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0, iind1].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_mc[model][station][0]]
            glon = lon[loc_indices_rec_mc[model][station][0]]
            if (model != 'HadGEM3-GC31-LL'):
                sim_lig_pi = \
                    lig_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values - \
                    pi_sst_alltime[model]['sm'].sel(
                        season='DJF')[iind0].values
                sigma1 = lig_sst_alltime[model]['sea'][
                    3::4, iind0].std(ddof=1).values
            elif (model == 'HadGEM3-GC31-LL'):
                sim_lig_pi = regrid(
                    lig_sst_alltime[model]['sm'].sel(season='DJF'),
                    ds_out = pi_sst[model],
                    )[iind0].values - \
                        pi_sst_alltime[model]['sm'].sel(
                            season='DJF')[iind0].values
                sigma1 = regrid(
                    lig_sst_alltime[model]['sea'],
                    ds_out = pi_sst[model])[
                        3::4, iind0].std(ddof=1).values
            sigma2 = pi_sst_alltime[model]['sea'][
                3::4, iind0].std(ddof=1).values
            sim_lig_pi_2s = (sigma1**2 + sigma2**2)**0.5 * 2
        
        sim_obs_lig_pi = sim_lig_pi - obs_lig_pi
        
        obs_sim_lig_pi_so_sst_mc = pd.concat([
            obs_sim_lig_pi_so_sst_mc,
            pd.DataFrame(data={
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig_pi': obs_lig_pi,
                'sim_lig_pi': sim_lig_pi,
                'sim_lig_pi_2s': sim_lig_pi_2s,
                'sim_obs_lig_pi': sim_obs_lig_pi,
                }, index=[0])], ignore_index=True,)


with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst_mc.pkl', 'wb') as f:
    pickle.dump(obs_sim_lig_pi_so_sst_mc, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst_mc.pkl', 'rb') as f:
    obs_sim_lig_pi_so_sst_mc = pickle.load(f)

site_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_so_sst_mc.slat, obs_sim_lig_pi_so_sst_mc.slon)]
grid_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_pi_so_sst_mc.glat, obs_sim_lig_pi_so_sst_mc.glon)]

from haversine import haversine_vector
distances = haversine_vector(
    site_pairs, grid_pairs, normalize=True,
    )
np.max(distances)




            # sim_lig_pi = lig_pi_am[iind0, iind1]
            # sim_lig_pi_2s = lig_pi_ann[:, iind0, iind1].std(ddof=1) * 2
            # sim_lig_pi = lig_pi_am[iind0]
            # sim_lig_pi_2s = lig_pi_ann[:, iind0].std(ddof=1) * 2

# Propogation of uncertainties
https://en.wikipedia.org/wiki/Propagation_of_uncertainty

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import MC reconstructions: SIC

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)

models=sorted(lig_sic.keys())

#-------- import MC reconstruction
with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

'''
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'rb') as f:
    pi_sic_alltime = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices

loc_indices_rec_mc_sic = {}

for model in models:
    # model = 'HadGEM3-GC31-LL'
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    
    loc_indices_rec_mc_sic[model] = {}
    
    lon = lig_sic[model].lon.values
    lat = lig_sic[model].lat.values
    
    for istation in chadwick_interp.index:
        # istation = 0
        station = chadwick_interp.sites[istation]
        print('#---- ' + str(istation) + ': ' + station)
        
        slon = chadwick_interp.lon[istation]
        slat = chadwick_interp.lat[istation]
        
        loc_indices_rec_mc_sic[model][station] = \
            find_ilat_ilon_general(slat, slon, lat, lon)

with open('scratch/cmip6/lig/loc_indices_rec_mc_sic.pkl', 'wb') as f:
    pickle.dump(loc_indices_rec_mc_sic, f)


'''
#---------------- check
with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
models=sorted(lig_sic.keys())
with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

from haversine import haversine
with open('scratch/cmip6/lig/loc_indices_rec_mc_sic.pkl', 'rb') as f:
    loc_indices_rec_mc_sic = pickle.load(f)

for model in models:
    # model = 'HadGEM3-GC31-LL'
    print('#-------------------------------- ' + model)
    
    lon = lig_sic[model].lon.values
    lat = lig_sic[model].lat.values
    
    for istation in chadwick_interp.index:
        # istation = 10
        station = chadwick_interp.sites[istation]
        print('#---------------- ' + str(istation) + ': ' + station)
        
        slon = chadwick_interp.lon[istation]
        slat = chadwick_interp.lat[istation]
        
        if (lon.ndim == 2):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_mc_sic[model][station][0],
                     loc_indices_rec_mc_sic[model][station][1]],
                 lon[loc_indices_rec_mc_sic[model][station][0],
                     loc_indices_rec_mc_sic[model][station][1]]],
                normalize=True,)
        elif (lon.ndim == 1):
            distance = haversine(
                [slat, slon],
                [lat[loc_indices_rec_mc_sic[model][station][0]],
                 lon[loc_indices_rec_mc_sic[model][station][0]]],
                normalize=True,)
        
        if (distance > 100):
            print(np.round(distance, 0))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract LIG-PI Sep SIC : MC


with open('scratch/cmip6/lig/loc_indices_rec_mc_sic.pkl', 'rb') as f:
    loc_indices_rec_mc_sic = pickle.load(f)


obs_sim_lig_so_sic_mc = pd.DataFrame(columns=(
    'stations', 'models',
    'slat', 'slon', 'glat', 'glon',
    'obs_lig', 'sim_lig', 'sim_lig_2s',
    'sim_obs_lig', ))

#---------------- Sep SIC

for istation in chadwick_interp.index:
    # istation = chadwick_interp.index[0]
    station = chadwick_interp.sites[istation]
    
    print('#---- ' + str(istation) + ': ' + station)
    
    slat = chadwick_interp.lat[istation]
    slon = chadwick_interp.lon[istation]
    obs_lig = chadwick_interp.sic_sep[istation]
    
    for model in models:
        # model = 'ACCESS-ESM1-5'
        # model = 'AWI-ESM-1-1-LR'
        # model = 'HadGEM3-GC31-LL'
        # model = 'GISS-E2-1-G'
        print(model)
        lon = lig_sic[model].lon.values
        lat = lig_sic[model].lat.values
        
        if (lon.shape == lig_sic_alltime[model]['am'].shape):
            iind0 = loc_indices_rec_mc_sic[model][station][0]
            iind1 = loc_indices_rec_mc_sic[model][station][1]
        else:
            iind0 = loc_indices_rec_mc_sic[model][station][1]
            iind1 = loc_indices_rec_mc_sic[model][station][0]
        
        if (lon.ndim == 2):
            glat = lat[loc_indices_rec_mc_sic[model][station][0],
                       loc_indices_rec_mc_sic[model][station][1]]
            glon = lon[loc_indices_rec_mc_sic[model][station][0],
                       loc_indices_rec_mc_sic[model][station][1]]
            sim_lig = \
                lig_sic_alltime[model]['mm'].sel(
                    month=9)[iind0, iind1].values
            sim_lig_2s = lig_sic_alltime[model]['mon'][
                8::12, iind0, iind1].std(ddof=1).values * 2
        elif (lon.ndim == 1):
            glat = lat[loc_indices_rec_mc_sic[model][station][0]]
            glon = lon[loc_indices_rec_mc_sic[model][station][0]]
            sim_lig = \
                lig_sic_alltime[model]['mm'].sel(
                    month=9)[iind0].values
            sim_lig_2s = lig_sic_alltime[model]['mon'][
                8::12, iind0].std(ddof=1).values * 2
        
        sim_obs_lig = sim_lig - obs_lig
        
        obs_sim_lig_so_sic_mc = pd.concat([
            obs_sim_lig_so_sic_mc,
            pd.DataFrame(data={
                'stations': station,
                'models': model,
                'slat': slat,
                'slon': slon,
                'glat': glat,
                'glon': glon,
                'obs_lig': obs_lig,
                'sim_lig': sim_lig,
                'sim_lig_2s': sim_lig_2s,
                'sim_obs_lig': sim_obs_lig,
                }, index=[0])], ignore_index=True,)

with open('scratch/cmip6/lig/obs_sim_lig_so_sic_mc.pkl', 'wb') as f:
    pickle.dump(obs_sim_lig_so_sic_mc, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/obs_sim_lig_so_sic_mc.pkl', 'rb') as f:
    obs_sim_lig_so_sic_mc = pickle.load(f)

site_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_so_sic_mc.slat, obs_sim_lig_so_sic_mc.slon)]
grid_pairs = [(x, y) for x, y in zip (
    obs_sim_lig_so_sic_mc.glat, obs_sim_lig_so_sic_mc.glon)]

from haversine import haversine_vector
distances = haversine_vector(
    site_pairs, grid_pairs, normalize=True,
    )
np.max(distances)

            # if (model != 'HadGEM3-GC31-LL'):
            # elif (model == 'HadGEM3-GC31-LL'):
            #     sim_lig = regrid(
            #         lig_sic_alltime[model]['mm'].sel(month=9),
            #         ds_out = pi_sic[model],
            #         )[iind0, iind1+1].values
            #     sim_lig_2s = regrid(
            #         lig_sic_alltime[model]['mon'],
            #         ds_out = pi_sic[model].siconc,
            #         )[8::12, iind0, iind1+1].std(ddof=1).values * 2

            # if (model != 'HadGEM3-GC31-LL'):
            # elif (model == 'HadGEM3-GC31-LL'):
            #     sim_lig = regrid(
            #         lig_sic_alltime[model]['mm'].sel(month=9),
            #         ds_out = pi_sic[model],
            #         )[iind0].values
            #     sim_lig_2s = regrid(
            #         lig_sic_alltime[model]['mon'],
            #         ds_out = pi_sic[model])[
            #             8::12, iind0].std(ddof=1).values * 2



'''
# endregion
# -----------------------------------------------------------------------------




