

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
# region get HadISST1.1 site values

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_hadisst.pkl', 'rb') as f:
    lig_recs_loc_indices_hadisst = pickle.load(f)

HadISST1_1 = {}
HadISST1_1['sst'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_sst.nc')
HadISST1_1['sic'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_ice.nc')

HadISST1_1['sst'].sst.values[HadISST1_1['sst'].sst.values == -1000] = np.nan

HadISST1_1['sst_alltime'] = mon_sea_ann(
    var_monthly=HadISST1_1['sst'].sst.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)
HadISST1_1['sic_alltime'] = mon_sea_ann(
    var_monthly=HadISST1_1['sic'].sic.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)


hadisst_site_values = {}
hadisst_site_values['sst'] = {}
hadisst_site_values['sic'] = {}

#-------------------------------- extract SST

for irec in ['MC', 'DC']:
    # irec = 'MC'
    print('#---------------- ' + irec)
    
    hadisst_site_values['sst'][irec] = {}
    
    # annual SST
    hadisst_site_values['sst'][irec]['annual'] = pd.DataFrame(
        columns={'Station', 'HadISST_annual'}
    )
    # JFM SST
    hadisst_site_values['sst'][irec]['JFM'] = pd.DataFrame(
        columns={'Station', 'HadISST_JFM'}
    )
    
    for istation in lig_recs_loc_indices_hadisst[irec].keys():
        # istation = list(lig_recs_loc_indices_hadisst[irec].keys())[0]
        print('#-------- ' + istation)
        
        # annual SST
        hadisst_site_values['sst'][irec]['annual'] = pd.concat([
            hadisst_site_values['sst'][irec]['annual'],
            pd.DataFrame(data={
                'Station': istation,
                'HadISST_annual': HadISST1_1['sst_alltime']['am'][
                    lig_recs_loc_indices_hadisst[irec][istation][0],
                    lig_recs_loc_indices_hadisst[irec][istation][1],
                ].values
            }, index=[0])
        ], ignore_index=True,)
        
        # JFM SST
        hadisst_site_values['sst'][irec]['JFM'] = pd.concat([
            hadisst_site_values['sst'][irec]['JFM'],
            pd.DataFrame(data={
                'Station': istation,
                'HadISST_JFM': HadISST1_1['sst_alltime']['sm'].sel(
                    month=3)[
                    lig_recs_loc_indices_hadisst[irec][istation][0],
                    lig_recs_loc_indices_hadisst[irec][istation][1],
                ].values
            }, index=[0])
        ], ignore_index=True,)

#-------------------------------- extract SIC

irec = 'MC'
print('#---------------- ' + irec)
# Sep sic
hadisst_site_values['sic'][irec] = {}
hadisst_site_values['sic'][irec]['Sep'] = pd.DataFrame(
        columns={'Station', 'HadISST_Sep'}
    )

for istation in lig_recs_loc_indices_hadisst[irec].keys():
    print('#-------- ' + istation)
    
    hadisst_site_values['sic'][irec]['Sep'] = pd.concat([
        hadisst_site_values['sic'][irec]['Sep'],
        pd.DataFrame(data={
                'Station': istation,
                'HadISST_Sep': HadISST1_1['sic_alltime']['mm'].sel(
                    month=9)[
                    lig_recs_loc_indices_hadisst[irec][istation][0],
                    lig_recs_loc_indices_hadisst[irec][istation][1],
                ].values * 100
            }, index=[0])
    ], ignore_index=True,)

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'wb') as f:
    pickle.dump(hadisst_site_values, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/rec/hadisst_site_values.pkl', 'rb') as f:
    hadisst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_hadisst.pkl', 'rb') as f:
    lig_recs_loc_indices_hadisst = pickle.load(f)

HadISST1_1 = {}
HadISST1_1['sst'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_sst.nc')
HadISST1_1['sic'] = xr.open_dataset(
    'data_sources/LIG/HadISST1.1/HadISST_ice.nc')

HadISST1_1['sst'].sst.values[HadISST1_1['sst'].sst.values == -1000] = np.nan

HadISST1_1['sst_alltime'] = mon_sea_ann(
    var_monthly=HadISST1_1['sst'].sst.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)
HadISST1_1['sic_alltime'] = mon_sea_ann(
    var_monthly=HadISST1_1['sic'].sic.isel(time=slice(0, 360)),
    seasons = 'Q-MAR',)

lig_recs = {}
lig_recs['DC'] = pd.read_csv(
    'data_sources/LIG/Chandler-Langebroek_2021_SST-anom.tab',
    sep='\t', header=0, skiprows=76,)
lig_recs['MC'] = pd.read_excel(
    'data_sources/LIG/Chadwick_et_al_2021/AICC2012 ages.xlsx',
    header=0,)

irec = 'DC'
# hadisst_site_values['sst'][irec]['annual'].Station
istation = 'DSDP-593'
hadisst_site_values['sst'][irec]['annual'][
    hadisst_site_values['sst'][irec]['annual'].Station == istation
].HadISST_annual

HadISST1_1['sst_alltime']['am'][
    lig_recs_loc_indices_hadisst[irec][istation][0],
    lig_recs_loc_indices_hadisst[irec][istation][1],
]

lig_recs[irec][lig_recs[irec]['Site'] == istation]['Latitude'].values[0]
lig_recs[irec][lig_recs[irec]['Site'] == istation]['Longitude'].values[0]
# Site; Core name



(HadISST1_1['sst'].sst.values == -1000).sum()
HadISST1_1['sst_alltime']['sm'].to_netcdf('scratch/test/test0.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get pmip3 site values

#-------------------------------- import data

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_pmip3ens.pkl', 'rb') as f:
    lig_recs_loc_indices_pmip3ens = pickle.load(f)

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc').sst
pmip3_lig_sim['summer_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sstdjf_c.nc').sstdjf

pmip3_lig_sim['annual_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfc_c.nc').sfc
pmip3_lig_sim['summer_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfcdjf_c.nc').sfcdjf

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)


#-------------------------------- settings

recs = ['EC', 'JH', 'DC', 'MC',]
lig_proxies = ['annual_sst', 'summer_sst', 'annual_sat', ]
lig_datasets = {}
lig_anom_name = {}
lig_twosigma_name = {}

# annual_sst
lig_datasets['annual_sst'] = {
    'EC': lig_recs['EC']['SO_ann'],
    'JH': lig_recs['JH']['SO_ann'],
    'DC': lig_recs['DC']['annual_128'],
    'MC': None,}
lig_anom_name['annual_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_ann',
    'MC': None,}
lig_twosigma_name['annual_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# summer_sst
lig_datasets['summer_sst'] = {
    'EC': lig_recs['EC']['SO_jfm'],
    'JH': lig_recs['JH']['SO_jfm'],
    'DC': lig_recs['DC']['JFM_128'],
    'MC': lig_recs['MC']['interpolated'],}
lig_anom_name['summer_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_jfm',
    'MC': 'sst_anom_hadisst_jfm',}
lig_twosigma_name['summer_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# annual_sat
lig_datasets['annual_sat'] = {
    'EC': lig_recs['EC']['AIS_am'],
    'JH': None,
    'DC': None,
    'MC': None,}
lig_anom_name['annual_sat'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}
lig_twosigma_name['annual_sat'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}


#-------------------------------- extract values

pmip3_anomalies_site_values = {}

for iproxy in lig_proxies:
    # iproxy = 'annual_sst'
    print('#-------------------------------- ' + iproxy)
    
    pmip3_anomalies_site_values[iproxy] = {}
    
    for irec in recs:
        # irec = 'EC'
        print('#---------------- ' + irec)
        
        if not (lig_datasets[iproxy][irec] is None):
            
            pmip3_anomalies_site_values[iproxy][irec] = pd.DataFrame(
                columns={
                    'Station',
                    'Latitude',
                    'Longitude',
                    'rec_lig_pi',
                    'rec_lig_pi_2sigma',
                    'sim_lig_pi',
                    'sim_rec_lig_pi',
                    }
                )
            
            for istation in lig_datasets[iproxy][irec].Station:
                # istation = lig_datasets[iproxy][irec].Station.iloc[0]
                print('#-------- ' + istation)
                
                Station = istation
                Latitude = lig_datasets[iproxy][irec].loc[
                    lig_datasets[iproxy][irec].Station == istation
                    ].Latitude.iloc[0]
                Longitude = lig_datasets[iproxy][irec].loc[
                    lig_datasets[iproxy][irec].Station == istation
                    ].Longitude.iloc[0]
                
                rec_lig_pi = lig_datasets[iproxy][irec].loc[
                    lig_datasets[iproxy][irec].Station == istation][
                        lig_anom_name[iproxy][irec]
                        ].iloc[0]
                
                if (lig_twosigma_name[iproxy][irec] is None):
                    rec_lig_pi_2sigma = np.nan
                else:
                    rec_lig_pi_2sigma = lig_datasets[iproxy][irec].loc[
                        lig_datasets[iproxy][irec].Station == istation][
                            lig_twosigma_name[iproxy][irec]
                            ].iloc[0]
                
                ind0 = lig_recs_loc_indices_pmip3ens[irec][istation][0]
                ind1 = lig_recs_loc_indices_pmip3ens[irec][istation][1]
                sim_lig_pi = pmip3_lig_sim[iproxy][ind0, ind1].values
                
                if (np.isnan(sim_lig_pi)):
                    print('land at marine sediment sites')
                
                sim_rec_lig_pi = sim_lig_pi - rec_lig_pi
                
                pmip3_anomalies_site_values[iproxy][irec] = pd.concat([
                    pmip3_anomalies_site_values[iproxy][irec],
                    pd.DataFrame(data={
                        'Station': Station,
                        'Latitude': Latitude,
                        'Longitude': Longitude,
                        'rec_lig_pi': rec_lig_pi,
                        'rec_lig_pi_2sigma': rec_lig_pi_2sigma,
                        'sim_lig_pi': sim_lig_pi,
                        'sim_rec_lig_pi': sim_rec_lig_pi,
                        }, index=[0])
                    ], ignore_index=True,)

with open('scratch/cmip6/lig/sst/pmip3_anomalies_site_values.pkl', 'wb') as f:
    pickle.dump(pmip3_anomalies_site_values, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/pmip3_anomalies_site_values.pkl', 'rb') as f:
    pmip3_anomalies_site_values = pickle.load(f)

# import data

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices_pmip3ens.pkl', 'rb') as f:
    lig_recs_loc_indices_pmip3ens = pickle.load(f)

pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc').sst
pmip3_lig_sim['summer_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sstdjf_c.nc').sstdjf

pmip3_lig_sim['annual_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfc_c.nc').sfc
pmip3_lig_sim['summer_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfcdjf_c.nc').sfcdjf

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

# setting

recs = ['EC', 'JH', 'DC', 'MC',]
lig_proxies = ['annual_sst', 'summer_sst', 'annual_sat', ]
lig_datasets = {}
lig_anom_name = {}
lig_twosigma_name = {}

# annual_sst
lig_datasets['annual_sst'] = {
    'EC': lig_recs['EC']['SO_ann'],
    'JH': lig_recs['JH']['SO_ann'],
    'DC': lig_recs['DC']['annual_128'],
    'MC': None,}
lig_anom_name['annual_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_ann',
    'MC': None,}
lig_twosigma_name['annual_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# summer_sst
lig_datasets['summer_sst'] = {
    'EC': lig_recs['EC']['SO_jfm'],
    'JH': lig_recs['JH']['SO_jfm'],
    'DC': lig_recs['DC']['JFM_128'],
    'MC': lig_recs['MC']['interpolated'],}
lig_anom_name['summer_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_jfm',
    'MC': 'sst_anom_hadisst_jfm',}
lig_twosigma_name['summer_sst'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,}

# annual_sat
lig_datasets['annual_sat'] = {
    'EC': lig_recs['EC']['AIS_am'],
    'JH': None,
    'DC': None,
    'MC': None,}
lig_anom_name['annual_sat'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}
lig_twosigma_name['annual_sat'] = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': None,
    'DC': None,
    'MC': None,}



iproxy = 'annual_sst'
irec = 'EC'

iproxy = 'annual_sst'
irec = 'DC'

iproxy = 'summer_sst'
irec = 'EC'

iproxy = 'summer_sst'
irec = 'DC'

iproxy = 'annual_sat'
irec = 'EC'

istation = lig_datasets[iproxy][irec].Station.iloc[-1]

pmip3_anomalies_site_values[iproxy][irec].loc[
    pmip3_anomalies_site_values[iproxy][irec].Station == istation]

np.round(lig_datasets[iproxy][irec].loc[
    lig_datasets[iproxy][irec].Station == istation][
        lig_anom_name[iproxy][irec]
        ].iloc[0], 2)

pmip3_lig_sim[iproxy][
    lig_recs_loc_indices_pmip3ens[irec][istation][0],
    lig_recs_loc_indices_pmip3ens[irec][istation][1]].values

'''
# endregion
# -----------------------------------------------------------------------------


