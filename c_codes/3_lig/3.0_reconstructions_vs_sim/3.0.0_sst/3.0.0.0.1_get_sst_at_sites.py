

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

#-------------------------------- import simulations

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

#-------------------------------- import reconstructions

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

#-------------------------------- import loc indices

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices.pkl', 'rb') as f:
    lig_recs_loc_indices = pickle.load(f)



'''
lig_recs_loc_indices.keys()
lig_recs_loc_indices['EC'].keys()
lig_recs_loc_indices['EC']['NEEM']

lig_sst_regrid_alltime.keys()
lig_sst_regrid_alltime['ACCESS-ESM1-5'].keys()
lig_sst_regrid_alltime['ACCESS-ESM1-5']['sm']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract SO annual SST

recs = ['EC', 'JH', 'DC',]
so_ann_sst_recs = {
    'EC': lig_recs['EC']['SO_ann'],
    'JH': lig_recs['JH']['SO_ann'],
    'DC': lig_recs['DC']['annual_128'],
}
# 2 + 12 + 20 = 34

sst_anom_127_name = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_ann',
}

sst_anom_127_2sigma_name = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
}

SO_ann_sst_site_values = {}

for irec in recs:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    SO_ann_sst_site_values[irec] = pd.DataFrame(
        columns={
            'Station',
            'Model',
            'Latitude',
            'Longitude',
            'rec_ann_sst_lig_pi',
            'rec_ann_sst_lig_pi_2sigma',
            'sim_ann_sst_lig_pi',
            'sim_ann_sst_lig_pi_2std',
            'sim_rec_ann_sst_lig_pi',
            }
    )
    
    for istation in so_ann_sst_recs[irec].Station:
        # istation = so_ann_sst_recs[irec].Station.iloc[0]
        print('#---- ' + istation)
        
        for imodel in lig_pi_sst_regrid_alltime.keys():
            # imodel = 'ACCESS-ESM1-5'
            print('#-------- ' + imodel)
            
            Station = istation
            Model = imodel
            Latitude = so_ann_sst_recs[irec].loc[
                so_ann_sst_recs[irec].Station == istation].Latitude.iloc[0]
            Longitude = so_ann_sst_recs[irec].loc[
                so_ann_sst_recs[irec].Station == istation].Longitude.iloc[0]
            
            rec_ann_sst_lig_pi = so_ann_sst_recs[irec].loc[
                so_ann_sst_recs[irec].Station == istation][
                    sst_anom_127_name[irec]
                ].iloc[0]
            
            if (sst_anom_127_2sigma_name[irec] is None):
                rec_ann_sst_lig_pi_2sigma = np.nan
            else:
                rec_ann_sst_lig_pi_2sigma = so_ann_sst_recs[irec].loc[
                    so_ann_sst_recs[irec].Station == istation][
                        sst_anom_127_2sigma_name[irec]
                        ].iloc[0]
            
            ind0 = lig_recs_loc_indices[irec][istation][0]
            ind1 = lig_recs_loc_indices[irec][istation][1]
            sim_ann_sst_lig_pi = lig_pi_sst_regrid_alltime[imodel]['am'][
                0, ind0, ind1].values
            
            if (np.isnan(sim_ann_sst_lig_pi)):
                print('land at marine sediment sites')
                ind0 = lig_recs_loc_indices[irec][istation][0] - 1
                ind1 = lig_recs_loc_indices[irec][istation][1]
                sim_ann_sst_lig_pi = lig_pi_sst_regrid_alltime[imodel]['am'][
                    0, ind0, ind1].values
                
                if (np.isnan(sim_ann_sst_lig_pi)):
                    print('land at marine sediment sites * 2')
                    ind0 = lig_recs_loc_indices[irec][istation][0]
                    ind1 = lig_recs_loc_indices[irec][istation][1] - 1
                    sim_ann_sst_lig_pi = \
                        lig_pi_sst_regrid_alltime[imodel]['am'][
                            0, ind0, ind1].values
                    
                    if (np.isnan(sim_ann_sst_lig_pi)):
                        print('land at marine sediment sites * 3')
                        ind0 = lig_recs_loc_indices[irec][istation][0] - 1
                        ind1 = lig_recs_loc_indices[irec][istation][1] + 1
                        sim_ann_sst_lig_pi = \
                            lig_pi_sst_regrid_alltime[imodel]['am'][
                                0, ind0, ind1].values
                        
                        if (np.isnan(sim_ann_sst_lig_pi)):
                            print('land at marine sediment sites * 4')
                            ind0 = lig_recs_loc_indices[irec][istation][0] + 1
                            ind1 = lig_recs_loc_indices[irec][istation][1]
                            sim_ann_sst_lig_pi = \
                                lig_pi_sst_regrid_alltime[imodel]['am'][
                                    0, ind0, ind1].values
                            
                            if (np.isnan(sim_ann_sst_lig_pi)):
                                print('land at marine sediment sites * 5')
                                print('warning: check!')
            
            sim_ann_sst_lig_pi_2std = lig_pi_sst_regrid_alltime[imodel]['ann'][
                :, ind0, ind1, ].values.std(ddof=1) * 2
            
            sim_rec_ann_sst_lig_pi = sim_ann_sst_lig_pi - rec_ann_sst_lig_pi
            
            SO_ann_sst_site_values[irec] = pd.concat([
                SO_ann_sst_site_values[irec],
                pd.DataFrame(data={
                    'Station': Station,
                    'Model': Model,
                    'Latitude': Latitude,
                    'Longitude': Longitude,
                    'rec_ann_sst_lig_pi': rec_ann_sst_lig_pi,
                    'rec_ann_sst_lig_pi_2sigma': rec_ann_sst_lig_pi_2sigma,
                    'sim_ann_sst_lig_pi': sim_ann_sst_lig_pi,
                    'sim_ann_sst_lig_pi_2std': sim_ann_sst_lig_pi_2std,
                    'sim_rec_ann_sst_lig_pi': sim_rec_ann_sst_lig_pi,
                }, index=[0])
            ], ignore_index=True,)

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'wb') as f:
    pickle.dump(SO_ann_sst_site_values, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)

irec = 'JH'
imodel = 'ACCESS-ESM1-5'
istation = 'MD97-2108'

SO_ann_sst_site_values[irec].loc[
    (SO_ann_sst_site_values[irec].Model == imodel) & \
        (SO_ann_sst_site_values[irec].Station == istation)]['rec_ann_sst_lig_pi']

lig_recs[irec]['SO_ann'].loc[
    lig_recs[irec]['SO_ann'].Station == istation]['127 ka SST anomaly (°C)']

SO_ann_sst_site_values[irec].loc[
    (SO_ann_sst_site_values[irec].Model == imodel) & \
        (SO_ann_sst_site_values[irec].Station == istation)]['sim_ann_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['am'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


irec = 'DC'
imodel = 'GISS-E2-1-G'
istation = 'FR1-94-GC3'

SO_ann_sst_site_values[irec].loc[
    (SO_ann_sst_site_values[irec].Model == imodel) & \
        (SO_ann_sst_site_values[irec].Station == istation)]['rec_ann_sst_lig_pi']

lig_recs[irec]['annual_128'].loc[
    lig_recs[irec]['annual_128'].Station == istation]['sst_anom_hadisst_ann']

SO_ann_sst_site_values[irec].loc[
    (SO_ann_sst_site_values[irec].Model == imodel) & \
        (SO_ann_sst_site_values[irec].Station == istation)]['sim_ann_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['am'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values




#---------------- draft

lig_pi_sst_regrid_alltime['ACCESS-ESM1-5'].keys()

lig_pi_sst_regrid_alltime[imodel]['am'][
    0,
    (lig_recs_loc_indices[irec][istation][0] - 1):(lig_recs_loc_indices[irec][istation][0] + 2),
    (lig_recs_loc_indices[irec][istation][1] - 1):(lig_recs_loc_indices[irec][istation][1] + 2),
    ]


# irec = 'JH'
# imodel = 'ACCESS-ESM1-5'
# istation = 'MD97-2121'

# irec = 'DC'
# imodel = 'FGOALS-g3'
# istation = 'SO136-GC3'

# irec = 'DC'
# imodel = 'GISS-E2-1-G'
# istation = 'MD06-2986'



lig_recs['EC']['SO_ann'].columns
lig_recs['JH']['SO_ann'].columns
lig_recs['DC']['annual_128'].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract SO summer SST

recs = ['EC', 'JH', 'DC', 'MC',]

so_jfm_sst_recs = {
    'EC': lig_recs['EC']['SO_jfm'],
    'JH': lig_recs['JH']['SO_jfm'],
    'DC': lig_recs['DC']['JFM_128'],
    'MC': lig_recs['MC']['interpolated'],
}
# 15 + 7 + 21 + 9 = 52
# in total 86 cores

sst_anom_127_name = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_jfm',
    'MC': 'sst_anom_hadisst_jfm',
}

sst_anom_127_2sigma_name = {
    'EC': '127 ka 2s PIAn [°C]',
    'JH': '127 ka 2σ (°C)',
    'DC': None,
    'MC': None,
}

SO_jfm_sst_site_values = {}

for irec in recs:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    SO_jfm_sst_site_values[irec] = pd.DataFrame(
        columns={
            'Station',
            'Model',
            'Latitude',
            'Longitude',
            'rec_jfm_sst_lig_pi',
            'rec_jfm_sst_lig_pi_2sigma',
            'sim_jfm_sst_lig_pi',
            'sim_jfm_sst_lig_pi_2std',
            'sim_rec_jfm_sst_lig_pi',
            }
    )
    
    for istation in so_jfm_sst_recs[irec].Station:
        # istation = so_jfm_sst_recs[irec].Station.iloc[0]
        print('#---- ' + istation)
        
        for imodel in lig_pi_sst_regrid_alltime.keys():
            # imodel = 'ACCESS-ESM1-5'
            print('#-------- ' + imodel)
            
            Station = istation
            Model = imodel
            Latitude = so_jfm_sst_recs[irec].loc[
                so_jfm_sst_recs[irec].Station == istation].Latitude.iloc[0]
            Longitude = so_jfm_sst_recs[irec].loc[
                so_jfm_sst_recs[irec].Station == istation].Longitude.iloc[0]
            
            rec_jfm_sst_lig_pi = so_jfm_sst_recs[irec].loc[
                so_jfm_sst_recs[irec].Station == istation][
                    sst_anom_127_name[irec]
                ].iloc[0]
            
            if (sst_anom_127_2sigma_name[irec] is None):
                rec_jfm_sst_lig_pi_2sigma = np.nan
            else:
                rec_jfm_sst_lig_pi_2sigma = so_jfm_sst_recs[irec].loc[
                    so_jfm_sst_recs[irec].Station == istation][
                        sst_anom_127_2sigma_name[irec]
                        ].iloc[0]
            
            ind0 = lig_recs_loc_indices[irec][istation][0]
            ind1 = lig_recs_loc_indices[irec][istation][1]
            sim_jfm_sst_lig_pi = lig_pi_sst_regrid_alltime[imodel]['sm'][
                0, ind0, ind1].values
            
            if (np.isnan(sim_jfm_sst_lig_pi)):
                print('land at marine sediment sites')
                ind0 = lig_recs_loc_indices[irec][istation][0] - 1
                ind1 = lig_recs_loc_indices[irec][istation][1]
                sim_jfm_sst_lig_pi = lig_pi_sst_regrid_alltime[imodel]['sm'][
                    0, ind0, ind1].values
                
                if (np.isnan(sim_jfm_sst_lig_pi)):
                    print('land at marine sediment sites * 2')
                    ind0 = lig_recs_loc_indices[irec][istation][0]
                    ind1 = lig_recs_loc_indices[irec][istation][1] - 1
                    sim_jfm_sst_lig_pi = \
                        lig_pi_sst_regrid_alltime[imodel]['sm'][
                            0, ind0, ind1].values
                    
                    if (np.isnan(sim_jfm_sst_lig_pi)):
                        print('land at marine sediment sites * 3')
                        ind0 = lig_recs_loc_indices[irec][istation][0] - 1
                        ind1 = lig_recs_loc_indices[irec][istation][1] + 1
                        sim_jfm_sst_lig_pi = \
                            lig_pi_sst_regrid_alltime[imodel]['sm'][
                                0, ind0, ind1].values
                        
                        if (np.isnan(sim_jfm_sst_lig_pi)):
                            print('land at marine sediment sites * 4')
                            ind0 = lig_recs_loc_indices[irec][istation][0] + 1
                            ind1 = lig_recs_loc_indices[irec][istation][1]
                            sim_jfm_sst_lig_pi = \
                                lig_pi_sst_regrid_alltime[imodel]['sm'][
                                    0, ind0, ind1].values
                            
                            if (np.isnan(sim_jfm_sst_lig_pi)):
                                print('land at marine sediment sites * 5')
                                print('warning: check!')
            
            sim_jfm_sst_lig_pi_2std = lig_pi_sst_regrid_alltime[imodel]['sea'][
                ::4, ind0, ind1].values.std(ddof=1) * 2
            
            sim_rec_jfm_sst_lig_pi = sim_jfm_sst_lig_pi - rec_jfm_sst_lig_pi
            
            SO_jfm_sst_site_values[irec] = pd.concat([
                SO_jfm_sst_site_values[irec],
                pd.DataFrame(data={
                    'Station': Station,
                    'Model': Model,
                    'Latitude': Latitude,
                    'Longitude': Longitude,
                    'rec_jfm_sst_lig_pi': rec_jfm_sst_lig_pi,
                    'rec_jfm_sst_lig_pi_2sigma': rec_jfm_sst_lig_pi_2sigma,
                    'sim_jfm_sst_lig_pi': sim_jfm_sst_lig_pi,
                    'sim_jfm_sst_lig_pi_2std': sim_jfm_sst_lig_pi_2std,
                    'sim_rec_jfm_sst_lig_pi': sim_rec_jfm_sst_lig_pi,
                }, index=[0])
            ], ignore_index=True,)

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'wb') as f:
    pickle.dump(SO_jfm_sst_site_values, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)

irec = 'EC'
imodel = 'ACCESS-ESM1-5'
istation = 'MD94-102'

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['rec_jfm_sst_lig_pi']

lig_recs[irec]['SO_jfm'].loc[
    lig_recs[irec]['SO_jfm'].Station == istation]['127 ka Median PIAn [°C]']

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['sim_jfm_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['sm'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


irec = 'JH'
imodel = 'ACCESS-ESM1-5'
istation = 'ODP-1089'

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['rec_jfm_sst_lig_pi']

lig_recs[irec]['SO_jfm'].loc[
    lig_recs[irec]['SO_jfm'].Station == istation]['127 ka SST anomaly (°C)']

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['sim_jfm_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['sm'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


irec = 'DC'
imodel = 'GISS-E2-1-G'
istation = 'E49-17'

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['rec_jfm_sst_lig_pi']

lig_recs[irec]['JFM_128'].loc[
    lig_recs[irec]['JFM_128'].Station == istation]['sst_anom_hadisst_jfm']

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['sim_jfm_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['sm'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


irec = 'MC'
imodel = 'GISS-E2-1-G'
istation = 'PC509'

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['rec_jfm_sst_lig_pi']

lig_recs[irec]['interpolated'].loc[
    lig_recs[irec]['interpolated'].Station == istation]['sst_anom_hadisst_jfm']

SO_jfm_sst_site_values[irec].loc[
    (SO_jfm_sst_site_values[irec].Model == imodel) & \
        (SO_jfm_sst_site_values[irec].Station == istation)]['sim_jfm_sst_lig_pi']
lig_pi_sst_regrid_alltime[imodel]['sm'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values



'''
# endregion
# -----------------------------------------------------------------------------


