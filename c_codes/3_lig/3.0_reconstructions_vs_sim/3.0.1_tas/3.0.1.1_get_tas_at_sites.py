

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

with open('scratch/cmip6/lig/tas/lig_pi_tas_regrid_alltime.pkl', 'rb') as f:
    lig_pi_tas_regrid_alltime = pickle.load(f)

#-------------------------------- import reconstructions

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

#-------------------------------- import loc indices

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices.pkl', 'rb') as f:
    lig_recs_loc_indices = pickle.load(f)



'''
lig_pi_tas_regrid_alltime.keys()
lig_pi_tas_regrid_alltime['ACCESS-ESM1-5'].keys()
lig_pi_tas_regrid_alltime['ACCESS-ESM1-5']['am']

lig_recs['EC'].keys()
lig_recs['EC']['AIS_am'].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract AIS ann SAT

recs = ['EC',]
ais_ann_tas_recs = {
    'EC': lig_recs['EC']['AIS_am'],
}
# 4

tas_anom_127_name = {
    'EC': '127 ka Median PIAn [°C]',
}

tas_anom_127_2sigma_name = {
    'EC': '127 ka 2s PIAn [°C]',
}

AIS_ann_tas_site_values = {}

for irec in recs:
    # irec = 'EC'
    print('#---------------- ' + irec)
    
    AIS_ann_tas_site_values[irec] = pd.DataFrame(
        columns={
            'Station',
            'Model',
            'Latitude',
            'Longitude',
            'rec_ann_tas_lig_pi',
            'rec_ann_tas_lig_pi_2sigma',
            'sim_ann_tas_lig_pi',
            'sim_ann_tas_lig_pi_2std',
            'sim_rec_ann_tas_lig_pi',
            }
    )
    
    for istation in ais_ann_tas_recs[irec].Station:
        # istation = ais_ann_tas_recs[irec].Station.iloc[0]
        print('#---- ' + istation)
        
        for imodel in lig_pi_tas_regrid_alltime.keys():
            # imodel = 'ACCESS-ESM1-5'
            print('#-------- ' + imodel)
            
            Station = istation
            Model = imodel
            
            Latitude = ais_ann_tas_recs[irec].loc[
                ais_ann_tas_recs[irec].Station == istation].Latitude.iloc[0]
            Longitude = ais_ann_tas_recs[irec].loc[
                ais_ann_tas_recs[irec].Station == istation].Longitude.iloc[0]
            
            rec_ann_tas_lig_pi = ais_ann_tas_recs[irec].loc[
                ais_ann_tas_recs[irec].Station == istation][
                    tas_anom_127_name[irec]
                ].iloc[0]
            
            rec_ann_tas_lig_pi_2sigma = ais_ann_tas_recs[irec].loc[
                ais_ann_tas_recs[irec].Station == istation][
                    tas_anom_127_2sigma_name[irec]
                ].iloc[0]
            
            ind0 = lig_recs_loc_indices[irec][istation][0]
            ind1 = lig_recs_loc_indices[irec][istation][1]
            
            sim_ann_tas_lig_pi = lig_pi_tas_regrid_alltime[imodel]['am'][
                0, ind0, ind1].values
            
            sim_ann_tas_lig_pi_2std = lig_pi_tas_regrid_alltime[imodel]['ann'][
                :, ind0, ind1].values.std(ddof=1) * 2
            
            sim_rec_ann_tas_lig_pi = sim_ann_tas_lig_pi - rec_ann_tas_lig_pi
            
            AIS_ann_tas_site_values[irec] = pd.concat([
                AIS_ann_tas_site_values[irec],
                pd.DataFrame(data={
                    'Station': Station,
                    'Model': Model,
                    'Latitude': Latitude,
                    'Longitude': Longitude,
                    'rec_ann_tas_lig_pi': rec_ann_tas_lig_pi,
                    'rec_ann_tas_lig_pi_2sigma': rec_ann_tas_lig_pi_2sigma,
                    'sim_ann_tas_lig_pi': sim_ann_tas_lig_pi,
                    'sim_ann_tas_lig_pi_2std': sim_ann_tas_lig_pi_2std,
                    'sim_rec_ann_tas_lig_pi': sim_rec_ann_tas_lig_pi,
                }, index=[0])
            ], ignore_index=True,)

with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'wb') as f:
    pickle.dump(AIS_ann_tas_site_values, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'rb') as f:
    AIS_ann_tas_site_values = pickle.load(f)

irec = 'EC'
imodel = 'CNRM-CM6-1'
istation = 'EDC'

AIS_ann_tas_site_values[irec].loc[
    (AIS_ann_tas_site_values[irec].Model == imodel) & \
        (AIS_ann_tas_site_values[irec].Station == istation)]['rec_ann_tas_lig_pi']


lig_recs[irec]['AIS_am'].loc[
    lig_recs[irec]['AIS_am'].Station == istation]['127 ka Median PIAn [°C]']

AIS_ann_tas_site_values[irec].loc[
    (AIS_ann_tas_site_values[irec].Model == imodel) & \
        (AIS_ann_tas_site_values[irec].Station == istation)]['sim_ann_tas_lig_pi']

lig_pi_tas_regrid_alltime[imodel]['am'][
    0,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


'''
# endregion
# -----------------------------------------------------------------------------


