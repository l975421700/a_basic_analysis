

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

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

#-------------------------------- import reconstructions

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

#-------------------------------- import loc indices

with open('scratch/cmip6/lig/rec/lig_recs_loc_indices.pkl', 'rb') as f:
    lig_recs_loc_indices = pickle.load(f)



'''
lig_pi_sic_regrid_alltime.keys()

lig_recs['MC'].keys()
lig_recs['MC']['interpolated'].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract SO Sep SIC

recs = ['MC',]

so_sep_sic_recs = {
    'MC': lig_recs['MC']['interpolated'],
}
# 9

sic_anom_127_name = {
    'MC': 'sic_anom_hadisst_sep',
}

sic_anom_127_2sigma_name = {
    'MC': None,
}

SO_sep_sic_site_values = {}

for irec in recs:
    # irec = 'MC'
    print('#---------------- ' + irec)
    
    SO_sep_sic_site_values[irec] = pd.DataFrame(
        columns={
            'Station',
            'Model',
            'Latitude',
            'Longitude',
            'rec_sep_sic_lig_pi',
            'rec_sep_sic_lig_pi_2sigma',
            'sim_sep_sic_lig_pi',
            'sim_sep_sic_lig_pi_2std',
            'sim_rec_sep_sic_lig_pi',
            }
    )
    
    for istation in so_sep_sic_recs[irec].Station:
        # istation = so_sep_sic_recs[irec].Station.iloc[0]
        print('#---- ' + istation)
        
        for imodel in lig_pi_sic_regrid_alltime.keys():
            # imodel = 'ACCESS-ESM1-5'
            print('#-------- ' + imodel)
            
            Station = istation
            Model = imodel
            
            Latitude = so_sep_sic_recs[irec].loc[
                so_sep_sic_recs[irec].Station == istation].Latitude.iloc[0]
            Longitude = so_sep_sic_recs[irec].loc[
                so_sep_sic_recs[irec].Station == istation].Longitude.iloc[0]
            
            rec_sep_sic_lig_pi = so_sep_sic_recs[irec].loc[
                so_sep_sic_recs[irec].Station == istation][
                    sic_anom_127_name[irec]
                ].iloc[0]
            
            # if (sic_anom_127_2sigma_name[irec] is None):
            rec_sep_sic_lig_pi_2sigma = np.nan
            
            ind0 = lig_recs_loc_indices[irec][istation][0]
            ind1 = lig_recs_loc_indices[irec][istation][1]
            
            sim_sep_sic_lig_pi = lig_pi_sic_regrid_alltime[imodel]['mm'][
                8, ind0, ind1].values
            
            # lig_pi_sic_regrid_alltime[imodel]['mon'][8::12]
            sim_sep_sic_lig_pi_2std = lig_pi_sic_regrid_alltime[imodel]['mon'][
                8::12, ind0, ind1].values.std(ddof=1) * 2
            
            sim_rec_sep_sic_lig_pi = sim_sep_sic_lig_pi - rec_sep_sic_lig_pi
            
            SO_sep_sic_site_values[irec] = pd.concat([
                SO_sep_sic_site_values[irec],
                pd.DataFrame(data={
                    'Station': Station,
                    'Model': Model,
                    'Latitude': Latitude,
                    'Longitude': Longitude,
                    'rec_sep_sic_lig_pi': rec_sep_sic_lig_pi,
                    'rec_sep_sic_lig_pi_2sigma': rec_sep_sic_lig_pi_2sigma,
                    'sim_sep_sic_lig_pi': sim_sep_sic_lig_pi,
                    'sim_sep_sic_lig_pi_2std': sim_sep_sic_lig_pi_2std,
                    'sim_rec_sep_sic_lig_pi': sim_rec_sep_sic_lig_pi,
                }, index=[0])
            ], ignore_index=True,)

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'wb') as f:
    pickle.dump(SO_sep_sic_site_values, f)




'''
irec = 'MC'
model = 'HadGEM3-GC31-LL'
np.round(SO_sep_sic_site_values[irec].loc[
    SO_sep_sic_site_values[irec].Model == model
    ]['sim_rec_sep_sic_lig_pi'].mean(), 1)

#-------------------------------- check

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)

irec = 'MC'
imodel = 'ACCESS-ESM1-5'
istation = 'TPC288'

SO_sep_sic_site_values[irec].loc[
    (SO_sep_sic_site_values[irec].Model == imodel) & \
        (SO_sep_sic_site_values[irec].Station == istation)]['rec_sep_sic_lig_pi']

lig_recs[irec]['interpolated'].loc[
    lig_recs[irec]['interpolated'].Station == istation]['sic_anom_hadisst_sep']

SO_sep_sic_site_values[irec].loc[
    (SO_sep_sic_site_values[irec].Model == imodel) & \
        (SO_sep_sic_site_values[irec].Station == istation)]['sim_sep_sic_lig_pi']
lig_pi_sic_regrid_alltime[imodel]['mm'][
    8,
    lig_recs_loc_indices[irec][istation][0],
    lig_recs_loc_indices[irec][istation][1]].values


'''
# endregion
# -----------------------------------------------------------------------------

