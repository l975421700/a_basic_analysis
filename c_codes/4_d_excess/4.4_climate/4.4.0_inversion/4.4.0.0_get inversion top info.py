

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import linregress

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
import cartopy.feature as cfeature
from matplotlib.ticker import AutoMinorLocator
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
    mean_over_ais,
    time_weighted_mean,
    inversion_top,
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
    cplot_ttest,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

zh_st_ml = {}
for i in range(len(expid)):
    print(str(i) + ' ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'rb') as f:
        zh_st_ml[expid[i]] = pickle.load(f)

temp2_alltime = {}
for i in range(len(expid)):
    print(str(i) + ' ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
        temp2_alltime[expid[i]] = pickle.load(f)

echam6_t63_geosp = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/pi_600_5.0/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get inversion height and strength

i = 0
inversion_height_strength = {}

for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------------------------------- ' + isite)
    
    inversion_height_strength[isite] = {}
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    site_elevation = echam6_t63_surface_height[ilat, ilon].values
    
    for ialltime in ['mon', 'mm', 'sea', 'sm', 'ann']:
        # ialltime = 'mon'
        # ialltime = 'sm'
        print('#---------------- ' + ialltime)
        
        inversion_height_strength[isite][ialltime] = {}
        
        nd_t_it = np.zeros(temp2_alltime[expid[i]][ialltime].shape[0])
        nd_h_it = np.zeros(temp2_alltime[expid[i]][ialltime].shape[0])
        nd_tas = np.zeros(temp2_alltime[expid[i]][ialltime].shape[0])
        nd_strength = np.zeros(temp2_alltime[expid[i]][ialltime].shape[0])
        
        for itime in range(temp2_alltime[expid[i]][ialltime].shape[0]):
            # itime = 0
            print('#-------- ' + str(itime))
            
            temperature = zh_st_ml[expid[i]]['st'][ialltime][itime, :, ilat, ilon].values
            height = zh_st_ml[expid[i]]['zh'][ialltime][itime, :, ilat, ilon].values / 1000
            tas = temp2_alltime[expid[i]][ialltime][itime, ilat, ilon].values + zerok
            
            t_it, h_it = inversion_top(temperature, height)
            
            nd_t_it[itime] = t_it
            nd_h_it[itime] = h_it * 1000 - site_elevation
            nd_tas[itime] = tas
            nd_strength[itime] = nd_t_it[itime] - nd_tas[itime]
        
        if (ialltime in ['mon', 'sea', 'ann']):
            time = temp2_alltime[expid[i]][ialltime].time
        elif (ialltime == 'mm'):
            time = month[temp2_alltime[expid[i]]['mm'].month - 1]
        elif (ialltime == 'sm'):
            time = temp2_alltime[expid[i]]['sm'].season.values
        
        inversion_height_strength[isite][ialltime]['Inversion temperature'] = \
            xr.DataArray(
                data = nd_t_it,
                dims=['time'], coords=dict(time=time), attrs=dict(units='K',),)
        inversion_height_strength[isite][ialltime]['Inversion height'] = \
            xr.DataArray(
                data = nd_h_it,
                dims=['time'], coords=dict(time=time), attrs=dict(units='m',),)
        inversion_height_strength[isite][ialltime]['Surface temperature'] = \
            xr.DataArray(
                data = nd_tas,
                dims=['time'], coords=dict(time=time), attrs=dict(units='K',),)
        inversion_height_strength[isite][ialltime]['Inversion strength'] = \
            xr.DataArray(
                data = nd_strength,
                dims=['time'], coords=dict(time=time), attrs=dict(units='K',),)


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.inversion_height_strength.pkl', 'wb') as f:
    pickle.dump(inversion_height_strength, f)




'''
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.inversion_height_strength.pkl', 'rb') as f:
    inversion_height_strength = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------

