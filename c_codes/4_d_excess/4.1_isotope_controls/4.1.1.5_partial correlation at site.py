

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
from scipy.stats import pearsonr
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
    regrid,
    mean_over_ais,
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
    monthini,
    plot_labels,
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
    xr_par_cor,
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


isotopes_alltime_icores = {}
temp2_alltime_icores = {}
pre_weighted_var_icores = {}

for i in range(len(expid)):
    print(i)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
        temp2_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
        pre_weighted_var_icores[expid[i]] = pickle.load(f)


par_corr_sources_isotopes = {}
par_corr_temp2_isotopes2 = {}

for i in range(len(expid)):
    print(i)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes.pkl', 'rb') as f:
        par_corr_sources_isotopes[expid[i]] = pickle.load(f)
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_temp2_isotopes2.pkl', 'rb') as f:
        par_corr_temp2_isotopes2[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region partial correlation d_ln, d_xs and source properties

icores = 'EDC'

cvar = 'wind10'

for i in [0, 1, 2, 3]:
    # i = 0
    print('#-------------------------------- ' + expid[i])
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ivar in ['sst']:
            # ivar = 'rh2m'
            # ['rh2m', 'wind10']
            print('#-------- ' + ivar)
            
            for ialltime in ['daily', 'mon', 'mm', 'ann']:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                ctl_var = pre_weighted_var_icores[expid[i]][icores][cvar][ialltime]
                
                r_squared = xr_par_cor(iso_var, src_var, ctl_var) ** 2
                p_value   = xr_par_cor(iso_var, src_var, ctl_var, output='p')
                
                if (p_value < 0.001):
                    plabel = '***'
                elif (p_value < 0.01):
                    plabel = '**'
                elif (p_value < 0.05):
                    plabel = '*'
                else:
                    plabel = ' '
                
                print(str(np.round(r_squared, 2)) + ' ' + plabel)
            
            print('#---- mon no mm')
            iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores]['mon']
            src_var = pre_weighted_var_icores[expid[i]][icores][ivar]['mon']
            ctl_var = pre_weighted_var_icores[expid[i]][icores][cvar]['mon']
            
            iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
            src_var = src_var.groupby('time.month') - src_var.groupby('time.month').mean(dim='time')
            ctl_var = ctl_var.groupby('time.month') - ctl_var.groupby('time.month').mean(dim='time')
            
            r_squared = xr_par_cor(iso_var, src_var, ctl_var) ** 2
            p_value   = xr_par_cor(iso_var, src_var, ctl_var, output='p')
            
            if (p_value < 0.001):
                plabel = '***'
            elif (p_value < 0.01):
                plabel = '**'
            elif (p_value < 0.05):
                plabel = '*'
            else:
                plabel = ' '
            
            print(str(np.round(r_squared, 2)) + ' ' + plabel)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region partial correlation d_ln, d_xs, dD and temp2

icores = 'EDC'

for i in [0,]:
    # i = 0
    print('#-------------------------------- ' + expid[i])
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_excess'
        print('#---------------- ' + iisotope)
        
        for iisotope1 in ['dD']:
            # iisotope1 = 'dD'
            print('#-------- ' + iisotope1)
            
            for ialltime in ['mon', 'mm', 'ann']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]
                
                r_squared = xr_par_cor(temp2_var, iso_var, ctl_var) ** 2
                p_value   = xr_par_cor(temp2_var, iso_var, ctl_var, output='p')
                
                if (p_value < 0.001):
                    plabel = '***'
                elif (p_value < 0.01):
                    plabel = '**'
                elif (p_value < 0.05):
                    plabel = '*'
                else:
                    plabel = ' '
                
                print(str(np.round(r_squared, 2)) + ' ' + plabel)
            
            print('#---- mon no mm')
            temp2_var = temp2_alltime_icores[expid[i]][icores]['mon']
            iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores]['mon']
            ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores]['mon']
            
            temp2_var = temp2_var.groupby('time.month') - temp2_var.groupby('time.month').mean(dim='time')
            iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
            ctl_var = ctl_var.groupby('time.month') - ctl_var.groupby('time.month').mean(dim='time')
            
            r_squared = xr_par_cor(temp2_var, iso_var, ctl_var) ** 2
            p_value   = xr_par_cor(temp2_var, iso_var, ctl_var, output='p')
            
            if (p_value < 0.001):
                plabel = '***'
            elif (p_value < 0.01):
                plabel = '**'
            elif (p_value < 0.05):
                plabel = '*'
            else:
                plabel = ' '
            
            print(str(np.round(r_squared, 2)) + ' ' + plabel)



'''
icores = 'EDC'
i = 0
iisotope = 'd_excess'
iisotope1 = 'dD'
ialltime = 'mon'
temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]
iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]

(pearsonr(temp2_var, iso_var).statistic) ** 2
xr_par_cor(temp2_var, iso_var, ctl_var) ** 2

(par_corr_temp2_isotopes2[expid[i]][iisotope][iisotope1]['mon']['r'][
    t63_sites_indices[icores]['ilat'],
    t63_sites_indices[icores]['ilon'],
].values) ** 2


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region partial correlation d_ln, dD and source SST

icores = 'EDC'

for i in [0, 1, 2, 3,]:
    # i = 0
    print('#-------------------------------- ' + expid[i])
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_excess'
        print('#---------------- ' + iisotope)
        
        for iisotope1 in ['dD',]:
            # iisotope1 = 'dD'
            print('#-------- ' + iisotope1)
            
            for ialltime in ['daily', 'mon', 'mm', 'ann']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                src_var = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]
                
                r_squared = xr_par_cor(src_var, iso_var, ctl_var) ** 2
                p_value   = xr_par_cor(src_var, iso_var, ctl_var, output='p')
                
                if (p_value < 0.001):
                    plabel = '***'
                elif (p_value < 0.01):
                    plabel = '**'
                elif (p_value < 0.05):
                    plabel = '*'
                else:
                    plabel = ' '
                
                print(str(np.round(r_squared, 2)) + ' ' + plabel)
            
            print('#---- mon no mm')
            src_var = pre_weighted_var_icores[expid[i]][icores]['sst']['mon']
            iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores]['mon']
            ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores]['mon']
            
            src_var = src_var.groupby('time.month') - src_var.groupby('time.month').mean(dim='time')
            iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
            ctl_var = ctl_var.groupby('time.month') - ctl_var.groupby('time.month').mean(dim='time')
            
            r_squared = xr_par_cor(src_var, iso_var, ctl_var) ** 2
            p_value   = xr_par_cor(src_var, iso_var, ctl_var, output='p')
            
            if (p_value < 0.001):
                plabel = '***'
            elif (p_value < 0.01):
                plabel = '**'
            elif (p_value < 0.05):
                plabel = '*'
            else:
                plabel = ' '
            
            print(str(np.round(r_squared, 2)) + ' ' + plabel)



'''
icores = 'EDC'
i = 0
iisotope = 'd_excess'
iisotope1 = 'dD'
ialltime = 'mon'
src_var = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]
iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
ctl_var = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]

(pearsonr(src_var, iso_var).statistic) ** 2
xr_par_cor(src_var, iso_var, ctl_var) ** 2

(par_corr_temp2_isotopes2[expid[i]][iisotope][iisotope1]['mon']['r'][
    t63_sites_indices[icores]['ilat'],
    t63_sites_indices[icores]['ilon'],
].values) ** 2


'''
# endregion
# -----------------------------------------------------------------------------
