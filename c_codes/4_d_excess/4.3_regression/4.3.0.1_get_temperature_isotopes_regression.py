

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
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
sys.path.append('/albedo/work/user/qigao001')

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
    xr_regression_y_x1,
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

dO18_alltime = {}
dD_alltime = {}
d_ln_alltime = {}
d_excess_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)


source_var = ['sst', ]
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_sst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)

temp2_alltime = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
        temp2_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression source SST = f(d_ln / d_xs)

regression_sst_d_AIS = {}

ivar = 'sst'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_sst_d_AIS[expid[i]] = {}
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        regression_sst_d_AIS[expid[i]][iisotope] = {}
        
        for ialltime in ['daily', 'mon', 'ann',]:
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            regression_sst_d_AIS[expid[i]][iisotope][ialltime] = {}
            
            if (iisotope == 'd_ln'):
                iso_var = d_ln_alltime[expid[i]][ialltime]
            elif (iisotope == 'd_excess'):
                iso_var = d_excess_alltime[expid[i]][ialltime]
            
            src_var = pre_weighted_var[expid[i]][ivar][ialltime]
            
            for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
                # ioutput = 'RMSE'
                print('#---- ' + ioutput)
                
                regression_sst_d_AIS[expid[i]][iisotope][ialltime][ioutput] = \
                    xr.apply_ufunc(
                        xr_regression_y_x1,
                        src_var, iso_var,
                        input_core_dims=[['time'], ['time']],
                        kwargs={'output': ioutput},
                        dask = 'allowed', vectorize = True,
                    )
        
        print('#-------- mon no mm')
        
        regression_sst_d_AIS[expid[i]][iisotope]['mon no mm'] = {}
        
        if (iisotope == 'd_ln'):
            iso_var = d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
                d_ln_alltime[expid[i]]['mon'].groupby('time.month').mean()
        elif (iisotope == 'd_excess'):
            iso_var = d_excess_alltime[expid[i]]['mon'].groupby('time.month')-\
                d_excess_alltime[expid[i]]['mon'].groupby('time.month').mean()
        
        src_var=pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month')-\
            pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month').mean()
        
        for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
            # ioutput = 'RMSE'
            print('#---- ' + ioutput)
            
            regression_sst_d_AIS[expid[i]][iisotope]['mon no mm'][ioutput] = \
                xr.apply_ufunc(
                    xr_regression_y_x1,
                    src_var, iso_var,
                    input_core_dims=[['time'], ['time']],
                    kwargs={'output': ioutput},
                    dask = 'allowed', vectorize = True,
                )
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_AIS.pkl', 'wb') as f:
        pickle.dump(regression_sst_d_AIS[expid[i]], f)




'''
#-------------------------------- check
ivar = 'sst'

regression_sst_d = {}
regression_sst_d_AIS = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_AIS.pkl', 'rb') as f:
        regression_sst_d_AIS[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d.pkl', 'rb') as f:
        regression_sst_d[expid[i]] = pickle.load(f)
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['daily', 'mon', 'mon no mm', 'ann',]:
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
                # ioutput = 'RMSE'
                print('#---- ' + ioutput)
                
                data1 = regression_sst_d_AIS[expid[i]][iisotope][ialltime][ioutput][
                    t63_sites_indices['EDC']['ilat'],
                    t63_sites_indices['EDC']['ilon'],
                    ].values
                
                if (ioutput in ['rsquared', 'RMSE']):
                    data2 = regression_sst_d[expid[i]][iisotope]['EDC'][ialltime][ioutput]
                elif (ioutput == 'slope'):
                    data2 = regression_sst_d[expid[i]][iisotope]['EDC'][ialltime]['params'][1]
                elif (ioutput == 'intercept'):
                    data2 = regression_sst_d[expid[i]][iisotope]['EDC'][ialltime]['params'][0]
                
                if (data1 != data2):
                    print(np.round(data1, 2))
                    print(np.round(data2, 2))


regression_sst_d_AIS[expid[i]]['d_ln']['mon']['RMSE']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression temp2 = f(dD / dO18)

regression_temp2_delta_AIS = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_temp2_delta_AIS[expid[i]] = {}
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        regression_temp2_delta_AIS[expid[i]][iisotope] = {}
        
        for ialltime in ['mon', 'ann',]:
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            regression_temp2_delta_AIS[expid[i]][iisotope][ialltime] = {}
            
            if (iisotope == 'dO18'):
                iso_var = dO18_alltime[expid[i]][ialltime]
            elif (iisotope == 'dD'):
                iso_var = dD_alltime[expid[i]][ialltime]
            
            temp2_var = temp2_alltime[expid[i]][ialltime]
            temp2_var['time'] = iso_var['time']
            
            for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
                # ioutput = 'RMSE'
                print('#---- ' + ioutput)
                
                regression_temp2_delta_AIS[expid[i]][iisotope][ialltime][ioutput] = \
                    xr.apply_ufunc(
                        xr_regression_y_x1,
                        temp2_var, iso_var,
                        input_core_dims=[['time'], ['time']],
                        kwargs={'output': ioutput},
                        dask = 'allowed', vectorize = True,
                    )
        
        print('#-------- mon no mm')
        
        regression_temp2_delta_AIS[expid[i]][iisotope]['mon no mm'] = {}
        
        if (iisotope == 'dO18'):
            iso_var = dO18_alltime[expid[i]]['mon'].groupby('time.month') - \
                dO18_alltime[expid[i]]['mon'].groupby('time.month').mean()
        elif (iisotope == 'dD'):
            iso_var = dD_alltime[expid[i]]['mon'].groupby('time.month')-\
                dD_alltime[expid[i]]['mon'].groupby('time.month').mean()
        
        temp2_var=temp2_alltime[expid[i]]['mon'].groupby('time.month')-\
            temp2_alltime[expid[i]]['mon'].groupby('time.month').mean()
        
        for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
            # ioutput = 'RMSE'
            print('#---- ' + ioutput)
            
            regression_temp2_delta_AIS[expid[i]][iisotope]['mon no mm'][ioutput] = \
                xr.apply_ufunc(
                    xr_regression_y_x1,
                    temp2_var, iso_var,
                    input_core_dims=[['time'], ['time']],
                    kwargs={'output': ioutput},
                    dask = 'allowed', vectorize = True,
                )
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_AIS.pkl', 'wb') as f:
        pickle.dump(regression_temp2_delta_AIS[expid[i]], f)




'''
#-------------------------------- check

regression_temp2_delta = {}
regression_temp2_delta_AIS = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_AIS.pkl', 'rb') as f:
        regression_temp2_delta_AIS[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta.pkl', 'rb') as f:
        regression_temp2_delta[expid[i]] = pickle.load(f)
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['mon', 'mon no mm', 'ann',]:
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            for ioutput in ['rsquared', 'RMSE', 'slope', 'intercept']:
                # ioutput = 'RMSE'
                print('#---- ' + ioutput)
                
                data1 = regression_temp2_delta_AIS[expid[i]][iisotope][ialltime][ioutput][
                    t63_sites_indices['EDC']['ilat'],
                    t63_sites_indices['EDC']['ilon'],
                    ].values
                
                if (ioutput in ['rsquared', 'RMSE']):
                    data2 = regression_temp2_delta[expid[i]][iisotope]['EDC'][ialltime][ioutput]
                elif (ioutput == 'slope'):
                    data2 = regression_temp2_delta[expid[i]][iisotope]['EDC'][ialltime]['params'][1]
                elif (ioutput == 'intercept'):
                    data2 = regression_temp2_delta[expid[i]][iisotope]['EDC'][ialltime]['params'][0]
                
                if (data1 != data2):
                    print(np.round(data1, 2))
                    print(np.round(data2, 2))


regression_temp2_delta_AIS[expid[i]]['dD']['mon']['RMSE']
'''
# endregion
# -----------------------------------------------------------------------------

