

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


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression source SST = f(d_ln / d_xs)

regression_sst_d = {}

ivar = 'sst'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_sst_d[expid[i]] = {}
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        regression_sst_d[expid[i]][iisotope] = {}
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            regression_sst_d[expid[i]][iisotope][icores] = {}
            
            for ialltime in ['daily', 'mon', 'mm', 'ann',]:
                # ialltime = 'ann'
                print('#---- ' + ialltime)
                
                regression_sst_d[expid[i]][iisotope][icores][ialltime] = {}
                
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                subset = np.isfinite(iso_var) & np.isfinite(src_var)
                iso_var = iso_var[subset]
                src_var = src_var[subset]
                
                ols_fit = sm.OLS(
                    src_var.values,
                    sm.add_constant(iso_var.values),
                    ).fit()
                
                predicted_y = ols_fit.params[0] + ols_fit.params[1] * isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime].values
                RMSE = np.sqrt(np.average(np.square(predicted_y[subset] - src_var.values)))
                
                regression_sst_d[expid[i]][iisotope][icores][ialltime]['ols_fit'] = ols_fit.summary()
                regression_sst_d[expid[i]][iisotope][icores][ialltime]['params'] = ols_fit.params
                regression_sst_d[expid[i]][iisotope][icores][ialltime]['rsquared'] = ols_fit.rsquared
                regression_sst_d[expid[i]][iisotope][icores][ialltime]['predicted_y'] = predicted_y
                regression_sst_d[expid[i]][iisotope][icores][ialltime]['RMSE'] = RMSE
                
                if (ialltime == 'mon'):
                    print('#---- mon no mm')
                    
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm'] = {}
                    
                    iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
                    src_var = src_var.groupby('time.month') - src_var.groupby('time.month').mean(dim='time')
                    
                    ols_fit = sm.OLS(
                        src_var.values,
                        sm.add_constant(iso_var.values),
                        ).fit()
                    
                    predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values
                    RMSE = np.sqrt(np.average(np.square(predicted_y - src_var.values)))
                    
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm']['ols_fit'] = ols_fit.summary()
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm']['params'] = ols_fit.params
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm']['rsquared'] = ols_fit.rsquared
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm']['predicted_y'] = predicted_y
                    regression_sst_d[expid[i]][iisotope][icores]['mon no mm']['RMSE'] = RMSE
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d.pkl', 'wb') as f:
        pickle.dump(regression_sst_d[expid[i]], f)




'''
#-------------------------------- check

icores = 'EDC'

for i in range(1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_sst_d = {}
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d.pkl', 'rb') as f:
        regression_sst_d[expid[i]] = pickle.load(f)
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann',]:
            # ialltime = 'ann'
            print('#---- ' + ialltime)
            
            print(np.round(regression_sst_d[expid[i]][iisotope]['EDC'][ialltime]['rsquared'], 3))
            
            data1 = regression_sst_d[expid[i]][iisotope]['EDC'][ialltime]['predicted_y']
            data2 = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]
            subset = np.isfinite(data1) & np.isfinite(data2)
            data1 = data1[subset]
            data2 = data2[subset]
            
            print(pearsonr(data1, data2,).statistic ** 2)




                # print(ols_fit.summary())
                # print("Parameters: ", ols_fit.params)
                # print("R2: ", ols_fit.rsquared)
                # print(pearsonr(src_var.values, predicted_y).statistic ** 2)
                # print(RMSE)
                

            ols_fit = sm.OLS(
                iso_var.values,
                sm.add_constant(src_var.values),
                ).fit()

            linearfit = linregress(x = iso_var.values, y = src_var.values,)
            print(linearfit)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression source SST = f(d_ln / d_xs, dD)

regression_sst_d_dD = {}

ivar = 'sst'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_sst_d_dD[expid[i]] = {}
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        regression_sst_d_dD[expid[i]][iisotope] = {}
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            regression_sst_d_dD[expid[i]][iisotope][icores] = {}
            
            for ialltime in ['daily', 'mon', 'mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime] = {}
                
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                dD_var = isotopes_alltime_icores[expid[i]]['dD'][icores][ialltime]
                subset = np.isfinite(iso_var) & np.isfinite(src_var) & np.isfinite(dD_var)
                iso_var = iso_var[subset]
                src_var = src_var[subset]
                dD_var = dD_var[subset]
                
                ols_fit = sm.OLS(
                    src_var.values,
                    sm.add_constant(np.column_stack((iso_var.values, dD_var.values))),
                    ).fit()
                
                # print(ols_fit.summary())
                # print("Parameters: ", ols_fit.params)
                # print("R2: ", ols_fit.rsquared)
                # print(pearsonr(src_var.values, predicted_y).statistic ** 2)
                # print(RMSE)
                
                predicted_y = ols_fit.params[0] + ols_fit.params[1] * isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime].values + ols_fit.params[2] * isotopes_alltime_icores[expid[i]]['dD'][icores][ialltime].values
                RMSE = np.sqrt(np.average(np.square(predicted_y[subset] - src_var.values)))
                
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['ols_fit'] = ols_fit.summary()
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['params'] = ols_fit.params
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['rsquared'] = ols_fit.rsquared
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['predicted_y'] = predicted_y
                regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['RMSE'] = RMSE
                
                if (ialltime == 'mon'):
                    print('#---- mon no mm')
                    
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm'] = {}
                    
                    iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
                    src_var = src_var.groupby('time.month') - src_var.groupby('time.month').mean(dim='time')
                    dD_var = dD_var.groupby('time.month') - dD_var.groupby('time.month').mean(dim='time')
                    
                    ols_fit = sm.OLS(
                        src_var.values,
                        sm.add_constant(np.column_stack((iso_var.values, dD_var.values))),
                        ).fit()
                    
                    predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values + ols_fit.params[2] * dD_var.values
                    RMSE = np.sqrt(np.average(np.square(predicted_y - src_var.values)))
                    
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm']['ols_fit'] = ols_fit.summary()
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm']['params'] = ols_fit.params
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm']['rsquared'] = ols_fit.rsquared
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm']['predicted_y'] = predicted_y
                    regression_sst_d_dD[expid[i]][iisotope][icores]['mon no mm']['RMSE'] = RMSE
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_dD.pkl', 'wb') as f:
        pickle.dump(regression_sst_d_dD[expid[i]], f)




'''
#-------------------------------- check

for i in range(1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_sst_d_dD = {}
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_dD.pkl', 'rb') as f:
        regression_sst_d_dD[expid[i]] = pickle.load(f)
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann',]:
            # ialltime = 'ann'
            # 
            print('#---- ' + ialltime)
            
            print(np.round(regression_sst_d_dD[expid[i]][iisotope]['EDC'][ialltime]['rsquared'], 3))
            
            data1 = regression_sst_d_dD[expid[i]][iisotope]['EDC'][ialltime]['predicted_y']
            data2 = pre_weighted_var_icores[expid[i]]['EDC']['sst'][ialltime]
            subset = np.isfinite(data1) & np.isfinite(data2)
            data1 = data1[subset]
            data2 = data2[subset]
            
            print(pearsonr(data1, data2,).statistic ** 2)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression temp2 = f(dD / dO18)

regression_temp2_delta = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_temp2_delta[expid[i]] = {}
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        regression_temp2_delta[expid[i]][iisotope] = {}
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            regression_temp2_delta[expid[i]][iisotope][icores] = {}
            
            for ialltime in ['mon', 'mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime] = {}
                
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]
                
                ols_fit = sm.OLS(
                    temp2_var.values,
                    sm.add_constant(iso_var.values),
                    ).fit()
                
                predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values
                RMSE = np.sqrt(np.average(np.square(predicted_y - temp2_var.values)))
                
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['ols_fit'] = ols_fit.summary()
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['params'] = ols_fit.params
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['rsquared'] = ols_fit.rsquared
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['predicted_y'] = predicted_y
                regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['RMSE'] = RMSE
                
                if (ialltime == 'mon'):
                    print('#---- mon no mm')
                    
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm'] = {}
                    
                    iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
                    temp2_var = temp2_var.groupby('time.month') - temp2_var.groupby('time.month').mean(dim='time')
                    
                    ols_fit = sm.OLS(
                        temp2_var.values,
                        sm.add_constant(iso_var.values),
                        ).fit()
                    
                    predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values
                    RMSE = np.sqrt(np.average(np.square(predicted_y - temp2_var.values)))
                    
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm']['ols_fit'] = ols_fit.summary()
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm']['params'] = ols_fit.params
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm']['rsquared'] = ols_fit.rsquared
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm']['predicted_y'] = predicted_y
                    regression_temp2_delta[expid[i]][iisotope][icores]['mon no mm']['RMSE'] = RMSE
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta.pkl', 'wb') as f:
        pickle.dump(regression_temp2_delta[expid[i]], f)




'''
#-------------------------------- check

icores = 'EDC'

for i in range(1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_temp2_delta = {}
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta.pkl', 'rb') as f:
        regression_temp2_delta[expid[i]] = pickle.load(f)
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['mon', 'mm', 'ann',]:
            # ialltime = 'ann'
            # , 'mon no mm'
            print('#---- ' + ialltime)
            
            print(np.round(regression_temp2_delta[expid[i]][iisotope]['EDC'][ialltime]['rsquared'], 3))
            
            data1 = regression_temp2_delta[expid[i]][iisotope]['EDC'][ialltime]['predicted_y']
            data2 = temp2_alltime_icores[expid[i]][icores][ialltime]
            
            print(pearsonr(data1, data2,).statistic ** 2)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regression temp2 = f(dD / dO18, d_ln / d_xs)


regression_temp2_delta_d = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_temp2_delta_d[expid[i]] = {}
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        regression_temp2_delta_d[expid[i]][iisotope] = {}
        
        for iisotope1 in ['d_ln', 'd_excess']:
            # iisotope1 = 'd_ln'
            print('#---------------- ' + iisotope1)
            
            regression_temp2_delta_d[expid[i]][iisotope][iisotope1] = {}
            
            for icores in ['EDC',]:
                # icores = 'EDC'
                print('#-------- ' + icores)
                
                regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores] = {}
                
                for ialltime in ['mon', 'mm', 'ann',]:
                    # ialltime = 'mon'
                    print('#---- ' + ialltime)
                    
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime] = {}
                    
                    iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                    iso_var1 = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]
                    temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]
                    
                    ols_fit = sm.OLS(
                        temp2_var.values,
                        sm.add_constant(np.column_stack((iso_var.values, iso_var1.values))),
                        ).fit()
                    
                    predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values + ols_fit.params[2] * iso_var1.values
                    RMSE = np.sqrt(np.average(np.square(predicted_y - temp2_var.values)))
                    
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime]['ols_fit'] = ols_fit.summary()
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime]['params'] = ols_fit.params
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime]['rsquared'] = ols_fit.rsquared
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime]['predicted_y'] = predicted_y
                    regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores][ialltime]['RMSE'] = RMSE
                    
                    if (ialltime == 'mon'):
                        print('#---- mon no mm')
                        
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm'] = {}
                        
                        iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean(dim='time')
                        iso_var1 = iso_var1.groupby('time.month') - iso_var1.groupby('time.month').mean(dim='time')
                        temp2_var = temp2_var.groupby('time.month') - temp2_var.groupby('time.month').mean(dim='time')
                        
                        ols_fit = sm.OLS(
                            temp2_var.values,
                            sm.add_constant(np.column_stack((iso_var.values, iso_var1.values))),
                            ).fit()
                        
                        predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var.values + ols_fit.params[2] * iso_var1.values
                        RMSE = np.sqrt(np.average(np.square(predicted_y - temp2_var.values)))
                        
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm']['ols_fit'] = ols_fit.summary()
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm']['params'] = ols_fit.params
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm']['rsquared'] = ols_fit.rsquared
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm']['predicted_y'] = predicted_y
                        regression_temp2_delta_d[expid[i]][iisotope][iisotope1][icores]['mon no mm']['RMSE'] = RMSE
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_d.pkl', 'wb') as f:
        pickle.dump(regression_temp2_delta_d[expid[i]], f)




'''
#-------------------------------- check

for i in range(1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    regression_temp2_delta_d = {}
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_d.pkl', 'rb') as f:
        regression_temp2_delta_d[expid[i]] = pickle.load(f)
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for iisotope1 in ['d_ln', 'd_excess']:
            # iisotope1 = 'd_ln'
            print('#---------------- ' + iisotope1)
            
            for ialltime in ['mon', 'mm', 'ann',]:
                # ialltime = 'ann'
                # , 'mon no mm'
                print('#---- ' + ialltime)
                
                print(np.round(regression_temp2_delta_d[expid[i]][iisotope][iisotope1]['EDC'][ialltime]['rsquared'], 3))
                
                data1 = regression_temp2_delta_d[expid[i]][iisotope][iisotope1]['EDC'][ialltime]['predicted_y']
                data2 = temp2_alltime_icores[expid[i]]['EDC'][ialltime]
                
                print(pearsonr(data1, data2,).statistic ** 2)

'''
# endregion
# -----------------------------------------------------------------------------



