

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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
from matplotlib.ticker import AutoMinorLocator
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

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
    plot_labels,
    plot_labels_no_unit,
    plot_labels_only_unit,
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

regression_sst_d = {}
regression_sst_d_dD = {}
# regression_temp2_delta = {}
# regression_temp2_delta_d = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d.pkl', 'rb') as f:
        regression_sst_d[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_dD.pkl', 'rb') as f:
        regression_sst_d_dD[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta.pkl', 'rb') as f:
    #     regression_temp2_delta[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_d.pkl', 'rb') as f:
    #     regression_temp2_delta_d[expid[i]] = pickle.load(f)


isotopes_alltime_icores = {}
# temp2_alltime_icores = {}
pre_weighted_var_icores = {}
# wisoaprt_alltime_icores = {}

for i in range(len(expid)):
    print(i)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    # with open(
    #     exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
    #     temp2_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
        pre_weighted_var_icores[expid[i]] = pickle.load(f)
    
    # with open(
    #     exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    #     wisoaprt_alltime_icores[expid[i]] = pickle.load(f)


# aprt_frc_alltime_icores = {}
# for i in range(len(expid)):
#     print(i)
    
#     with open(
#         exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl', 'rb') as f:
#         aprt_frc_alltime_icores[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)


regression_sst_d[expid[i]]['d_ln']['EDC']['ann']['RMSE']
regression_sst_d_dD[expid[i]]['d_excess']['EDC']['ann']['RMSE']
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructed source SST based on d_ln / d_xs

ivar = 'sst'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am', ]:
                # ialltime = 'mon'
                # ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']
                print('#---- ' + ialltime)
                
                params = regression_sst_d[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_sst_d[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_sst_d[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_sst_d[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                
                # if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                #     sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                # elif (ialltime == 'mon no mm'):
                #     sim_src = pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month') - pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(len(rec_src))
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=12, lw=1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                # plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',
                #          va='bottom', ha='right')
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] > 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] < 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] == 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                
                plt.text(
                    0.05, 0.95, eq_text,
                    transform=ax.transAxes, linespacing=2,
                    va='top', ha='left',)
                
                ax.set_xlabel('Simulated ' + plot_labels[ivar],)
                ax.set_xlim(xymin-0.1, xymax+0.1)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel('Estimated ' + plot_labels[ivar],)
                ax.set_ylim(xymin-0.1, xymax+0.1)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both',)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.16, right=0.98, bottom=0.16, top=0.98)
                fig.savefig(output_png)




'''
                xmax_value = np.max(sim_src)
                xmin_value = np.min(sim_src)
                ymax_value = np.max(rec_src)
                ymin_value = np.min(rec_src)
                ax.set_xlim(xmin_value, xmax_value)
                ax.set_ylim(ymin_value, ymax_value)


regression_sst_d[expid[i]]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructed source SST based on d_ln / d_xs and dD

ivar = 'sst'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln', 'd_excess',]:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                
                # if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                #     sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                # elif (ialltime == 'mon no mm'):
                #     sim_src = pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month') - pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.1_sst_d_dD/8.1.6.1.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + ' and dD.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=12, lw=1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                # plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] > 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] < 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                elif (params[0] == 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[ivar]
                
                plt.text(
                    0.05, 0.95, eq_text,
                    transform=ax.transAxes, linespacing=2,
                    va='top', ha='left',)
                
                ax.set_xlabel('Simulated ' + plot_labels[ivar],)
                ax.set_xlim(xymin-0.1, xymax+0.1)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel('Estimated ' + plot_labels[ivar],)
                ax.set_ylim(xymin-0.1, xymax+0.1)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both',)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.16, right=0.98, bottom=0.16, top=0.98)
                fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check regression statistics


icores = 'EDC'
iisotope = 'd_ln'
ialltime = 'mon'

# ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    print('#---------------- rsquared')
    print(np.round(regression_sst_d[expid[i]][iisotope][icores][ialltime]['rsquared'], 2))
    
    print('#---------------- RMSE')
    print(np.round(regression_sst_d[expid[i]][iisotope][icores][ialltime]['RMSE'], 2))
    
    print('#---------------- Slope')
    print(np.round(regression_sst_d[expid[i]][iisotope][icores][ialltime]['params'][1], 2))
    
    print('#---------------- ols_fit')
    print(regression_sst_d[expid[i]][iisotope][icores][ialltime]['ols_fit'])



# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot reconstructed temp2 based on dD / dO18


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['dD',]:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                
                # if (ialltime in ['mon', 'mm', 'ann',]):
                #     sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                # elif (ialltime == 'mon no mm'):
                #     sim_src = temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month') - temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.2_temp2_delta/8.1.6.2.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec temp2 using '+ iisotope + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=6, lw=0.1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] > 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] < 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] == 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                
                plt.text(
                    0.1, 0.05, eq_text,
                    transform=ax.transAxes, fontsize=6, linespacing=1.5)
                
                ax.set_xlabel(
                    'Simulated ' + plot_labels['temp2'],
                    labelpad=2, fontsize=8)
                ax.set_xlim(xymin, xymax)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel(
                    'Derived ' + plot_labels['temp2'],
                    labelpad=2, fontsize=8)
                ax.set_ylim(xymin, xymax)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.25, top=0.95)
                fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructed temp2 based on dD / dO18 and d_ln / d_xs

isotope1 = 'dD'

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['params']
                rsquared = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['predicted_y']
                
                sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                
                # if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                #     sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                # elif (ialltime == 'mon no mm'):
                #     sim_src = temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month') - temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.3_temp2_delta_d/8.1.6.3.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec temp2 using dD and '+ iisotope + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=6, lw=0.1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] > 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] < 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit[iisotope] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] == 0):
                    eq_text = 'temp2 = ' + str(np.round(params[1], 2)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                
                plt.text(
                    0.1, 0.05, eq_text,
                    transform=ax.transAxes, fontsize=4, linespacing=1.5)
                
                ax.set_xlabel(
                    'Simulated ' + plot_labels['temp2'],
                    labelpad=2, fontsize=8)
                ax.set_xlim(xymin, xymax)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel(
                    'Derived ' + plot_labels['temp2'],
                    labelpad=2, fontsize=8)
                ax.set_ylim(xymin, xymax)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.25, top=0.95)
                fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructed source SST based on d_ln / d_xs, controlling aprt_frc, wisoaprt

ivar = 'sst'
scatter_size = 1

# frc_threshold = 80
wisoaprt_threshold = 0.05

# pltlevel = np.array([0.05, 0.1, 0.5, 1, 2, 4, 8,])
# pltticks = np.array([0.05, 0.1, 0.5, 1, 2, 4, 8,])
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

ivar1 = 'd_excess'
if (ivar1 == 'dO18'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -80, cm_max = -40, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'dD'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -500, cm_max = -300, cm_interval1 = 20, cm_interval2 = 40,
        cmap = 'viridis_r',)
elif (ivar1 == 'd_ln'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -20, cm_max = 20, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'd_excess'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -20, cm_max = 20, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'sst'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 0, cm_max = 24, cm_interval1 = 2, cm_interval2 = 4,
        cmap = 'viridis_r',)
elif (ivar1 == 'rh2m'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 70, cm_max = 90, cm_interval1 = 2, cm_interval2 = 4,
        cmap = 'viridis_r',)
elif (ivar1 == 'lat'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -60, cm_max = -20, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'wind10'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 4, cm_max = 16, cm_interval1 = 1, cm_interval2 = 2,
        cmap = 'viridis_r',)


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_ln'
        # ['d_ln', 'd_excess']
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily']:
                # ialltime = 'mon'
                # ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']
                print('#---- ' + ialltime)
                
                src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime].copy()
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime].copy()
                # aprt_frc = aprt_frc_alltime_icores[expid[i]][icores][ialltime]
                wisoaprt_var = wisoaprt_alltime_icores[expid[i]][icores][ialltime].copy()
                # subset = (np.isfinite(src_var) & np.isfinite(iso_var)) & (aprt_frc >= frc_threshold)
                subset = (np.isfinite(src_var) & np.isfinite(iso_var)) & (wisoaprt_var >= (wisoaprt_threshold / 2.628e6))
                src_var = src_var[subset]
                iso_var = iso_var[subset]
                
                # color_var = wisoaprt_var[subset] * 2.628e6
                # color_var = pre_weighted_var_icores[expid[i]][icores][ivar1][ialltime][subset]
                color_var = isotopes_alltime_icores[expid[i]][ivar1][icores][ialltime][subset]
                
                ols_fit = sm.OLS(
                    src_var.values,
                    sm.add_constant(iso_var.values),
                    ).fit()
                
                params = ols_fit.params
                rsquared = ols_fit.rsquared
                predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var
                RMSE = np.sqrt(np.average(np.square(predicted_y - src_var)))
                
                xymax = np.max(np.concatenate((src_var, predicted_y)))
                xymin = np.min(np.concatenate((src_var, predicted_y)))
                
                # output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + '_frc' + str(frc_threshold) + '.png'
                # output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + '_aprt' + str(wisoaprt_threshold) + '.png'
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + '_color_' + str(ivar1) + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 5]) / 2.54)
                
                plt_scatter = ax.scatter(
                    src_var, predicted_y,
                    s=scatter_size, lw=0.1,
                    c= color_var, norm=pltnorm, cmap=pltcmp,
                    # facecolors='white', edgecolors='k',
                    )
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] > 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] < 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] == 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                
                plt.text(
                    0.1, 0.05, eq_text,
                    transform=ax.transAxes, fontsize=6, linespacing=1.5)
                
                ax.set_xlabel(
                    'Simulated ' + plot_labels[ivar],
                    labelpad=2, fontsize=8)
                ax.set_xlim(xymin, xymax)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel(
                    'Derived ' + plot_labels[ivar],
                    labelpad=2, fontsize=8)
                ax.set_ylim(xymin, xymax)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                cbar = fig.colorbar(
                    plt_scatter, ax=ax,
                    orientation="horizontal",shrink=1.4,aspect=30,
                    anchor=(1, -3), extend='both',
                    pad=0.25, fraction=0.03,
                    ticks=pltticks, format=remove_trailing_zero_pos, )
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(
                    # 'Precipitation [$mm \; day^{-1}$]',
                    plot_labels[ivar1],
                    fontsize=8, linespacing=1.5)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.22, top=0.95)
                fig.savefig(output_png)





'''
#-------- check excluded wisoaprt
i = 0
icores = 'EDC'
ialltime = 'daily'
wisoaprt_threshold = 1

total_sum = wisoaprt_alltime_icores[expid[i]][icores][ialltime].values.sum()
threshold_sum = wisoaprt_alltime_icores[expid[i]][icores][ialltime].where(
    wisoaprt_alltime_icores[expid[i]][icores][ialltime] >= (wisoaprt_threshold / 2.628e6),
    0).sum()
excluded_frc = np.round((total_sum - threshold_sum).values / total_sum * 100, 1)
print('Excluded pre in this threshold: ' + str(excluded_frc) + '%')



                linearfit = linregress(x = src_var, y = iso_var,)
                ols_fit = sm.OLS(
                    iso_var.values,
                    sm.add_constant(src_var.values),
                    ).fit()
                print(ols_fit.summary())
                print("R2: ", ols_fit.rsquared)
                pearsonr(src_var, iso_var).statistic **2
                
                linearfit = linregress(x = iso_var, y = src_var,)
                print(linearfit)
                ols_fit = sm.OLS(
                    src_var.values,
                    sm.add_constant(iso_var.values),
                    ).fit()
                print(ols_fit.summary())
                print("R2: ", ols_fit.rsquared)
                pearsonr(iso_var, src_var).statistic **2
                # 2.4664 * 0.3082

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region detailed regression analysis, source SST vs. d_ln

icores = 'EDC'
# 'EDC', 'DOME F', 'Vostok', 'EDML',
ivar = 'sst'
iisotope = 'd_ln'
ialltime = 'ann no am'

i = 0

src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
subset = (np.isfinite(src_var) & np.isfinite(iso_var))
src_var = src_var[subset]
iso_var = iso_var[subset]

ols_fit = sm.OLS(
    src_var.values,
    sm.add_constant(iso_var.values),
    ).fit()

params = ols_fit.params
rsquared = ols_fit.rsquared
predicted_y = ols_fit.params[0] + ols_fit.params[1] * iso_var
RMSE = np.sqrt(np.average(np.square(predicted_y - src_var)))

print(np.round(rsquared, 2))
print(np.round(RMSE, 2))
print(ols_fit.summary())

# slope: 0.3082 (95% interval: 0.263 to 0.354)

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    print(np.round(regression_sst_d[expid[i]]['d_ln'][icores]['ann no am']['rsquared'], 2))
    print(np.round(regression_sst_d[expid[i]]['d_ln'][icores]['ann no am']['RMSE'], 2))
    print(np.round(regression_sst_d[expid[i]]['d_ln'][icores]['ann no am']['params'][1], 2))



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region daily wisoaprt vs. isotopes

i = 0
icores = 'EDC'
ialltime = 'daily'

xticks = np.array([0.01, 0.1, 1, 10, 100])
xticklabels = np.array(['0.01', '0.1', '1', '10', '100'])

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


for iisotope in ['d_ln', 'd_excess', 'dO18', 'dD']:
    # iisotope = 'd_ln'
    # 'd_ln', 'd_excess', 'dO18', 'dD'
    print('#---------------- ' + iisotope)
    
    aprt_var = wisoaprt_alltime_icores[expid[i]][icores][ialltime] * 2.628e6
    iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
    
    subset = np.isfinite(aprt_var) & np.isfinite(iso_var)
    
    aprt_var = aprt_var[subset]
    iso_var = iso_var[subset]
    
    output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.1 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' wisoaprt vs. '+ iisotope + '.png'
    
    # fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    fig = plt.figure(figsize=np.array([4.4, 4]) / 2.54,)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
    ax.scatter_density(
        aprt_var, iso_var,
        cmap=white_viridis)
    # ax.scatter(
    #     aprt_var, iso_var,
    #     s=1, lw=0.1, facecolors='white', edgecolors='k', alpha=0.5)
    plt.text(0.7, 0.05, icores, transform=ax.transAxes, color='k',)
    
    ax.set_xlabel(plot_labels['wisoaprt'], labelpad=2, fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.axvline(0.05, lw=0.5)
    ax.axvline(0.5, lw=0.5, ls='--')
    
    ax.set_ylabel(plot_labels[iisotope], labelpad=2, fontsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', labelsize=8)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(
        left=0.32, right=0.95, bottom=0.25, top=0.95)
    fig.savefig(output_png)




'''
np.nanmax(wisoaprt_alltime_icores[expid[i]][icores][ialltime] * 2.628e6)
np.nanmin(wisoaprt_alltime_icores[expid[i]][icores][ialltime] * 2.628e6)
np.nanmax(wisoaprt_alltime_icores[expid[i]][icores][ialltime])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily source SST vs. d_ln / d_xs, colored by source rh2m, wind10, lat, lon

ivar = 'sst'
scatter_size = 1

ivar1 = 'rh2m'
if (ivar1 == 'rh2m'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 70, cm_max = 90, cm_interval1 = 2, cm_interval2 = 4,
        cmap = 'viridis_r',)
elif (ivar1 == 'wind10'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 6, cm_max = 16, cm_interval1 = 1, cm_interval2 = 2,
        cmap = 'viridis_r',)
elif (ivar1 == 'lat'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -60, cm_max = -20, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'lon'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = -180, cm_max = 180, cm_interval1 = 30, cm_interval2 = 60,
        cmap = 'twilight',)
elif (ivar1 == 'distance'):
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = 20, cm_max = 60, cm_interval1 = 4, cm_interval2 = 8,
        cmap = 'viridis_r',)
elif (ivar1 == 'wisoaprt'):
    pltlevel = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 4,])
    pltticks = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 4,])
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
    pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_ln'
        # ['d_ln', 'd_excess']
        print('#---------------- ' + iisotope)
        
        for icores in ['Rothera',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily']:
                # ialltime = 'mon'
                # ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']
                print('#---- ' + ialltime)
                
                src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime].copy()
                iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime].copy()
                # wisoaprt_var = wisoaprt_alltime_icores[expid[i]][icores][ialltime].copy()
                
                subset = (np.isfinite(src_var) & np.isfinite(iso_var))
                #  & (wisoaprt_var <= (2 / 2.628e6))
                src_var = src_var[subset]
                iso_var = iso_var[subset]
                
                color_var = pre_weighted_var_icores[expid[i]][icores][ivar1][ialltime][subset]
                if (ivar1 == 'lon'):
                    color_var = calc_lon_diff(
                        color_var,
                        t63_sites_indices[icores]['lon'],
                    )
                elif (ivar1 == 'distance'):
                    color_var = color_var / 100
                
                # color_var = wisoaprt_alltime_icores[expid[i]][icores][ialltime][subset] * 2.628e6
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.2 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' source ' + ivar + ' vs. '+ iisotope + '_color_' + ivar1 + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 5]) / 2.54)
                
                plt_scatter = ax.scatter(
                    src_var, iso_var,
                    s=scatter_size, lw=0.1,
                    c=color_var, norm=pltnorm, cmap=pltcmp,
                    # facecolors='white', edgecolors='k',
                    )
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                ax.set_xlabel(plot_labels[ivar], labelpad=2, fontsize=8)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_ylabel(plot_labels[iisotope], labelpad=2, fontsize=8)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                cbar = fig.colorbar(
                    plt_scatter, ax=ax,
                    orientation="horizontal",shrink=1.4,aspect=30,
                    anchor=(1, -3), extend='both',
                    pad=0.25, fraction=0.03,
                    ticks=pltticks, format=remove_trailing_zero_pos, )
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(
                    # 'Precipitation [$mm \; day^{-1}$]',
                    plot_labels[ivar1],
                    fontsize=8, linespacing=1.5)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.22, top=0.95)
                fig.savefig(output_png)





'''
'''
# endregion
# -----------------------------------------------------------------------------

