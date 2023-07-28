

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
    plot_labels,
    plot_labels_no_unit,
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
regression_temp2_delta = {}
regression_temp2_delta_d = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d.pkl', 'rb') as f:
        regression_sst_d[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_dD.pkl', 'rb') as f:
        regression_sst_d_dD[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta.pkl', 'rb') as f:
        regression_temp2_delta[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_d.pkl', 'rb') as f:
        regression_temp2_delta_d[expid[i]] = pickle.load(f)


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
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_sst_d[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_sst_d[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_sst_d[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_sst_d[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                    sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                elif (ialltime == 'mon no mm'):
                    sim_src = pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month') - pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.0_sst_d/8.1.6.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + '.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=6, lw=0.1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (params[0] > 0):
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
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.25, top=0.95)
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
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_sst_d_dD[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                    sim_src = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                elif (ialltime == 'mon no mm'):
                    sim_src = pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month') - pre_weighted_var_icores[expid[i]][icores][ivar]['mon'].groupby('time.month').mean(dim='time')
                
                subset = np.isfinite(rec_src) & np.isfinite(sim_src)
                rec_src = rec_src[subset]
                sim_src = sim_src[subset]
                
                # print(np.round(pearsonr(rec_src, sim_src).statistic ** 2, 3))
                # print(np.round(rsquared, 3))
                
                xymax = np.max(np.concatenate((sim_src, rec_src)))
                xymin = np.min(np.concatenate((sim_src, rec_src)))
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.1_sst_d_dD/8.1.6.1.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' sim vs. rec source ' + ivar + ' using '+ iisotope + ' and dD.png'
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
                
                ax.scatter(
                    sim_src, rec_src,
                    s=6, lw=0.1, facecolors='white', edgecolors='k',)
                ax.axline((0, 0), slope = 1, lw=0.5, color='k')
                plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (params[0] > 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + ' + ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] < 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + ' ' + \
                        str(np.round(params[0], 1)) + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                elif (params[0] == 0):
                    eq_text = plot_labels_no_unit[ivar] + ' = ' + str(np.round(params[1], 2)) + plot_labels_no_unit[iisotope] + ' + ' + \
                        str(np.round(params[2], 3)) + plot_labels_no_unit['dD'] + \
                            '\n$R^2 = $' + str(np.round(rsquared, 2)) + \
                                '\n$RMSE = $' + str(np.round(RMSE, 1))
                
                plt.text(
                    0.1, 0.05, eq_text,
                    transform=ax.transAxes, fontsize=4, linespacing=1.5)
                
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
# region plot reconstructed temp2 based on dD / dO18


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['dD', 'dO18']:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['mon', 'mm', 'mon no mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['params']
                rsquared = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_temp2_delta[expid[i]][iisotope][icores][ialltime]['predicted_y']
                
                if (ialltime in ['mon', 'mm', 'ann',]):
                    sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                elif (ialltime == 'mon no mm'):
                    sim_src = temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month') - temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month').mean(dim='time')
                
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
                
                if (params[0] > 0):
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
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for icores in ['EDC',]:
            # icores = 'EDC'
            print('#-------- ' + icores)
            
            for ialltime in ['mon', 'mm', 'mon no mm', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                params = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['params']
                rsquared = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['rsquared']
                RMSE = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['RMSE']
                rec_src = regression_temp2_delta_d[expid[i]][isotope1][iisotope][icores][ialltime]['predicted_y']
                
                if (ialltime in ['daily', 'mon', 'mm', 'ann',]):
                    sim_src = temp2_alltime_icores[expid[i]][icores][ialltime]
                elif (ialltime == 'mon no mm'):
                    sim_src = temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month') - temp2_alltime_icores[expid[i]][icores]['mon'].groupby('time.month').mean(dim='time')
                
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
                
                if (params[0] > 0):
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

