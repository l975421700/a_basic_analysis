

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
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


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. temp2 at all times

i = 0

for icores in ['EDC', 'DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['dO18', 'd_excess', 'd_ln']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['mon', 'ann',]:
            # ['mon', 'ann',]:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            #---------------- settings
            
            xdata = temp2_alltime_icores[expid[i]][icores][ialltime].values
            ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime].values
            
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            xmax_value = np.max(xdata)
            xmin_value = np.min(xdata)
            ymax_value = np.max(ydata)
            ymin_value = np.min(ydata)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.1_isotopes_temp2/8.1.3.1.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' temp2 vs. ' + iisotope + '.png'
            
            linearfit = linregress(x = xdata, y = ydata,)
            
            #---------------- plot
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
            
            ax.scatter(
                xdata, ydata,
                s=6, lw=0.1, facecolors='white', edgecolors='k',)
            ax.axline(
                (0, linearfit.intercept), slope = linearfit.slope,
                lw=0.5, color='k')
            plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
            
            plt.text(
                0.5, 0.05,
                '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
                transform=ax.transAxes, fontsize=6, linespacing=1.5)
            
            ax.set_ylabel(plot_labels[iisotope], labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel('temp2 [$°C$]', labelpad=2)
            ax.set_xlim(xmin_value, xmax_value)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis='both', labelsize=8)
            
            ax.grid(True, which='both',
                    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
            fig.subplots_adjust(
                left=0.32, right=0.95, bottom=0.25, top=0.95)
            fig.savefig(output_png)




#-------- partial correlation

i = 0
icores = 'EDC'
# icores = 'DOME F'
ialltime = 'mon'

iisotope1 = 'd_ln'
iisotope2 = 'dO18'

temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]
iso_var = isotopes_alltime_icores[expid[i]][iisotope1][icores][ialltime]
ctl_var = isotopes_alltime_icores[expid[i]][iisotope2][icores][ialltime]

xr_par_cor(temp2_var, iso_var, ctl_var) ** 2
xr_par_cor(temp2_var, ctl_var, iso_var) ** 2

(pearsonr(temp2_var, iso_var).statistic) ** 2
(pearsonr(temp2_var, ctl_var).statistic) ** 2




i = 0
ialltime = 'mon'
# icores = 'EDC'
icores = 'DOME F'

dO18_var = isotopes_alltime_icores[expid[i]]['dO18'][icores][ialltime]
d_ln_var = isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime]
sst_var = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]
temp2_var = temp2_alltime_icores[expid[i]][icores][ialltime]

xr_par_cor(dO18_var, sst_var, temp2_var) ** 2
xr_par_cor(d_ln_var, temp2_var, sst_var) ** 2
xr_par_cor(temp2_var, d_ln_var, dO18_var) ** 2
xr_par_cor(sst_var, dO18_var, d_ln_var) ** 2

# (pearsonr(iso_var, sst_var).statistic) ** 2
# (pearsonr(iso_var, temp2_var).statistic) ** 2
# (pearsonr(temp2_var, sst_var).statistic) ** 2





'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. temp2 at all times at monthly scale without mm

i = 0

for icores in ['EDC', 'DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['dO18', 'd_excess', 'd_ln']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['mon', ]:
            # ['mon', ]:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            #---------------- settings
            
            xdata = temp2_alltime_icores[expid[i]][icores][ialltime]
            ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
            
            xdata = xdata.groupby('time.month') - xdata.groupby('time.month').mean()
            ydata = ydata.groupby('time.month') - ydata.groupby('time.month').mean()
            
            xmax_value = np.max(xdata)
            xmin_value = np.min(xdata)
            ymax_value = np.max(ydata)
            ymin_value = np.min(ydata)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.1_isotopes_temp2/8.1.3.1.1 ' + expid[i] + ' ' + icores + ' no mm ' + ialltime + ' temp2 vs. ' + iisotope + '.png'
            
            linearfit = linregress(x = xdata, y = ydata,)
            
            #---------------- plot
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
            
            ax.scatter(
                xdata, ydata,
                s=6, lw=0.1, facecolors='white', edgecolors='k',)
            ax.axline(
                (0, linearfit.intercept), slope = linearfit.slope,
                lw=0.5, color='k')
            plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
            
            plt.text(
                0.5, 0.05,
                '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
                transform=ax.transAxes, fontsize=6, linespacing=1.5)
            
            ax.set_ylabel(plot_labels[iisotope], labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel('temp2 [$°C$]', labelpad=2)
            ax.set_xlim(xmin_value, xmax_value)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
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
# region source sst vs. temp2

i = 0

for icores in ['EDC', 'DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for ivar in ['sst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        for ialltime in ['mon', 'ann',]:
            # ['mon', 'ann',]:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
            ydata = temp2_alltime_icores[expid[i]][icores][ialltime]
            
            xmax_value = np.max(xdata)
            xmin_value = np.min(xdata)
            ymax_value = np.max(ydata)
            ymin_value = np.min(ydata)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.1_isotopes_temp2/8.1.3.1.2 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar + ' vs. temp2.png'
            
            linearfit = linregress(x = xdata, y = ydata,)
            
            #---------------- plot
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
            
            ax.scatter(
                xdata, ydata,
                s=6, lw=0.1, facecolors='white', edgecolors='k',)
            ax.axline(
                (0, linearfit.intercept), slope = linearfit.slope,
                lw=0.5, color='k')
            plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
            
            plt.text(
                0.5, 0.05,
                '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
                transform=ax.transAxes, fontsize=6, linespacing=1.5)
            
            ax.set_ylabel('temp2 [$°C$]', labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel(plot_labels[ivar], labelpad=2)
            ax.set_xlim(xmin_value, xmax_value)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis='both', labelsize=8)
            
            ax.grid(True, which='both',
                    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
            fig.subplots_adjust(
                left=0.32, right=0.95, bottom=0.25, top=0.95)
            fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


