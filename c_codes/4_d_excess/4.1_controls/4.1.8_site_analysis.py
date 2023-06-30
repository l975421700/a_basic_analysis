

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
pre_weighted_var_icores = {}
wisoaprt_alltime_icores = {}

for i in range(len(expid)):
    print(i)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
        pre_weighted_var_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
        wisoaprt_alltime_icores[expid[i]] = pickle.load(f)



'''

aprt_frc_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl', 'rb') as f:
    aprt_frc_alltime_icores[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm isotopes

i = 0
Sites = ['EDC', 'DOME F']
colors = {'EDC': 'dodgerblue', 'DOME F': 'darkorange'}

isotope_label = {
    'd_ln': '$d_{ln}$',
    'd_excess': 'd-excess',
}

iisotope = 'd_ln'
iisotope = 'd_excess'


output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0 pi_600_3 ' + iisotope + ' mm EDC and Dome Fuji.png'


fig, ax = plt.subplots(1, 1, figsize=np.array([6, 6]) / 2.54)

for icores in Sites:
    # icores = 'EDC'
    monstd = isotopes_alltime_icores[expid[i]][iisotope][icores][
        'mon'].groupby('time.month').std(ddof=1)
    
    ax.errorbar(
        month,
        isotopes_alltime_icores[expid[i]][iisotope][icores]['mm'],
        yerr = monstd, capsize=2,
        fmt='.-', lw=1, markersize=6, alpha=0.75,
        label=icores, color=colors[icores],)
    
    ax.hlines(
        isotopes_alltime_icores[expid[i]][iisotope][icores]['am'].values,
        -1, 13,
        linestyles='dashed', colors=colors[icores], lw=1,
    )

ax.set_title(isotope_label[iisotope] + ' [$‰$]', fontsize=10,)
ax.set_xticklabels(monthini)
ax.set_xlim(-0.55, 11.55)

ax.set_ylabel(None)
# ax.set_yticks(np.arange(-42, -31 + 1e-4, 2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.14, right=0.99, bottom=0.1, top=0.9)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region d_ln vs. source SST

i = 0
icores = 'EDC'
iisotope = 'd_ln'
ivar = 'sst'

ialltime = 'sea'

output_png = 'figures/test/test.png'
fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)

ax.scatter(
    pre_weighted_var_icores[expid[i]][icores][ivar][ialltime],
    isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime],
    s=6, lw=0.1, facecolors='white', edgecolors='k',
    )
fig.tight_layout()
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. source properties at all times

i = 0

for icores in ['DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['dO18', 'd_excess', 'd_ln']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ivar in pre_weighted_var_icores[expid[i]][icores].keys():
            # ivar = 'sst'
            print('#-------- ' + ivar)
            
            for ialltime in ['daily', 'mon', 'ann',]:
                # ['daily', 'mon', 'ann',]:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                #---------------- settings
                
                xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                subset = (np.isfinite(xdata) & np.isfinite(ydata))
                xdata = xdata[subset]
                ydata = ydata[subset]
                
                xmax_value = np.max(xdata)
                xmin_value = np.min(xdata)
                ymax_value = np.max(ydata)
                ymin_value = np.min(ydata)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar + ' vs. ' + iisotope + '.png'
                
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
                
                ax.set_ylabel(iisotope, labelpad=2)
                ax.set_ylim(ymin_value, ymax_value)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_xlabel(ivar, labelpad=2)
                ax.set_xlim(xmin_value, xmax_value)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.25, top=0.95)
                fig.savefig(output_png)




'''
#-------- partial correlation

i = 0
icores = 'EDC'
icores = 'DOME F'
iisotope = 'd_ln'
ialltime = 'mon'

ivar = 'rh2m'
ivar = 'wind10'
# ivar = 'sst'

iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
ctl_var = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]

xr_par_cor(iso_var, src_var, ctl_var) ** 2


# (pearsonr(iso_var, src_var).statistic) ** 2
# (pearsonr(iso_var, ctl_var).statistic) ** 2


len(isotopes_alltime_icores[expid[i]].keys()) * len(pre_weighted_var_icores[expid[i]][icores].keys()) * 3
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. source properties at monthly scale without mm

i = 0

for icores in ['DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['dO18', 'd_excess', 'd_ln']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ivar in pre_weighted_var_icores[expid[i]][icores].keys():
            # ivar = 'sst'
            print('#-------- ' + ivar)
            
            for ialltime in ['mon',]:
                # ['daily', 'mon', 'ann',]:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                #---------------- settings
                
                xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                subset = (np.isfinite(xdata) & np.isfinite(ydata))
                xdata = xdata[subset]
                ydata = ydata[subset]
                
                xdata = xdata.groupby('time.month') - xdata.groupby('time.month').mean()
                ydata = ydata.groupby('time.month') - ydata.groupby('time.month').mean()
                
                xmax_value = np.max(xdata)
                xmin_value = np.min(xdata)
                ymax_value = np.max(ydata)
                ymin_value = np.min(ydata)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.1 ' + expid[i] + ' ' + icores + ' no mm ' + ialltime + ' ' + ivar + ' vs. ' + iisotope + '.png'
                
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
                
                ax.set_ylabel(iisotope, labelpad=2)
                ax.set_ylim(ymin_value, ymax_value)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_xlabel(ivar, labelpad=2)
                ax.set_xlim(xmin_value, xmax_value)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both', labelsize=8)
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.32, right=0.95, bottom=0.25, top=0.95)
                fig.savefig(output_png)



'''
#-------- partial correlation

i = 0
# icores = 'EDC'
icores = 'DOME F'
iisotope = 'd_ln'
ialltime = 'mon'

ivar = 'rh2m'
ivar = 'wind10'
ivar = 'sst'

iso_var = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
src_var = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
ctl_var = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime]

iso_var = iso_var.groupby('time.month') - iso_var.groupby('time.month').mean()
src_var = src_var.groupby('time.month') - src_var.groupby('time.month').mean()
ctl_var = ctl_var.groupby('time.month') - ctl_var.groupby('time.month').mean()


xr_par_cor(iso_var, src_var, ctl_var) ** 2

# (pearsonr(iso_var, src_var).statistic) ** 2
# (pearsonr(iso_var, ctl_var).statistic) ** 2


len(isotopes_alltime_icores[expid[i]].keys()) * len(pre_weighted_var_icores[expid[i]][icores].keys()) * 3
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source SST vs. other source properties

i = 0

ivar1 = 'sst'

for icores in ['DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for ivar in ['lat', 'lon', 'rh2m', 'wind10', 'distance']:
        # ivar = 'sst'
        print('#-------- ' + ivar)
        
        for ialltime in ['daily', 'mon', 'ann',]:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            xdata = pre_weighted_var_icores[expid[i]][icores][ivar1][ialltime]
            ydata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            xmax_value = np.max(xdata)
            xmin_value = np.min(xdata)
            ymax_value = np.max(ydata)
            ymin_value = np.min(ydata)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.2 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar1 + ' vs. ' + ivar + '.png'
            
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
            
            ax.set_ylabel(ivar, labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel(ivar1, labelpad=2)
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


# -----------------------------------------------------------------------------
# region source SST vs. other source properties without mm


i = 0

ivar1 = 'sst'

for icores in ['DOME F']:
    # ['EDC', 'DOME F']:
    # icores = 'EDC'
    # pre_weighted_var_icores[expid[i]].keys()
    print('#-------------------------------- ' + icores)
    
    for ivar in ['lat', 'lon', 'rh2m', 'wind10', 'distance']:
        # ivar = 'sst'
        print('#-------- ' + ivar)
        
        for ialltime in ['mon',]:
            # ialltime = 'mon'
            print('#---- ' + ialltime)
            
            xdata = pre_weighted_var_icores[expid[i]][icores][ivar1][ialltime]
            ydata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            xdata = xdata.groupby('time.month') - xdata.groupby('time.month').mean()
            ydata = ydata.groupby('time.month') - ydata.groupby('time.month').mean()
            
            xmax_value = np.max(xdata)
            xmin_value = np.min(xdata)
            ymax_value = np.max(ydata)
            ymin_value = np.min(ydata)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.3 ' + expid[i] + ' ' + icores + ' no mm ' + ialltime + ' ' + ivar1 + ' vs. ' + ivar + '.png'
            
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
            
            ax.set_ylabel(ivar, labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel(ivar1, labelpad=2)
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


