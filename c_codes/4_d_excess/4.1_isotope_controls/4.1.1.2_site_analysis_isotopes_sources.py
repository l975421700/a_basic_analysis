

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'hist_700_5.0',
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
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
pre_weighted_var_icores = {}
wisoaprt_alltime_icores = {}
# temp2_alltime_icores = {}

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
    
    # with open(
    #     exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.temp2_alltime_icores.pkl', 'rb') as f:
    #     temp2_alltime_icores[expid[i]] = pickle.load(f)

# aprt_frc_alltime_icores = {}
# for i in range(len(expid)):
#     print(i)
    
#     with open(
#         exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl', 'rb') as f:
#         aprt_frc_alltime_icores[expid[i]] = pickle.load(f)


'''

wisoaprt_alltime_icores = {}
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
        wisoaprt_alltime_icores[expid[i]] = pickle.load(f)


aprt_frc_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.aprt_frc_alltime_icores.pkl', 'rb') as f:
    aprt_frc_alltime_icores[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. source properties at all times

i = 0

for icores in ['EDC',]:
    # icores = 'EDC'
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_ln'
        # ['d_ln', 'd_excess', 'dO18', 'dD',]
        print('#---------------- ' + iisotope)
        
        for ivar in ['sst',]:
            # ivar = 'sst'
            # 'sst', 'RHsst', 'rh2m', 'wind10'
            print('#-------- ' + ivar)
            
            for ialltime in ['mon']:
                # ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am', 'sm']:
                # ialltime = 'sea'
                print('#---- ' + ialltime)
                
                #---------------- settings
                
                xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                subset = (np.isfinite(xdata) & np.isfinite(ydata))
                xdata = xdata[subset]
                ydata = ydata[subset]
                
                xmax_value = np.max(xdata) + 1
                xmin_value = np.min(xdata) - 1
                ymax_value = np.max(ydata) + 1
                ymin_value = np.min(ydata) - 1
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar + ' vs. ' + iisotope + '.png'
                
                linearfit = linregress(x = xdata, y = ydata,)
                
                #---------------- plot
                
                fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
                
                sns.scatterplot(
                    x=xdata, y=ydata,
                    hue=xdata.time.dt.season,
                    s=12, lw=0.5, facecolors='white', edgecolors='k',
                )
                # ax.scatter(
                #     xdata, ydata,
                #     s=12, lw=0.5, facecolors='white', edgecolors='k',)
                ax.axline(
                    (0, linearfit.intercept), slope = linearfit.slope,
                    lw=0.5, color='k')
                # plt.text(0.05, 0.9, icores, transform=ax.transAxes, color='k',)
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x$' + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2))
                elif (linearfit.intercept >= 0):
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                        str(np.round(linearfit.intercept, 1)) + \
                            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2))
                else:
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x $' + \
                        str(np.round(linearfit.intercept, 1)) + \
                            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2))
                
                plt.text(
                    0.95, 0.05, eq_text, transform=ax.transAxes,
                    linespacing=2, ha='right', va='bottom')
                
                ax.set_ylabel(plot_labels[iisotope], labelpad=2)
                ax.set_ylim(ymin_value, ymax_value)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                
                ax.set_xlabel(plot_labels[ivar], labelpad=2)
                ax.set_xlim(xmin_value, xmax_value)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis='both')
                
                ax.grid(True, which='both',
                        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
                fig.subplots_adjust(
                    left=0.18, right=0.95, bottom=0.15, top=0.95)
                fig.savefig(output_png)



#-------- partial correlation

i = 0
# icores = 'EDC'
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


'''
'''
# endregion
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# region source SST vs. other source properties

i = 0

ivar1 = 'sst'

for icores in ['EDC']:
    # icores = 'EDC'
    print('#-------------------------------- ' + icores)
    
    for ivar in ['rh2m', 'wind10']:
        # ivar = 'sst'
        print('#-------- ' + ivar)
        
        for ialltime in ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']:
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
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.1 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar1 + ' vs. ' + ivar + '.png'
            
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
            
            if (linearfit.intercept >= 0):
                eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3))
            else:
                eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3))
            
            plt.text(
                0.5, 0.05,
                eq_text,
                transform=ax.transAxes, fontsize=6, linespacing=1.5)
            
            ax.set_ylabel(plot_labels[ivar], labelpad=2)
            ax.set_ylim(ymin_value, ymax_value)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_xlabel(plot_labels[ivar1], labelpad=2)
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
# region d_ln vs. source SST, plot scatter density of daily values

import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

i = 0
icores = 'DOME F'
iisotope = 'd_ln'
ivar = 'sst'

ialltime = 'daily'

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
subset = (np.isfinite(xdata) & np.isfinite(ydata))
xdata = xdata[subset]
ydata = ydata[subset]

xmax_value = np.max(xdata)
xmin_value = np.min(xdata)
ymax_value = np.max(ydata)
ymin_value = np.min(ydata)

output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.0 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar + ' vs. ' + iisotope + '_scatter_density.png'

linearfit = linregress(x = xdata, y = ydata,)

#---------------- plot

fig = plt.figure(figsize=np.array([4.4, 4]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

ax.scatter_density(
    xdata,
    ydata,
    cmap=white_viridis)

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

ax.set_xlabel(plot_labels[ivar], labelpad=2)
ax.set_xlim(xmin_value, xmax_value)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', labelsize=8)

ax.grid(
    True, which='both',
    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.32, right=0.95, bottom=0.25, top=0.95)
fig.savefig(output_png)


'''
https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region isotopes vs. source properties, controlling aprt_frc

frc_threshold = 80

i = 0

for icores in ['EDC',]:
    # icores = 'EDC'
    print('#-------------------------------- ' + icores)
    
    for iisotope in ['d_ln']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ivar in ['sst',]:
            # ivar = 'sst'
            print('#-------- ' + ivar)
            
            for ialltime in ['daily',]:
                # ['daily', 'mon', 'ann',]:
                # ialltime = 'daily'
                # ['daily', 'mon', 'mm', 'mon no mm', 'ann', 'ann no am']
                print('#---- ' + ialltime)
                
                #---------------- settings
                
                xdata = pre_weighted_var_icores[expid[i]][icores][ivar][ialltime]
                ydata = isotopes_alltime_icores[expid[i]][iisotope][icores][ialltime]
                aprt_frc = aprt_frc_alltime_icores[expid[i]][icores][ialltime]
                if (ialltime == 'mon no mm'):
                    aprt_frc = aprt_frc_alltime_icores[expid[i]][icores]['mon']
                elif (ialltime == 'ann no am'):
                    aprt_frc = aprt_frc_alltime_icores[expid[i]][icores]['ann']
                
                subset = (np.isfinite(xdata) & np.isfinite(ydata) & (aprt_frc >= frc_threshold))
                xdata = xdata[subset]
                ydata = ydata[subset]
                
                xmax_value = np.max(xdata)
                xmin_value = np.min(xdata)
                ymax_value = np.max(ydata)
                ymin_value = np.min(ydata)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.0_isotopes_sources_alltimes/8.1.3.0.2 ' + expid[i] + ' ' + icores + ' ' + ialltime + ' ' + ivar + ' vs. ' + iisotope + '_frc' + str(frc_threshold) + '.png'
                
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
                
                if (ialltime in ['mon no mm', 'ann no am']):
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x$' + \
                        '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3))
                elif (linearfit.intercept >= 0):
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
                        str(np.round(linearfit.intercept, 1)) + \
                            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3))
                else:
                    eq_text = '$y = $' + str(np.round(linearfit.slope, 1)) + '$x $' + \
                        str(np.round(linearfit.intercept, 1)) + \
                            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 3))
                
                plt.text(
                    0.5, 0.05, eq_text,
                    transform=ax.transAxes, fontsize=6, linespacing=1.5)
                
                ax.set_ylabel(plot_labels[iisotope], labelpad=2)
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



frc_threshold = 80
icores = 'EDC'
ialltime = 'daily'
i = 0

min_values = np.nanmin(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime].values)
print(min_values)
print(np.nanmin(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime].values[aprt_frc_alltime_icores[expid[i]][icores][ialltime].values >= frc_threshold]))


wheremin = np.where(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime].values == min_values)[0][0]

isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime].values[wheremin]
isotopes_alltime_icores[expid[i]]['dD'][icores][ialltime].values[wheremin]
isotopes_alltime_icores[expid[i]]['dO18'][icores][ialltime].values[wheremin]
isotopes_alltime_icores[expid[i]]['d_excess'][icores][ialltime].values[wheremin]
aprt_frc_alltime_icores[expid[i]][icores][ialltime].values[wheremin]
wisoaprt_alltime_icores[expid[i]][icores][ialltime].values[wheremin]

# 0.05 / 2.628e6, 1.902587519025875e-08

'''
np.std(aprt_frc_alltime_icores[expid[i]][icores]['daily'].values)

aprt_frc_alltime_icores[expid[i]][icores]['ann'].values
aprt_frc_alltime_icores[expid[i]][icores]['ann no am'].values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily time series of temp2/aprt, source SST/rh2m, dD/dln

icores = 'EDC'
ialltime = 'daily'

istartday = 3000
# istartday = np.where(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime] == np.nanmin(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime]))[0][0] - 7
iendday   = istartday + 15

ts_temp2 = temp2_alltime_icores[expid[i]][icores][ialltime][istartday:iendday]
ts_aprt = wisoaprt_alltime_icores[expid[i]][icores][ialltime][istartday:iendday] * seconds_per_d

ts_dln = isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime][istartday:iendday]
ts_dD = isotopes_alltime_icores[expid[i]]['dD'][icores][ialltime][istartday:iendday]
ts_dO18 = isotopes_alltime_icores[expid[i]]['dO18'][icores][ialltime][istartday:iendday]

ts_sst = pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime][istartday:iendday]
ts_rh2m = pre_weighted_var_icores[expid[i]][icores]['rh2m'][ialltime][istartday:iendday]
ts_wind10 = pre_weighted_var_icores[expid[i]][icores]['wind10'][ialltime][istartday:iendday]

start_time = temp2_alltime_icores[expid[i]][icores][ialltime][istartday].time.values
end_time = temp2_alltime_icores[expid[i]][icores][ialltime][iendday-1].time.values

output_png = 'figures/8_d-excess/8.1_controls/8.1.3_site_analysis/8.1.3.2_daily_timeseries/8.1.3.2.0 ' + expid[i] + ' ' + icores + ' daily timeseries startday_' + str(istartday) + '.png'

fig, axs = plt.subplots(7, 1, figsize=np.array([15, 20]) / 2.54, sharex=True)

# 1st row
axs[0].scatter(np.arange(1, 15.5, 1), ts_temp2)

axs[0].set_ylabel('temp2\n[$°C$]', labelpad=2)
# axs[0].set_ylim(-76, -19)
# axs[0].set_yticks(np.arange(-70, -20 + 1e-4, 10))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))

# 2nd row
axs[1].scatter(np.arange(1, 15.5, 1), ts_aprt)

axs[1].set_ylabel('Precipitation\n[$mm \; day^{-1}$]', labelpad=2)
axs[1].set_ylim(0.0016, 2.8)
axs[1].set_yscale('log')
axs[1].set_yticks(np.array([0.01, 0.1, 1,]))
axs[1].set_yticklabels(np.array([0.01, 0.1, 1,]))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))

# 3rd row
axs[2].scatter(np.arange(1, 15.5, 1), ts_dln)

axs[2].set_ylabel('$d_{ln}$\n[$‰$]', labelpad=2)
# axs[2].set_ylim(-100, 100)
# axs[2].set_yticks(np.arange(-100, 100 + 1e-4, 40))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(2))

# 4th row
axs[3].scatter(np.arange(1, 15.5, 1), ts_dD)

axs[3].set_ylabel('$\delta D$\n[$‰$]', labelpad=2)
# axs[3].set_ylim(-830, -132)
# axs[3].set_yticks(np.arange(-800, -200 + 1e-4, 100))
axs[3].yaxis.set_minor_locator(AutoMinorLocator(2))

# 5th row
axs[4].scatter(np.arange(1, 15.5, 1), ts_sst)

axs[4].set_ylabel('Source SST\n[$°C$]', labelpad=2)
# axs[4].set_ylim(0, 26)
# axs[4].set_yticks(np.arange(2, 24 + 1e-4, 4))
axs[4].yaxis.set_minor_locator(AutoMinorLocator(2))

# 6th row
axs[5].scatter(np.arange(1, 15.5, 1), ts_rh2m)

axs[5].set_ylabel('Source rh2m\n[$\%$]', labelpad=2)
# axs[5].set_ylim(64, 91)
# axs[5].set_yticks(np.arange(66, 88 + 1e-4, 4))
axs[5].yaxis.set_minor_locator(AutoMinorLocator(2))

# 7th row
axs[6].scatter(np.arange(1, 15.5, 1), ts_wind10)

axs[6].set_ylabel('Source wind10\n[$m \; s^{-1}$]', labelpad=2)
# axs[6].set_ylim(6, 18)
# axs[6].set_yticks(np.arange(8, 18 + 1e-4, 2))
axs[6].yaxis.set_minor_locator(AutoMinorLocator(2))

axs[6].set_xlabel('Date starting from ' + str(start_time)[:10], labelpad=2)
axs[6].set_xlim(0.5, 15.5)
axs[6].set_xticks(np.arange(1, 15.5, 1))
# axs[6].tick_params(axis='both', labelsize=8)

for irow in range(7):
    axs[irow].grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.14, right=0.97, bottom=0.06, top=0.98)
fig.savefig(output_png)









'''
temp2: [-76, -19]; [-70, -60, -50, -40, -30, -20]
aprt: [0.0016, 2.8]; [0.01, 0.1, 1,]
d_ln: [-630, 100]; [-600, -500, -400, -300, -200, -100, 0, 100]
dD: [-830, -132]; [-800, -700, -600, -500, -400, -300, -200]

sst: [0, 26], np.arange(2, 26, 2)
rh2m: [64, 91], np.arange(66, 92, 4)
wind10: [6, 18], np.arange(8, 18, 2)

stats.describe(temp2_alltime_icores[expid[i]][icores][ialltime])
stats.describe(wisoaprt_alltime_icores[expid[i]][icores][ialltime] * seconds_per_d)

stats.describe(isotopes_alltime_icores[expid[i]]['d_ln'][icores][ialltime], nan_policy='omit')
stats.describe(isotopes_alltime_icores[expid[i]]['dD'][icores][ialltime], nan_policy='omit')

stats.describe(pre_weighted_var_icores[expid[i]][icores]['sst'][ialltime], nan_policy='omit')
stats.describe(pre_weighted_var_icores[expid[i]][icores]['rh2m'][ialltime], nan_policy='omit')
stats.describe(pre_weighted_var_icores[expid[i]][icores]['wind10'][ialltime], nan_policy='omit')

2e-8 * seconds_per_d = 0.001728
0.05 / 2.628e6 * seconds_per_d = 0.00164

'''
# endregion
# -----------------------------------------------------------------------------



