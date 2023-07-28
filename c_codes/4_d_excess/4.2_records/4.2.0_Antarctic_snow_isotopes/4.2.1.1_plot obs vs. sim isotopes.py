

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_605_5.5',
    'pi_606_5.6',
    'pi_609_5.7',
    
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
    find_ilat_ilon_general,
    find_multi_gridvalue_at_site,
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
    expid_colours,
    expid_labels,
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


Antarctic_snow_isotopes_simulations = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_simulations.pkl', 'rb') as f:
        Antarctic_snow_isotopes_simulations[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot observed vs. simulated isotopes


for iisotopes in ['dO18', 'dD', 'd_ln', 'd_excess',]:
    # iisotopes = 'd_ln'
    print('#-------- ' + iisotopes)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.0 pi_60_6 observed vs. simulated ' + iisotopes + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = Antarctic_snow_isotopes_simulations[expid[i]][iisotopes]
        ydata = Antarctic_snow_isotopes_simulations[expid[i]][iisotopes + '_sim']
        subset = (np.isfinite(xdata) & np.isfinite(ydata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.4, facecolors='white', alpha=0.5,
            edgecolors=expid_colours[expid[i]],)
        
        linearfit = linregress(x = xdata, y = ydata,)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=1, color=expid_colours[expid[i]], alpha=0.5)
        
        if (linearfit.intercept >= 0):
            eq_text = '$y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        if (linearfit.intercept < 0):
            eq_text = '$y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        
        plt.text(
            0.2, 0.35 - i * 0.06, eq_text,
            transform=ax.transAxes, fontsize=10,
            color=expid_colours[expid[i]], ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[iisotopes], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[iisotopes], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d_excess & d_ln vs. dD


for iisotopes in ['d_excess', 'd_ln', ]:
    # iisotopes = 'd_ln'
    print('#-------- ' + iisotopes)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.1 pi_60_6 observed and simulated ' + iisotopes + ' vs. dD.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = Antarctic_snow_isotopes_simulations[expid[0]]['dD']
    ydata = Antarctic_snow_isotopes_simulations[expid[0]][iisotopes]
    
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    ax.scatter(
        xdata, ydata,
        s=6, lw=0.4, facecolors='white', alpha=0.5,
        edgecolors='gray',)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = Antarctic_snow_isotopes_simulations[expid[i]]['dD_sim']
        ydata = Antarctic_snow_isotopes_simulations[expid[i]][iisotopes + '_sim']
        
        subset = (np.isfinite(xdata) & np.isfinite(ydata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.4, facecolors='white', alpha=0.5,
            edgecolors=expid_colours[expid[i]],)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_xlabel(plot_labels['dD'], labelpad=6)
    ax.set_ylabel(plot_labels[iisotopes], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)







# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot legend for expid

fig, ax = plt.subplots(1, 1, figsize=np.array([2, 3.6]) / 2.54)

symbol_size = 20
linewidth = 1
alpha = 1

l1 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[0]],
    lw=linewidth, alpha=alpha,)
l2 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[1]],
    lw=linewidth, alpha=alpha,)
l3 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[2]],
    lw=linewidth, alpha=alpha,)
l4 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[3]],
    lw=linewidth, alpha=alpha,)
l5 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[4]],
    lw=linewidth, alpha=alpha,)
l6 = plt.scatter(
    [],[],
    s=symbol_size, c='white', edgecolors=expid_colours[expid[5]],
    lw=linewidth, alpha=alpha,)

plt.legend(
    [l1, l2, l3, l4, l5, l6,],
    [expid_labels[expid[0]],
     expid_labels[expid[1]],
     expid_labels[expid[2]],
     expid_labels[expid[3]],
     expid_labels[expid[4]],
     expid_labels[expid[5]],],
    title = 'Experiments', title_fontsize = 10,
    ncol=1, frameon=False, loc = 'center', handletextpad=0.01)

plt.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.0 pi_60_6 legend.png'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot observed vs. simulated isotopes, control simulations


i = 0
Antarctic_snow_isotopes_sim_interpn = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_interpn.pkl', 'rb') as f:
    Antarctic_snow_isotopes_sim_interpn[expid[i]] = pickle.load(f)


for iisotopes in ['d_ln',]:
    # iisotopes = 'd_ln'
    print('#-------- ' + iisotopes)
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in [0]:
        print(str(i) + ': ' + expid[i])
        
        # output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.0 ' + expid[i] + ' observed vs. simulated ' + iisotopes + '.png'
        output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.0 ' + expid[i] + ' observed vs. simulated ' + iisotopes + '_interpn.png'
        
        # xdata = Antarctic_snow_isotopes_simulations[expid[i]][iisotopes]
        # ydata = Antarctic_snow_isotopes_simulations[expid[i]][iisotopes + '_sim']
        xdata = Antarctic_snow_isotopes_sim_interpn[expid[i]][iisotopes]
        ydata = Antarctic_snow_isotopes_sim_interpn[expid[i]][iisotopes + '_sim']
        subset = (np.isfinite(xdata) & np.isfinite(ydata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.4, facecolors='white', alpha=0.5,
            edgecolors=expid_colours[expid[i]],)
        
        linearfit = linregress(x = xdata, y = ydata,)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=1, color=expid_colours[expid[i]], alpha=0.5)
        plt.text(
            0.4, 0.1,
            '$y = $' + str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 3)),
            transform=ax.transAxes, fontsize=10, linespacing=1.5,
            color=expid_colours[expid[i]], ha='left')
    
    # xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    # xylim_min = np.min(xylim)
    # xylim_max = np.max(xylim)
    # ax.set_xlim(xylim_min, xylim_max)
    # ax.set_ylim(xylim_min, xylim_max)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    
    ax.axline((0, 0), slope = 1, lw=1, color='k', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[iisotopes], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[iisotopes], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------

