

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_600_5.0',
    'pi_610_5.8',
    
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


Antarctic_snow_isotopes_sim_grouped = {}
Antarctic_snow_isotopes_sim_grouped_all = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped.pkl', 'rb') as f:
        Antarctic_snow_isotopes_sim_grouped[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl', 'rb') as f:
        Antarctic_snow_isotopes_sim_grouped_all[expid[i]] = pickle.load(f)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot grouped observed vs. simulated isotopes


for iisotopes in ['dO18', 'dD', 'd_ln', 'd_excess',]:
    # iisotopes = 'd_ln'
    print('#-------- ' + iisotopes)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.2 pi_60_6 grouped observed vs. simulated ' + iisotopes + '.png'
    # output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.2 pi_60_6 all grouped observed vs. simulated ' + iisotopes + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = Antarctic_snow_isotopes_sim_grouped[expid[i]][iisotopes]
        ydata = Antarctic_snow_isotopes_sim_grouped[expid[i]][iisotopes + '_sim']
        # xdata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]][iisotopes]
        # ydata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]][iisotopes + '_sim']
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
            0.32, 0.24 - i * 0.045, eq_text,
            transform=ax.transAxes, fontsize=8,
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
# region plot grouped am values_AIS

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

lon = d_ln_alltime[expid[0]]['am'].lon
lat = d_ln_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    for iisotopes in ['d_ln']:
        # iisotopes = 'd_ln'
        # ['dO18', 'dD', 'd_ln', 'd_excess',]
        print('#-------- ' + iisotopes)
        
        if (iisotopes == 'dO18'):
            isotopevar = dO18_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-60, cm_max=-20, cm_interval1=5, cm_interval2=5,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'dD'):
            isotopevar = dD_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-450, cm_max=-100, cm_interval1=25, cm_interval2=50,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'd_ln'):
            isotopevar = d_ln_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=30, cm_interval1=2.5, cm_interval2=5,
                cmap='viridis', reversed=False)
            
        elif (iisotopes == 'd_excess'):
            isotopevar = d_excess_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=20, cm_interval1=2, cm_interval2=4,
                cmap='viridis', reversed=False)
        
        # output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.1_sim_vs_obs_spatial/8.0.3.1.2 ' + expid[i] + ' grouped observed vs. simulated ' + iisotopes + '_AIS.png'
        output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.1_sim_vs_obs_spatial/8.0.3.1.2 ' + expid[i] + ' all grouped observed vs. simulated ' + iisotopes + '_AIS.png'
        
        fig, ax = hemisphere_plot(northextent=-60)
        
        # xdata = Antarctic_snow_isotopes_sim_grouped[expid[i]]['lon']
        # ydata = Antarctic_snow_isotopes_sim_grouped[expid[i]]['lat']
        # cdata = Antarctic_snow_isotopes_sim_grouped[expid[i]][iisotopes]
        xdata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]]['lon']
        ydata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]]['lat']
        cdata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]][iisotopes]
        subset = (np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(cdata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        cdata = cdata[subset]
        
        plt_scatter = ax.scatter(
            xdata, ydata, s=8, c=cdata,
            edgecolors='k', linewidths=0.1, zorder=3,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        plt1 = plot_t63_contourf(
            lon, lat, isotopevar, ax,
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
        ax.add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        cbar = fig.colorbar(
            plt_scatter, ax=ax, aspect=30,
            orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
            pad=0.02, fraction=0.2,)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel(
            'Annual mean ' + plot_labels[iisotopes] + '\n' + \
                expid_labels[expid[i]],
            linespacing=1.5)
        fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot grouped observed vs. simulated isotopes, PI_control

i = 0

for ivar in ['d_ln', ]:
    # ivar = 'd_ln'
    # 'temperature', 'accumulation', 'dD', 'dO18', 'd_excess', 'd_ln',
    print('#-------- ' + ivar)
    
    output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0_sim_vs_obs/8.0.3.0.2 ' + expid[i] + ' all grouped observed vs. simulated ' + ivar + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in range(1):
        print(str(i) + ': ' + expid[i])
        
        xdata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]][ivar]
        ydata = Antarctic_snow_isotopes_sim_grouped_all[expid[i]][ivar + '_sim']
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
        elif (linearfit.intercept < 0):
            eq_text = '$y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        
        plt.text(
            0.2, 0.1, eq_text,
            transform=ax.transAxes, fontsize=10,
            color=expid_colours[expid[i]], ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[ivar], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[ivar], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


