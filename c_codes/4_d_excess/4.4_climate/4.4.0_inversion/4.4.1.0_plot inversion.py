

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    'nudged_703_6.0_k52',
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
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import linregress
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
    inversion_top,
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

# st_plev = {}
# for i in range(len(expid)):
#     with open(
#         exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.st_plev.pkl',
#         'rb') as f:
#         st_plev[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

zh_st_ml = {}
for i in range(len(expid)):
    print(str(i) + ' ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'rb') as f:
        zh_st_ml[expid[i]] = pickle.load(f)

i = 0
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.inversion_height_strength.pkl', 'rb') as f:
    inversion_height_strength = pickle.load(f)

echam6_t63_geosp = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/' + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot an example of inversion on model level

i = 0
imon = 0

for isite in t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    x_data = zh_st_ml[expid[i]]['st']['mon'][imon, :, ilat, ilon].values
    y_data = zh_st_ml[expid[i]]['zh']['mon'][imon, :, ilat, ilon].values / 1000
    
    subset = (y_data <= 6)
    x_data = x_data[subset]
    y_data = y_data[subset]
    
    t_it, h_it = inversion_top(x_data, y_data)
    
    xlim_min = np.min(x_data) - 2
    xlim_max = np.max(x_data) + 2
    x_interval = 5
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.2 ' + expid[i] + ' example of monthly inveresion against height at ' + isite + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
    
    ax.plot(x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
    
    if (not np.isnan(t_it)):
        plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel('Temperature [$K$]', labelpad=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(0, 5.1, 0.5))
    ax.set_ylim(0, 5)
    ax.set_ylabel('Height [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt_text = plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    plt.savefig(output_png)
    plt.close()





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate mon/mm/sm inversion on model level

i = 0
ialltime = 'mon'
# ialltime = 'mm'
# ialltime = 'sm'

for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.3 ' + expid[i] + ' ' + ialltime + ' inveresion against height at ' + isite + '.mp4'
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    site_elevation = echam6_t63_surface_height[ilat, ilon].values
    
    #-------------------------------- plot
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
    
    ims = []
    
    for itime in range(zh_st_ml[expid[i]]['st'][ialltime].shape[0]):
        # itime = 0
        # print(itime)
        
        x_data = zh_st_ml[expid[i]]['st'][ialltime][itime, :, ilat, ilon].values
        y_data = zh_st_ml[expid[i]]['zh'][ialltime][itime, :, ilat, ilon].values / 1000
        
        plt_line = ax.plot(
            x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
        
        # t_it, h_it = inversion_top(x_data, y_data)
        # if (not np.isnan(t_it)):
        #     plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
        plt_scatter = ax.scatter(
            inversion_height_strength[isite][ialltime]['Inversion temperature'][itime].values,
            inversion_height_strength[isite][ialltime]['Inversion height'][itime].values / 1000 + site_elevation / 1000,
            s=5, c='red', zorder=3)
        
        if (ialltime in ['mon', 'sea', 'ann']):
            timelabel = str(zh_st_ml[expid[i]]['st'][ialltime][itime].time.values)[:7]
        elif (ialltime == 'mm'):
            timelabel = month[zh_st_ml[expid[i]]['st']['mm'][itime].month.values-1]
        elif (ialltime == 'sm'):
            timelabel = zh_st_ml[expid[i]]['st']['sm'][itime].season.values
        
        plt_text = ax.text(
            0.1, 0.1, timelabel,
            transform=ax.transAxes, color='k',
            ha='left', va = 'center')
        
        ims.append(plt_line + [plt_text, plt_scatter])
    
    ax.set_xticks(np.arange(190, 260.1, 10))
    ax.set_xlim(190, 260)
    ax.set_xlabel('Temperature [$K$]')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(3, 5.1, 0.5))
    ax.set_ylim(3, 5)
    ax.set_ylabel('Height [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='k',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)




'''


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot an example of inversion on model level daily data

i = 0
iday = 0

for isite in t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    x_data = zh_st_ml[expid[i]]['st']['daily'][iday, :, ilat, ilon].values
    y_data = zh_st_ml[expid[i]]['zh']['daily'][iday, :, ilat, ilon].values / 1000
    
    subset = (y_data <= 6)
    x_data = x_data[subset]
    y_data = y_data[subset]
    
    t_it, h_it = inversion_top(x_data, y_data)
    
    xlim_min = np.min(x_data) - 2
    xlim_max = np.max(x_data) + 2
    x_interval = 5
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.2 ' + expid[i] + ' example of daily inveresion against height at ' + isite + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
    
    ax.plot(x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
    
    if (not np.isnan(t_it)):
        plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel('Temperature [$K$]', labelpad=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(0, 5.1, 0.5))
    ax.set_ylim(0, 5)
    ax.set_ylabel('Height [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt_text = plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    plt.savefig(output_png)
    plt.close()





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate inversion on model level daily data


i = 0
ialltime = 'daily'
# ialltime = 'mon'
# ialltime = 'mm'
# ialltime = 'sm'

itimestart = np.where(zh_st_ml[expid[i]]['st'][ialltime].time.values == np.datetime64('2014-12-25T23:52:30.000000000'))[0][0]
itimeend = np.where(zh_st_ml[expid[i]]['st'][ialltime].time.values == np.datetime64('2015-01-16T23:52:30.000000000'))[0][0]


for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.3 ' + expid[i] + ' ' + ialltime + ' inveresion against height at ' + isite + '.mp4'
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    site_elevation = echam6_t63_surface_height[ilat, ilon].values
    
    #-------------------------------- plot
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
    
    ims = []
    
    for itime in np.arange(itimestart, itimeend+1, 1):
        # range(zh_st_ml[expid[i]]['st'][ialltime].shape[0])
        # itime = 0
        # print(itime)
        
        x_data = zh_st_ml[expid[i]]['st'][ialltime][itime, :, ilat, ilon].values
        y_data = zh_st_ml[expid[i]]['zh'][ialltime][itime, :, ilat, ilon].values / 1000
        
        plt_line = ax.plot(
            x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
        
        # t_it, h_it = inversion_top(x_data, y_data)
        # if (not np.isnan(t_it)):
        #     plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
        plt_scatter = ax.scatter(
            inversion_height_strength[isite][ialltime]['Inversion temperature'][itime].values,
            inversion_height_strength[isite][ialltime]['Inversion height'][itime].values / 1000 + site_elevation / 1000,
            s=5, c='red', zorder=3)
        
        if (ialltime in ['mon', 'sea', 'ann']):
            timelabel = str(zh_st_ml[expid[i]]['st'][ialltime][itime].time.values)[:7]
        elif (ialltime == 'mm'):
            timelabel = month[zh_st_ml[expid[i]]['st']['mm'][itime].month.values-1]
        elif (ialltime == 'sm'):
            timelabel = zh_st_ml[expid[i]]['st']['sm'][itime].season.values
        elif (ialltime == 'daily'):
            timelabel = str(zh_st_ml[expid[i]]['st'][ialltime][itime].time.values)[:10]
        
        plt_text = ax.text(
            0.1, 0.1, timelabel,
            transform=ax.transAxes, color='k',
            ha='left', va = 'center')
        
        ims.append(plt_line + [plt_text, plt_scatter])
    
    ax.set_xticks(np.arange(190, 260.1, 10))
    ax.set_xlim(190, 260)
    ax.set_xlabel('Temperature [$K$]')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(3, 5.1, 0.5))
    ax.set_ylim(3, 5)
    ax.set_ylabel('Altitude [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='k',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)




'''


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate inversion on model level daily data with radiosounding


i = 0
ialltime = 'daily'

itimestart = np.where(zh_st_ml[expid[i]]['st'][ialltime].time.values == np.datetime64('2014-12-25T23:52:30.000000000'))[0][0]
itimeend = np.where(zh_st_ml[expid[i]]['st'][ialltime].time.values == np.datetime64('2015-01-16T23:52:30.000000000'))[0][0]
# itimeend = np.where(zh_st_ml[expid[i]]['st'][ialltime].time.values == np.datetime64('2014-12-26T23:52:30.000000000'))[0][0]

EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.3 ' + expid[i] + ' and radiosonde ' + ialltime + ' inveresion against height at ' + isite + '.mp4'
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    site_elevation = echam6_t63_surface_height[ilat, ilon].values
    
    #-------------------------------- plot
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
    ims = []
    for itime in np.arange(itimestart, itimeend+1, 1):
        print(itime)
        # range(zh_st_ml[expid[i]]['st'][ialltime].shape[0])
        # itime = 0
        # print(itime)
        
        x_data = zh_st_ml[expid[i]]['st'][ialltime][itime, :, ilat, ilon].values
        y_data = zh_st_ml[expid[i]]['zh'][ialltime][itime, :, ilat, ilon].values / 1000
        
        plt_line = ax.plot(
            x_data, y_data, '.-', color='tab:blue', lw=0.5, markersize=2.5,)
        
        # t_it, h_it = inversion_top(x_data, y_data)
        # if (not np.isnan(t_it)):
        #     plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
        plt_scatter = ax.scatter(
            inversion_height_strength[isite][ialltime]['Inversion temperature'][itime].values,
            inversion_height_strength[isite][ialltime]['Inversion height'][itime].values / 1000 + site_elevation / 1000,
            s=5, c='red', zorder=3)
        
        if (ialltime in ['mon', 'sea', 'ann']):
            timelabel = str(zh_st_ml[expid[i]]['st'][ialltime][itime].time.values)[:7]
        elif (ialltime == 'mm'):
            timelabel = month[zh_st_ml[expid[i]]['st']['mm'][itime].month.values-1]
        elif (ialltime == 'sm'):
            timelabel = zh_st_ml[expid[i]]['st']['sm'][itime].season.values
        elif (ialltime == 'daily'):
            timelabel = str(zh_st_ml[expid[i]]['st'][ialltime][itime].time.values)[:10]
        
        plt_text = ax.text(
            0.1, 0.1, timelabel,
            transform=ax.transAxes, color='k', ha='left', va = 'center')
        
        # radiosonde data
        idate = np.where(date == zh_st_ml[expid[i]]['st'][ialltime][itime].time.dt.floor('12h').values)[0]
        if (len(idate) > 0):
            altitude = EDC_df_drvd.iloc[
                np.where(EDC_df_drvd.date == date[idate[0]])[0]][
                'calculated_height'].values / 1000
            temperature = EDC_df_drvd.iloc[
                np.where(EDC_df_drvd.date == date[idate[0]])[0]][
                'temperature'].values
            plt_line2 = ax.plot(
                temperature, altitude,
                '.-', color='black', lw=0.5, markersize=2.5,)
            
            t_it, h_it = inversion_top(temperature, altitude)
            if (not np.isnan(t_it)):
                plt_scatter2 = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
                ims.append(plt_line + [plt_text, plt_scatter] + plt_line2 + [plt_scatter2])
            else:
                ims.append(plt_line + [plt_text, plt_scatter] + plt_line2)
        else:
            ims.append(plt_line + [plt_text, plt_scatter])
    
    ax.set_xticks(np.arange(232, 250.1, 2))
    ax.set_xlim(232, 250)
    ax.set_xlabel('Temperature [$K$]')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(3, 5.1, 0.5))
    ax.set_ylim(3, 5)
    ax.set_ylabel('Altitude [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    # plt.text(
    #     0.9, 0.9, isite, transform=ax.transAxes, color='k',
    #     ha='right', va = 'center')
    ax.legend(
        [plt_line[0], plt_line2[0]],
        ['ECHAM6', 'Radiosonde'],
        loc='upper right',
    )
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)




'''


'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot an example of inversion

i = 0
imon = 0

for isite in t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    x_data = st_plev[expid[i]]['mon'][imon, :, ilat, ilon].values - zerok
    y_data = st_plev[expid[i]]['mon'][imon, :, ilat, ilon].plev.values / 100
    
    subset = np.isfinite(x_data) & (y_data >= 200)
    x_data = x_data[subset]
    y_data = y_data[subset]
    
    xlim_min = np.min(x_data) - 2
    xlim_max = np.max(x_data) + 2
    x_interval = 10
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    ylim_min = np.min(y_data) - 10
    ylim_max = np.max(y_data) + 10
    y_interval = 100
    ytickmin = np.ceil(ylim_min / y_interval) * y_interval
    ytickmax = np.floor(ylim_max / y_interval) * y_interval
    
    output_png = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.0 ' + expid[i] + ' example of monthly inveresion at ' + isite + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        x_data,
        y_data,
    )
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel('Temperature [$°C$]', labelpad=3)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, y_interval))
    ax.invert_yaxis()
    ax.set_ylabel('Pressure [$hPa$]', labelpad=3)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    plt_text = plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.25, top=0.95)
    plt.savefig(output_png)
    plt.close()




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate mon/mm/sm inversion

i = 0
ialltime = 'mon'
# ialltime = 'mm'
# ialltime = 'sm'

for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0.1 ' + expid[i] + ' ' + ialltime + ' inveresion at ' + isite + '.mp4'
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    #-------------------------------- plot
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ims = []
    
    for itime in range(st_plev[expid[i]][ialltime].shape[0]):
        # itime = 0
        # print(itime)
        
        x_data = st_plev[expid[i]][ialltime][itime, :, ilat, ilon].values - zerok
        y_data = st_plev[expid[i]][ialltime][itime, :, ilat, ilon].plev.values / 100
        
        subset = np.isfinite(x_data) & (y_data >= 200)
        x_data = x_data[subset]
        y_data = y_data[subset]
        
        plt_line = ax.plot(
            x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
        
        if (ialltime in ['mon', 'sea', 'ann']):
            timelabel = str(st_plev[expid[i]][ialltime][itime].time.values)[:7]
        elif (ialltime == 'mm'):
            timelabel = month[st_plev[expid[i]]['mm'][itime].month.values-1]
        elif (ialltime == 'sm'):
            timelabel = st_plev[expid[i]]['sm'][itime].season.values
        
        plt_text = ax.text(
            0.1, 0.1, timelabel,
            transform=ax.transAxes, color='gray',
            ha='left', va = 'center')
        
        ims.append(plt_line + [plt_text])
    
    xlim_min = -80
    xlim_max = -20
    x_interval = 20
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    ylim_min = 200
    ylim_max = 700
    y_interval = 100
    ytickmin = np.ceil(ylim_min / y_interval) * y_interval
    ytickmax = np.floor(ylim_max / y_interval) * y_interval
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel('Temperature [$°C$]', labelpad=3)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, y_interval))
    ax.invert_yaxis()
    ax.set_ylabel('Pressure [$hPa$]', labelpad=3)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.25, top=0.95)
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)




'''
t63_sites_indices['EDC']

era5_data = xr.open_dataset('scratch/products/era5/evap/era5_mon_evap_1979_2021.nc')
era5_data.e.sel(longitude=123.35, latitude=-75.1, method='nearest')

'''
# endregion
# -----------------------------------------------------------------------------


