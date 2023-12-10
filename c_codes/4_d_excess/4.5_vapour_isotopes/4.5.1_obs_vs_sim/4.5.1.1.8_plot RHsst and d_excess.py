

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    ]
i = 0


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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import pearsonr
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath

# self defined
from a_basic_analysis.b_module.mapplot import (
    remove_trailing_zero,
    remove_trailing_zero_pos,
    hemisphere_conic_plot,
    hemisphere_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
    expid_colours,
    expid_labels,
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

SO_vapor_isotopes_SLMSIC = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl', 'rb') as f:
    SO_vapor_isotopes_SLMSIC[expid[i]] = pickle.load(f)

q_sfc_weighted_RHsst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + 'RHsst' + '.pkl',
          'rb') as f:
    q_sfc_weighted_RHsst[expid[i]] = pickle.load(f)

q_sfc_weighted_sst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + 'sst' + '.pkl',
          'rb') as f:
    q_sfc_weighted_sst[expid[i]] = pickle.load(f)


RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)

ERA5_daily_RHsst_2013_2022 = xr.open_dataset('scratch/ERA5/RHsst/ERA5_daily_RHsst_2013_2022.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source RHsst&sst against d_excess

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (pd.DatetimeIndex(pd.to_datetime(SO_vapor_isotopes_SLMSIC[expid[i]]['time'], utc=True)).year < 2019))

SO_vapour_src_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily'].values,
) * 100

SO_vapour_src_sst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    q_sfc_weighted_sst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_sst[expid[i]]['daily'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['sst', 'RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'sst'):
            xdata = SO_vapour_src_sst.copy()
        elif (ivar == 'RHsst'):
            xdata = SO_vapour_src_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' SO daily ' + iisotope + ' vs. source ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot local RHsst&sst against d_excess

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (pd.DatetimeIndex(pd.to_datetime(SO_vapor_isotopes_SLMSIC[expid[i]]['time'], utc=True)).year < 2019))

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = SO_vapour_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' SO daily ' + iisotope + ' vs. local ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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
        
        ax.set_xlabel('RHsst [$\%$]', labelpad=2)
        ax.set_xlim(xmin_value, xmax_value)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', labelsize=8)
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.32, right=0.95, bottom=0.25, top=0.95)
        fig.savefig(output_png)








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot NK16 RHsst against d_excess

subset = ((NK16_Australia_Syowa['1d']['lat'] <= -20) & (NK16_Australia_Syowa['1d']['lat'] >= -60))

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = NK16_Australia_Syowa['1d'][subset]['rh_sst'].copy()
        
        ydata = NK16_Australia_Syowa['1d'][subset][iisotope].copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.4 NK16 daily ' + iisotope + ' vs. local ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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
        
        ax.set_xlabel('RHsst [$\%$]', labelpad=2)
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
# region plot source RHsst&sst against d_excess: single dataset

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (pd.DatetimeIndex(pd.to_datetime(SO_vapor_isotopes_SLMSIC[expid[i]]['time'], utc=True)).year < 2019) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)'))

SO_vapour_src_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily'].values,
) * 100

SO_vapour_src_sst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    q_sfc_weighted_sst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_sst[expid[i]]['daily'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['sst', 'RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'sst'):
            xdata = SO_vapour_src_sst.copy()
        elif (ivar == 'RHsst'):
            xdata = SO_vapour_src_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' NK16 daily ' + iisotope + ' vs. source ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot local RHsst&sst against d_excess: single dataset

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (pd.DatetimeIndex(pd.to_datetime(SO_vapor_isotopes_SLMSIC[expid[i]]['time'], utc=True)).year < 2019) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)'))

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = SO_vapour_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' NK16 daily ' + iisotope + ' vs. local ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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
        
        ax.set_xlabel('RHsst [$\%$]', labelpad=2)
        ax.set_xlim(xmin_value, xmax_value)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', labelsize=8)
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.32, right=0.95, bottom=0.25, top=0.95)
        fig.savefig(output_png)








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ERA5 local RHsst&sst against d_excess

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['time'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['latitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = ERA5_SO_vapour_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ERA5 SO daily ' + iisotope + ' vs. local ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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
        
        ax.set_xlabel('RHsst [$\%$]', labelpad=2)
        ax.set_xlim(xmin_value, xmax_value)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', labelsize=8)
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.32, right=0.95, bottom=0.25, top=0.95)
        fig.savefig(output_png)








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ERA5 local RHsst&sst against d_excess: single dataset

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)'))

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['time'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['latitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__']['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = ERA5_SO_vapour_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ERA5 NK16 daily ' + iisotope + ' vs. local ' + ivar + '.png'
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.1, facecolors='white', edgecolors='k',)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=0.5, color='k')
        
        if (linearfit.intercept >= 0):
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
        
        ax.set_xlabel('RHsst [$\%$]', labelpad=2)
        ax.set_xlim(xmin_value, xmax_value)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', labelsize=8)
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.32, right=0.95, bottom=0.25, top=0.95)
        fig.savefig(output_png)








'''
xdata = SO_vapour_src_sst.copy()
ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()
subset1 = np.isfinite(xdata) & np.isfinite(ydata)
xdata = xdata[subset1]
ydata = ydata[subset1]
pearsonr(xdata, ydata)
'''
# endregion
# -----------------------------------------------------------------------------
