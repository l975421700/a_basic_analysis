

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
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
import statsmodels.api as sm

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

from a_basic_analysis.b_module.statistics import (
    xr_par_cor,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


SO_vapor_isotopes_SLMSIC = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl', 'rb') as f:
    SO_vapor_isotopes_SLMSIC[expid[i]] = pickle.load(f)


# source RHsst and SST
q_sfc_weighted_RHsst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_RHsst' + '.pkl',
          'rb') as f:
    q_sfc_weighted_RHsst[expid[i]] = pickle.load(f)

q_sfc_weighted_sst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_sst' + '.pkl',
          'rb') as f:
    q_sfc_weighted_sst[expid[i]] = pickle.load(f)


# local RHsst and SST
RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)


# ERA5 RHsst and SST
ERA5_daily_RHsst_2013_2022 = xr.open_dataset('scratch/ERA5/RHsst/ERA5_daily_RHsst_2013_2022.nc')

ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc')


'''
with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# plot
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot observed d_ln/d_xs vs. ERA5 RHsst/sst

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst', 'sst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = ERA5_SO_vapour_RHsst.copy()
            xlabel = 'RHsst [$\%$]'
        elif (ivar == 'sst'):
            xdata = ERA5_SO_vapour_SST.copy()
            xlabel = 'SST [$°C$]'
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        linearfit = linregress(x = xdata, y = ydata,)
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 SO daily observed ' + iisotope + ' vs. ERA5 ' + ivar + '.png'
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=12, lw=0.5, facecolors='white', edgecolors='k',)
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
                0.95, 0.95, eq_text,
                transform=ax.transAxes, linespacing=1.5,
                ha='right', va='top')
        
        ax.set_ylabel(plot_labels[iisotope], labelpad=6)
        ax.set_ylim(ymin_value-2, ymax_value+2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_xlim(xmin_value-2, xmax_value+2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both')
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.22, right=0.98, bottom=0.20, top=0.98)
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
# region plot simulated d_ln/d_xs vs. simulated source RHsst/sst

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

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
    
    for ivar in ['RHsst', 'sst', ]:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'sst'):
            xdata = SO_vapour_src_sst.copy()
        elif (ivar == 'RHsst'):
            xdata = SO_vapour_src_RHsst.copy()
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope + '_sim'].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        linearfit = linregress(x = xdata, y = ydata,)
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' SO daily simulated ' + iisotope + ' vs. source ' + ivar + '.png'
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=12, lw=0.5, facecolors='white', edgecolors='k',)
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
                0.95, 0.95, eq_text,
                transform=ax.transAxes, linespacing=1.5,
                va='top', ha='right')
        
        ax.set_ylabel(plot_labels[iisotope])
        ax.set_ylim(ymin_value-2, ymax_value+2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.set_xlabel(plot_labels[ivar])
        ax.set_xlim(xmin_value-2, xmax_value+2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both',)
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.22, right=0.98, bottom=0.20, top=0.98)
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
# region plot simulated d_ln/d_xs vs. simulated RHsst/sst

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    tsw_alltime[expid[i]]['daily']['time'].values,
    tsw_alltime[expid[i]]['daily']['lat'].values,
    tsw_alltime[expid[i]]['daily']['lon'].values,
    tsw_alltime[expid[i]]['daily'].values,
)

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_ln'
    print('#-------------------------------- ' + iisotope)
    
    for ivar in ['RHsst', 'sst']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        if (ivar == 'RHsst'):
            xdata = SO_vapour_RHsst.copy()
            xlabel = 'RHsst [$\%$]'
        elif (ivar == 'sst'):
            xdata = SO_vapour_SST.copy()
            xlabel = 'SST [$°C$]'
        
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][iisotope + '_sim'].values.copy()
        
        subset1 = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        linearfit = linregress(x = xdata, y = ydata,)
        
        # print(pearsonr(xdata, ydata))
        
        xmax_value = np.max(xdata)
        xmin_value = np.min(xdata)
        ymax_value = np.max(ydata)
        ymin_value = np.min(ydata)
        
        output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.3 ' + expid[i] + ' SO daily simulated ' + iisotope + ' vs. local ' + ivar + '.png'
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
        
        ax.scatter(
            xdata, ydata,
            s=12, lw=0.5, facecolors='white', edgecolors='k',)
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
                0.95, 0.95, eq_text,
                transform=ax.transAxes, linespacing=1.5,
                ha='right', va='top')
        
        ax.set_ylabel(plot_labels[iisotope])
        ax.set_ylim(ymin_value-2, ymax_value+2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.set_xlabel(xlabel)
        ax.set_xlim(xmin_value-2, xmax_value+2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both')
        
        ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        fig.subplots_adjust(
            left=0.22, right=0.98, bottom=0.20, top=0.98)
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


# check
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check correlation between observed d_ln/d_xs and ERA5 RHsst/sst all data


#-------------------------------- check d_ln & RHsst/sst all data
# When controlling ERA5 RHsst, partial correlation between observed d_ln and ERA5 SST are insignificant.

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)


SO_vapour_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_ln'].values.copy()

subset1 = np.isfinite(SO_vapour_d_ln) & np.isfinite(ERA5_SO_vapour_RHsst) & np.isfinite(ERA5_SO_vapour_SST)
SO_vapour_d_ln = SO_vapour_d_ln[subset1]
ERA5_SO_vapour_RHsst = ERA5_SO_vapour_RHsst[subset1]
ERA5_SO_vapour_SST = ERA5_SO_vapour_SST[subset1]


pearsonr(SO_vapour_d_ln, ERA5_SO_vapour_SST)
pearsonr(SO_vapour_d_ln, ERA5_SO_vapour_RHsst)

xr_par_cor(SO_vapour_d_ln, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst)
xr_par_cor(SO_vapour_d_ln, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst,output='p')




#-------------------------------- check d_xs & RHsst/sst all data
# When controlling ERA5 RHsst, partial correlation between observed d_xs and ERA5 SST are insignificant.

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)


SO_vapour_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()

subset1 = np.isfinite(SO_vapour_d_xs) & np.isfinite(ERA5_SO_vapour_RHsst) & np.isfinite(ERA5_SO_vapour_SST)
SO_vapour_d_xs = SO_vapour_d_xs[subset1]
ERA5_SO_vapour_RHsst = ERA5_SO_vapour_RHsst[subset1]
ERA5_SO_vapour_SST = ERA5_SO_vapour_SST[subset1]


pearsonr(SO_vapour_d_xs, ERA5_SO_vapour_RHsst)
pearsonr(SO_vapour_d_xs, ERA5_SO_vapour_SST)


xr_par_cor(SO_vapour_d_xs, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst)
xr_par_cor(SO_vapour_d_xs, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    SO_vapour_d_xs,
    sm.add_constant(np.column_stack((
        ERA5_SO_vapour_RHsst, ERA5_SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * ERA5_SO_vapour_RHsst + ols_fit.params[2] * ERA5_SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - SO_vapour_d_xs)))
RMSE



#-------------------------------- check d_xs & RHsst/sst single dataset
# For each single dataset, when controlling ERA5 RHsst, partial correlation between observed d_xs and ERA5 SST are insignificant.

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)'))

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)


SO_vapour_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs'].values.copy()

subset1 = np.isfinite(SO_vapour_d_xs) & np.isfinite(ERA5_SO_vapour_RHsst) & np.isfinite(ERA5_SO_vapour_SST)
SO_vapour_d_xs = SO_vapour_d_xs[subset1]
ERA5_SO_vapour_RHsst = ERA5_SO_vapour_RHsst[subset1]
ERA5_SO_vapour_SST = ERA5_SO_vapour_SST[subset1]

pearsonr(SO_vapour_d_xs, ERA5_SO_vapour_RHsst)
pearsonr(SO_vapour_d_xs, ERA5_SO_vapour_SST)

xr_par_cor(SO_vapour_d_xs, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst)
xr_par_cor(SO_vapour_d_xs, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    SO_vapour_d_xs,
    sm.add_constant(np.column_stack((
        ERA5_SO_vapour_RHsst, ERA5_SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * ERA5_SO_vapour_RHsst + ols_fit.params[2] * ERA5_SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - SO_vapour_d_xs)))
RMSE


#-------------------------------- check d_ln & RHsst/sst single dataset
# For each single dataset, when controlling ERA5 RHsst, partial correlation between observed d_ln and ERA5 SST are insignificant.

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)'))

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)


SO_vapour_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_ln'].values.copy()

subset1 = np.isfinite(SO_vapour_d_ln) & np.isfinite(ERA5_SO_vapour_RHsst) & np.isfinite(ERA5_SO_vapour_SST)
SO_vapour_d_ln = SO_vapour_d_ln[subset1]
ERA5_SO_vapour_RHsst = ERA5_SO_vapour_RHsst[subset1]
ERA5_SO_vapour_SST = ERA5_SO_vapour_SST[subset1]

pearsonr(SO_vapour_d_ln, ERA5_SO_vapour_RHsst)
pearsonr(SO_vapour_d_ln, ERA5_SO_vapour_SST)

xr_par_cor(SO_vapour_d_ln, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst)
xr_par_cor(SO_vapour_d_ln, ERA5_SO_vapour_SST, ERA5_SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    SO_vapour_d_ln,
    sm.add_constant(np.column_stack((
        ERA5_SO_vapour_RHsst, ERA5_SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * ERA5_SO_vapour_RHsst + ols_fit.params[2] * ERA5_SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - SO_vapour_d_ln)))
RMSE


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check correlation between observed d_ln/d_xs and RHsst/sst in NK16
# While controlling observed RHsst, partial correlation between observed d_xs and SST from NK16 is still significantly correlated. This is consistent with the finding of Bonne et al. (2019).

with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)

T63GR15_jan_surf = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/input/echam/unit.24')
NK16_1d_SLM = find_multi_gridvalue_at_site(
    NK16_Australia_Syowa['1d']['lat'].values,
    NK16_Australia_Syowa['1d']['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
    )

ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})
NK16_1d_SIC = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'],
    NK16_Australia_Syowa['1d']['lat'],
    NK16_Australia_Syowa['1d']['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc
    )

# check d_xs

NK_1d_d_xs = NK16_Australia_Syowa['1d']['d_xs'].values
NK_1d_RHsst = NK16_Australia_Syowa['1d']['rh_sst'].values
NK_1d_SST = NK16_Australia_Syowa['1d']['sst'].values

subset = np.isfinite(NK_1d_d_xs) & np.isfinite(NK_1d_RHsst) & np.isfinite(NK_1d_SST) & (NK16_1d_SLM == 0) & (NK16_1d_SIC == 0) & (NK16_Australia_Syowa['1d']['lat'] <= -20) & (NK16_Australia_Syowa['1d']['lat'] >= -60)
NK_1d_d_xs = NK_1d_d_xs[subset]
NK_1d_RHsst = NK_1d_RHsst[subset]
NK_1d_SST = NK_1d_SST[subset]

pearsonr(NK_1d_d_xs, NK_1d_RHsst)
pearsonr(NK_1d_d_xs, NK_1d_SST)


xr_par_cor(NK_1d_d_xs, NK_1d_SST, NK_1d_RHsst)
xr_par_cor(NK_1d_d_xs, NK_1d_SST, NK_1d_RHsst,output='p')


ols_fit = sm.OLS(
    NK_1d_d_xs,
    sm.add_constant(np.column_stack((
        NK_1d_RHsst, NK_1d_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * NK_1d_RHsst + ols_fit.params[2] * NK_1d_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - NK_1d_d_xs)))
RMSE



# check d_ln

NK_1d_d_ln = NK16_Australia_Syowa['1d']['d_ln'].values
NK_1d_RHsst = NK16_Australia_Syowa['1d']['rh_sst'].values
NK_1d_SST = NK16_Australia_Syowa['1d']['sst'].values

subset = np.isfinite(NK_1d_d_ln) & np.isfinite(NK_1d_RHsst) & np.isfinite(NK_1d_SST) & (NK16_1d_SLM == 0) & (NK16_1d_SIC == 0) & (NK16_Australia_Syowa['1d']['lat'] <= -20) & (NK16_Australia_Syowa['1d']['lat'] >= -60)
NK_1d_d_ln = NK_1d_d_ln[subset]
NK_1d_RHsst = NK_1d_RHsst[subset]
NK_1d_SST = NK_1d_SST[subset]

pearsonr(NK_1d_d_ln, NK_1d_RHsst)
pearsonr(NK_1d_d_ln, NK_1d_SST)


xr_par_cor(NK_1d_d_ln, NK_1d_SST, NK_1d_RHsst)
xr_par_cor(NK_1d_d_ln, NK_1d_SST, NK_1d_RHsst,output='p')


ols_fit = sm.OLS(
    NK_1d_d_ln,
    sm.add_constant(np.column_stack((
        NK_1d_RHsst, NK_1d_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * NK_1d_RHsst + ols_fit.params[2] * NK_1d_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - NK_1d_d_ln)))
RMSE


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check correlation between simulated d_ln/d_xs and local RHsst/sst in Nudge_control

#-------------------------------- check d_xs & RHsst/sst all data
# When controlling RHsst, partial correlation between d_xs and sst is still significant

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    tsw_alltime[expid[i]]['daily']['time'].values,
    tsw_alltime[expid[i]]['daily']['lat'].values,
    tsw_alltime[expid[i]]['daily']['lon'].values,
    tsw_alltime[expid[i]]['daily'].values,
)


sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs_sim'].values.copy()

pearsonr(sim_d_xs, SO_vapour_RHsst)
pearsonr(sim_d_xs, SO_vapour_SST)


xr_par_cor(sim_d_xs, SO_vapour_SST, SO_vapour_RHsst)
xr_par_cor(sim_d_xs, SO_vapour_SST, SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_xs,
    sm.add_constant(np.column_stack((
        SO_vapour_RHsst, SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_RHsst + ols_fit.params[2] * SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_xs)))
RMSE


#-------------------------------- check d_xs & RHsst/sst single dataset
# For each dataset, when controlling RHsst, partial correlation between d_xs and sst is still significant


subset = np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln']) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    tsw_alltime[expid[i]]['daily']['time'].values,
    tsw_alltime[expid[i]]['daily']['lat'].values,
    tsw_alltime[expid[i]]['daily']['lon'].values,
    tsw_alltime[expid[i]]['daily'].values,
)


sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs_sim'].values.copy()

pearsonr(sim_d_xs, SO_vapour_RHsst)
pearsonr(sim_d_xs, SO_vapour_SST)


xr_par_cor(sim_d_xs, SO_vapour_SST, SO_vapour_RHsst)
xr_par_cor(sim_d_xs, SO_vapour_SST, SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_xs,
    sm.add_constant(np.column_stack((
        SO_vapour_RHsst, SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_RHsst + ols_fit.params[2] * SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_xs)))
RMSE



#-------------------------------- check d_ln & RHsst/sst single dataset
# For each dataset, when controlling RHsst, partial correlation between d_ln and sst is still significant


subset = np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln']) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'][subset].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'][subset].values,
    tsw_alltime[expid[i]]['daily']['time'].values,
    tsw_alltime[expid[i]]['daily']['lat'].values,
    tsw_alltime[expid[i]]['daily']['lon'].values,
    tsw_alltime[expid[i]]['daily'].values,
)


sim_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_ln_sim'].values.copy()

pearsonr(sim_d_ln, SO_vapour_RHsst)
pearsonr(sim_d_ln, SO_vapour_SST)


xr_par_cor(sim_d_ln, SO_vapour_SST, SO_vapour_RHsst)
xr_par_cor(sim_d_ln, SO_vapour_SST, SO_vapour_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_ln,
    sm.add_constant(np.column_stack((
        SO_vapour_RHsst, SO_vapour_SST))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_RHsst + ols_fit.params[2] * SO_vapour_SST

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_ln)))
RMSE



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check correlation between simulated d_ln/d_xs and source RHsst/sst in Nudge_control

#-------------------------------- check d_xs & source RHsst/sst all data
# While controlling source RHsst, partial correlation between d_xs and source sst is still significant.

subset = np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln']) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60)

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


sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs_sim'].values.copy()

pearsonr(sim_d_xs, SO_vapour_src_RHsst)
pearsonr(sim_d_xs, SO_vapour_src_sst)


xr_par_cor(sim_d_xs, SO_vapour_src_sst, SO_vapour_src_RHsst)
xr_par_cor(sim_d_xs, SO_vapour_src_sst, SO_vapour_src_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_xs,
    sm.add_constant(np.column_stack((
        SO_vapour_src_RHsst, SO_vapour_src_sst))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_src_RHsst + ols_fit.params[2] * SO_vapour_src_sst

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_xs)))
RMSE


#-------------------------------- check d_xs & source RHsst/sst single dataset
# For each dataset, while controlling source RHsst, partial correlation between d_xs and source sst is still significant.

subset = np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln']) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

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


sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_xs_sim'].values.copy()

pearsonr(sim_d_xs, SO_vapour_src_RHsst)
pearsonr(sim_d_xs, SO_vapour_src_sst)


xr_par_cor(sim_d_xs, SO_vapour_src_sst, SO_vapour_src_RHsst)
xr_par_cor(sim_d_xs, SO_vapour_src_sst, SO_vapour_src_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_xs,
    sm.add_constant(np.column_stack((
        SO_vapour_src_RHsst, SO_vapour_src_sst))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_src_RHsst + ols_fit.params[2] * SO_vapour_src_sst

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_xs)))
RMSE



#-------------------------------- check d_ln & source RHsst/sst single dataset
# For each dataset, while controlling source RHsst, partial correlation between d_ln and source sst is still significant.

subset = np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln']) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

# Kurita et al. (2016), Bonne et al. (2019), 'Thurnherr et al. (2020)'

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


sim_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d_ln_sim'].values.copy()

pearsonr(sim_d_ln, SO_vapour_src_RHsst)
pearsonr(sim_d_ln, SO_vapour_src_sst)


xr_par_cor(sim_d_ln, SO_vapour_src_sst, SO_vapour_src_RHsst)
xr_par_cor(sim_d_ln, SO_vapour_src_sst, SO_vapour_src_RHsst,output='p')


ols_fit = sm.OLS(
    sim_d_ln,
    sm.add_constant(np.column_stack((
        SO_vapour_src_RHsst, SO_vapour_src_sst))),
    ).fit()

ols_fit.summary()
ols_fit.params
ols_fit.rsquared

predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_src_RHsst + ols_fit.params[2] * SO_vapour_src_sst

RMSE = np.sqrt(np.average(np.square(predicted_y - sim_d_ln)))
RMSE



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
