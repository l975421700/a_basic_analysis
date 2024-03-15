

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
    plot_labels_no_unit,
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
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_RHsst.pkl', 'rb') as f:
    q_sfc_weighted_RHsst[expid[i]] = pickle.load(f)

q_sfc_weighted_sst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_sst.pkl', 'rb') as f:
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


# -----------------------------------------------------------------------------
# region plot regression of observed d_ln/d_xs and RHsst/sst in NK16


#-------------------------------- get data

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

NK_1d_d_ln = NK16_Australia_Syowa['1d']['d_ln'].values
NK_1d_d_xs = NK16_Australia_Syowa['1d']['d_xs'].values
NK_1d_RHsst = NK16_Australia_Syowa['1d']['rh_sst'].values
NK_1d_SST = NK16_Australia_Syowa['1d']['sst'].values

subset = np.isfinite(NK_1d_d_ln) & np.isfinite(NK_1d_d_xs) & np.isfinite(NK_1d_RHsst) & np.isfinite(NK_1d_SST) & \
    (NK16_1d_SLM == 0) & (NK16_1d_SIC == 0) & (NK16_Australia_Syowa['1d']['lat'] <= -20) & (NK16_Australia_Syowa['1d']['lat'] >= -60)

NK_1d_d_ln = NK_1d_d_ln[subset]
NK_1d_d_xs = NK_1d_d_xs[subset]
NK_1d_RHsst = NK_1d_RHsst[subset]
NK_1d_SST = NK_1d_SST[subset]


#-------------------------------- regression and plot

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_xs'
    print('#-------------------------------- ' + iisotope)
    
    if (iisotope == 'd_xs'):
        NK_1d_isotope = NK_1d_d_xs.copy()
    if (iisotope == 'd_ln'):
        NK_1d_isotope = NK_1d_d_ln.copy()
    
    ols_fit = sm.OLS(
        NK_1d_isotope,
        sm.add_constant(np.column_stack((
            NK_1d_RHsst, NK_1d_SST))),
        ).fit()
    
    # ols_fit.summary()
    # ols_fit.params
    # ols_fit.rsquared
    
    predicted_y = ols_fit.params[0] + ols_fit.params[1] * NK_1d_RHsst + ols_fit.params[2] * NK_1d_SST
    
    rsquared = pearsonr(NK_1d_isotope, predicted_y).statistic ** 2
    RMSE = np.sqrt(np.average(np.square(predicted_y - NK_1d_isotope)))
    
    eq_text = plot_labels_no_unit[iisotope] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'RHsst+' + \
            str(np.round(ols_fit.params[2], 2)) + 'SST+' + \
                str(np.round(ols_fit.params[0], 1)) + '$' + \
            '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
                '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'
    
    xymax = np.max(np.concatenate((NK_1d_isotope, predicted_y)))
    xymin = np.min(np.concatenate((NK_1d_isotope, predicted_y)))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.4 daily obs ' + iisotope + ' vs. obs RHsst and SST from NK16.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    ax.scatter(
        NK_1d_isotope, predicted_y,
        s=12, lw=1, facecolors='white', edgecolors='k',)
    ax.axline((0, 0), slope = 1, lw=0.5, color='k')
    
    plt.text(
        0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
        va='top', ha='left',)
    
    ax.set_xlabel('Observed ' + plot_labels[iisotope] + ' from Kurita et al. (2016)',)
    ax.set_xlim(xymin-2, xymax+2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Estimated ' + plot_labels[iisotope],)
    ax.set_ylim(xymin-2, xymax+2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot regression of simulated d_ln/d_xs and RHsst/sst in Nudge_control

#-------------------------------- get data

SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'].values,
    RHsst_alltime[expid[i]]['daily']['time'].values,
    RHsst_alltime[expid[i]]['daily']['lat'].values,
    RHsst_alltime[expid[i]]['daily']['lon'].values,
    RHsst_alltime[expid[i]]['daily'].values,
)

SO_vapour_SST = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'].values,
    tsw_alltime[expid[i]]['daily']['time'].values,
    tsw_alltime[expid[i]]['daily']['lat'].values,
    tsw_alltime[expid[i]]['daily']['lon'].values,
    tsw_alltime[expid[i]]['daily'].values,
)

obs_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln'].values.copy()
obs_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]]['d_xs'].values.copy()
sim_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln_sim'].values.copy()
sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]]['d_xs_sim'].values.copy()

subset = np.isfinite(obs_d_ln) & np.isfinite(obs_d_xs) & np.isfinite(sim_d_ln) & np.isfinite(sim_d_xs) & np.isfinite(SO_vapour_RHsst) & np.isfinite(SO_vapour_SST) & \
    (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

sim_d_ln = sim_d_ln[subset]
sim_d_xs = sim_d_xs[subset]
SO_vapour_RHsst = SO_vapour_RHsst[subset]
SO_vapour_SST = SO_vapour_SST[subset]


#-------------------------------- regression and plot

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_xs'
    print('#-------------------------------- ' + iisotope)
    
    if (iisotope == 'd_xs'):
        sim_isotope = sim_d_xs.copy()
    if (iisotope == 'd_ln'):
        sim_isotope = sim_d_ln.copy()
    
    ols_fit = sm.OLS(
        sim_isotope,
        sm.add_constant(np.column_stack((SO_vapour_RHsst, SO_vapour_SST))),
        ).fit()
    
    # ols_fit.summary()
    # ols_fit.params
    # ols_fit.rsquared
    
    predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_RHsst + ols_fit.params[2] * SO_vapour_SST
    
    rsquared = pearsonr(sim_isotope, predicted_y).statistic ** 2
    RMSE = np.sqrt(np.average(np.square(predicted_y - sim_isotope)))
    
    eq_text = plot_labels_no_unit[iisotope] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'RHsst+' + \
            str(np.round(ols_fit.params[2], 2)) + 'SST+' + \
                str(np.round(ols_fit.params[0], 1)) + '$' + \
            '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
                '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'
    
    xymax = np.max(np.concatenate((sim_isotope, predicted_y)))
    xymin = np.min(np.concatenate((sim_isotope, predicted_y)))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.4 ' + expid[i] + ' daily ' + iisotope + ' vs. local RHsst and SST.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    ax.scatter(
        sim_isotope, predicted_y,
        s=12, lw=1, facecolors='white', edgecolors='k',)
    ax.axline((0, 0), slope = 1, lw=0.5, color='k')
    
    plt.text(
        0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
        va='top', ha='left',)
    
    ax.set_xlabel('Simulated ' + plot_labels[iisotope] + ' from ' + expid_labels[expid[i]],)
    ax.set_xlim(xymin-2, xymax+2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Estimated ' + plot_labels[iisotope],)
    ax.set_ylim(xymin-2, xymax+2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot regression of simulated d_ln/d_xs and source RHsst/sst in Nudge_control

#-------------------------------- get data

SO_vapour_src_RHsst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_RHsst[expid[i]]['daily'].values,
) * 100

SO_vapour_src_sst = find_multi_gridvalue_at_site_time(
    SO_vapor_isotopes_SLMSIC[expid[i]]['time'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lat'].values,
    SO_vapor_isotopes_SLMSIC[expid[i]]['lon'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['time'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lat'].values,
    q_sfc_weighted_sst[expid[i]]['daily']['lon'].values,
    q_sfc_weighted_sst[expid[i]]['daily'].values,
)

obs_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln'].values.copy()
obs_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]]['d_xs'].values.copy()
sim_d_ln = SO_vapor_isotopes_SLMSIC[expid[i]]['d_ln_sim'].values.copy()
sim_d_xs = SO_vapor_isotopes_SLMSIC[expid[i]]['d_xs_sim'].values.copy()

subset = np.isfinite(obs_d_ln) & np.isfinite(obs_d_xs) & np.isfinite(sim_d_ln) & np.isfinite(sim_d_xs) & np.isfinite(SO_vapour_src_RHsst) & np.isfinite(SO_vapour_src_sst) & \
    (SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == 'Kurita et al. (2016)')

sim_d_ln = sim_d_ln[subset]
sim_d_xs = sim_d_xs[subset]
SO_vapour_src_RHsst = SO_vapour_src_RHsst[subset]
SO_vapour_src_sst = SO_vapour_src_sst[subset]


#-------------------------------- regression and plot

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_xs'
    print('#-------------------------------- ' + iisotope)
    
    if (iisotope == 'd_xs'):
        sim_isotope = sim_d_xs.copy()
    if (iisotope == 'd_ln'):
        sim_isotope = sim_d_ln.copy()
    
    ols_fit = sm.OLS(
        sim_isotope,
        sm.add_constant(np.column_stack((SO_vapour_src_RHsst, SO_vapour_src_sst))),
        ).fit()
    
    # ols_fit.summary()
    # ols_fit.params
    # ols_fit.rsquared
    
    predicted_y = ols_fit.params[0] + ols_fit.params[1] * SO_vapour_src_RHsst + ols_fit.params[2] * SO_vapour_src_sst
    
    rsquared = pearsonr(sim_isotope, predicted_y).statistic ** 2
    RMSE = np.sqrt(np.average(np.square(predicted_y - sim_isotope)))
    
    eq_text = plot_labels_no_unit[iisotope] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'srcRHsst+' + \
            str(np.round(ols_fit.params[2], 2)) + 'srcSST+' + \
                str(np.round(ols_fit.params[0], 1)) + '$' + \
            '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
                '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'
    
    xymax = np.max(np.concatenate((sim_isotope, predicted_y)))
    xymin = np.min(np.concatenate((sim_isotope, predicted_y)))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.4 ' + expid[i] + ' daily ' + iisotope + ' vs. source RHsst and SST.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    ax.scatter(
        sim_isotope, predicted_y,
        s=12, lw=1, facecolors='white', edgecolors='k',)
    ax.axline((0, 0), slope = 1, lw=0.5, color='k')
    
    plt.text(
        0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
        va='top', ha='left',)
    
    ax.set_xlabel('Simulated ' + plot_labels[iisotope] + ' from ' + expid_labels[expid[i]],)
    ax.set_xlim(xymin-2, xymax+2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Estimated ' + plot_labels[iisotope],)
    ax.set_ylim(xymin-2, xymax+2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot regression of observed d_ln/d_xs in NK16 and ERA5 RHsst/sst


#-------------------------------- get data

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

ERA5_SO_vapour_RHsst = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'],
    NK16_Australia_Syowa['1d']['lat'],
    NK16_Australia_Syowa['1d']['lon'],
    ERA5_daily_RHsst_2013_2022['time'].values,
    ERA5_daily_RHsst_2013_2022['latitude'].values,
    ERA5_daily_RHsst_2013_2022['longitude'].values,
    ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'],
)

ERA5_SO_vapour_SST = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'],
    NK16_Australia_Syowa['1d']['lat'],
    NK16_Australia_Syowa['1d']['lon'],
    ERA5_daily_SST_2013_2022['time'].values,
    ERA5_daily_SST_2013_2022['latitude'].values,
    ERA5_daily_SST_2013_2022['longitude'].values,
    ERA5_daily_SST_2013_2022['sst'],
)

NK_1d_d_ln = NK16_Australia_Syowa['1d']['d_ln'].values
NK_1d_d_xs = NK16_Australia_Syowa['1d']['d_xs'].values
# NK_1d_RHsst = NK16_Australia_Syowa['1d']['rh_sst'].values
# NK_1d_SST = NK16_Australia_Syowa['1d']['sst'].values

subset = np.isfinite(NK_1d_d_ln) & np.isfinite(NK_1d_d_xs) & np.isfinite(ERA5_SO_vapour_RHsst) & np.isfinite(ERA5_SO_vapour_SST) & \
    (NK16_1d_SLM == 0) & (NK16_1d_SIC == 0) & (NK16_Australia_Syowa['1d']['lat'] <= -20) & (NK16_Australia_Syowa['1d']['lat'] >= -60)

NK_1d_d_ln = NK_1d_d_ln[subset]
NK_1d_d_xs = NK_1d_d_xs[subset]
ERA5_SO_vapour_RHsst = ERA5_SO_vapour_RHsst[subset]
ERA5_SO_vapour_SST = ERA5_SO_vapour_SST[subset]


#-------------------------------- regression and plot

for iisotope in ['d_xs', 'd_ln']:
    # iisotope = 'd_xs'
    print('#-------------------------------- ' + iisotope)
    
    if (iisotope == 'd_xs'):
        NK_1d_isotope = NK_1d_d_xs.copy()
    if (iisotope == 'd_ln'):
        NK_1d_isotope = NK_1d_d_ln.copy()
    
    ols_fit = sm.OLS(
        NK_1d_isotope,
        sm.add_constant(np.column_stack((
            ERA5_SO_vapour_RHsst, ERA5_SO_vapour_SST))),
        ).fit()
    
    # ols_fit.summary()
    # ols_fit.params
    # ols_fit.rsquared
    
    predicted_y = ols_fit.params[0] + ols_fit.params[1] * ERA5_SO_vapour_RHsst + ols_fit.params[2] * ERA5_SO_vapour_SST
    
    rsquared = pearsonr(NK_1d_isotope, predicted_y).statistic ** 2
    RMSE = np.sqrt(np.average(np.square(predicted_y - NK_1d_isotope)))
    
    eq_text = plot_labels_no_unit[iisotope] + '$=' + \
        str(np.round(ols_fit.params[1], 2)) + 'RHsst+' + \
            str(np.round(ols_fit.params[2], 2)) + 'SST+' + \
                str(np.round(ols_fit.params[0], 1)) + '$' + \
            '\n$R^2=' + str(np.round(rsquared, 2)) + '$' + \
                '\n$RMSE=' + str(np.round(RMSE, 1)) + '‰$'
    
    xymax = np.max(np.concatenate((NK_1d_isotope, predicted_y)))
    xymin = np.min(np.concatenate((NK_1d_isotope, predicted_y)))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.4 daily obs ' + iisotope + ' from NK16 vs. ERA5 RHsst and SST.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    ax.scatter(
        NK_1d_isotope, predicted_y,
        s=12, lw=1, facecolors='white', edgecolors='k',)
    ax.axline((0, 0), slope = 1, lw=0.5, color='k')
    
    plt.text(
        0.05, 0.95, eq_text, transform=ax.transAxes, linespacing=2,
        va='top', ha='left',)
    
    ax.set_xlabel('Observed ' + plot_labels[iisotope] + ' from Kurita et al. (2016)',)
    ax.set_xlim(xymin-2, xymax+2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_ylabel('Estimated ' + plot_labels[iisotope],)
    ax.set_ylim(xymin-2, xymax+2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------




