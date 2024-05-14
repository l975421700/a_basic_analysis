

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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    remove_trailing_zero_pos_abs,
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
    plot_labels_no_unit,
    time_labels,
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


corr_sources_isotopes_q = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q.pkl', 'rb') as f:
    corr_sources_isotopes_q[expid[i]] = pickle.load(f)

par_corr_sources_isotopes_q={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q.pkl', 'rb') as f:
    par_corr_sources_isotopes_q[expid[i]] = pickle.load(f)

corr_sources_isotopes_q_zm = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)

par_corr_sources_isotopes_q_zm={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)


lon = corr_sources_isotopes_q[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes_q[expid[i]]['sst']['d_ln']['mon']['r'].lat
plev = corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']['r'].plev

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_sources_isotopes_q_zm

for ivar in ['sst', 'RHsst']:
    # ivar = 'sst'
    # ['sst', 'RHsst']
    print('#-------------------------------- ' + ivar)
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        # ['d_ln', 'd_xs',]
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['daily', ]:
            # ialltime = 'daily'
            # ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0 ' + expid[i] + ' zm ' + ialltime + ' corr. ' + iisotopes + ' vs. ' + ivar + ' SH.png'
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
            
            plt_mesh = ax.contourf(
                lat.sel(lat=slice(3, -90)),
                plev.sel(plev=slice(1e+5, 2e+4)) / 100,
                corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['r'].sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
                norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='neither',)
            
            # x-axis
            ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
            ax.set_xlim(0, -88.57)
            ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            # y-axis
            ax.invert_yaxis()
            ax.set_ylim(1000, 200)
            ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
            ax.set_ylabel('Pressure [$hPa$]')
            
            ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
            
            cbar = fig.colorbar(
                plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.7, ticks=pltticks,
                extend='neither', pad=0.1, fraction=0.04, anchor=(0.4, -1),)
            cbar.ax.set_xlabel('Correlation: ' + time_labels[ialltime] + ' ' + plot_labels_no_unit[ivar] + ' and ' + plot_labels_no_unit[iisotopes],)
            
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
            fig.savefig(output_png)





'''
corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_sources_isotopes_q_zm

for ivar in ['sst', 'RHsst']:
    # ivar = 'sst'
    # ['sst', 'RHsst']
    print('#-------------------------------- ' + ivar)
    
    ctr_var = list(set(['sst', 'RHsst']) - set([ivar]))[0]
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        # ['d_ln', 'd_xs',]
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['ann no am',]:
            # ialltime = 'mon'
            # ['mon', 'mon no mm', 'ann', 'ann no am']
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.1 ' + expid[i] + ' zm ' + ialltime + ' partial corr. ' + iisotopes + ' vs. ' + ivar + ' controlling ' + ctr_var + ' SH.png'
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
            
            plt_mesh = ax.contourf(
                lat.sel(lat=slice(3, -90)),
                plev.sel(plev=slice(1e+5, 2e+4)) / 100,
                par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
                norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='neither',)
            
            # x-axis
            ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
            ax.set_xlim(0, -88.57)
            ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            # y-axis
            ax.invert_yaxis()
            ax.set_ylim(1000, 200)
            ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
            ax.set_ylabel('Pressure [$hPa$]')
            
            ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
            
            cbar = fig.colorbar(
                plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.7, ticks=pltticks,
                extend='neither', pad=0.1, fraction=0.04, anchor=(0.4, -1),)
            cbar.ax.set_xlabel('Partial correlation: ' + plot_labels_no_unit[iisotopes] + ' & ' + plot_labels_no_unit[ivar] + ' while controlling ' + plot_labels_no_unit[ctr_var],)
            
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
            fig.savefig(output_png)





'''
corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_sources_isotopes_q

isite = 'EDC'

for ivar in ['sst', 'RHsst']:
    # ivar = 'sst'
    # ['sst', 'RHsst']
    print('#-------------------------------- ' + ivar)
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        # ['d_ln', 'd_xs',]
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['mon', 'mon no mm', 'ann', 'ann no am']:
            # ialltime = 'mon'
            # ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.2 ' + expid[i] + ' ' + ialltime + ' corr. ' + iisotopes + ' vs. ' + ivar + ' across ' + isite + ' SH.png'
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
            
            plt_mesh = ax.contourf(
                lat.sel(lat=slice(3, -90)),
                plev.sel(plev=slice(1e+5, 2e+4)) / 100,
                corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['r'].sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)).sel(lon=t63_sites_indices[isite]['lon'], method='nearest'),
                norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='neither',)
            
            # x-axis
            ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
            ax.set_xlim(0, -88.57)
            ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            # y-axis
            ax.invert_yaxis()
            ax.set_ylim(1000, 200)
            ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
            ax.set_ylabel('Pressure [$hPa$]')
            
            ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
            
            cbar = fig.colorbar(
                plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.7, ticks=pltticks,
                extend='neither', pad=0.1, fraction=0.04, anchor=(0.4, -1),)
            cbar.ax.set_xlabel('Correlation: ' + time_labels[ialltime] + ' ' + plot_labels_no_unit[ivar] + ' and ' + plot_labels_no_unit[iisotopes] + ' across ' + isite,)
            
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
            fig.savefig(output_png)





'''
corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_sources_isotopes_q

for ivar in ['sst', 'RHsst',]:
    # ivar = 'sst'
    # ['sst', 'RHsst']
    print('#-------------------------------- ' + ivar)
    
    ctr_var = list(set(['sst', 'RHsst']) - set([ivar]))[0]
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        # ['d_ln', 'd_xs',]
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['mon', 'mon no mm', 'ann', 'ann no am']:
            # ialltime = 'mon'
            # ['mon', 'mon no mm', 'ann', 'ann no am']
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.3 ' + expid[i] + ' ' + ialltime + ' partial corr. ' + iisotopes + ' vs. ' + ivar + ' controlling ' + ctr_var + ' across ' + isite + ' SH.png'
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
            
            plt_mesh = ax.contourf(
                lat.sel(lat=slice(3, -90)),
                plev.sel(plev=slice(1e+5, 2e+4)) / 100,
                par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)).sel(lon=t63_sites_indices[isite]['lon'], method='nearest'),
                norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='neither',)
            
            # x-axis
            ax.set_xticks(np.arange(0, -90 - 1e-4, -10))
            ax.set_xlim(0, -88.57)
            ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            # y-axis
            ax.invert_yaxis()
            ax.set_ylim(1000, 200)
            ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
            ax.set_ylabel('Pressure [$hPa$]')
            
            ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
            
            cbar = fig.colorbar(
                plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.7, ticks=pltticks,
                extend='neither', pad=0.1, fraction=0.04, anchor=(0.4, -1),)
            cbar.ax.set_xlabel('Partial correlation: ' + time_labels[ialltime] + ' ' + plot_labels_no_unit[ivar] + ' and ' + plot_labels_no_unit[iisotopes] + ', controlling ' + plot_labels_no_unit[ctr_var] + ' across ' + isite,)
            
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
            fig.savefig(output_png)





'''
corr_sources_isotopes_q_zm[expid[i]]['sst']['d_ln']['mon']
'''
# endregion
# -----------------------------------------------------------------------------



