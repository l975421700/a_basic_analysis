

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
    plot_labels_no_unit,
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

corr_sources_isotopes = {}
par_corr_sources_isotopes = {}
corr_temp2_isotopes = {}
par_corr_temp2_isotopes2 = {}
par_corr_sst_isotopes2 = {}
par_corr_isotopes_temp2_sst = {}
corr_sam_isotopes_sources_temp2 = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl', 'rb') as f:
        corr_sources_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes.pkl', 'rb') as f:
        par_corr_sources_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_temp2_isotopes.pkl', 'rb') as f:
        corr_temp2_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_temp2_isotopes2.pkl', 'rb') as f:
        par_corr_temp2_isotopes2[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sst_isotopes2.pkl', 'rb') as f:
        par_corr_sst_isotopes2[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_isotopes_temp2_sst.pkl', 'rb') as f:
        par_corr_isotopes_temp2_sst[expid[i]] = pickle.load(f)
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sam_isotopes_sources_temp2.pkl', 'rb') as f:
        corr_sam_isotopes_sources_temp2[expid[i]] = pickle.load(f)

lon = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lat

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
#-------------------------------- check

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

for icores in ['EDC', 'DOME F']:
    # icores = 'EDC'
    print('#---------------- ' + icores)
    
    print(np.round((par_corr_sst_isotopes2[expid[i]]['dO18']['d_ln']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))
    
    print(np.round((corr_sources_isotopes[expid[i]]['sst']['dO18']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))
    
    print(np.round((par_corr_temp2_isotopes2[expid[i]]['d_ln']['dO18']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))
    
    print(np.round((corr_temp2_isotopes[expid[i]]['dO18']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))
    
    print(np.round((par_corr_isotopes_temp2_sst[expid[i]]['d_ln']['temp2']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))
    
    print('#-------- Corr. sources vs. d_ln controlling sst')
    print(np.round((par_corr_sources_isotopes[expid[i]]['wind10']['d_ln']['mon']['r'][
        t63_sites_indices[icores]['ilat'],
        t63_sites_indices[icores]['ilon'],
    ].values) ** 2, 3))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_sources_isotopes

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in ['sst', 'rh2m', 'wind10']:
        # ivar = 'sst'
        # 'lat', 'lon', 'distance',
        print('#---------------- ' + ivar)
        
        for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
            # iisotope = 'd_ln'
            print('#-------- ' + iisotope)
            
            for ialltime in ['daily', 'mon', 'mon_no_mm', 'ann']:
                # ialltime = 'mon_no_mm'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.0_sources_isotopes/8.1.5.0.0 ' + expid[i] + ' ' + ialltime + ' corr. ' + ivar + ' vs. ' + iisotope + '.png'
                
                cbar_label = 'Correlation: ' + plot_labels_no_unit[ivar] + ' & ' + plot_labels_no_unit[iisotope]
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    corr_sources_isotopes[expid[i]][ivar][iisotope][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.15,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label)
                fig.savefig(output_png)



'''
6*5*5
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_sources_isotopes


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in ['rh2m', 'wind10',]:
        # ivar = 'rh2m'
        # 'lat', 'lon', 'distance'
        print('#---------------- ' + ivar)
        
        for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
            # iisotope = 'd_ln'
            print('#-------- ' + iisotope)
            
            for ialltime in ['mon', 'mon_no_mm',]:
                # ialltime = 'mon_no_mm'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.1_sources_isotopes_sst/8.1.5.1.0 ' + expid[i] + ' ' + ialltime + ' partial corr. ' + ivar + ' vs. ' + iisotope + ' controlling source SST.png'
                
                cbar_label = 'Partial correlation: ' + plot_labels_no_unit[ivar] + ' & ' + plot_labels_no_unit[iisotope] + '\ncontrolling Source SST'
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7.5]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    par_corr_sources_isotopes[expid[i]][ivar][iisotope][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.18,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label, fontsize=9, linespacing=1.5)
                fig.savefig(output_png)


'''
i = 0
ivar = 'rh2m'
iisotope = 'd_ln'
ialltime = 'mon'
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_temp2_isotopes


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in ['temp2']:
        # ivar = 'temp2'
        print('#---------------- ' + ivar)
        
        for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
            # iisotope = 'd_ln'
            print('#-------- ' + iisotope)
            
            for ialltime in ['mon', 'mon_no_mm', 'ann']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.2_temp2_isotopes/8.1.5.2.0 ' + expid[i] + ' ' + ialltime + ' corr. ' + ivar + ' vs. ' + iisotope + '.png'
                
                cbar_label = 'Correlation: temp2 & ' + plot_labels_no_unit[iisotope]
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    corr_temp2_isotopes[expid[i]][iisotope][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.15,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label)
                fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_temp2_isotopes2


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for iisotope2 in list(set(['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']) - set([iisotope])):
            # iisotope2 = 'dO18'
            print('#-------- ' + iisotope2)
            
            for ialltime in ['mon', 'mon_no_mm',]:
                # ialltime = 'mon_no_mm'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.3_temp2_isotopes2/8.1.5.3.0 ' + expid[i] + ' ' + ialltime + ' partial corr. temp2 vs. ' + iisotope + ' controlling ' + iisotope2 + '.png'
                
                cbar_label = 'Partial correlation: temp2 & ' + plot_labels_no_unit[iisotope] + '\ncontrolling ' + plot_labels_no_unit[iisotope2]
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7.5]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    par_corr_temp2_isotopes2[expid[i]][iisotope][iisotope2][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.18,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label, fontsize=9, linespacing=1.5)
                fig.savefig(output_png)





'''
iisotope = 'd_ln'
iisotope2 = 'dO18'
ialltime = 'mon'

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_sst_isotopes2

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for iisotope2 in list(set(['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']) - set([iisotope])):
            # iisotope2 = 'dO18'
            print('#-------- ' + iisotope2)
            
            for ialltime in ['mon', 'mon_no_mm',]:
                # ialltime = 'mon_no_mm'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.4_sst_isotopes2/8.1.5.4.0 ' + expid[i] + ' ' + ialltime + ' partial corr. source SST vs. ' + iisotope + ' controlling ' + iisotope2 + '.png'
                
                cbar_label = 'Partial correlation: source SST & ' + plot_labels_no_unit[iisotope] + '\ncontrolling ' + plot_labels_no_unit[iisotope2]
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7.5]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    par_corr_sst_isotopes2[expid[i]][iisotope][iisotope2][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.18,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label, fontsize=9, linespacing=1.5)
                fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_isotopes_temp2_sst


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ivar1, ivar2 in zip(['temp2', 'sst'], ['sst', 'temp2', ]):
            # ivar1 = 'temp2'; ivar2 = 'sst'
            print('#-------- ' + ivar1 + ' vs. ' + ivar2)
            
            for ialltime in ['mon', 'mon_no_mm',]:
                # ialltime = 'mon_no_mm'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.5_isotopes_temp2_sst/8.1.5.5.0 ' + expid[i] + ' ' + ialltime + ' partial corr. ' + iisotope + ' vs. ' + ivar1 + ' controlling ' + ivar2 + '.png'
                
                cbar_label = 'Partial correlation: ' + plot_labels_no_unit[iisotope] + ' & ' + plot_labels_no_unit[ivar1] + '\ncontrolling ' + plot_labels_no_unit[ivar2]
                
                fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7.5]) / 2.54,)
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    par_corr_isotopes_temp2_sst[expid[i]][iisotope][ivar1][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                ax.add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                    pad=0.02, fraction=0.18,
                    )
                
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.set_xlabel(cbar_label, fontsize=9, linespacing=1.5)
                fig.savefig(output_png)


'''
i = 0
iisotope = 'd_ln'
ivar1 = 'temp2'
ivar2 = 'sst'
ialltime = 'mon'
par_corr_isotopes_temp2_sst[expid[i]][iisotope][ivar1][ialltime]['r']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_sam_isotopes_sources_temp2


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in ['sst', 'rh2m', 'wind10',
                 'wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',
                 'temp2',]:
        # ivar = 'sst'
        # 'lat', 'lon', 'distance',
        print('#---------------- ' + ivar)
        
        for ialltime in ['mon', 'mon_no_mm',]:
            # ialltime = 'mon_no_mm'
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.6_sam_isotopes_sources_temp2/8.1.5.6.0 ' + expid[i] + ' ' + ialltime + ' corr. sam vs. ' + ivar + '.png'
            
            cbar_label = 'Correlation: SAM & ' + plot_labels_no_unit[ivar]
            
            fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)
            
            cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
            
            plt1 = plot_t63_contourf(
                lon, lat,
                corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r'],
                ax,
                pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
            
            ax.add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            cbar = fig.colorbar(
                plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                pad=0.02, fraction=0.15,
                )
            
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.set_xlabel(cbar_label)
            fig.savefig(output_png)



'''
corr_sam_isotopes_sources_temp2[expid[i]]['sst']['mon']['r']
'''
# endregion
# -----------------------------------------------------------------------------

