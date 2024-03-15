

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_703_6.0_k52',
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

corr_sources_isotopes_q_sfc = {}
par_corr_sources_isotopes_q_sfc={}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

lon = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lat

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot corr_sources_isotopes_q_sfc globe

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in ['sst', 'RHsst',]:
        # ivar = 'sst'
        # 'lat', 'lon', 'distance', 'rh2m', 'wind10'
        print('#---------------- ' + ivar)
        
        for iisotope in ['d_ln', 'd_excess']:
            # iisotope = 'd_ln'
            # 'wisoaprt', 'dO18', 'dD',
            print('#-------- ' + iisotope)
            
            for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.0_sources_isotopes/8.1.5.0.1 ' + expid[i] + ' q_sfc ' + ialltime + ' corr. ' + ivar + ' vs. ' + iisotope + '_global.png'
                
                cbar_label = 'Correlation: ' + plot_labels_no_unit[ivar] + ' & ' + plot_labels_no_unit[iisotope] + ' in surface vapour'
                
                fig, ax = globe_plot(
                    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
                    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotope][ialltime]['r'],
                    ax,
                    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                cbar = fig.colorbar(
                    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                    orientation="horizontal", shrink=0.7, ticks=pltticks,
                    pad=0.05, fraction=0.12,
                    )
                cbar.ax.tick_params(length=2, width=0.4)
                cbar.ax.set_xlabel(cbar_label, linespacing=2)
                
                fig.savefig(output_png)



'''
6*5*5
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


#-------------------------------- Daily negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- Daily positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])


#-------------------------------- Annual negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_min = np.min(annual_par_corr)
where_annual_min = np.where(annual_par_corr == annual_min)
# print(annual_min)
# print(annual_par_corr[where_annual_min[0][0], where_annual_min[1][0]])

annual_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lon.values
annual_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lat.values

annual_min_ilon = np.where(lon == annual_min_lon)[0][0]
annual_min_ilat = np.where(lat == annual_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_min_ilat, annual_min_ilon])


#-------------------------------- Annual positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_max = np.max(annual_par_corr)
where_annual_max = np.where(annual_par_corr == annual_max)
# print(annual_max)
# print(annual_par_corr[where_annual_max[0][0], where_annual_max[1][0]])

annual_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lon.values
annual_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lat.values

annual_max_ilon = np.where(lon == annual_max_lon)[0][0]
annual_max_ilat = np.where(lat == annual_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_max_ilat, annual_max_ilon])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot par_corr_sources_isotopes_q_sfc

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotopes in ['d_ln',]:
        # iisotopes = 'd_ln'
        # ['d_ln', 'd_excess',]
        print('#---------------- ' + iisotopes)
        
        for ivar in ['sst',]:
            # ivar = 'sst'
            # ['sst', 'RHsst']
            print('#---------------- ' + ivar)
            
            for ctr_var in list(set(['sst', 'RHsst']) - set([ivar])):
                # ctr_var = 'RHsst'
                print('#-------- ' + ctr_var)
                
                for ialltime in ['daily', 'ann',]:
                    # ialltime = 'mon'
                    # ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']
                    print('#---- ' + ialltime)
                    
                    output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.0_sources_isotopes/8.1.5.0.3 ' + expid[i] + ' ' + ialltime + ' corr. ' + iisotopes + ' vs. ' + ivar + ' while controlling ' + ctr_var + '.png'
                    
                    cbar_label = 'Partial correlation: ' + plot_labels_no_unit[iisotopes] + ' & ' + plot_labels_no_unit[ivar] + '\nwhile controlling ' + plot_labels_no_unit[ctr_var]
                    
                    fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.5]) / 2.54,)
                    
                    plt1 = plot_t63_contourf(
                        lon, lat,
                        par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'],
                        ax, pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
                    
                    if ((iisotopes == 'd_ln') & (ivar == 'sst') & (ialltime == 'daily')):
                        cplot_ice_cores(daily_min_lon, daily_min_lat, ax, s=12)
                        cplot_ice_cores(daily_max_lon, daily_max_lat, ax, s=12)
                    elif ((iisotopes == 'd_ln') & (ivar == 'sst') & (ialltime == 'ann')):
                        cplot_ice_cores(annual_min_lon, annual_min_lat, ax,s=12)
                        cplot_ice_cores(annual_max_lon, annual_max_lat, ax,s=12)
                    
                    cbar = fig.colorbar(
                        plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                        pad=0.02, fraction=0.2,
                        )
                    
                    # cbar.ax.tick_params(labelsize=8)
                    cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
                    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------



