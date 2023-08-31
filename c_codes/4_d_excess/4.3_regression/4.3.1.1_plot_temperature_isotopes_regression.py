

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
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
from scipy.stats import pearsonr
import statsmodels.api as sm
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

regression_sst_d_AIS = {}
regression_temp2_delta_AIS = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_AIS.pkl', 'rb') as f:
        regression_sst_d_AIS[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_temp2_delta_AIS.pkl', 'rb') as f:
        regression_temp2_delta_AIS[expid[i]] = pickle.load(f)

lon = regression_sst_d_AIS[expid[i]]['d_ln']['ann']['RMSE'].lon
lat = regression_sst_d_AIS[expid[i]]['d_ln']['ann']['RMSE'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot regression temp2 = f(dD / dO18)

#---------------- settings
slope_interval  = np.arange(0, 1 + 1e-4, 0.1)
RMSE_interval   = np.arange(0, 10 + 1e-4, 0.5)

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2,
    cmap='PuOr', asymmetric=True, reversed=True)


for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['dD',]:
        # iisotope = 'dD'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.5_temp2_delta_spatial/8.1.6.5.0 ' + expid[i] + ' ' + ialltime + ' regression temp2 vs. ' + iisotope + '.png'
            
            cbar_label = '$R^2$: temp2 & ' + plot_labels_no_unit[iisotope]
            
            fig, ax = hemisphere_plot(northextent=-60,)
            
            cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
            
            plt1 = plot_t63_contourf(
                lon, lat,
                regression_temp2_delta_AIS[expid[i]][iisotope][ialltime]['rsquared'],
                ax, pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)
            
            # RMSE
            plt_ctr1 = ax.contour(
                lon, lat.sel(lat=slice(-62, -90)),
                regression_temp2_delta_AIS[expid[i]][iisotope][ialltime][
                    'RMSE'].sel(lat=slice(-62, -90)),
                colors='k', levels=RMSE_interval, linewidths=0.6,
                clip_on=True, zorder=1, transform=ccrs.PlateCarree(),)
            ax_clabel = ax.clabel(
                plt_ctr1, inline=1, colors='k', fmt=remove_trailing_zero,
                levels=RMSE_interval, inline_spacing=1, fontsize=8,
                zorder=1, )
            
            # slope
            plt_ctr2 = ax.contour(
                lon, lat.sel(lat=slice(-62, -90)),
                regression_temp2_delta_AIS[expid[i]][iisotope][ialltime][
                    'slope'].sel(lat=slice(-62, -90)),
                colors='b', levels=slope_interval, linewidths=0.6,
                clip_on=True, zorder=1, transform=ccrs.PlateCarree(),)
            ax_clabel = ax.clabel(
                plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
                levels=slope_interval, inline_spacing=1, fontsize=8,
                zorder=1, )
            
            ax.add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            cbar = fig.colorbar(
                plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.9, ticks=pltticks,
                extend='neither', pad=0.02, fraction=0.2,
                )
            
            # contours legend
            h1, _ = plt_ctr1.legend_elements()
            h2, _ = plt_ctr2.legend_elements()
            ax_legend = ax.legend(
                [h1[0], h2[0]],
                ['RMSE [$‰$]',
                 'Slope [$°C / ‰$]'],
                loc='lower center', frameon=False, ncol = 2,
                bbox_to_anchor=(0.5, -0.39),
                handlelength=1, columnspacing=1,
                )
            
            cbar.ax.set_xlabel(cbar_label)
            fig.savefig(output_png)




'''
'rsquared', 'RMSE', 'slope', 'intercept'

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot regression source SST = f(d_ln / d_xs), revised

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0.2, cm_max=1, cm_interval1=0.05, cm_interval2=0.1,
    cmap='viridis', asymmetric=False, reversed=True)

# #---------------- settings
RMSE_interval       = np.arange(0.5, 2.5 + 1e-4, 2)
rsquared_interval   = np.arange(0.5, 2.5 + 1e-4, 2)

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for iisotope in ['d_ln',]:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
            # ialltime = 'mon'
            # ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']
            print('#-------- ' + ialltime)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.6_regression_analysis/8.1.6.4_sst_d_spatial/8.1.6.4.0 ' + expid[i] + ' ' + ialltime + ' regression source sst vs. ' + iisotope + '.png'
            
            cbar_label = 'Slope: ' + plot_labels_no_unit['sst'] + ' vs. ' + plot_labels_no_unit[iisotope]
            
            fig, ax = hemisphere_plot(northextent=-60,)
            
            cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
            
            plt1 = plot_t63_contourf(
                lon, lat,
                regression_sst_d_AIS[expid[i]][iisotope][ialltime]['slope'],
                ax, pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
            
            # RMSE
            plt_ctr1 = ax.contour(
                lon, lat.sel(lat=slice(-62, -90)),
                regression_sst_d_AIS[expid[i]][iisotope][ialltime][
                    'RMSE'].sel(lat=slice(-62, -90)),
                colors='k', levels=RMSE_interval, linewidths=0.6,
                clip_on=True, zorder=1, transform=ccrs.PlateCarree(),)
            ax_clabel = ax.clabel(
                plt_ctr1, inline=1, colors='k', fmt=remove_trailing_zero,
                levels=RMSE_interval, inline_spacing=1, fontsize=8,
                zorder=1, )
            
            # rsquared
            plt_ctr2 = ax.contour(
                lon, lat.sel(lat=slice(-62, -90)),
                regression_sst_d_AIS[expid[i]][iisotope][ialltime][
                    'rsquared'].sel(lat=slice(-62, -90)),
                colors='b', levels=rsquared_interval, linewidths=0.6,
                clip_on=True, zorder=1, transform=ccrs.PlateCarree(),)
            ax_clabel = ax.clabel(
                plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
                levels=rsquared_interval, inline_spacing=1, fontsize=8,
                zorder=1, )
            
            ax.add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            cbar = fig.colorbar(
                plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.9, ticks=pltticks,
                extend='neither', pad=0.02, fraction=0.2,
                )
            
            # contours legend
            h1, _ = plt_ctr1.legend_elements()
            h2, _ = plt_ctr2.legend_elements()
            ax_legend = ax.legend(
                [h1[0], h2[0]],
                ['RMSE [$°C$]',
                 '$R^2$ [-]'],
                loc='lower center', frameon=False, ncol = 2,
                bbox_to_anchor=(0.5, -0.39),
                handlelength=1, columnspacing=1,
                )
            
            cbar.ax.set_xlabel(cbar_label)
            fig.savefig(output_png)





stats.describe(regression_sst_d_AIS[expid[i]][iisotope][ialltime]['slope'].values[echam6_t63_ais_mask['mask']['AIS']])

stats.describe(regression_sst_d_AIS[expid[i]][iisotope][ialltime]['RMSE'].values[echam6_t63_ais_mask['mask']['AIS']])

rsquared_AIS = regression_sst_d_AIS[expid[i]][iisotope][ialltime][
    'rsquared'].values[echam6_t63_ais_mask['mask']['AIS']]
stats.describe(rsquared_AIS)



'''
icores = 'EDC'
iisotope = 'd_ln'
ialltime = 'ann no am'

regression_sst_d_AIS[expid[i]][iisotope][ialltime].keys()

regression_sst_d_AIS[expid[i]][iisotope][ialltime]['slope'][
    t63_sites_indices[icores]['ilat'],
    t63_sites_indices[icores]['ilon'],
].values

'rsquared', 'RMSE', 'slope', 'intercept'

'''
# endregion
# -----------------------------------------------------------------------------




