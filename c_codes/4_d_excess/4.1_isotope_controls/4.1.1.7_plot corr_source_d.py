

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
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

corr_sources_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl', 'rb') as f:
    corr_sources_isotopes[expid[i]] = pickle.load(f)

corr_sources_isotopes_q_sfc = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

lon = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lat

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot data

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.2,
    cmap='PuOr', reversed=True)
pltticks[-6] = 0

nrow = 2
ncol = 2

for ialltime in ['daily', 'mon', 'mon no mm', 'ann no am']:
    # ialltime = 'daily'
    print('#-------------------------------- ' + ialltime)
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.0_sources_isotopes/8.1.5.0.2 ' + expid[i] + ' ' + ialltime + ' SH corr. sources vs. ' + iisotope + '.png'
        print(output_png)
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([5.8*ncol+1, 5.8*nrow+2.5]) / 2.54,
            subplot_kw={'projection': ccrs.SouthPolarStereo()},
            )
        
        ipanel=0
        for irow in range(nrow):
            for jcol in range(ncol):
                axs[irow, jcol] = hemisphere_plot(
                    northextent=-20, ax_org = axs[irow, jcol])
                
                plt.text(
                    0.05, 1, panel_labels[ipanel],
                    transform=axs[irow, jcol].transAxes,
                    ha='center', va='center', rotation='horizontal')
                ipanel += 1
        
        # (a)
        plt1 = plot_t63_contourf(
            lon, lat,
            corr_sources_isotopes_q_sfc[expid[i]]['RHsst'][iisotope][ialltime]['r'],
            axs[0, 0], pltlevel, 'neither', pltnorm, pltcmp,ccrs.PlateCarree(),)
        plt1 = plot_t63_contourf(
            lon, lat,
            corr_sources_isotopes_q_sfc[expid[i]]['sst'][iisotope][ialltime]['r'],
            axs[0, 1], pltlevel, 'neither', pltnorm, pltcmp,ccrs.PlateCarree(),)
        
        plt1 = plot_t63_contourf(
            lon, lat,
            corr_sources_isotopes[expid[i]]['RHsst'][iisotope][ialltime]['r'],
            axs[1, 0], pltlevel, 'neither', pltnorm, pltcmp,ccrs.PlateCarree(),)
        plt1 = plot_t63_contourf(
            lon, lat,
            corr_sources_isotopes[expid[i]]['sst'][iisotope][ialltime]['r'],
            axs[1, 1], pltlevel, 'neither', pltnorm, pltcmp,ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, 'Source RHsst and ' + plot_labels_no_unit[iisotope],
            transform=axs[0, 0].transAxes,
            ha='center', va='center', rotation='horizontal')
        plt.text(
            0.5, 1.05, 'Source SST and ' + plot_labels_no_unit[iisotope],
            transform=axs[0, 1].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        plt.text(
            -0.05, 0.5, 'Surface vapour',
            transform=axs[0, 0].transAxes,
            ha='center', va='center', rotation='vertical')
        plt.text(
            -0.05, 0.5, 'Precipitation',
            transform=axs[1, 0].transAxes,
            ha='center', va='center', rotation='vertical')
        
        cbar = fig.colorbar(
            plt1, ax=axs, aspect=25,
            orientation="horizontal", shrink=0.75, ticks=pltticks,
            extend='neither', anchor=(0.5, -0.25),
            format=remove_trailing_zero_pos,
            )
        cbar.ax.set_xlabel('Correlation coefficient [-]', linespacing=1.5)
        
        fig.subplots_adjust(
            left=0.05, right = 0.99, bottom = 0.12, top = 0.95,
            wspace=0.01, hspace=0.01,)
        fig.savefig(output_png)




'''
for ivar in ['RHsst', 'sst']:
    # ivar = 'RHsst'
    print('#-------------------------------- ' + ialltime)
    
    for iisotope in ['d_ln', 'd_excess']:
        # iisotope = 'd_ln'
        print('#---------------- ' + iisotope)
        
        data1 = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotope]['ann no am']['r'].values
        data2 = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotope]['ann']['r'].values
        
        # print(np.isnan(data1).sum())
        # print(np.isnan(data2).sum())
        print(np.max(abs(data1 - data2) / data2))

'''
# endregion
# -----------------------------------------------------------------------------


