

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
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

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

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
# region plot corr_sources_isotopes globe

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


