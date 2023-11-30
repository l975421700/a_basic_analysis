

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp
from sklearn.metrics import mean_squared_error

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
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
    find_ilat_ilon,
    regrid,
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
    marker_recs,
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
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values.pkl', 'rb') as f:
    hadcm3_output_site_values = pickle.load(f)



'''
for iproxy in hadcm3_output_site_values.keys():
    # iproxy = 'annual_sst'
    print('#-------------------------------- ' + iproxy)
    
    for irec in hadcm3_output_site_values[iproxy].keys():
        # irec = 'EC'
        print('#---------------- ' + irec)
        
        print(hadcm3_output_site_values[iproxy][irec])
'''
# endregion
# -----------------------------------------------------------------------------


symbol_size = 60
linewidth = 1
alpha = 0.75


# -----------------------------------------------------------------------------
# region Q-Q plot

output_png = 'figures/7_lig/7.1_hadcm3/7.1.0.1 sim_rec SST, SAT, SIC.png'

axis_mins = [-8, -8, -8, -80]
axis_maxs = [12, 12, 12, 20]
axis_intervals = [4, 4, 4, 20]
cbar_labels = ['Annual SST [$°C$]', 'Summer SST [$°C$]', 'Annual SAT [$°C$]', 'Sep SIC [$\%$]']

ncol = 4

fig, axs = plt.subplots(1, ncol, figsize=np.array([6.6 * ncol, 8]) / 2.54,
                        gridspec_kw={'hspace': 0, 'wspace': 0.15},)

ipanel=0
for icol, iproxy in zip(range(ncol), hadcm3_output_site_values.keys()):
    # iproxy = 'annual_sst'
    print('#-------------------------------- ' + str(icol) + ' ' +  iproxy)
    
    plt.text(
            0, 1.02, panel_labels[ipanel],
            transform=axs[icol].transAxes,
            ha='left', va='bottom', rotation='horizontal')
    ipanel += 1
    plt.text(0.5, 1.02, cbar_labels[icol], transform=axs[icol].transAxes,
             ha='center', va='bottom', rotation='horizontal', weight='bold')
    
    for irec in hadcm3_output_site_values[iproxy].keys():
        # irec = 'EC'
        print('#---------------- ' + irec)
        
        axs[icol].scatter(
            hadcm3_output_site_values[iproxy][irec]['rec_lig_pi'],
            hadcm3_output_site_values[iproxy][irec]['sim_lig_pi'],
            marker=marker_recs[irec],
            s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )
        
        axs[icol].scatter(
            hadcm3_output_site_values[iproxy][irec]['rec_lig_pi'],
            hadcm3_output_site_values[iproxy][irec]['sim_lig0.25Sv_pi'],
            marker=marker_recs[irec],
            s=symbol_size, c='white', edgecolors='b', lw=linewidth, alpha=alpha,
        )
    
    axs[icol].axline((0, 0), slope = 1, c='k', lw=0.5, ls='--')
    axs[icol].axhline(0, c='k', lw=0.5, ls='--')
    axs[icol].axvline(0, c='k', lw=0.5, ls='--')
    
    axs[icol].set_xlim(axis_mins[icol], axis_maxs[icol])
    axs[icol].set_xticks(np.arange(axis_mins[icol], axis_maxs[icol] + 1e-4, axis_intervals[icol]))
    axs[icol].xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[icol].set_ylim(axis_mins[icol], axis_maxs[icol])
    axs[icol].set_yticks(np.arange(axis_mins[icol], axis_maxs[icol] + 1e-4, axis_intervals[icol]))
    axs[icol].yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[icol].set_xlabel('Reconstructions')
    
    axs[icol].grid(True, which='both',
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

axs[0].set_ylabel('Simulations')
# plt.text(0.5, -0.3, 'Reconstructions',
#          ha='center', va='center', rotation='horizontal')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.92)
fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------

