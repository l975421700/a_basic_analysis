

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

with open('scratch/cmip6/lig/sst/pmip3_anomalies_site_values.pkl', 'rb') as f:
    pmip3_anomalies_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)
with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)
with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'rb') as f:
    AIS_ann_tas_site_values = pickle.load(f)
with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)

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

output_png = 'figures/7_lig/7.1_hadcm3/7.1.1.1 PMIP3, PMIP4, HadCM3, SST, SAT, SIC anomalies scatters.png'

axis_mins = [-8, -8, -8, -80]
axis_maxs = [12, 12, 12, 20]
axis_intervals = [2, 2, 2, 20]
cbar_labels = ['Annual SST [$°C$]', 'Summer SST [$°C$]', 'Annual SAT [$°C$]', 'Sep SIC [$\%$]']

nrow = 4
ncol = 4
fm_bottom = 1 / (5.8*nrow + 1)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8 * ncol, 5.8 * nrow + 1]) / 2.54,
    gridspec_kw={'hspace': 0.2, 'wspace': 0.2}, )

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            plt.text(
                0, 1.02, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='left', va='bottom', rotation='horizontal')
            ipanel += 1
        
        if (irow == (nrow-1)):
            axs[irow, jcol].set_xlabel('Reconstructed ' + cbar_labels[jcol])
        if (irow != (nrow-1)):
            axs[irow, jcol].set_xticklabels([])
    
    # axs[irow, 0].set_ylabel('Simulations')

# PMIP3 Annual SST
for irec in ['EC', 'JH', 'DC']:
    axs[0, 0].scatter(
        pmip3_anomalies_site_values['annual_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['annual_sst'][irec]['sim_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )

# PMIP3 Summer SST
for irec in ['EC', 'JH', 'DC', 'MC']:
    axs[0, 1].scatter(
        pmip3_anomalies_site_values['summer_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['summer_sst'][irec]['sim_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )

# PMIP3 Annual SAT
axs[0, 2].scatter(
    pmip3_anomalies_site_values['annual_sat']['EC']['rec_lig_pi'],
    pmip3_anomalies_site_values['annual_sat']['EC']['sim_lig_pi'],
    marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

# PMIP4 Annual SST
data_to_plot = {}
data_to_plot['EC'] = SO_ann_sst_site_values['EC'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
data_to_plot['JH'] = SO_ann_sst_site_values['JH'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
data_to_plot['DC'] = SO_ann_sst_site_values['DC'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]

for irec in ['EC', 'JH', 'DC']:
    axs[1, 0].scatter(
        data_to_plot[irec]['rec_ann_sst_lig_pi'],
        data_to_plot[irec]['sim_ann_sst_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )

# PMIP4 Summer SST
data_to_plot = {}
data_to_plot['EC'] = SO_jfm_sst_site_values['EC'].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
data_to_plot['JH'] = SO_jfm_sst_site_values['JH'].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
data_to_plot['DC'] = SO_jfm_sst_site_values['DC'].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
data_to_plot['MC'] = SO_jfm_sst_site_values['MC'].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]

for irec in ['EC', 'JH', 'DC', 'MC']:
    axs[1, 1].scatter(
        data_to_plot[irec]['rec_jfm_sst_lig_pi'],
        data_to_plot[irec]['sim_jfm_sst_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )

# PMIP4 Annual SAT
data_to_plot = {}
data_to_plot['EC_tas'] = AIS_ann_tas_site_values['EC'].groupby(['Station']).mean()[
    ['rec_ann_tas_lig_pi', 'sim_ann_tas_lig_pi']]

axs[1, 2].scatter(
    data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
    data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
    marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

# PMIP4 Sep SIC
data_to_plot = {}
data_to_plot['MC'] = SO_sep_sic_site_values['MC'].groupby(['Station']).mean()[
    ['rec_sep_sic_lig_pi', 'sim_sep_sic_lig_pi']]

axs[1, 3].scatter(
    data_to_plot['MC']['rec_sep_sic_lig_pi'],
    data_to_plot['MC']['sim_sep_sic_lig_pi'],
    marker=marker_recs['MC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

for irow, iperiod in zip(np.arange(2, 4, 1), ['lig', 'lig0.25Sv']):
    print('#-------------------------------- ' + str(irow) + ' ' +  iperiod)
    
    for jcol, jproxy in zip(range(ncol), hadcm3_output_site_values.keys()):
        print('#---------------- ' + str(jcol) + ' ' +  jproxy)
        
        for irec in hadcm3_output_site_values[jproxy].keys():
            print('#-------- ' + irec)
            
            axs[irow, jcol].scatter(
                hadcm3_output_site_values[jproxy][irec]['rec_lig_pi'],
                hadcm3_output_site_values[jproxy][irec]['sim_'+iperiod+'_pi'],
                marker=marker_recs[irec], s=symbol_size, c='white',
                edgecolors='k', lw=linewidth, alpha=alpha,
            )

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol].axline((0, 0), slope = 1, c='k', lw=0.5, ls='--')
        axs[irow, jcol].axhline(0, c='k', lw=0.5, ls='--')
        axs[irow, jcol].axvline(0, c='k', lw=0.5, ls='--')
        
        axs[irow, jcol].set_xlim(axis_mins[jcol], axis_maxs[jcol])
        axs[irow, jcol].set_xticks(np.arange(
            axis_mins[jcol], axis_maxs[jcol] + 1e-4, axis_intervals[jcol]))
        axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
        
        axs[irow, jcol].set_ylim(axis_mins[jcol], axis_maxs[jcol])
        axs[irow, jcol].set_yticks(np.arange(
            axis_mins[jcol], axis_maxs[jcol] + 1e-4, axis_intervals[jcol]))
        axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
        
        axs[irow, jcol].grid(
            True, which='both',
            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

axs[0, 3].remove()

axs[0, 0].set_ylabel('PMIP3 model ensemble')
axs[1, 0].set_ylabel('PMIP4 model ensemble')
axs[2, 0].set_ylabel('HadCM3')
axs[3, 0].set_ylabel('HadCM3 with 0.25 $Sv$ freshwater forcing')

fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.97)
fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------

