

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')
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

with open('scratch/cmip6/lig/sst/pmip3_anomalies_site_values.pkl', 'rb') as f:
    pmip3_anomalies_site_values = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec ann_sst

mean_err = {}
rms_err  = {}

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec ann_sst ens_pmip3.png'

axis_min = -8
axis_max = 12

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for irec in ['EC', 'JH', 'DC']:
    # irec = 'EC'
    print(irec)
    
    ax.scatter(
        pmip3_anomalies_site_values['annual_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['annual_sst'][irec]['sim_lig_pi'],
        marker=marker_recs[irec],
        s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,
        )
    
    mean_err[irec] = np.round(
        pmip3_anomalies_site_values[
            'annual_sst'][irec]['sim_rec_lig_pi'].mean(), 1)
    rms_err[irec] = np.round(mean_squared_error(
        pmip3_anomalies_site_values['annual_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['annual_sst'][irec]['sim_lig_pi'],
        squared=False), 1)

ax.scatter(
    pmip3_anomalies_site_values['annual_sat']['EC']['rec_lig_pi'],
    pmip3_anomalies_site_values['annual_sat']['EC']['sim_lig_pi'],
    marker=marker_recs['EC'],
    s=25, c='white', edgecolors='b', lw=0.8, alpha=0.75,
    )
mean_err['EC_tas'] = np.round(
    pmip3_anomalies_site_values[
            'annual_sat']['EC']['sim_rec_lig_pi'].mean(), 1)
rms_err['EC_tas'] = np.round(
    mean_squared_error(
        pmip3_anomalies_site_values['annual_sat']['EC']['rec_lig_pi'],
        pmip3_anomalies_site_values['annual_sat']['EC']['sim_lig_pi'],
        squared=False), 1)

ax.plot([0, 1], [0, 1], transform=ax.transAxes,
        c='k', lw=0.5, ls='--')

ax.set_ylabel('Simulated annual SST/SAT anomalies [$°C$]')
ax.set_ylim(axis_min, axis_max)
ax.set_yticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Reconstructed annual SST/SAT anomalies [$°C$]')
ax.set_xlim(axis_min, axis_max)
ax.set_xticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

l1 = plt.scatter(
    [],[], marker=marker_recs['EC'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
l1_1 = plt.scatter(
    [],[], marker=marker_recs['EC'],
    s=25, c='white', edgecolors='b', lw=0.8, alpha=0.75,)
l2 = plt.scatter(
    [],[], marker=marker_recs['JH'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
l3 = plt.scatter(
    [],[], marker=marker_recs['DC'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
plt.legend(
    [l1, l1_1, l2, l3,],
    ['Capron et al. (2017), SST: ' + \
        str(mean_err['EC']) + ', ' + str(rms_err['EC']),
     'Capron et al. (2017), SAT: ' + \
        str(mean_err['EC_tas']) + ', ' + str(rms_err['EC_tas']),
     'Hoffman et al. (2017):       ' + \
        str(mean_err['JH']) + ', ' + str(rms_err['JH']),
     'Chandler et al. (2021):       ' + \
        str(mean_err['DC']) + ', ' + str(rms_err['DC']),],
    ncol=1, frameon=True,
    loc = 'upper left', handletextpad=0.05,)

plt.text(
    0.95, 0.05, 'PMIP3 model ensembles',
    horizontalalignment='right', verticalalignment='bottom',
    transform=ax.transAxes, backgroundcolor='white',)

ax.grid(True, which='both', linewidth=0.4, color='lightgray', linestyle=':')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec jfm_sst

mean_err = {}
rms_err  = {}

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec jfm_sst ens_pmip3.png'

axis_min = -5
axis_max = 7

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print(irec)
    
    ax.scatter(
        pmip3_anomalies_site_values['summer_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['summer_sst'][irec]['sim_lig_pi'],
        marker=marker_recs[irec],
        s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,
        )
    
    mean_err[irec] = np.round(
        pmip3_anomalies_site_values[
            'summer_sst'][irec]['sim_rec_lig_pi'].mean(), 1)
    rms_err[irec] = np.round(mean_squared_error(
        pmip3_anomalies_site_values['summer_sst'][irec]['rec_lig_pi'],
        pmip3_anomalies_site_values['summer_sst'][irec]['sim_lig_pi'],
        squared=False), 1)

ax.plot([0, 1], [0, 1], transform=ax.transAxes,
        c='k', lw=0.5, ls='--')

ax.set_ylabel('Simulated summer SST anomalies [$°C$]')
ax.set_ylim(axis_min, axis_max)
ax.set_yticks(np.arange(axis_min, axis_max + 1e-4, 1))
ax.yaxis.set_minor_locator(AutoMinorLocator(1))

ax.set_xlabel('Reconstructed summer SST anomalies [$°C$]')
ax.set_xlim(axis_min, axis_max)
ax.set_xticks(np.arange(axis_min, axis_max + 1e-4, 1))
ax.xaxis.set_minor_locator(AutoMinorLocator(1))

l1 = plt.scatter(
    [],[], marker=marker_recs['EC'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
l2 = plt.scatter(
    [],[], marker=marker_recs['JH'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
l3 = plt.scatter(
    [],[], marker=marker_recs['DC'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
l4 = plt.scatter(
    [],[], marker=marker_recs['MC'],
    s=25, c='white', edgecolors='k', lw=0.8, alpha=0.75,)
plt.legend(
    [l1, l2, l3, l4,],
    ['Capron et al. (2017)    : ' + \
        str(mean_err['EC']) + ', ' + str(rms_err['EC']),
     'Hoffman et al. (2017)  : ' + \
        str(mean_err['JH']) + ', ' + str(rms_err['JH']),
     'Chandler et al. (2021)  : ' + \
        str(mean_err['DC']) + ', ' + str(rms_err['DC']),
     'Chadwick et al. (2021): ' + \
        str(mean_err['MC']) + ', ' + str(rms_err['MC']),],
    ncol=1, frameon=True,
    loc = 'upper left', handletextpad=0.05,)

plt.text(
    0.95, 0.05, 'PMIP3 model ensembles',
    horizontalalignment='right', verticalalignment='bottom',
    transform=ax.transAxes, backgroundcolor='white',)

ax.grid(True, which='both', linewidth=0.4, color='lightgray', linestyle=':')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------




