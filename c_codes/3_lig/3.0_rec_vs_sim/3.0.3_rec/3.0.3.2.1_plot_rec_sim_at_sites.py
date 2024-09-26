

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

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'rb') as f:
    AIS_ann_tas_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',]

# endregion
# -----------------------------------------------------------------------------


symbol_size = 60
linewidth = 1
alpha = 0.75


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region compare sim_rec ann_sst

data_to_plot = {}
data_to_plot['EC'] = SO_ann_sst_site_values['EC'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]

data_to_plot['JH'] = SO_ann_sst_site_values['JH'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]

data_to_plot['DC'] = SO_ann_sst_site_values['DC'].groupby(['Station']).mean()[
    ['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]

data_to_plot['EC_tas'] = AIS_ann_tas_site_values['EC'].groupby(['Station']).mean()[
    ['rec_ann_tas_lig_pi', 'sim_ann_tas_lig_pi']]

mean_err = {}
rms_err  = {}

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec ann_sst ens.png'

axis_min = -8
axis_max = 12

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for irec in ['EC', 'JH', 'DC']:
    # irec = 'EC'
    print(irec)
    
    ax.scatter(
        data_to_plot[irec]['rec_ann_sst_lig_pi'],
        data_to_plot[irec]['sim_ann_sst_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )
    
    mean_err[irec] = np.round(
        (SO_ann_sst_site_values[irec].groupby(['Station']).mean()[
            ['sim_rec_ann_sst_lig_pi']]).mean().values[0], 1)
    rms_err[irec] = np.round(mean_squared_error(
        data_to_plot[irec]['rec_ann_sst_lig_pi'],
        data_to_plot[irec]['sim_ann_sst_lig_pi'],
        squared=False), 1)

ax.scatter(
    data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
    data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
    marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='b', lw=linewidth, alpha=alpha,
    )
mean_err['EC_tas'] = np.round(
    (AIS_ann_tas_site_values['EC'].groupby(['Station']).mean()[
        ['sim_rec_ann_tas_lig_pi']]).mean().values[0], 1)
rms_err['EC_tas'] = np.round(
    mean_squared_error(
        data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
        data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
        squared=False), 1)

ax.plot([-100, 100], [-100, 100], c='k', lw=0.5, ls='--')
ax.hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
ax.vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')

ax.set_ylabel('Simulations [$°C$]')
ax.set_ylim(axis_min, axis_max)
ax.set_yticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.set_ylim(-2, 2)

ax.set_xlabel('Reconstructions [$°C$]')
ax.set_xlim(axis_min, axis_max)
ax.set_xticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# l1 = plt.scatter(
#     [],[], marker=marker_recs['EC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# l1_1 = plt.scatter(
#     [],[], marker=marker_recs['EC'],
#     s=symbol_size, c='white', edgecolors='b', lw=linewidth, alpha=alpha,)
# l2 = plt.scatter(
#     [],[], marker=marker_recs['JH'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# l3 = plt.scatter(
#     [],[], marker=marker_recs['DC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# plt.legend(
#     [l1, l1_1, l2, l3,],
#     [str(rms_err['EC']) + ' (SST)',
#      str(rms_err['EC_tas']) + ' (SAT)',
#      str(rms_err['JH']),
#      str(rms_err['DC']),],
#     ncol=1, frameon=True, title='RMSE',
#     loc = 'upper right', handletextpad=0.05,)

ax.grid(True, which='both',
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)



'''
plt.text(
    0.95, 0.05, 'PMIP4 model ensemble',
    horizontalalignment='right', verticalalignment='bottom',
    transform=ax.transAxes, backgroundcolor='white',)


print((SO_ann_sst_site_values['EC'].groupby(['Station']).mean()[
    ['sim_rec_ann_sst_lig_pi']]).mean().values[0])

print((SO_ann_sst_site_values['JH'].groupby(['Station']).mean()[
    ['sim_rec_ann_sst_lig_pi']]).mean().values[0])

print((SO_ann_sst_site_values['DC'].groupby(['Station']).mean()[
    ['sim_rec_ann_sst_lig_pi']]).mean().values[0])

print((AIS_ann_tas_site_values['EC'].groupby(['Station']).mean()[
    ['sim_rec_ann_tas_lig_pi']]).mean().values[0])



mean_squared_error(
    data_to_plot['EC']['rec_ann_sst_lig_pi'],
    data_to_plot['EC']['sim_ann_sst_lig_pi'],
    squared=False
)

(((SO_ann_sst_site_values['EC'].groupby(['Station']).mean()[['sim_rec_ann_sst_lig_pi']]) ** 2).mean()) ** 0.5

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec jfm_sst

data_to_plot = {}
data_to_plot['EC'] = SO_jfm_sst_site_values['EC'][['Station', 'rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']].groupby(['Station']).mean()[['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]

data_to_plot['EC']['std'] = SO_jfm_sst_site_values['EC'][['Station', 'sim_jfm_sst_lig_pi']].groupby(['Station']).std(ddof=1)[['sim_jfm_sst_lig_pi']]

data_to_plot['JH'] = SO_jfm_sst_site_values['JH'][['Station', 'rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]

data_to_plot['JH']['std'] = SO_jfm_sst_site_values['JH'][['Station', 'sim_jfm_sst_lig_pi']].groupby(['Station']).std(ddof=1)[['sim_jfm_sst_lig_pi']]

data_to_plot['DC'] = SO_jfm_sst_site_values['DC'][['Station', 'rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]

data_to_plot['DC']['std'] = SO_jfm_sst_site_values['DC'][['Station', 'sim_jfm_sst_lig_pi']].groupby(['Station']).std(ddof=1)[['sim_jfm_sst_lig_pi']]

data_to_plot['MC'] = SO_jfm_sst_site_values['MC'][['Station', 'rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']].groupby(['Station']).mean()[
    ['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]

data_to_plot['MC']['std'] = SO_jfm_sst_site_values['MC'][['Station', 'sim_jfm_sst_lig_pi']].groupby(['Station']).std(ddof=1)[['sim_jfm_sst_lig_pi']]

# mean_err = {}
# rms_err  = {}

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec jfm_sst ens.png'

axis_min = -8
axis_max = 12

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for irec in ['EC', 'JH', 'DC', 'MC']:
    # irec = 'EC'
    print(irec)
    
    ax.errorbar(
        x=data_to_plot[irec]['rec_jfm_sst_lig_pi'].values,
        y=data_to_plot[irec]['sim_jfm_sst_lig_pi'].values,
        yerr=data_to_plot[irec]['std'],
        linestyle='None', c='tab:blue', lw=0.75, alpha=0.75,
    )
    
    ax.scatter(
        data_to_plot[irec]['rec_jfm_sst_lig_pi'].values,
        data_to_plot[irec]['sim_jfm_sst_lig_pi'].values,
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )
    
    # mean_err[irec] = np.round(
    #     (SO_jfm_sst_site_values[irec].groupby(['Station']).mean()[
    #         ['sim_rec_jfm_sst_lig_pi']]).mean().values[0], 1)
    # rms_err[irec] = np.round(mean_squared_error(
    #     data_to_plot[irec]['rec_jfm_sst_lig_pi'],
    #     data_to_plot[irec]['sim_jfm_sst_lig_pi'],
    #     squared=False), 1)

ax.plot([0, 1], [0, 1], transform=ax.transAxes,
        c='k', lw=0.5, ls='--')
ax.hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
ax.vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')

ax.set_ylabel('Simulations [$°C$]')
ax.set_ylim(axis_min, axis_max)
ax.set_yticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Reconstructions [$°C$]')
ax.set_xlim(axis_min, axis_max)
ax.set_xticks(np.arange(axis_min, axis_max + 1e-4, 2))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# l1 = plt.scatter(
#     [],[], marker=marker_recs['EC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# l2 = plt.scatter(
#     [],[], marker=marker_recs['JH'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# l3 = plt.scatter(
#     [],[], marker=marker_recs['DC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# l4 = plt.scatter(
#     [],[], marker=marker_recs['MC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# plt.legend(
#     [l1, l2, l3, l4,],
#     [str(rms_err['EC']),
#      str(rms_err['JH']),
#      str(rms_err['DC']),
#      str(rms_err['MC']),],
#     ncol=1, frameon=True, title='RMSE',
#     loc = 'upper right', handletextpad=0.05,)

ax.grid(True, which='both',
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)



'''
plt.text(
    0.95, 0.05, 'PMIP4 model ensemble',
    horizontalalignment='right', verticalalignment='bottom',
    transform=ax.transAxes, backgroundcolor='white',)

print((SO_jfm_sst_site_values['EC'].groupby(['Station']).mean()[
    ['sim_rec_jfm_sst_lig_pi']]).mean().values[0])

print((SO_jfm_sst_site_values['JH'].groupby(['Station']).mean()[
    ['sim_rec_jfm_sst_lig_pi']]).mean().values[0])

print((SO_jfm_sst_site_values['DC'].groupby(['Station']).mean()[
    ['sim_rec_jfm_sst_lig_pi']]).mean().values[0])

print((SO_jfm_sst_site_values['MC'].groupby(['Station']).mean()[
    ['sim_rec_jfm_sst_lig_pi']]).mean().values[0])

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec sep SIC



data_to_plot = {}

data_to_plot['MC'] = SO_sep_sic_site_values['MC'].groupby(['Station']).mean()[
    ['rec_sep_sic_lig_pi', 'sim_sep_sic_lig_pi']]

mean_err = {}
rms_err  = {}

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.2_sim_rec_sic/7.0.3.2.0 sim_rec sep_sic ens.png'

axis_min = -70
axis_max = 15

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

for irec in ['MC']:
    # irec = 'MC'
    print(irec)
    
    ax.scatter(
        data_to_plot[irec]['rec_sep_sic_lig_pi'],
        data_to_plot[irec]['sim_sep_sic_lig_pi'],
        marker=marker_recs[irec],
        s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
        )
    
    mean_err[irec] = np.int(
        (SO_sep_sic_site_values[irec].groupby(['Station']).mean()[
            ['sim_rec_sep_sic_lig_pi']]).mean().values[0])
    rms_err[irec] = np.int(mean_squared_error(
        data_to_plot[irec]['rec_sep_sic_lig_pi'],
        data_to_plot[irec]['sim_sep_sic_lig_pi'],
        squared=False))

ax.plot([0, 1], [0, 1], transform=ax.transAxes,
        c='k', lw=0.5, ls='--')
ax.hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
ax.vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')

ax.set_ylabel('Simulations [$\%$]')
ax.set_ylim(axis_min, axis_max)
ax.set_yticks(np.arange(axis_min, axis_max + 1e-4, 10))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel('Reconstructions [$\%$]')
ax.set_xlim(axis_min, axis_max)
ax.set_xticks(np.arange(axis_min, axis_max + 1e-4, 10))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# l1 = plt.scatter(
#     [],[], marker=marker_recs['MC'],
#     s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,)
# plt.legend(
#     [l1,],
#     [str(rms_err['MC']),],
#     ncol=1, frameon=True, title='RMSE',
#     loc = (0.7, 0.8), handletextpad=0.05,)

ax.grid(True, which='both',
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.97)
fig.savefig(output_png)


'''
plt.text(
    0.95, 0.05, 'PMIP4 model ensemble',
    horizontalalignment='right', verticalalignment='bottom',
    transform=ax.transAxes, backgroundcolor='white',)

'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region compare sim_rec ann_sst multiple models

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec ann_sst multiple_models.png'

axis_min = -8
axis_max = 12

nrow = 3
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([7.5*ncol, 7.5*nrow]) / 2.54,
    gridspec_kw={'hspace': 0.14, 'wspace': 0.12},
    sharex=True, sharey=True,)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        plt.text(
            0, 1.02, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='bottom', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        plt.text(
            0.5, 1.05,
            model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        data_to_plot = {}
        data_to_plot['EC'] = SO_ann_sst_site_values['EC'].loc[SO_ann_sst_site_values['EC']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
        data_to_plot['JH'] = SO_ann_sst_site_values['JH'].loc[SO_ann_sst_site_values['JH']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
        data_to_plot['DC'] = SO_ann_sst_site_values['DC'].loc[SO_ann_sst_site_values['DC']['Model'] == model][['rec_ann_sst_lig_pi', 'sim_ann_sst_lig_pi']]
        data_to_plot['EC_tas'] = AIS_ann_tas_site_values['EC'].loc[AIS_ann_tas_site_values['EC']['Model'] == model][['rec_ann_tas_lig_pi', 'sim_ann_tas_lig_pi']]
        
        rms_err  = {}
        for irec in ['EC', 'JH', 'DC']:
            # irec = 'EC'
            # print(irec)
            axs[irow, jcol].scatter(
                data_to_plot[irec]['rec_ann_sst_lig_pi'],
                data_to_plot[irec]['sim_ann_sst_lig_pi'],
                marker=marker_recs[irec],
                s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
                )
            rms_err[irec] = np.round(mean_squared_error(
                data_to_plot[irec]['rec_ann_sst_lig_pi'],
                data_to_plot[irec]['sim_ann_sst_lig_pi'],
                squared=False), 1)
        
        axs[irow, jcol].scatter(
            data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
            data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
            marker=marker_recs['EC'],
            s=symbol_size, c='white', edgecolors='b', lw=linewidth, alpha=alpha,
            )
        rms_err['EC_tas'] = np.round(
            mean_squared_error(
                data_to_plot['EC_tas']['rec_ann_tas_lig_pi'],
                data_to_plot['EC_tas']['sim_ann_tas_lig_pi'],
                squared=False), 1)
        
        l1 = plt.scatter(
            [],[], marker=marker_recs['EC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        l1_1 = plt.scatter(
            [],[], marker=marker_recs['EC'],
            s=symbol_size, c='white', edgecolors='b',
            lw=linewidth, alpha=alpha,)
        l2 = plt.scatter(
            [],[], marker=marker_recs['JH'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        l3 = plt.scatter(
            [],[], marker=marker_recs['DC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        axs[irow, jcol].legend(
            [l1, l1_1, l2, l3,],
            [str(rms_err['EC']) + ' (SST)',
             str(rms_err['EC_tas']) + ' (SAT)',
             str(rms_err['JH']),
             str(rms_err['DC']),],
            ncol=1, frameon=True, title='RMSE',
            loc = 'upper right', handletextpad=0.05,)
        
        axs[irow, jcol].plot([0, 1], [0, 1], transform=axs[irow, jcol].transAxes, c='k', lw=0.5, ls='--')
        axs[irow, jcol].hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        axs[irow, jcol].vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        
        if (jcol==0):
            axs[irow, jcol].set_ylabel('Simulations [$°C$]')
        axs[irow, jcol].set_ylim(axis_min, axis_max)
        axs[irow, jcol].set_yticks(np.arange(axis_min, axis_max + 1e-4, 2))
        axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
        
        if (irow== (nrow-1)):
            axs[irow, jcol].set_xlabel('Reconstructions [$°C$]')
        axs[irow, jcol].set_xlim(axis_min, axis_max)
        axs[irow, jcol].set_xticks(np.arange(axis_min, axis_max + 1e-4, 2))
        axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
        
        axs[irow, jcol].grid(
            True, which='both',
            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


fig.subplots_adjust(left=0.05, right = 0.99, bottom = 0.05, top = 0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec jfm_sst multiple models

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0_sim_rec_sst/7.0.3.0.0 sim_rec jfm_sst multiple_models.png'

axis_min = -8
axis_max = 12

nrow = 3
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([7.5*ncol, 7.5*nrow]) / 2.54,
    gridspec_kw={'hspace': 0.14, 'wspace': 0.12},
    sharex=True, sharey=True,)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        plt.text(
            0, 1.02, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='bottom', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        plt.text(
            0.5, 1.05,
            model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        data_to_plot = {}
        data_to_plot['EC'] = SO_jfm_sst_site_values['EC'].loc[SO_jfm_sst_site_values['EC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
        data_to_plot['JH'] = SO_jfm_sst_site_values['JH'].loc[SO_jfm_sst_site_values['JH']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
        data_to_plot['DC'] = SO_jfm_sst_site_values['DC'].loc[SO_jfm_sst_site_values['DC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
        data_to_plot['MC'] = SO_jfm_sst_site_values['MC'].loc[SO_jfm_sst_site_values['MC']['Model'] == model][['rec_jfm_sst_lig_pi', 'sim_jfm_sst_lig_pi']]
        
        rms_err  = {}
        for irec in ['EC', 'JH', 'DC', 'MC']:
            # irec = 'EC'
            # print(irec)
            axs[irow, jcol].scatter(
                data_to_plot[irec]['rec_jfm_sst_lig_pi'],
                data_to_plot[irec]['sim_jfm_sst_lig_pi'],
                marker=marker_recs[irec],
                s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
                )
            rms_err[irec] = np.round(mean_squared_error(
                data_to_plot[irec]['rec_jfm_sst_lig_pi'],
                data_to_plot[irec]['sim_jfm_sst_lig_pi'],
                squared=False), 1)
        
        l1 = plt.scatter(
            [],[], marker=marker_recs['EC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        l2 = plt.scatter(
            [],[], marker=marker_recs['JH'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        l3 = plt.scatter(
            [],[], marker=marker_recs['DC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        l4 = plt.scatter(
            [],[], marker=marker_recs['MC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        axs[irow, jcol].legend(
            [l1, l2, l3, l4,],
            [str(rms_err['EC']),
             str(rms_err['JH']),
             str(rms_err['DC']),
             str(rms_err['MC']),],
            ncol=1, frameon=True, title='RMSE',
            loc = 'upper right', handletextpad=0.05,)
        
        axs[irow, jcol].plot([0, 1], [0, 1], transform=axs[irow, jcol].transAxes, c='k', lw=0.5, ls='--')
        axs[irow, jcol].hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        axs[irow, jcol].vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        
        if (jcol==0):
            axs[irow, jcol].set_ylabel('Simulations [$°C$]')
        axs[irow, jcol].set_ylim(axis_min, axis_max)
        axs[irow, jcol].set_yticks(np.arange(axis_min, axis_max + 1e-4, 2))
        axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
        
        if (irow== (nrow-1)):
            axs[irow, jcol].set_xlabel('Reconstructions [$°C$]')
        axs[irow, jcol].set_xlim(axis_min, axis_max)
        axs[irow, jcol].set_xticks(np.arange(axis_min, axis_max + 1e-4, 2))
        axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
        
        axs[irow, jcol].grid(
            True, which='both',
            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


fig.subplots_adjust(left=0.05, right = 0.99, bottom = 0.05, top = 0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare sim_rec sep SIC multiple models

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.2_sim_rec_sic/7.0.3.2.0 sim_rec sep_sic multiple_models.png'

axis_min = -70
axis_max = 15

nrow = 3
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([7.5*ncol, 7.5*nrow]) / 2.54,
    gridspec_kw={'hspace': 0.14, 'wspace': 0.12},
    sharex=True, sharey=True,)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        plt.text(
            0, 1.02, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='bottom', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        plt.text(
            0.5, 1.05,
            model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        data_to_plot = {}
        data_to_plot['MC'] = SO_sep_sic_site_values['MC'].loc[SO_sep_sic_site_values['MC']['Model'] == model][['rec_sep_sic_lig_pi', 'sim_sep_sic_lig_pi']]
        
        rms_err  = {}
        axs[irow, jcol].scatter(
            data_to_plot['MC']['rec_sep_sic_lig_pi'],
            data_to_plot['MC']['sim_sep_sic_lig_pi'],
            marker=marker_recs['MC'],
            s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
            )
        
        rms_err['MC'] = np.int(mean_squared_error(
            data_to_plot['MC']['rec_sep_sic_lig_pi'],
            data_to_plot['MC']['sim_sep_sic_lig_pi'],
            squared=False))
        
        l1 = plt.scatter(
            [],[], marker=marker_recs['MC'],
            s=symbol_size, c='white', edgecolors='k',
            lw=linewidth, alpha=alpha,)
        axs[irow, jcol].legend(
            [l1,],
            [str(rms_err['MC']),],
            ncol=1, frameon=True, title='RMSE',
            loc = 'lower right', handletextpad=0.05,)
        
        axs[irow, jcol].plot([0, 1], [0, 1], transform=axs[irow, jcol].transAxes, c='k', lw=0.5, ls='--')
        axs[irow, jcol].hlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        axs[irow, jcol].vlines(0, axis_min, axis_max, colors='k', lw=0.5, linestyles='--')
        
        if (jcol==0):
            axs[irow, jcol].set_ylabel('Simulations [$°C$]')
        axs[irow, jcol].set_ylim(axis_min, axis_max)
        axs[irow, jcol].set_yticks(np.arange(axis_min, axis_max + 1e-4, 10))
        axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
        
        if (irow== (nrow-1)):
            axs[irow, jcol].set_xlabel('Reconstructions [$°C$]')
        axs[irow, jcol].set_xlim(axis_min, axis_max)
        axs[irow, jcol].set_xticks(np.arange(axis_min, axis_max + 1e-4, 10))
        axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
        
        axs[irow, jcol].grid(
            True, which='both',
            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


fig.subplots_adjust(left=0.05, right = 0.99, bottom = 0.05, top = 0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------




