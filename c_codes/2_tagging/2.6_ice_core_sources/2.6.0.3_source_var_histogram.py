

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
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
sys.path.append('/work/ollie/qigao001')
os.chdir('/work/ollie/qigao001/')
# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
from scipy.stats import pearsonr
import pingouin as pg

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
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
    remove_trailing_zero_pos_abs,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
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


pre_weighted_var_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
    pre_weighted_var_icores[expid[i]] = pickle.load(f)

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

'''
wisoaprt_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_alltime_icores.pkl', 'rb') as f:
    wisoaprt_alltime_icores[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source latitude

ivar = 'lat'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_' + ivar + '/histgram/6.1.3 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xmin = np.nanmin(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xlim_min = xmin - 1
    xlim_max = xmax + 1
    
    xtickmin = np.ceil(xlim_min / 10) * 10
    xtickmax = np.floor(xlim_max / 10) * 10
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
        binwidth=1,
        )
    # ? shall we use weights?
    
    ax.axvline(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Source latitude [$°\;S$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 10))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)



'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source longitude

ivar = 'lon'

for isite in stations_sites.Site:
    # isite = 'Halley'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_' + ivar + '/histgram/6.1.3 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(
        calc_lon_diff(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
                      t63_sites_indices[isite]['lon'],)
        )
    xmin = np.nanmin(
        calc_lon_diff(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
                      t63_sites_indices[isite]['lon'],)
    )
    xlim_min = xmin - 1
    xlim_max = xmax + 1
    
    xtickmin = np.ceil(xlim_min / 60) * 60
    xtickmax = np.floor(xlim_max / 60) * 60
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        calc_lon_diff(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
                      t63_sites_indices[isite]['lon'],),
        binwidth=5,
        )
    # ? shall we use weights?
    
    ax.axvline(
        calc_lon_diff(pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
                      t63_sites_indices[isite]['lon'],),
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Relative source longitude [$°$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 60))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)




'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source sst

ivar = 'sst'

for isite in stations_sites.Site:
    # isite = 'Halley'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_' + ivar + '/histgram/6.1.3 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xmin = np.nanmin(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xlim_min = xmin - 0.5
    xlim_max = xmax + 0.5
    
    xtickmin = abs(np.ceil(xlim_min / 5)) * 5
    xtickmax = np.floor(xlim_max / 5) * 5
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
        binwidth=1,
        )
    # ? shall we use weights?
    
    ax.axvline(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Source SST [$°C$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 5))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)



'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source-sink distance

ivar = 'distance'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.5_transport_distance/histgram/6.1.3 ' + expid[i] + ' histogram of daily source-sink distance at ' + isite + '.png'
    
    xmax = np.nanmax(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'] / 100)
    xmin = np.nanmin(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'] / 100)
    xlim_min = xmin - 1
    xlim_max = xmax + 1
    
    xtickmin = abs(np.ceil(xlim_min / 10)) * 10
    xtickmax = np.floor(xlim_max / 10) * 10
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        pre_weighted_var_icores[expid[i]][isite][ivar]['daily'] / 100,
        binwidth=1,
        )
    # ? shall we use weights?
    
    ax.axvline(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'] / 100,
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Source-sink distance [$10^2 \; km$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 10))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)



'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source rh2m

ivar = 'rh2m'

for isite in stations_sites.Site:
    # isite = 'Halley'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_' + ivar + '/histgram/6.1.3 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xmin = np.nanmin(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xlim_min = xmin - 1
    xlim_max = xmax + 1
    
    if((xlim_max - xlim_min) > 50):
        tickbin = 10
    else:
        tickbin = 5
    
    xtickmin = np.ceil(xlim_min / tickbin) * tickbin
    xtickmax = np.floor(xlim_max / tickbin) * tickbin
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
        binwidth=1,
        )
    # ? shall we use weights?
    
    ax.axvline(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Source rh2m [$\%$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, tickbin))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)



'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot histogram of daily source wind10

ivar = 'wind10'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.4_' + ivar + '/histgram/6.1.3 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xmin = np.nanmin(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'])
    xlim_min = xmin - 0.5
    xlim_max = xmax + 0.5
    
    xtickmin = np.ceil(xlim_min / 2) * 2
    xtickmax = np.floor(xlim_max / 2) * 2
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(
        pre_weighted_var_icores[expid[i]][isite][ivar]['daily'],
        binwidth=0.5,
        )
    # ? shall we use weights?
    
    ax.axvline(
        pre_weighted_var_icores[expid[i]][isite][ivar]['am'],
        c = 'red', linewidth=0.5,
        )
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Source wind10 [$m\;s^{-1}$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 2))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.22, top=0.97)
    fig.savefig(output_png)



'''
pre_weighted_var_icores[expid[i]][isite][ivar]['am'].values
np.nanmean(pre_weighted_var_icores[expid[i]][isite][ivar]['daily'].values)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region correlation among source properties

#-------------------------------- lat vs. ???

ivar1 = 'lat'

# ivar2 = 'sst'
# ivar2 = 'rh2m'
# ivar2 = 'distance'
ivar2 = 'wind10'


#-------------------------------- lon vs. ???

# ivar1 = 'lon'

# ivar2 = 'sst'
# ivar2 = 'rh2m'
# ivar2 = 'distance'
# ivar2 = 'wind10'


for isite in ['EDC', 'Halley']:
    
    # isite = 'EDC'
    print('#---------------- ' + isite + ': ' + ivar1 + ' vs. ' + ivar2)
    
    data1 = pre_weighted_var_icores[expid[i]][isite][ivar1]['daily']
    data2 = pre_weighted_var_icores[expid[i]][isite][ivar2]['daily']
    
    daily_cor = pearsonr(
        data1[np.isfinite(data1) & np.isfinite(data2)],
        data2[np.isfinite(data1) & np.isfinite(data2)],
    )
    
    if(daily_cor.pvalue > 0.001):
        print('daily pvalue larger than 0.001')
    
    mon_cor = pearsonr(
        pre_weighted_var_icores[expid[i]][isite][ivar1]['mon'],
        pre_weighted_var_icores[expid[i]][isite][ivar2]['mon'],)
    
    if(mon_cor.pvalue > 0.001):
        print('monthly pvalue larger than 0.001')
    
    ann_cor = pearsonr(
        pre_weighted_var_icores[expid[i]][isite][ivar1]['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar2]['ann'],)
    
    if(ann_cor.pvalue > 0.001):
        print('annual pvalue larger than 0.001')
    
    print(
        ivar1 + ' vs. ' + ivar2 + ', day, mon, ann: ' + \
            str(np.round(daily_cor.statistic, 2)) + ', ' + \
                str(np.round(mon_cor.statistic, 2)) + ', ' + \
                    str(np.round(ann_cor.statistic, 2)) +  ' at ' + isite
            )


#-------------------------------- partial correlation

ivar1 = 'lat'
ivar2 = 'lon'

ivar3 = 'sst'
# ivar3 = 'rh2m'
# ivar3 = 'distance'
ivar3 = 'wind10'


for isite in ['EDC', 'Halley']:
    
    # isite = 'EDC'
    print('#---------------- ' + isite + ': ' + ivar2 + ' vs. ' + ivar3 + ' controlling ' + ivar1)
    
    data1 = pre_weighted_var_icores[expid[i]][isite][ivar1]['daily']
    data2 = calc_lon_diff(
        pre_weighted_var_icores[expid[i]][isite][ivar2]['daily'],
        t63_sites_indices[isite]['lon'],)
    data3 = pre_weighted_var_icores[expid[i]][isite][ivar3]['daily']
    
    dataframe = pd.DataFrame(data={
        'ivar1': data1,
        'ivar2': data2,
        'ivar3': data3,
    })
    
    partial_cor = pg.partial_corr(
        data=dataframe, x='ivar2', y='ivar3', covar='ivar1')
    
    print(np.round(partial_cor.r, 2).values)
    
    print(partial_cor['p-val'].values)




'''
#---- lat

lat vs. sst, day, mon, ann: 0.91, 0.86, 0.92 at EDC
lat vs. sst, day, mon, ann: 0.98, 0.91, 0.93 at Halley

lat vs. rh2m, day, mon, ann: -0.68, -0.45, -0.78 at EDC
lat vs. rh2m, day, mon, ann: -0.92, -0.91, -0.85 at Halley

lat vs. distance, day, mon, ann: 0.61, 0.65, 0.9 at EDC
lat vs. distance, day, mon, ann: 0.96, 0.9, 0.79 at Halley

lat vs. wind10, day, mon, ann: -0.62, -0.66, -0.42 at EDC
lat vs. wind10, day, mon, ann: 0.79, 0.52, -0.18 at Halley




#---- lon

lon vs. sst, day, mon, ann: -0.02, -0.05, 0.1 at EDC
lon vs. sst, day, mon, ann: -0.7, -0.52, -0.36 at Halley

lon vs. rh2m, day, mon, ann: 0.07, 0.14, -0.07 at EDC
lon vs. rh2m, day, mon, ann: 0.67, 0.61, 0.32 at Halley

lon vs. distance, day, mon, ann: 0.39, 0.17, -0.38 at EDC
lon vs. distance, day, mon, ann: -0.81, -0.87, -0.85 at Halley

lon vs. wind10, day, mon, ann: -0.01, 0.01, 0.05 at EDC
lon vs. wind10, day, mon, ann: -0.61, -0.47, 0.0 at Halley


'''




'''
#-------------------------------- lat vs. rh2m

ivar1 = 'lat'
ivar2 = 'rh2m'


for isite in ['EDC', 'Halley']:
    
    # isite = 'EDC'
    print('#---------------- ' + isite + ': lat vs. rh2m')
    
    data1 = pre_weighted_var_icores[expid[i]][isite][ivar1]['daily']
    data2 = pre_weighted_var_icores[expid[i]][isite][ivar2]['daily']
    
    daily_cor = pearsonr(
        data1[np.isfinite(data1) & np.isfinite(data2)],
        data2[np.isfinite(data1) & np.isfinite(data2)],
    )
    
    mon_cor = pearsonr(
        pre_weighted_var_icores[expid[i]][isite][ivar1]['mon'],
        pre_weighted_var_icores[expid[i]][isite][ivar2]['mon'],)
    
    ann_cor = pearsonr(
        pre_weighted_var_icores[expid[i]][isite][ivar1]['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar2]['ann'],)
    
    print('daily: ' + str(np.round(daily_cor.statistic, 2)))
    print('monthly: ' + str(np.round(mon_cor.statistic, 2)))
    print('annual:  ' + str(np.round(ann_cor.statistic, 2)))
    
    # -0.45, -0.91
    # -0.78, -0.85

(np.isfinite(data1)).sum()
(np.isfinite(data2)).sum()

'''
# endregion
# -----------------------------------------------------------------------------


