

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
from scipy.stats import linregress
from scipy.stats import pearsonr

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
    calc_lon_diff_np,
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

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# normal sources
pre_weighted_var_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
    pre_weighted_var_icores[expid[i]] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pre_weighted_lat vs. _sst

ivar = 'sst'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    linearfit = linregress(
        x = pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        y = pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
    )
    
    xmax_value = np.max(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xmin_value = np.min(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xlimmax = xmax_value + 0.25
    xlimmin = xmin_value - 0.25
    xtickmax = np.ceil(xlimmax)
    xtickmin = np.floor(xlimmin)
    
    ymax_value = np.max(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ymin_value = np.min(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ylimmax = ymax_value + 0.25
    ylimmin = ymin_value - 0.25
    ytickmax = np.ceil(ylimmax)
    ytickmin = np.floor(ylimmin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 correlation between ann source lat_' + ivar + ' at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
        '.', markersize=1.5,
        )
    
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    plt.text(
        0.55, 0.05,
        '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
            str(np.round(linearfit.intercept, 1)) + \
                '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, fontsize=6, linespacing=1.5)
    
    ax.set_ylabel('Source SST [$°C$]')
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 1))
    ax.set_ylim(ylimmin, ylimmax)
    
    ax.set_xlabel('Source latitude [$°\;S$]')
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 1))
    ax.set_xlim(xlimmin, xlimmax)
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)



for ivar in ['sst', 'rh2m', 'wind10', 'distance']:
    print('#---------------- ' + ivar)
    for isite in ['EDC', 'Halley']:
        print('#-------- ' + isite)
        for ialltime in ['mon', 'sea', 'ann']:
            print('#---- ' + ialltime)
            
            corr = pearsonr(
                pre_weighted_var_icores[expid[i]][isite]['lat'][ialltime],
                pre_weighted_var_icores[expid[i]][isite][ivar][ialltime],
                ).statistic**2
            
            print(np.round(corr, 1))


'''

# 0.85
pearsonr(
    pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
    pre_weighted_var_icores[expid[i]][isite]['sst']['ann'],
).statistic**2
# 0.73
pearsonr(
    pre_weighted_var_icores[expid[i]][isite]['lat']['mon'],
    pre_weighted_var_icores[expid[i]][isite]['sst']['mon'],
).statistic**2
# 0.67
pearsonr(
    pre_weighted_var_icores[expid[i]][isite]['lat']['sea'],
    pre_weighted_var_icores[expid[i]][isite]['sst']['sea'],
).statistic**2
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pre_weighted_lat vs. _rh2m

ivar = 'rh2m'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    linearfit = linregress(
        x = pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        y = pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
    )
    
    xmax_value = np.max(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xmin_value = np.min(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xlimmax = xmax_value + 0.25
    xlimmin = xmin_value - 0.25
    xtickmax = np.ceil(xlimmax)
    xtickmin = np.floor(xlimmin)
    
    ymax_value = np.max(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ymin_value = np.min(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ylimmax = ymax_value + 0.1
    ylimmin = ymin_value - 0.1
    ytickmax = np.ceil(ylimmax)
    ytickmin = np.floor(ylimmin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 correlation between ann source lat_' + ivar + ' at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
        '.', markersize=1.5,
        )
    
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    plt.text(
        0.55, 0.05,
        '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
            str(np.round(linearfit.intercept, 1)) + \
                '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, fontsize=6, linespacing=1.5)
    
    ax.set_ylabel('Source rh2m [$\%$]', labelpad=0.5)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 0.5))
    ax.set_ylim(ylimmin, ylimmax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Source latitude [$°\;S$]')
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 1))
    ax.set_xlim(xlimmin, xlimmax)
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pre_weighted_lat vs. _wind10

ivar = 'wind10'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    linearfit = linregress(
        x = pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        y = pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
    )
    
    xmax_value = np.max(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xmin_value = np.min(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xlimmax = xmax_value + 0.25
    xlimmin = xmin_value - 0.25
    xtickmax = np.ceil(xlimmax)
    xtickmin = np.floor(xlimmin)
    
    ymax_value = np.max(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ymin_value = np.min(pre_weighted_var_icores[expid[i]][isite][ivar]['ann'])
    ylimmax = ymax_value + 0.1
    ylimmin = ymin_value - 0.1
    ytickmax = np.ceil(ylimmax)
    ytickmin = np.floor(ylimmin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 correlation between ann source lat_' + ivar + ' at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann'],
        '.', markersize=1.5,
        )
    
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    plt.text(
        0.55, 0.05,
        '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
            str(np.round(linearfit.intercept, 1)) + \
                '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, fontsize=6, linespacing=1.5)
    
    ax.set_ylabel('Source wind10 [$m \; s^{-1}$]', labelpad=0)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 0.2))
    ax.set_ylim(ylimmin, ylimmax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Source latitude [$°\;S$]')
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 1))
    ax.set_xlim(xlimmin, xlimmax)
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pre_weighted_lat vs. _distance

ivar = 'distance'

for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    linearfit = linregress(
        x = pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        y = pre_weighted_var_icores[expid[i]][isite][ivar]['ann'] / 100,
    )
    
    xmax_value = np.max(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xmin_value = np.min(pre_weighted_var_icores[expid[i]][isite]['lat']['ann'])
    xlimmax = xmax_value + 0.25
    xlimmin = xmin_value - 0.25
    xtickmax = np.ceil(xlimmax)
    xtickmin = np.floor(xlimmin)
    
    ymax_value = np.max(
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann']) / 100
    ymin_value = np.min(
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann']) / 100
    ylimmax = ymax_value + 0.25
    ylimmin = ymin_value - 0.25
    ytickmax = np.ceil(ylimmax)
    ytickmin = np.floor(ylimmin)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_' + ivar + '/6.1.3.6.0 correlation between ann source lat_' + ivar + ' at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    ax.plot(
        pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
        pre_weighted_var_icores[expid[i]][isite][ivar]['ann'] / 100,
        '.', markersize=1.5,
        )
    
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
    
    plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    plt.text(
        0.55, 0.05,
        '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
            str(np.round(linearfit.intercept, 1)) + \
                '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, fontsize=6, linespacing=1.5)
    
    ax.set_ylabel('Source-sink distance [$10^{2} \; km$]', y=0.4)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, 2))
    ax.set_ylim(ylimmin, ylimmax)
    
    ax.set_xlabel('Source latitude [$°\;S$]')
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 1))
    ax.set_xlim(xlimmin, xlimmax)
    
    ax.grid(True, linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.25, top=0.97)
    fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region pre_weighted_lat vs. westerlies

wind10_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_alltime.pkl', 'rb') as f:
    wind10_alltime[expid[i]] = pickle.load(f)

t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')


#-------------------------------- mon relation

b_cellarea = np.broadcast_to(
    t63_cellarea.cell_area.sel(lat=slice(-45, -55)),
    wind10_alltime[expid[i]]['mon'].sel(lat=slice(-45, -55)).shape
)

westerlies_mon = np.average(
    wind10_alltime[expid[i]]['mon'].sel(lat=slice(-45, -55)),
    weights = b_cellarea,
    axis=(1, 2)
)

isite = 'EDC'
pearsonr(
    westerlies_mon,
    pre_weighted_var_icores[expid[i]][isite]['lat']['mon'],
    ).statistic
# -0.45



#-------------------------------- mm relation


b_cellarea = np.broadcast_to(
    t63_cellarea.cell_area.sel(lat=slice(-45, -55)),
    wind10_alltime[expid[i]]['mm'].sel(lat=slice(-45, -55)).shape
)

westerlies_mm = np.average(
    wind10_alltime[expid[i]]['mm'].sel(lat=slice(-45, -55)),
    weights = b_cellarea,
    axis=(1, 2)
)

isite = 'Halley'
pearsonr(
    westerlies_mm,
    pre_weighted_var_icores[expid[i]][isite]['wind10']['mm'],
    ).statistic
# 0.9427280753237633



#-------------------------------- ann relation

b_cellarea = np.broadcast_to(
    t63_cellarea.cell_area.sel(lat=slice(-45, -55)),
    wind10_alltime[expid[i]]['ann'].sel(lat=slice(-45, -55)).shape
)

westerlies_ann = np.average(
    wind10_alltime[expid[i]]['ann'].sel(lat=slice(-45, -55)),
    weights = b_cellarea,
    axis=(1, 2)
)

pearsonr(
    westerlies_ann,
    pre_weighted_var_icores[expid[i]][isite]['lat']['ann'],
    ).statistic
# 0.05301844265636421

pearsonr(
    westerlies_ann,
    pre_weighted_var_icores[expid[i]][isite]['wind10']['ann'],
    ).statistic
# 0.024436680744233854

# endregion
# -----------------------------------------------------------------------------

