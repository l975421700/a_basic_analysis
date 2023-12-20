
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import pearsonr
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath
from metpy.calc import pressure_to_height_std
from metpy.units import units

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
    ticks_labels,
    hemisphere_conic_plot,
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
    plot_labels,
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

dD_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
    dD_q_alltime[expid[i]] = pickle.load(f)

zh_st_ml = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'rb') as f:
    zh_st_ml[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot an example of vertical dD distribution


idate = '2014-12-25'

for isite in ['EDC']:
    # isite = 'EDC'
    print('#--------------------------------' + isite)
    
    x_data = dD_q_alltime[expid[i]]['daily'].sel(
        time=idate + 'T05:52:30',).sel(
            lat=t63_sites_indices[isite]['lat'],
            lon=t63_sites_indices[isite]['lon'],
            method='nearest',).values
    y_data = zh_st_ml[expid[i]]['zh']['daily'].sel(
        time=idate + 'T23:52:30',).sel(
            lat=t63_sites_indices[isite]['lat'],
            lon=t63_sites_indices[isite]['lon'],
            method='nearest',).values / 1000
    
    subset = (y_data <= 5.1)
    x_data = x_data[subset]
    y_data = y_data[subset]
    
    xlim_min = np.min(x_data) - 10
    xlim_max = np.max(x_data) + 10
    x_interval = 10
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.4_vertical_isotopes/8.3.1.4.0 ' + expid[i] + ' example of daily dD against height at ' + isite + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    
    ax.plot(x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel(plot_labels['dD'], labelpad=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(3, 5.1, 0.5))
    ax.set_ylim(3, 5)
    ax.set_ylabel('Height [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt_text = plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    plt_text = plt.text(
        0.1, 0.1, idate, transform=ax.transAxes, color='gray',
        ha='left', va = 'bottom')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    plt.savefig(output_png)
    plt.close()








# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate vertical dD distribution

idatestart = '2014-12-25'
idateend   = '2015-01-16'

for isite in ['EDC']:
    # isite = 'EDC'
    print('#--------------------------------' + isite)
    
    output_mp4 = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.4_vertical_isotopes/8.3.1.4.0 ' + expid[i] + ' daily dD against height at ' + isite + ' ' + idatestart + ' to ' + idateend + '.mp4'
    
    xlim_min = -450
    xlim_max = -350
    x_interval = 20
    xtickmin = np.ceil(xlim_min / x_interval) * x_interval
    xtickmax = np.floor(xlim_max / x_interval) * x_interval
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    
    ims = []
    
    for idate in pd.date_range(idatestart, idateend).values:
        print('#---------------- ' + str(idate)[:10])
        
        x_data = dD_q_alltime[expid[i]]['daily'].sel(
            time=str(idate)[:10] + 'T05:52:30',).sel(
                lat=t63_sites_indices[isite]['lat'],
                lon=t63_sites_indices[isite]['lon'],
                method='nearest',).values
        y_data = zh_st_ml[expid[i]]['zh']['daily'].sel(
            time=str(idate)[:10] + 'T23:52:30',).sel(
                lat=t63_sites_indices[isite]['lat'],
                lon=t63_sites_indices[isite]['lon'],
                method='nearest',).values / 1000
        
        subset = (y_data <= 5.1)
        x_data = x_data[subset]
        y_data = y_data[subset]
        
        plt_line = ax.plot(x_data, y_data, '.-', color='black', lw=0.5, markersize=2.5)
        plt_text = plt.text(
            0.1, 0.1, str(idate)[:10], transform=ax.transAxes, color='gray',
            ha='left', va = 'bottom')
        
        ims.append(plt_line + [plt_text])
        
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
    ax.set_xlabel(plot_labels['dD'], labelpad=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.set_yticks(np.arange(3, 5.1, 0.5))
    ax.set_ylim(3, 5)
    ax.set_ylabel('Height [$km$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    plt_text = plt.text(
        0.9, 0.9, isite, transform=ax.transAxes, color='gray',
        ha='right', va = 'center')
    
    ax.grid(
        True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)



'''
'''
# endregion
# -----------------------------------------------------------------------------
