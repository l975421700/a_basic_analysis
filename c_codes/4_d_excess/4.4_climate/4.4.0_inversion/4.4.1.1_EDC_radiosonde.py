

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')
from datetime import datetime

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
from siphon.simplewebservice.igra2 import IGRAUpperAir

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
    inversion_top,
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
# region animate EDC vertical temperature profiles

EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

idatestart = np.where(date == np.datetime64('2014-12-25T12:00:00.000000000'))[0][0]
idateend = np.where(date == np.datetime64('2015-01-16T12:00:00.000000000'))[0][0]

# stats.describe(EDC_df_drvd.loc[EDC_df_drvd.calculated_height <= 5000, 'temperature'])

# output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0 vertical sounding profile at EDC 2006_2022.mp4'
output_mp4 = 'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0 vertical sounding profile at EDC 2014-12-25_2015-01-16.mp4'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)
ims = []
for i in np.arange(idatestart, idateend+1, 1):
    # range(len(date))
    # i=0
    altitude = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'calculated_height'].values / 1000
    temperature = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'temperature'].values
    
    plt_line = ax.plot(
        temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
    plt_text = ax.text(
        0.1, 0.1,
        str(date[i])[0:10],
        transform=ax.transAxes, color='k', ha='left', va = 'center')
    
    t_it, h_it = inversion_top(temperature, altitude)
    if (not np.isnan(t_it)):
        plt_scatter = ax.scatter(t_it, h_it, s=5, c='red', zorder=3)
        ims.append(plt_line + [plt_text, plt_scatter])
    else:
        ims.append(plt_line + [plt_text])
    
    print(str(i) + '/' + str(len(date)))

plt.text(
    0.9, 0.9, 'EDC', transform=ax.transAxes, color='k',
    ha='right', va = 'center')

ax.set_xticks(np.arange(190, 260.1, 10))
ax.set_xlim(190, 260)
ax.set_xlabel('Temperature [$K$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_yticks(np.arange(3, 5.1, 0.5))
ax.set_ylim(3, 5)
ax.set_ylabel('Altitude [$km$]')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(
    True, which='both',
    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)




'''
#-------------------------------- check

EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

i=2000
outputfile = 'figures/test/test.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

altitude = EDC_df_drvd.iloc[
    np.where(EDC_df_drvd.date == date[i])[0]][
    'calculated_height'].values / 1000
temperature = EDC_df_drvd.iloc[
    np.where(EDC_df_drvd.date == date[i])[0]][
    'temperature'].values

plt_line = ax.plot(
    temperature, altitude, '.-', color='black', lw=0.5, markersize=2.5)
plt_text = ax.text(
    0.5, 0.5,
    '#' + str(i) + '   ' +
    str(date[i])[0:10] + ' ' + str(date[i])[11:13] + ':00 UTC',
    transform=ax.transAxes,)

ax.set_yticks(np.arange(3, 5.1, 0.5))
ax.set_ylim(3, 5)

ax.set_xticks(np.arange(190, 260.1, 10))
ax.set_xlim(190, 260)

ax.set_ylabel('Height [km]')
ax.set_xlabel('Temperature [K]')
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.96)
fig.savefig(outputfile)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate EDC vertical temperature profiles against pressure

EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54, dpi=600)
ims = []
for i in range(len(date)):
    # i=0
    pressure = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'pressure'].values
    temperature = EDC_df_drvd.iloc[
        np.where(EDC_df_drvd.date == date[i])[0]][
        'temperature'].values - zerok
    
    plt_line = ax.plot(
        temperature, pressure, '.-', color='black', lw=0.5, markersize=2.5)
    plt_text = ax.text(
        0.9, 0.8,
        str(date[i])[0:10],
        transform=ax.transAxes, ha='right', va = 'center')
    ims.append(plt_line + [plt_text])
    print(str(i) + '/' + str(len(date)))

xlim_min = -80
xlim_max = -20
x_interval = 20
xtickmin = np.ceil(xlim_min / x_interval) * x_interval
xtickmax = np.floor(xlim_max / x_interval) * x_interval

ylim_min = 200
ylim_max = 700
y_interval = 100
ytickmin = np.ceil(ylim_min / y_interval) * y_interval
ytickmax = np.floor(ylim_max / y_interval) * y_interval

ax.set_xlim(xlim_min, xlim_max)
ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, x_interval))
ax.set_xlabel('Temperature [$°C$]', labelpad=3)
ax.xaxis.set_tick_params(labelsize=8)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))

ax.set_ylim(ylim_min, ylim_max)
ax.set_yticks(np.arange(ytickmin, ytickmax + 1e-4, y_interval))
ax.invert_yaxis()
ax.set_ylabel('Pressure [$hPa$]', labelpad=3)
ax.yaxis.set_tick_params(labelsize=8)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
fig.subplots_adjust(left=0.28, right=0.95, bottom=0.25, top=0.95)

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/8_d-excess/8.2_climate/8.2.0_inversion/8.2.0.0_site_examples/8.2.0.0 vertical sounding profile against pressure at EDC 2006_2022.mp4',
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)

# endregion
# -----------------------------------------------------------------------------
