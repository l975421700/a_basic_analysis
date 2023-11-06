

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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

# self defined
from a_basic_analysis.b_module.mapplot import (
    remove_trailing_zero,
    remove_trailing_zero_pos,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_multi_gridvalue_at_site,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

NK16_Australia_Syowa_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.NK16_Australia_Syowa_1d_sim.pkl', 'rb') as f:
    NK16_Australia_Syowa_1d_sim[expid[i]] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
NK16_Australia_Syowa_1d_sim[expid[i]].iloc[np.argmax(NK16_Australia_Syowa_1d_sim[expid[i]]['d_ln'])]

#---------------- check correlation
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    subset = np.isfinite(NK16_Australia_Syowa_1d_sim[expid[i]][var_name]) & np.isfinite(NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim'])
    
    print(np.round(pearsonr(NK16_Australia_Syowa_1d_sim[expid[i]][var_name][subset], NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim'][subset], ).statistic ** 2, 3))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by year

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = NK16_Australia_Syowa_1d_sim[expid[i]]['period']
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline(
        (0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by latitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_lat.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = abs(NK16_Australia_Syowa_1d_sim[expid[i]]['lat'])
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='Latitude [$°\;S$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by temperature bias

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_T_bias.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = (abs(NK16_Australia_Syowa_1d_sim[expid[i]]['t_air'] - NK16_Australia_Syowa_1d_sim[expid[i]]['t_air_sim']) >= 7.5)
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='Large T bias')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by temperature bias continuous

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_T_bias_continuous.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = abs(NK16_Australia_Syowa_1d_sim[expid[i]]['t_air'] - NK16_Australia_Syowa_1d_sim[expid[i]]['t_air_sim'])
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='T bias [$°C$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by SIC

ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

NK16_1d_SIC = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa_1d_sim[expid[i]]['time'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lat'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc.values * 100
    )

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_SIC.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = NK16_1d_SIC.copy()
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='SIC [$\%$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by open ocean contribution to q

q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)

NK16_1d_oo2q = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa_1d_sim[expid[i]]['time'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lat'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lon'],
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].time.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].sel(geo_regions='Open Ocean').values,
    )

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_oo2q.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = NK16_1d_oo2q.copy()
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='Open ocean contribution to q [$\%$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot, colored by SLM

T63GR15_jan_surf = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/input/echam/unit.24')

NK16_1d_SLM = find_multi_gridvalue_at_site(
    NK16_Australia_Syowa_1d_sim[expid[i]]['lat'].values,
    NK16_Australia_Syowa_1d_sim[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
)

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.0 ' + expid[i] + ' NK16 observed vs. simulated daily ' + var_name + ' colored_by_SLM.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    ydata = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = NK16_1d_SLM.copy()
    cdata = cdata[subset]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata,
        s=12, palette='viridis',
        # marker="o",
    )
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        ', $RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.32, 0.15, eq_text,
        transform=ax.transAxes, fontsize=8, ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    ax.legend(title='Land [1] or Ocean [0]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region map plot

#-------------------------------- basic settings
projections = ccrs.LambertConformal(
        central_longitude=(10 + 160)/2,
        central_latitude=(-77 -25)/2,
        cutoff=30, standard_parallels=(-33, -45),)

col_names = ['2013/14 Austral Summer', '2014/15 Austral Summer',
             '2015/16 Austral Summer', '2016/17 Austral Summer',
             '2017/18 Austral Summer', '2018/19 Austral Summer',
             '2019/20 Austral Summer',]

periods = ['13summer', '14summer', '15summer', '16summer', '17summer',
           '18summer', '19summer']

nrow = 2
ncol = 4
wspace = 0.02
hspace = 0.04
fm_left = 0.01
fm_bottom = 0.005
fm_right = 0.99
fm_top = 0.965

#-------------------------------- variables related settings
cm_mins = [-240, -34, -5, 0, -9, 1, ]
cm_maxs = [-80, -10, 45, 45, 21, 10, ]
cm_interval1s = [20, 2, 5, 5, 3, 1, ]
cm_interval2s = [40, 4, 10, 10, 6, 2, ]
cmaps = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']

min_size = 6
scale_size = [0.2, 1, 1, 1, 1, 5, ]
size_interval = [25, 5, 5, 5, 5, 1]

for icount, var_name in enumerate(['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']):
    # ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']
    # var_name = 'd_ln'
    # icount=0; var_name='dD'
    # icount=3; var_name='d_ln'
    # icount=5; var_name='q'
    print('#---------------- ' + str(icount) + ' ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.1 ' + expid[i] + ' maps of NK16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([5.5*ncol, 3.3*nrow]) / 2.54,
        subplot_kw={'projection': projections},)
    
    #---------------- plot background
    
    for irow in range(nrow):
        for jcol in range(ncol):
            
            if ((irow != (nrow-1)) | (jcol != (ncol-1))):
                axs[irow, jcol] = hemisphere_conic_plot(ax_org=axs[irow, jcol])
                cplot_ice_cores(
                    lon=ten_sites_loc.lon[ten_sites_loc['Site']=='EDC'],
                    lat=ten_sites_loc.lat[ten_sites_loc['Site']=='EDC'],
                    ax=axs[irow, jcol], s=12, marker='*',)
                plt.text(
                    0, 1, panel_labels[irow * ncol + jcol] + ' ' + col_names[irow * ncol + jcol],
                    transform=axs[irow, jcol].transAxes,
                    ha='left', va='bottom', rotation='horizontal')
    
    axs[nrow-1, ncol-1].axis('off')
    
    #---------------- plot data
    
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = cm_mins[icount],
        cm_max = cm_maxs[icount],
        cm_interval1 = cm_interval1s[icount],
        cm_interval2 = cm_interval2s[icount],
        cmap = cmaps[icount],)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow = 0; jcol = 0
            if ((irow != (nrow-1)) | (jcol != (ncol-1))):
                print('#-------- ' + periods[irow * ncol + jcol])
                
                data_iperiod = NK16_Australia_Syowa_1d_sim[expid[i]][NK16_Australia_Syowa_1d_sim[expid[i]]['period'] == periods[irow * ncol + jcol]]
                
                subset = np.isfinite(data_iperiod[var_name]) & np.isfinite(data_iperiod[var_name + '_sim'])
                lat_subset = data_iperiod['lat'][subset]
                lon_subset = data_iperiod['lon'][subset]
                var_subset = data_iperiod[var_name][subset]
                var_sim_subset = data_iperiod[var_name + '_sim'][subset]
                var_diff_subset = (var_sim_subset - var_subset)
                
                if (var_name == 'q'):
                    var_subset = var_subset * 1000
                    var_sim_subset = var_sim_subset * 1000
                    var_diff_subset = var_diff_subset * 1000
                
                edgecolors = np.repeat('darkred', len(var_diff_subset))
                edgecolors[var_diff_subset < 0] = 'gray'
                
                # here
                plt_scatter = axs[irow, jcol].scatter(
                    x=lon_subset,
                    y=lat_subset,
                    c=var_subset,
                    s=min_size + scale_size[icount]*abs(var_diff_subset),
                    edgecolors=edgecolors,
                    lw=0.5, marker='o', zorder=2,
                    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),
                    )
    
    #---------------- plot color legend
    cbar = fig.colorbar(
        plt_scatter, ax=axs[nrow-1, ncol-1], aspect=25,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        fraction=1.05,
        )
    # cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('Observed ' + plot_labels[var_name])
    
    #---------------- plot size legend
    l0 = plt.scatter(
        [],[], marker='o', lw=0.5,
        c='black',# edgecolors = 'black',
        s=min_size + scale_size[icount] * 0 * size_interval[icount],)
    l1 = plt.scatter(
        [],[], marker='o', lw=0.5,
        c='black',# edgecolors = 'black',
        s=min_size + scale_size[icount] * 1 * size_interval[icount],)
    l2 = plt.scatter(
        [],[], marker='o', lw=0.5,
        c='black',# edgecolors = 'black',
        s=min_size + scale_size[icount] * 2 * size_interval[icount],)
    l3 = plt.scatter(
        [],[], marker='o', lw=0.5,
        c='black',# edgecolors = 'black',
        s=min_size + scale_size[icount] * 3 * size_interval[icount],)
    l4 = plt.scatter(
        [],[], marker='o', lw=0.5,
        c='black',# edgecolors = 'black',
        s=min_size + scale_size[icount] * 4 * size_interval[icount],)
    
    l5 = plt.scatter(
        [],[], marker='o', c='white', edgecolors = 'darkred', lw=0.75, s=20,)
    l6 = plt.scatter(
        [],[], marker='o', c='white', edgecolors = 'gray', lw=0.75, s=20,)
    
    axs[nrow-1, ncol-1].legend(
        [l0, l1, l2,
         l3, l4,
         l5, l6,
         ],
        [size_interval[icount] * 0, size_interval[icount] * 1,
         size_interval[icount] * 2, size_interval[icount] * 3,
         size_interval[icount] * 4, 'Positive', 'Negative',],
        title = 'Differences between simulated and\nobserved ' + plot_labels[var_name],
        ncol=3, frameon=False,
        loc = 'center', bbox_to_anchor=(0.5, 4.2, 0.5, 0.5),
        labelspacing=0.3, handletextpad=0, columnspacing=0.2,)
    
    fig.subplots_adjust(
        left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
        wspace=wspace, hspace=hspace,)
    fig.savefig(output_png)



'''
#-------------------------------- check variable ranges
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    print('#---------------- ' + var_name)
    
    print(stats.describe(NK16_Australia_Syowa_1d_sim[expid[i]][var_name], nan_policy='omit'))


#-------------------------------- plot locations of NK16 daily observations

output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.1 NK16 daily records locations.png'
fig, ax = hemisphere_conic_plot(add_grid_labels=True,)
cplot_ice_cores(
    lon=ten_sites_loc.lon[ten_sites_loc['Site']=='EDC'],
    lat=ten_sites_loc.lat[ten_sites_loc['Site']=='EDC'],
    ax=ax, s=12,)
ax.scatter(
        x=NK16_Australia_Syowa_1d_sim[expid[i]]['lon'],
        y=NK16_Australia_Syowa_1d_sim[expid[i]]['lat'],
        s=8, c='none', lw=0.5, marker='o',
        edgecolors='black', zorder=4, alpha=0.5,
        transform=ccrs.PlateCarree(),)
fig.savefig(output_png)

# np.unique(NK16_Australia_Syowa_1d_sim[expid[i]]['period'], return_counts=True)
stats.describe(NK16_Australia_Syowa_1d_sim[expid[i]]['lat']) # [-70, -30]
stats.describe(NK16_Australia_Syowa_1d_sim[expid[i]]['lon']) # [18, 153]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot against latitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.2 ' + expid[i] + ' NK16 observed and simulated daily ' + var_name + ' against latitude.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
    
    var_obs = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    var_sim = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(var_obs) & np.isfinite(var_sim))
    var_obs = var_obs[subset]
    var_sim = var_sim[subset]
    var_lat = NK16_Australia_Syowa_1d_sim[expid[i]]['lat'][subset]
    
    if (var_name == 'q'):
        var_obs = var_obs * 1000
        var_sim = var_sim * 1000
    
    sns.scatterplot(x=var_lat, y=var_obs, s=12, marker='+', alpha=0.5,
                    label='Observations',)
    sns.scatterplot(x=var_lat, y=var_sim, s=12, marker='x', alpha=0.5,
                    label='Simulations',)
    ax.legend(handletextpad=0)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xticks(np.arange(-70, -30 + 1e-3, 10))
    ax.set_xticklabels([remove_trailing_zero(x) for x in ax.get_xticks()*(-1)])
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    # if (var_name == 'q'):
    #     ax.set_yticks(np.arange(0, 0.015 + 1e-3, 0.003))
    
    ax.set_xlabel('Latitude [$°\;S$]', labelpad=3)
    ax.set_ylabel(plot_labels[var_name], labelpad=3)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.18, top=0.96)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot differences against latitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_air', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.0_NK16/8.3.0.0.2 ' + expid[i] + ' NK16 diff. in observed and simulated daily ' + var_name + ' against latitude.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
    
    var_obs = NK16_Australia_Syowa_1d_sim[expid[i]][var_name]
    var_sim = NK16_Australia_Syowa_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(var_obs) & np.isfinite(var_sim))
    var_obs = var_obs[subset]
    var_sim = var_sim[subset]
    var_lat = NK16_Australia_Syowa_1d_sim[expid[i]]['lat'][subset]
    
    if (var_name == 'q'):
        var_obs = var_obs * 1000
        var_sim = var_sim * 1000
    
    var_diff = var_sim - var_obs
    
    sns.scatterplot(x=var_lat, y=var_diff, s=12, marker='+', alpha=0.5,)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xticks(np.arange(-70, -30 + 1e-3, 10))
    ax.set_xticklabels([remove_trailing_zero(x) for x in ax.get_xticks()*(-1)])
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    ax.set_xlabel('Latitude [$°\;S$]', labelpad=3)
    ax.set_ylabel('Sim. vs. Obs. ' + plot_labels[var_name], labelpad=3)
    
    ax.axhline(y=0, linewidth=0.8, color='gray', alpha=0.75, linestyle='-')
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.18, top=0.96)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

