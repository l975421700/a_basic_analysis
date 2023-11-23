

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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    ticks_labels,
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

IT20_ACE_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.IT20_ACE_1d_sim.pkl', 'rb') as f:
    IT20_ACE_1d_sim[expid[i]] = pickle.load(f)

T63GR15_jan_surf = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/input/echam/unit.24')

IT20_1d_SLM = find_multi_gridvalue_at_site(
    IT20_ACE_1d_sim[expid[i]]['lat'].values,
    IT20_ACE_1d_sim[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
)

echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

IT20_1d_height = find_multi_gridvalue_at_site(
    IT20_ACE_1d_sim[expid[i]]['lat'].values,
    IT20_ACE_1d_sim[expid[i]]['lon'].values,
    echam6_t63_surface_height.lat.values,
    echam6_t63_surface_height.lon.values,
    echam6_t63_surface_height.values,
)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

IT20_1d_SIC = find_multi_gridvalue_at_site_time(
    IT20_ACE_1d_sim[expid[i]]['time'],
    IT20_ACE_1d_sim[expid[i]]['lat'],
    IT20_ACE_1d_sim[expid[i]]['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc.values * 100
    )

q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)

IT20_1d_oo2q = find_multi_gridvalue_at_site_time(
    IT20_ACE_1d_sim[expid[i]]['time'],
    IT20_ACE_1d_sim[expid[i]]['lat'],
    IT20_ACE_1d_sim[expid[i]]['lon'],
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].time.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].sel(geo_regions='Open Ocean').values,
    )


'''
((IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20)).sum()
#---------------- check correlation
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    subset = np.isfinite(IT20_ACE_1d_sim[expid[i]][var_name]) & np.isfinite(IT20_ACE_1d_sim[expid[i]][var_name + '_sim'])
    
    print(np.round(pearsonr(IT20_ACE_1d_sim[expid[i]][var_name][subset], IT20_ACE_1d_sim[expid[i]][var_name + '_sim'][subset], ).statistic ** 2, 3))

'''
# endregion
# -----------------------------------------------------------------------------


# clean
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region Q-Q plot colored by SLM

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_SLM.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = IT20_1d_SLM.copy()
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
    
    ax.legend(title='Land [1] or Ocean [0]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot colored by SIC

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_SIC.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = IT20_1d_SIC.copy()
    cdata = np.round(cdata[subset], 0)
    
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
    
    ax.legend(title='SIC [$\%$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot colored by q

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_q.png'
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_q SLM0 SIC0.png'
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_q SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    # subset = (np.isfinite(xdata) & np.isfinite(ydata))
    # subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0))
    subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = IT20_ACE_1d_sim[expid[i]]['q'].copy() * 1000
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
    
    ax.legend(title=plot_labels['q'])
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot colored by oo2q

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_oo2q.png'
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_oo2q SLM0 SIC0.png'
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_oo2q SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    # subset = (np.isfinite(xdata) & np.isfinite(ydata))
    subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0))
    # subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = IT20_1d_oo2q.copy()
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
    
    ax.legend(title='Open ocean contributions to q [$\%$]', loc='upper left')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot colored by latitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_latitude.png'
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_latitude SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    # subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = abs(IT20_ACE_1d_sim[expid[i]]['lat'].copy())
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
    
    ax.legend(title='Latitude [$°\;S$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot colored by altitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_altitude.png'
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 observed vs. simulated daily ' + var_name + ' colored_by_altitude SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = IT20_ACE_1d_sim[expid[i]][var_name].copy()
    ydata = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'].copy()
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    # subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = IT20_1d_height.copy()
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
    
    ax.legend(title='Height [$m$]')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot oo2q against latitude

xdata = IT20_ACE_1d_sim[expid[i]]['lat'].copy()
ydata = IT20_1d_oo2q.copy()

# subset = np.isfinite(xdata) & np.isfinite(ydata)
# output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 daily oo2q vs. latitude.png'
# subset = np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0)
# output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 daily oo2q vs. latitude SLM0 SIC0.png'
subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 ' + expid[i] + ' IT20 daily oo2q vs. latitude SLM0 SIC0 lat_20_60.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
ax.scatter(xdata[subset], ydata[subset], marker='x', lw=0.5, )

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Latitude [$°\;S$]', labelpad=6)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_ylabel('Open ocean contributions to q [$\%$]', labelpad=6)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# plot map
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot observation locations

xdata = IT20_ACE_1d_sim[expid[i]]['lon']
ydata = IT20_ACE_1d_sim[expid[i]]['lat']

subset = (np.isfinite(xdata) & np.isfinite(ydata) & (IT20_1d_SLM == 0) & (IT20_1d_SIC == 0) & (IT20_ACE_1d_sim[expid[i]]['lat'] >= -60) & (IT20_ACE_1d_sim[expid[i]]['lat'] <= -20))
xdata = xdata[subset]
ydata = ydata[subset]

output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 IT20 daily observation locations SLM0 SIC0 lat_20_60.png'

output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.0 IT20 daily observation locations.png'

# globe
fig, ax = globe_plot(add_grid_labels=False)

ax.scatter(
    xdata,
    ydata,
    marker='x', s=4, lw=0.2,
    transform=ccrs.PlateCarree(),
)
fig.savefig(output_png)

# SH
fig, ax = hemisphere_conic_plot(
    lat_min=-85, lat_max=-20, lon_min=-75, lon_max=155,
    add_grid_labels=True, lat_min_tick = -80, lon_min_tick = -60,)
ax.scatter(
    xdata,
    ydata,
    marker='x', s=12, lw=1,
    transform=ccrs.PlateCarree(),
)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region map plot

#-------------------------------- variables related settings
cm_mins = [-200, -30, -5, 0, 1, ]
cm_maxs = [-80, -10, 15, 25, 15, ]
cm_interval1s = [15, 2, 2.5, 2.5, 1, ]
cm_interval2s = [30, 4, 5, 5, 2, ]
cmaps = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis']

min_size = 6
scale_size = [0.2, 1, 1, 1, 5, ]
size_interval = [25, 5, 5, 5, 1]

for icount, var_name in enumerate(['dD', 'd18O', 'd_xs', 'd_ln', 'q']):
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q']
    # var_name = 'd_ln'
    # icount=0; var_name='dD'
    # icount=3; var_name='d_ln'
    # icount=4; var_name='q'
    print('#---------------- ' + str(icount) + ' ' + var_name)
    
    subset = np.isfinite(IT20_ACE_1d_sim[expid[i]][var_name]) & np.isfinite(IT20_ACE_1d_sim[expid[i]][var_name + '_sim'])
    
    lat_subset = IT20_ACE_1d_sim[expid[i]]['lat'][subset]
    lon_subset = IT20_ACE_1d_sim[expid[i]]['lon'][subset]
    var_subset = IT20_ACE_1d_sim[expid[i]][var_name][subset]
    var_sim_subset = IT20_ACE_1d_sim[expid[i]][var_name + '_sim'][subset]
    var_diff_subset = (var_sim_subset - var_subset)
    # print(stats.describe(var_subset, axis=None))
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.1 ' + expid[i] + ' maps of IT20 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = hemisphere_plot(northextent=-20)
    cplot_ice_cores(
        lon=ten_sites_loc.lon[ten_sites_loc['Site']=='EDC'],
        lat=ten_sites_loc.lat[ten_sites_loc['Site']=='EDC'],
        ax=ax, s=12, marker='*',)
    
    #---------------- plot data
    
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min = cm_mins[icount],
        cm_max = cm_maxs[icount],
        cm_interval1 = cm_interval1s[icount],
        cm_interval2 = cm_interval2s[icount],
        cmap = cmaps[icount],)
    
    if (var_name == 'q'):
        var_subset = var_subset * 1000
        var_sim_subset = var_sim_subset * 1000
        var_diff_subset = var_diff_subset * 1000
    
    edgecolors = np.repeat('darkred', len(var_diff_subset))
    edgecolors[var_diff_subset < 0] = 'gray'
    
    plt_scatter = ax.scatter(
        x=lon_subset,
        y=lat_subset,
        c=var_subset,
        s=min_size + scale_size[icount]*abs(var_diff_subset),
        edgecolors=edgecolors,
        lw=0.5, marker='o', zorder=2,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),
        )
    
    cbar = fig.colorbar(
        plt_scatter, ax=ax, aspect=30,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.2,
        )
    cbar.ax.set_xlabel('Observed ' + plot_labels[var_name])
    
    fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot differences against latitude

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'dD'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.3_IT20/8.3.0.3.2 ' + expid[i] + ' IT20 diff. in observed and simulated daily ' + var_name + ' against latitude.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6.6]) / 2.54)
    
    var_obs = IT20_ACE_1d_sim[expid[i]][var_name]
    var_sim = IT20_ACE_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(var_obs) & np.isfinite(var_sim))
    var_obs = var_obs[subset]
    var_sim = var_sim[subset]
    var_lat = IT20_ACE_1d_sim[expid[i]]['lat'][subset]
    
    if (var_name == 'q'):
        var_obs = var_obs * 1000
        var_sim = var_sim * 1000
    
    var_diff = var_sim - var_obs
    
    sns.scatterplot(x=var_lat, y=var_diff, s=12, marker='+', alpha=0.5,)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xticks(np.arange(-70, -20 + 1e-3, 10))
    ax.set_xticklabels([remove_trailing_zero(x) for x in ax.get_xticks()*(-1)])
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xlim(-70, -20)
    
    ax.set_xlabel('Latitude [$°\;S$]', labelpad=3)
    ax.set_ylabel('Sim. vs. Obs. ' + plot_labels[var_name], labelpad=3)
    
    ax.axhline(y=0, linewidth=0.8, color='gray', alpha=0.75, linestyle='-')
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.18, top=0.96)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


