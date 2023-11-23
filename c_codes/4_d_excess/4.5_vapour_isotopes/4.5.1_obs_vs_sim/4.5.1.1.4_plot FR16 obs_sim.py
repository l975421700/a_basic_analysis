

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
    remove_trailing_zero,
    remove_trailing_zero_pos,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_gridvalue_at_site_time,
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

with open('data_sources/water_isotopes/FR16/FR16_Kohnen.pkl', 'rb') as f:
    FR16_Kohnen = pickle.load(f)

FR16_Kohnen_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.FR16_Kohnen_1d_sim.pkl', 'rb') as f:
    FR16_Kohnen_1d_sim[expid[i]] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)
FR16_1d_oo2q = find_multi_gridvalue_at_site_time(
    FR16_Kohnen_1d_sim[expid[i]]['time'],
    FR16_Kohnen_1d_sim[expid[i]]['lat'],
    FR16_Kohnen_1d_sim[expid[i]]['lon'],
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].time.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].sel(geo_regions='Open Ocean').values,
    )

'''
echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)
FR16_1d_height = find_gridvalue_at_site(
    FR16_Kohnen_1d_sim[expid[i]]['lat'].values[0],
    FR16_Kohnen_1d_sim[expid[i]]['lon'].values[0],
    echam6_t63_surface_height.lat.values,
    echam6_t63_surface_height.lon.values,
    echam6_t63_surface_height.values,
)
print('Height of Kohnen in T63 ECHAM6: ' + str(np.round(FR16_1d_height, 1)))


#---------------- check correlation
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 'temp2']:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    subset = np.isfinite(FR16_Kohnen_1d_sim[expid[i]][var_name]) & np.isfinite(FR16_Kohnen_1d_sim[expid[i]][var_name + '_sim'])
    
    print(np.round(pearsonr(FR16_Kohnen_1d_sim[expid[i]][var_name][subset], FR16_Kohnen_1d_sim[expid[i]][var_name + '_sim'][subset], ).statistic ** 2, 3))



print(stats.describe(FR16_1d_oo2q))
# 62.9%
print(find_gridvalue_at_site(
    FR16_Kohnen_1d_sim[expid[i]]['lat'][0],
    FR16_Kohnen_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='Open Ocean').values,
    ))
# 75.2%
print(find_gridvalue_at_site(
    FR16_Kohnen_1d_sim[expid[i]]['lat'][0],
    FR16_Kohnen_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='AIS').values,
    ))
# 13.8%
print(find_gridvalue_at_site(
    FR16_Kohnen_1d_sim[expid[i]]['lat'][0],
    FR16_Kohnen_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='Land excl. AIS').values,
    ))
# 6.7%
print(find_gridvalue_at_site(
    FR16_Kohnen_1d_sim[expid[i]]['lat'][0],
    FR16_Kohnen_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='SH seaice').values,
    ))
# 4.3%


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 'temp2']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.4_FR16/8.3.0.4.0 ' + expid[i] + ' FR16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = FR16_Kohnen_1d_sim[expid[i]][var_name]
    ydata = FR16_Kohnen_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata,
        s=12,
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
# region plot time series

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'temp2', 'q']:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.4_FR16/8.3.0.4.0 ' + expid[i] + ' FR16 time series of observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 6.6]) / 2.54)
    
    xdata = FR16_Kohnen_1d_sim[expid[i]]['time'].values.copy()
    ydata = FR16_Kohnen_1d_sim[expid[i]][var_name].values.copy()
    ydata_sim = FR16_Kohnen_1d_sim[expid[i]][var_name + '_sim'].values.copy()
    
    if (var_name == 'q'):
        ydata = ydata * 1000
        ydata_sim = ydata_sim * 1000
    
    ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5, label='Simulation',)
    ax.plot(xdata, ydata, 'o', ls='-', ms=2, lw=0.5, label='Observation',)
    ax.set_xticks(xdata[::4])
    
    # plot original observations
    if (var_name == 'dD'):
        xdata = FR16_Kohnen['isotopes']['time'].values.copy()
        ydata = FR16_Kohnen['isotopes']['dD'].values.copy()
    elif (var_name == 'd18O'):
        xdata = FR16_Kohnen['isotopes']['time'].values.copy()
        ydata = FR16_Kohnen['isotopes']['d18O'].values.copy()
    elif (var_name == 'd_xs'):
        xdata = FR16_Kohnen['isotopes']['time'].values.copy()
        ydata = FR16_Kohnen['isotopes']['dD'].values.copy()-8*FR16_Kohnen['isotopes']['d18O'].values.copy()
    elif (var_name == 'd_ln'):
        xdata = FR16_Kohnen['isotopes']['time'].values.copy()
        ln_dD = 1000 * np.log(1 + FR16_Kohnen['isotopes']['dD'].values.copy() / 1000)
        ln_d18O = 1000 * np.log(1 + FR16_Kohnen['isotopes']['d18O'].values.copy() / 1000)
        ydata = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)
    elif (var_name == 'q'):
        xdata = FR16_Kohnen['isotopes']['time'].values.copy()
        ydata = FR16_Kohnen['isotopes']['humidity'].values.copy() * 18.01528 / (28.9645 * 1e6) * 1000
    elif (var_name == 'temp2'):
        xdata = FR16_Kohnen['T']['time'].values.copy()
        ydata = FR16_Kohnen['T']['temp2'].values.copy()
    ax.plot(xdata, ydata, ls='-', lw=0.2, label='Hourly Observation',)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Date', labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[var_name], labelpad=6)
    plt.xticks(rotation=30, ha='right')
    
    ax.legend(handlelength=1, loc='upper right', framealpha=0.25,)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.3, top=0.98)
    fig.savefig(output_png)



# MC16_Dome_C_1d_sim[expid[i]]['time']

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot station location

fig, ax = hemisphere_plot(northextent=-60)

cplot_ice_cores(FR16_Kohnen_1d_sim[expid[i]]['lon'][0], FR16_Kohnen_1d_sim[expid[i]]['lat'][0], ax=ax, s=50, marker='*', edgecolors='b')

fig.savefig('figures/test/trial.png')

from metpy.calc import geopotential_to_height
from metpy.units import units
echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

echam6_t63_surface_height.sel(lon = FR16_Kohnen_1d_sim[expid[i]]['lon'][0], lat = FR16_Kohnen_1d_sim[expid[i]]['lat'][0], method='nearest').values

# endregion
# -----------------------------------------------------------------------------


