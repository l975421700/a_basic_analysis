

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
    # 'nudged_706_6.0_k52_88',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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
    hemisphere_plot,
    globe_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_multi_gridvalue_at_site,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
    expid_colours,
    expid_labels,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


SO_vapor_isotopes_SLMSIC = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl', 'rb') as f:
    SO_vapor_isotopes_SLMSIC[expid[i]] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
SO_vapor_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes.pkl', 'rb') as f:
    SO_vapor_isotopes[expid[i]] = pickle.load(f)

T63GR15_jan_surf = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/input/echam/unit.24')
SO_vapor_SLM = find_multi_gridvalue_at_site(
    SO_vapor_isotopes[expid[i]]['lat'].values,
    SO_vapor_isotopes[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot subset data: SLM=0, SIC=0, lat:[-60, -20]

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))

# (pd.DatetimeIndex(pd.to_datetime(SO_vapor_isotopes_SLMSIC[expid[i]][subset]['time'], utc=True)).year >= 2019).sum()

# SO_vapor_isotopes_SLMSIC[expid[i]][subset]['time']


for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', ]:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.1 ' + expid[i] + ' SO_cruise observed vs. simulated daily ' + var_name + ' SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][var_name].copy()
    ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][var_name + '_sim'].copy()
    subset1 = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset1]
    ydata = ydata[subset1]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    cdata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['Reference'].copy()
    cdata = cdata[subset1]
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata, hue=cdata, style=cdata,
        s=12, palette='magma',
        # marker="o",
    )
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline(
        (0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.65, 0.05, eq_text,
        transform=ax.transAxes, fontsize=10, ha='left', va='bottom',
        linespacing=2,)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    if (var_name == 'q'):
        ax.legend(title='Data sources', handletextpad=0.4)
    else:
        ax.legend().set_visible(False)
    
    if (var_name == 'q'):
        ax.set_xticks(np.arange(2, 18+1e-4, 2))
        ax.set_yticks(np.arange(2, 18+1e-4, 2))
    elif (var_name == 'd18O'):
        ax.set_xticks(np.arange(-22, -10+1e-4, 2))
        ax.set_yticks(np.arange(-22, -10+1e-4, 2))
    
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



'''

for idataset in ['Kurita et al. (2016)', 'Bonne et al. (2019)', 'Thurnherr et al. (2020)']:
    print('#-------------------------------- ' + idataset)
    
    subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes_SLMSIC[expid[i]]['Reference'] == idataset))
    
    print('d18O')
    print(np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d18O']).sum())
    print('dD')
    print(np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]][subset]['dD']).sum())
    print('d18O_sim')
    print(np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]][subset]['d18O_sim']).sum())
    print('dD_sim')
    print(np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]][subset]['dD_sim']).sum())

# Kurita et al. (2016)
# Bonne et al. (2019)
# Thurnherr et al. (2020)

    # ax.legend(title=plot_labels['q'])
    # cdata = SO_vapor_isotopes_SLMSIC[expid[i]][subset]['q'].copy() * 1000

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot maps of subset data

subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60) & np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['d18O']) & np.isfinite(SO_vapor_isotopes_SLMSIC[expid[i]]['dD']))

output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.2 SO_cruise daily observation locations SLM0 SIC0 lat_20_60.png'

fig, ax = hemisphere_plot(northextent=-20, fm_bottom=0.22,)

sns.scatterplot(
        x=SO_vapor_isotopes_SLMSIC[expid[i]][subset]['lon'],
        y=SO_vapor_isotopes_SLMSIC[expid[i]][subset]['lat'],
        hue=SO_vapor_isotopes_SLMSIC[expid[i]][subset]['Reference'],
        style=SO_vapor_isotopes_SLMSIC[expid[i]][subset]['Reference'],
        s=12, palette='magma', transform=ccrs.PlateCarree(),
        # linewidth=1,
    )
ax.legend(
    # labels=[
    #     'Kurita et al. (2016)',
    #     'Thurnherr et al. (2020)',
    #     'Bonne et al. (2019)',
    # ],
    title='Data sources', loc=(0.18, -0.32), fontsize=8)

cplot_ice_cores(123.35, -75.10, ax)
cplot_ice_cores(0.04, -75, ax)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot multiple models' subset data: SLM=0, SIC=0, lat:[-60, -20]

SO_vapor_isotopes_SLMSIC = {}
for i in range(len(expid)):
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl', 'rb') as f:
        SO_vapor_isotopes_SLMSIC[expid[i]] = pickle.load(f)

i = 0
subset = ((SO_vapor_isotopes_SLMSIC[expid[i]]['SLM'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['SIC'] == 0) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] <= -20) & (SO_vapor_isotopes_SLMSIC[expid[i]]['lat'] >= -60))


for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', ]:
    # var_name = 'q'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.7_SO_cruise/8.3.0.7.2 nudged_705_11 SO_cruise observed vs. simulated daily ' + var_name + ' SLM0 SIC0 lat_20_60.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][var_name].copy()
        ydata = SO_vapor_isotopes_SLMSIC[expid[i]][subset][var_name + '_sim'].copy()
        subset1 = (np.isfinite(xdata) & np.isfinite(ydata))
        xdata = xdata[subset1]
        ydata = ydata[subset1]
        
        if (var_name == 'q'):
            xdata = xdata * 1000
            ydata = ydata * 1000
        
        RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.4, alpha=0.5,
            facecolors='white',
            edgecolors=expid_colours[expid[i]],
            )
        
        linearfit = linregress(x = xdata, y = ydata,)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=1, alpha=0.5,
            color=expid_colours[expid[i]],
            )
        
        if (linearfit.intercept >= 0):
            eq_text = expid_labels[expid[i]] + ':       $y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        if (linearfit.intercept < 0):
            eq_text = expid_labels[expid[i]] + ':       $y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        
        plt.text(
            0.05, 0.95 - 0.06*i, eq_text,
            transform=ax.transAxes, fontsize=6,
            color=expid_colours[expid[i]],
            ha='left')
    
    
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
    
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


