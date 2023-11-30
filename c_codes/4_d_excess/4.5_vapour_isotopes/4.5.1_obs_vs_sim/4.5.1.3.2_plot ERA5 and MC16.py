

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
    hemisphere_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
    zerok,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

MC16_Dome_C_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.MC16_Dome_C_1d_sim.pkl', 'rb') as f:
    MC16_Dome_C_1d_sim[expid[i]] = pickle.load(f)

with open('scratch/ERA5/temp2/MC16_Dome_C_1d_era5.pkl', 'rb') as f:
    MC16_Dome_C_1d_era5 = pickle.load(f)

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})
ERA5_daily_q_2013_2022 = xr.open_dataset('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc', chunks={'time': 720})
ERA5_daily_rh2m_2013_2022 = xr.open_dataset('scratch/ERA5/rh2m/ERA5_daily_rh2m_2013_2022.nc', chunks={'time': 720})

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate data

itime_start = np.datetime64('2014-12-25')
itime_end   = np.datetime64('2015-01-16')

north_extent = -60

for var_name in ['local_rh2m', ]:
    # var_name = 'q'
    # ['t_3m', 'q', 'local_rh2m', ]
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 't_3m'):
        plt_data = ERA5_daily_temp2_2013_2022.sel(latitude=slice(north_extent+1, -90), time=slice(itime_start, itime_end)).t2m.copy()-zerok
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-36, cm_max=0, cm_interval1=2, cm_interval2=4,
            cmap='viridis', reversed=False)
    elif (var_name == 'q'):
        plt_data = ERA5_daily_q_2013_2022.sel(latitude=slice(north_extent+1, -90), time=slice(itime_start, itime_end)).q.copy()
        pltlevel = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6,])
        pltticks = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = cm.get_cmap('viridis', len(pltlevel)-1)
    elif (var_name == 'local_rh2m'):
        plt_data = ERA5_daily_rh2m_2013_2022.sel(latitude=slice(north_extent+1, -90), time=slice(itime_start, itime_end)).rh2m.copy()
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=100, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    
    start_time = str(plt_data.time.values[0])[:10]
    end_time   = str(plt_data.time.values[-1])[:10]
    
    output_mp4 = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.8_era5/8.3.0.8.0 ERA5 vs. MC16 daily_sfc ' + var_name + ' ' + start_time + ' to ' + end_time + '.mp4'
    
    fig, ax = hemisphere_plot(northextent=north_extent, fm_top=0.92,)
    
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
        format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.15,
        )
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('Daily surface ' + plot_labels[var_name], linespacing=1.5)
    
    plt_objs = []
    
    def update_frames(itime):
        # itime = 0
        global plt_objs
        for plt_obj in plt_objs:
            plt_obj.remove()
        plt_objs = []
        
        plt_mesh = plot_t63_contourf(
            plt_data.longitude, plt_data.latitude, plt_data[itime], ax,
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
        
        plt_ocean = ax.add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        plt_txt = plt.text(
            0.5, 1, str(plt_data.time[itime].values)[:10],
            transform=ax.transAxes,
            ha='center', va='bottom', rotation='horizontal')
        
        if (var_name != 'local_rh2m'):
            scatter_values = MC16_Dome_C_1d_sim[expid[i]][var_name][itime].copy()
            if (var_name == 'q'):
                scatter_values = scatter_values * 1000
            plt_scatter = ax.scatter(
                MC16_Dome_C_1d_sim[expid[i]]['lon'][itime],
                MC16_Dome_C_1d_sim[expid[i]]['lat'][itime],
                c=scatter_values,
                marker='o', edgecolors='k', lw=0.5, s=12,
                cmap=pltcmp, norm=pltnorm, transform=ccrs.PlateCarree(),
            )
        else:
            plt_scatter = ax.scatter(
                MC16_Dome_C_1d_sim[expid[i]]['lon'][itime],
                MC16_Dome_C_1d_sim[expid[i]]['lat'][itime],
                marker='x', c='k', lw=0.5, s=12,
                transform=ccrs.PlateCarree(),
            )
        
        plt_objs = plt_mesh.collections + [plt_txt, plt_ocean, plt_scatter,]
        
        return(plt_objs)
    
    ani = animation.FuncAnimation(
        fig, update_frames, frames=len(MC16_Dome_C_1d_sim[expid[i]]['lon']),
        interval=500, blit=False)
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)




'''
ERA5_daily_temp2_2013_2022.longitude
ERA5_daily_temp2_2013_2022.latitude
ERA5_daily_temp2_2013_2022.time
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot


output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.0 MC16 vs. ERA5 daily 2m temperature.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

xdata = MC16_Dome_C_1d_era5['t_3m']
ydata = MC16_Dome_C_1d_era5['t_3m_era5'] - zerok
subset = (np.isfinite(xdata) & np.isfinite(ydata))
xdata = xdata[subset]
ydata = ydata[subset]

RMSE = np.sqrt(np.average(np.square(xdata - ydata)))

sns.scatterplot( x=xdata, y=ydata, s=12,)
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

plt.text(0.32, 0.15, eq_text, transform=ax.transAxes, fontsize=8, ha='left')

xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
xylim_min = np.min(xylim)
xylim_max = np.max(xylim)
ax.set_xlim(xylim_min, xylim_max)
ax.set_ylim(xylim_min, xylim_max)

ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Observed '  + plot_labels['t_air'], labelpad=6)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_ylabel('ERA5 ' + plot_labels['t_air'], labelpad=6)

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------



