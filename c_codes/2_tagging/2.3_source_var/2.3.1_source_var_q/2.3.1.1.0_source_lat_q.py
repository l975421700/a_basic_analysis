

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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
from metpy.calc import pressure_to_height_std
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
    cplot_lon180,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

q_weighted_lat = {}

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lat.pkl',
          'rb') as f:
    q_weighted_lat[expid[i]] = pickle.load(f)

lon = q_weighted_lat[expid[i]]['am'].lon
lat = q_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)
plevs = q_weighted_lat[expid[i]]['am'].plev.sel(plev=slice(1e+5, 2e+4))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot q_weighted_lat am

ilat = 80
output_png = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am ' + str(np.round(lat[80].values, 1)) + '.png'

pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
pltticks = np.arange(-55, -20 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = cplot_lon180(
    lon, plevs / 100, ax, pltnorm, pltcmp,
    q_weighted_lat[expid[i]]['am'].sel(
        lat = lat[80].values, plev=slice(1e+5, 2e+4)),)

# x-axis
ax.set_xticks(np.arange(-180, 180 + 1e-4, 60))
ax.set_xlim(-180, 180)
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', which='both',)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticklabels(np.flip(height.magnitude), c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# which lat cross-section
plt.text(
    0.02, 0.98, str(np.negative(np.round(lat[80].values, 1))) + '$°\;S$',
    transform=ax.transAxes, weight='bold',
    ha='left', va='top', rotation='horizontal',
    bbox=dict(boxstyle='round', fc='white', ec='gray', lw=1, alpha=0.7),)

# cbar
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.25, anchor=(0.5, -0.6),)
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.set_xlabel('Humidity-weighted open-oceanic source latitude [$°\;S$]',)

fig.subplots_adjust(left=0.11, right=0.89, bottom=0.22, top=0.98)
fig.savefig(output_png)



'''
#-------------------------------- plot [-180, 180] manually
plt_data = q_weighted_lat[expid[i]]['am'].sel(
    lat = lat[80].values, plev=slice(1e+5, 2e+4))

lon_180 = np.concatenate([lon[96:] - 360, lon[:96], ])
plt_data_180 = xr.concat([plt_data.sel(lon=slice(180, 360)),
                          plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')

plt_mesh = ax.pcolormesh(
    lon_180,
    plevs / 100,
    plt_data_180,
    norm=pltnorm, cmap=pltcmp,)

#-------------------------------- plot 0-360 as check
ilat = 80
output_png = 'figures/trial.png'
plt_data = q_weighted_lat[expid[i]]['am'].sel(
    lat = lat[80].values, plev=slice(1e+5, 2e+4))

pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
pltticks = np.arange(-55, -20 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

ax.pcolormesh(
    lon,
    plevs / 100,
    plt_data,
    norm=pltnorm, cmap=pltcmp,)

ax.set_xticks(np.arange(0, 360 + 1e-4, 60))
ax.set_xlim(0, 360)

ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))

ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')

from matplotlib.ticker import AutoMinorLocator
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
        which='both',)

fig.subplots_adjust(left=0.11, right=0.89, bottom=0.22, top=0.98)
fig.savefig(output_png)


q_weighted_lat[expid[i]]['am'].sel(lat = lat[80].values).to_netcdf(
    'scratch/test/test.nc')
q_weighted_lat[expid[i]]['am'].to_netcdf('scratch/test/test.nc')

# ax.set_yticks(np.arange(100000, 20000 - 1e-4, -10000))
# ax.set_yticklabels(
#     [remove_trailing_zero(x) for x in np.arange(1000, 200 - 1e-4, -100)],
#     )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate q_weighted_lat am


#-------- settings

output_mp4 = 'figures/6_awi/6.1_echam6/6.1.5_source_var_q/6.1.5.0_lat/6.1.5.0 ' + expid[i] + ' q_weighted_lat am SH.mp4'

pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
pltticks = np.arange(-55, -20 + 1e-4, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


#-------- plot framework

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

# x-axis
ax.set_xticks(np.arange(-180, 180 + 1e-4, 60))
ax.set_xlim(-180, 180)
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))

# y-axis
ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_ylabel('Pressure [$hPa$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# grid
ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', which='both',
        zorder=2)

# 2nd y-axis
height = np.round(
    pressure_to_height_std(
        pressure=np.arange(1000, 200 - 1e-4, -100) * units('hPa')), 1,)
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 200)
ax2.set_yticklabels(np.flip(height.magnitude), c = 'gray')
ax2.set_ylabel('Height assuming a standard atmosphere [$km$]', c = 'gray')

# cbar
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    pad=0.01, fraction=0.25, anchor=(0.5, -0.6),)
cbar.ax.set_xticklabels(
    [remove_trailing_zero(x) for x in np.negative(pltticks)])
cbar.ax.set_xlabel('Humidity-weighted open-oceanic source latitude [$°\;S$]',)

fig.subplots_adjust(left=0.11, right=0.89, bottom=0.22, top=0.98)


plt_objs = []

def update_frames(ilat):
    # ilat = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_mesh = cplot_lon180(
        lon, plevs / 100, ax, pltnorm, pltcmp,
        q_weighted_lat[expid[i]]['am'].sel(
            lat = lat[ilat].values, plev=slice(1e+5, 2e+4)),)
    
    # which lat cross-section
    plt_text = plt.text(
        0.02, 0.98, str(np.negative(np.round(lat[ilat].values, 1))) + '$°\;S$',
        transform=ax.transAxes, weight='bold',
        ha='left', va='top', rotation='horizontal',
        bbox=dict(boxstyle='round', fc='white', ec='gray', lw=1, alpha=0.7),)
    
    plt_objs = [plt_mesh, plt_text]
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=np.arange(48, 96, 1), interval=1000, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


# endregion
# -----------------------------------------------------------------------------

