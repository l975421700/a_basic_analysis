

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]
i = 0

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os

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
import pickle

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

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_lon = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

pre_weighted_sinlon = {}
pre_weighted_coslon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon.pkl', 'rb') as f:
    pre_weighted_sinlon[expid[i]] = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon.pkl', 'rb') as f:
    pre_weighted_coslon[expid[i]] = pickle.load(f)

'''
# pre_weighted_lon[expid[i]]['am'].to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lon


#-------- basic set

lon = pre_weighted_lon[expid[i]]['am'].lon
lat = pre_weighted_lon[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'


pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-30, 30 + 1e-4, 5)
pltticks2 = np.arange(-30, 30 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'), pre_weighted_lon[expid[i]]['sm'].sel(season='JJA')),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lon Antarctic


#-------- basic set

lon = pre_weighted_lon[expid[i]]['am'].lon
lat = pre_weighted_lon[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-30, 30 + 1e-4, 5)
pltticks2 = np.arange(-30, 30 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-45, ax_org = axs[jcol])

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'), pre_weighted_lon[expid[i]]['sm'].sel(season='JJA')),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean values

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon am Antarctica.png'

pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 45)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-45, figsize=np.array([5.8, 7]) / 2.54)

plt1 = ax.pcolormesh(
    pre_weighted_lon[expid[i]]['am'].lon,
    pre_weighted_lon[expid[i]]['am'].lat,
    pre_weighted_lon[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Source longitude [$°$]\n ', linespacing=2)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

import xarray as xr
import pickle
data1 = {}
data1['data1_0'] = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/restart/echam/restart_pi_m_416_4.9_accw_20001231.nc')
data1['data1_1'] = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/restart/echam/restart_pi_m_416_4.9_accw_20011231.nc')

with open('scratch/test/test.pkl', 'wb') as f:
    pickle.dump(data1, f)


import pickle
import xarray as xr
with open('scratch/test/test.pkl', 'rb') as f:
    data1 = pickle.load(f)


# -----------------------------------------------------------------------------


