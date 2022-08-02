

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


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Times New Roman', size=9)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
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
)

# endregion
# -----------------------------------------------------------------------------


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_402_4.7',
    'pi_m_411_4.9',
    ]

# -----------------------------------------------------------------------------
# region animate daily pre and daily pre-weighted longitude


#-------------------------------- import total pre

# i = 0
# expid[i]

# tot_pre_alltime = {}

# with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tot_pre_alltime.pkl', 'rb') as f:
#     tot_pre_alltime[expid[i]] = pickle.load(f)


#-------------------------------- import pre_weighted_lon and ocean pre

j = 1
expid[j]

pre_weighted_lon = {}

with open(exp_odir + expid[j] + '/analysis/echam/' + expid[j] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[j]] = pickle.load(f)

ocean_pre_alltime = {}
with open(exp_odir + expid[j] + '/analysis/echam/' + expid[j] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[j]] = pickle.load(f)


#-------------------------------- basic settings

itimestart_djf = np.where(ocean_pre_alltime[expid[j]]['daily'].time == np.datetime64('2025-12-01T23:52:30'))[0][0]
itimestart_jja = np.where(ocean_pre_alltime[expid[j]]['daily'].time == np.datetime64('2026-06-01T23:52:30'))[0][0]


pltlevel = np.concatenate(
    (np.arange(0, 0.5, 0.05), np.arange(0.5, 5 + 1e-4, 0.5)))
pltticks = np.concatenate(
    (np.arange(0, 0.5, 0.1), np.arange(0.5, 5 + 1e-4, 1)))
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel))


pltlevel2 = np.arange(0, 360 + 1e-4, 15)
pltticks2 = np.arange(0, 360 + 1e-4, 60)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


#-------------------------------- plot

nrow = 2
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

djf_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_djf + 89,].copy() * 3600 * 24
jja_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_jja + 89,].copy() * 3600 * 24
djf_pre.values[djf_pre.values < 2e-8] = np.nan
jja_pre.values[jja_pre.values < 2e-8] = np.nan

plt1 = axs[0, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat,
    djf_pre,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat,
    jja_pre,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt2 = axs[0, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat,
    pre_weighted_lon[expid[j]]['daily'][itimestart_djf + 89,],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat,
    pre_weighted_lon[expid[j]]['daily'][itimestart_jja + 89,],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf + 89,].values)[:10], transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja + 89,].values)[:10], transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'DJF',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'JJA',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.6,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.6,aspect=40,extend='neither',
    anchor=(1.3,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)
fig.savefig('figures/test.png')


#-------------------------------- animate

nrow = 2
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

plt.text(
    -0.05, 0.5, 'DJF',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'JJA',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')


ims = []

for itime in range(90):
    # itime = 0
    
    #---- daily precipitation
    
    djf_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_djf + itime,].copy() * 3600 * 24
    jja_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_jja + itime,].copy() * 3600 * 24
    djf_pre.values[djf_pre.values < 2e-8] = np.nan
    jja_pre.values[jja_pre.values < 2e-8] = np.nan
    
    #---- plot daily precipitation
    plt1 = axs[0, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        djf_pre, rasterized = True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[1, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        jja_pre, rasterized = True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #---- plot daily pre_weighted_lon
    plt3 = axs[0, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        pre_weighted_lon[expid[j]]['daily'][itimestart_djf + itime,],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized = True,)
    
    plt4 = axs[1, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        pre_weighted_lon[expid[j]]['daily'][itimestart_jja + itime,],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized = True,)
    
    plt5 = plt.text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_djf + itime,].values)[:10],
        transform=axs[0, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt6 = plt.text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_jja + itime,].values)[:10],
        transform=axs[1, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    ims.append([plt1, plt2, plt3, plt4, plt5, plt6, ])
    print(str(itime))


cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.6,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.6,aspect=40,extend='neither',
    anchor=(1.3,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0_Antarctic DJF_JJA daily precipitation and pre_weighted_lon ' + expid[j] + '.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


'''
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf].values
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja].values
'''
# endregion
# -----------------------------------------------------------------------------
