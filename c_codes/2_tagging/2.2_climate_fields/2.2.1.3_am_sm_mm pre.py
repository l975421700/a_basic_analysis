

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
    month_dec_num,
    month_dec,
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
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]

# region import output

i = 0
expid[i]

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm aprt Antarctica

#-------- basic set

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt am_sm Antarctica.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in precipitation [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh2 = axs[0, 1].pcolormesh(
    lon, lat,
    (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') - 1) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat,
    (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON') - 1) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

for jcol in range(ncol):
    #-------- sm
    axs[1, jcol].pcolormesh(
        lon, lat,
        wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(
            season=seasons[jcol]) * 3600 * 24,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm/am - 1
    axs[2, jcol].pcolormesh(
        lon, lat,
        (wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season=seasons[jcol]) / wisoaprt_alltime[expid[i]]['am'][0] - 1) * 100,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    print(seasons[jcol])


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF/JJA - 1', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM/SON - 1', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

for jcol in range(ncol):
    plt.text(
        0.5, 1.05, seasons[jcol], transform=axs[1, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[jcol] + '/Annual mean - 1',
        transform=axs[2, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm aprt Antarctica


#-------- basic set

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt mm Antarctica.png'
# cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in precipitation [$\%$]'

# pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()



nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

for jcol in range(ncol):
    for irow in range(nrow):
        plt_mesh1 = axs[irow, jcol].pcolormesh(
            lon, lat, (wisoaprt_alltime[expid[i]]['mm'].sel(month=month_dec_num[jcol*3+irow])[0] / wisoaprt_alltime[expid[i]]['am'][0] - 1) * 100,
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar2 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------







