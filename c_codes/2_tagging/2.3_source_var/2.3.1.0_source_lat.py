

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

pre_weighted_lat = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

'''
# pre_weighted_lat[expid[i]]['am'].to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA mean source lat


#-------- basic settings

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'


pltlevel = np.arange(-60, 60 + 1e-4, 10)
pltticks = np.arange(-60, 60 + 1e-4, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
pltticks2 = np.arange(-20, 20 + 1e-4, 5)
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
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
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
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
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
# region plot am/DJF/JJA/DJF-JJA mean source lat Antarctica


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-10, 10 + 1e-4, 2)
pltticks2 = np.arange(-10, 10 + 1e-4, 4)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['ann'].std(
        dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr2 = axs[1].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7)

axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr3 = axs[2].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[2].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
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
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
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
# region plot ann/DJF/JJA standard deviation of source lat


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat ann_DJF_JJA std.png'
cbar_label1 = 'Standard deviation of precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(0, 10 + 1e-4, 1)
pltticks = np.arange(0, 10 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

nrow = 1
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Annual, DJF, JJA std
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['ann'].std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs, ticks=pltticks,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(0.5, 0.8),
    )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
np.isnan(pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True)).sum()
np.isnan(pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=False)).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ann/DJF/JJA standard deviation of source lat Antarctic


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat ann_DJF_JJA std Antarctica.png'
cbar_label1 = 'Standard deviation of precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(0, 5 + 1e-4, 0.5)
pltticks = np.arange(0, 5 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)


nrow = 1
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

#-------- Annual, DJF, JJA std
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['ann'].std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(0.5, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean values

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am Antarctica.png'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

plt1 = ax.pcolormesh(
    pre_weighted_lat[expid[i]]['am'].lon,
    pre_weighted_lat[expid[i]]['am'].lat,
    pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Source latitude [$°$]\n ', linespacing=2)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm source lat


#-------- basic settings

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_sm.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'


pltlevel = np.arange(-60, 60 + 1e-4, 10)
pltticks = np.arange(-60, 60 + 1e-4, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
pltticks2 = np.arange(-20, 20 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 3
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol == 0)):
            axs[irow, jcol] = globe_plot(
                ax_org = axs[irow, jcol], add_grid_labels=False)
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0, 0].contour(
    lon, lat,
    pre_weighted_lat[expid[i]]['ann'].std(
        dim='time', skipna=True),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- sm
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr2 = axs[1, 0].contour(
    lon, lat,
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 0].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr3 = axs[1, 1].contour(
    lon, lat,
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 5)
        ).std(dim='time', skipna=True),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 1].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr4 = axs[1, 2].contour(
    lon, lat,
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 2].clabel(plt_ctr4, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr5 = axs[1, 3].contour(
    lon, lat,
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 11)
        ).std(dim='time', skipna=True),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 3].clabel(plt_ctr5, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- am - sm
plt_mesh2 = axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'MAM', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'SON', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Annual mean - DJF', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Annual mean - MAM', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Annual mean - JJA', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Annual mean - SON', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm source lat Antarctica


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_sm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

ctr_level = np.array([1, 2, 3, 4, 5, ])

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
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0, 0].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['ann'].std(
        dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0, 0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

#-------- DJF - JJA
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
#-------- MAM -SON
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM') - pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- sm
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr2 = axs[1, 0].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 0].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr3 = axs[1, 1].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 5)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 1].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr4 = axs[1, 2].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 2].clabel(plt_ctr4, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1, 3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr5 = axs[1, 3].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 11)
        ).std(dim='time', skipna=True).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1, 3].clabel(plt_ctr5, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- differences
plt_mesh2 = axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'] - pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Annual mean - DJF', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean - MAM', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean - JJA', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean - SON', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source lat Antarctica


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat mm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


ctr_level = np.array([1, 2, 3, 4, 5, ])

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
            lon, lat, pre_weighted_lat[expid[i]]['mm'].sel(
                month=month_dec_num[jcol*3+irow]),
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        plt_ctr1 = axs[irow, jcol].contour(
            lon, lat.sel(lat=slice(-60, -90)),
            pre_weighted_lat[expid[i]]['mon'].sel(time=(pre_weighted_lat[expid[
                i]]['mon'].time.dt.month == month_dec_num[jcol*3+irow])).std(
            dim='time', skipna=True).sel(lat=slice(-60, -90)),
            levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
            linewidths=0.5, linestyles='solid',
        )
        axs[irow, jcol].clabel(
            plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
            levels=ctr_level, inline_spacing=10, fontsize=7,)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------

