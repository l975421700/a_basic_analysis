

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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_402_4.7',
    'pi_m_411_4.9',
    ]

# region import output

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso_daily'] = xr.open_mfdataset(filenames_wiso_daily, data_vars='minimal', coords='minimal', parallel=True)
        
        filenames_echam_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_echam.nc'))
        exp_org_o[expid[i]]['echam_daily'] = xr.open_mfdataset(filenames_echam_daily[120:], data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/snn pre

i = 0
expid[i]

tot_pre = {}

tot_pre[expid[i]] = (
    exp_org_o[expid[i]]['echam_daily'].aprl + \
    exp_org_o[expid[i]]['echam_daily'].aprc.values).compute()
tot_pre[expid[i]] = tot_pre[expid[i]].rename('tot_pre')
tot_pre[expid[i]].values[tot_pre[expid[i]].values < 2e-8] = 0

tot_pre_alltime = {}
tot_pre_alltime[expid[i]] = mon_sea_ann(tot_pre[expid[i]])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tot_pre_alltime.pkl',
          'wb') as f:
    pickle.dump(tot_pre_alltime[expid[i]], f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check annual mean precipitation distribution over Antarctica

i = 0
expid[i]

tot_pre_alltime = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tot_pre_alltime.pkl', 'rb') as f:
    tot_pre_alltime[expid[i]] = pickle.load(f)


with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


pltlevel = np.concatenate(
    (np.arange(0, 0.5, 0.05), np.arange(0.5, 5 + 1e-4, 0.5)))
pltticks = np.concatenate(
    (np.arange(0, 0.5, 0.1), np.arange(0.5, 5 + 1e-4, 1)))
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel))

test = tot_pre_alltime[expid[i]]['am'].values * 3600 * 24
test[echam6_t63_ais_mask['mask']['ais'] == False] = np.nan

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

plt1 = ax.pcolormesh(
    tot_pre_alltime[expid[i]]['am'].lon, tot_pre_alltime[expid[i]]['am'].lat,
    # tot_pre_alltime[expid[i]]['am'] * 3600 * 24,
    test,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Annual mean precipitation [$mm \; day^{-1}$]\n ', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0.0.0_Antarctic annual mean precipitation ' + expid[i] + '.png')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check daily precipitation distribution over Antarctica

i = 0
expid[i]

tot_pre_alltime = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tot_pre_alltime.pkl', 'rb') as f:
    tot_pre_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

tot_pre_daily = tot_pre_alltime[expid[i]]['daily'].copy()
b_ais_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['ais'][None, :, :], tot_pre_daily.shape)
b_cell_area = np.broadcast_to(
    echam6_t63_cellarea.cell_area.values, tot_pre_daily.shape)

# 6% of grid cells has daily precipitation
tot_pre_daily_flatten = tot_pre_daily.values[b_ais_mask & (tot_pre_daily.values >= 2e-8) & (tot_pre_daily.values * 3600 * 24 < 0.05)]
b_cell_area_flatten = b_cell_area[b_ais_mask & (tot_pre_daily.values >= 2e-8) & (tot_pre_daily.values * 3600 * 24 < 0.05)]


#-------- plot histogram

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

plt_hist = plt.hist(
    x=tot_pre_daily_flatten * 3600 * 24,
    weights=b_cell_area_flatten,
    color=['lightgray', ],
    # bins=np.arange(0, 2.5 + 1e-4, 0.1),
    bins=np.arange(0, 0.05 + 1e-4, 0.002),
    density=True,
    rwidth=1,
)

# ax.set_xticks(np.arange(0, 2.5 + 1e-4, 0.5))
# ax.set_xticklabels(np.arange(0, 2.5 + 1e-4, 0.5), size=8)
ax.set_xticks(np.arange(0, 0.05 + 1e-4, 0.01))
ax.set_xticklabels(np.arange(0, 0.05 + 1e-4, 0.01), size=8)
ax.set_xlabel('Precipitation over AIS [$mm \; day^{-1}$]', size=10)

# ax.set_yticks(np.arange(0, 6 + 1e-4, 1))
# ax.set_yticklabels(np.arange(0, 6 + 1e-4, 1), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.97)

fig.savefig('figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0.0.1_Antarctic daily precipitation distribution ' + expid[i] + '.png')

stats.describe(tot_pre_daily_flatten * 3600 * 24, )

'''
# np.max(tot_pre_daily.values[tot_pre_daily.values < 2e-8])
'''
# endregion
# -----------------------------------------------------------------------------


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
# pltcmp = cm.get_cmap('RdBu', len(pltlevel))
pltcmp = cm.get_cmap('PuOr', len(pltlevel))

pltlevel2 = np.arange(0, 360 + 1e-4, 20)
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
# djf_pre.values[djf_pre.values < 2e-8 * 3600 * 24] = np.nan
# jja_pre.values[jja_pre.values < 2e-8 * 3600 * 24] = np.nan

plt1 = axs[0, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    djf_pre.sel(lat=slice(-60, -90)), rasterized=True,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    jja_pre.sel(lat=slice(-60, -90)), rasterized=True,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt3 = axs[0, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_djf + 89,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized=True,)
axs[1, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_jja + 89,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized=True,)

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
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)
fig.savefig('figures/test1.png')


#-------------------------------- animate with animation.FuncAnimation

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

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    #---- daily precipitation
    
    djf_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_djf + itime,].copy() * 3600 * 24
    jja_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_jja + itime,].copy() * 3600 * 24
    # djf_pre.values[djf_pre.values < 2e-8 * 3600 * 24] = np.nan
    # jja_pre.values[jja_pre.values < 2e-8 * 3600 * 24] = np.nan
    
    #---- plot daily precipitation
    plt1 = axs[0, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        djf_pre.sel(lat=slice(-60, -90)), rasterized=True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[1, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        jja_pre.sel(lat=slice(-60, -90)), rasterized=True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #---- plot daily pre_weighted_lon
    plt3 = axs[0, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_djf + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized=True,)
    
    plt4 = axs[1, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_jja + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized=True,)
    
    plt5 = axs[0, 0].text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_djf + itime,].values)[:10],
        transform=axs[0, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt6 = axs[1, 0].text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_jja + itime,].values)[:10],
        transform=axs[1, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6]
    # plt_objs = [plt6]
    
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames,
    frames=90, interval=250, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0_Antarctic DJF_JJA daily precipitation and pre_weighted_lon ' + expid[j] + '.mp4',
    # 'figures/test.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)




'''
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf].values
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja].values

#-------------------------------- animate with animation.ArtistAnimation

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

for itime in range(2):
    # itime = 0
    
    #---- daily precipitation
    
    djf_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_djf + itime,].copy() * 3600 * 24
    jja_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_jja + itime,].copy() * 3600 * 24
    
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
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0_Antarctic DJF_JJA daily precipitation and pre_weighted_lon ' + expid[j] + '_1.mp4',
    # 'figures/test.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)

(djf_pre.values < 2e-8).sum()
(djf_pre.values < 2e-8 * 3600 * 24).sum()
(ocean_pre_alltime[expid[j]]['daily'].values < 2e-8).sum()
(ocean_pre_alltime[expid[j]]['daily'].values < 2e-8 * 3600 * 24).sum()

'''
# endregion
# -----------------------------------------------------------------------------


