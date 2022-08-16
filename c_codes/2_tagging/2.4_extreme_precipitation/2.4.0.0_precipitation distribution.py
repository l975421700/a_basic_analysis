

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
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_402_4.7',
    # 'pi_m_411_4.9',
    'pi_m_416_4.9',
    ]

# region import output

exp_org_o = {}

i = 0
exp_org_o[expid[i]] = {}

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso[120:], data_vars='minimal', coords='minimal', parallel=True)



'''
#-------- check pre
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:348], data_vars='minimal', coords='minimal', parallel=True)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl[-1].values - exp_org_o[expid[i]]['wiso'].wisoaprl[-31:, 0].mean(dim='time').values))


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

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann pre

i = 0
expid[i]

wisoaprt = {}


wisoaprt[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl[:, :3] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, :3].values).compute()

wisoaprt[expid[i]] = wisoaprt[expid[i]].rename('wisoaprt')

wisoaprt_alltime = {}
wisoaprt_alltime[expid[i]] = mon_sea_ann(wisoaprt[expid[i]])


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl',
          'wb') as f:
    pickle.dump(wisoaprt_alltime[expid[i]], f)


'''
#-------- check
i = 0
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:], data_vars='minimal', coords='minimal', parallel=True)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl.values + exp_org_o[expid[i]]['echam'].aprc.values - wisoaprt_alltime[expid[i]]['mon'][:, 0].values))

#-------- memory profiling

sys.getsizeof(wisoaprt[expid[i]])

import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 2**30)

import gc
gc.collect()

from pympler import asizeof
from pympler import muppy
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean precipitation over Antarctica

i = 0
expid[i]

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


#-------- basic settings

pltctr1 = np.array([0.004, 0.008, 0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

plt_data = wisoaprt_alltime[expid[i]]['am'][0].values * 3600 * 24

output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' am precipitation Antarctica.png'

#-------- plot

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 6.5]) / 2.54, lw=0.1,
    fm_bottom=0.1, fm_top=0.99)

plt2 = ax.contour(
    wisoaprt_alltime[expid[i]]['am'].lon, wisoaprt_alltime[expid[i]]['am'].lat,
    plt_data,
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)

plt3 = ax.contour(
    wisoaprt_alltime[expid[i]]['am'].lon, wisoaprt_alltime[expid[i]]['am'].lat,
    plt_data,
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

plt.text(
    0.5, -0.08, 'Annual mean precipitation [$mm \; day^{-1}$]',
    transform=ax.transAxes, ha='center', va='center', rotation='horizontal')

fig.savefig(output_png)


'''
# pltlevel = np.concatenate(
#     (np.arange(0, 0.5, 0.05), np.arange(0.5, 5 + 1e-4, 0.5)))
# pltticks = np.concatenate(
#     (np.arange(0, 0.5, 0.1), np.arange(0.5, 5 + 1e-4, 1)))
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
# pltcmp = cm.get_cmap('RdBu', len(pltlevel))

# pltctr = np.array([0.004, 0.008, 0.05, 0.1, 0.5, 1, 2, 4, 8])

# plt1 = ax.pcolormesh(
#     wisoaprt_alltime[expid[i]]['am'].lon, wisoaprt_alltime[expid[i]]['am'].lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


# cbar = fig.colorbar(
#     plt1, ax=ax, aspect=30,
#     orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
#     pad=0.02, fraction=0.15,
#     )
# cbar.ax.set_xlabel('Annual mean precipitation [$mm \; day^{-1}$]\n ', linespacing=2)

# plt_data[echam6_t63_ais_mask['mask']['ais'] == False] = np.nan
# output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' am precipitation AIS.png'


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily precipitation distribution over Antarctica


#-------- import data

i = 0
expid[i]

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')


#-------- extract data

wisoaprt_daily = wisoaprt_alltime[expid[i]]['daily'][:, 0].copy()
b_cell_area = np.broadcast_to(
    echam6_t63_cellarea.cell_area.values, wisoaprt_daily.shape)

b_mask = {}
for imask in echam6_t63_ais_mask['mask'].keys():
    b_mask[imask] = np.broadcast_to(
        echam6_t63_ais_mask['mask'][imask][None, :, :], wisoaprt_daily.shape)

wisoaprt_daily_pd = pd.DataFrame(columns=('Region', 'daily_pre', 'weights'))
wisoaprt_daily_flatten = {}
b_cell_area_flatten = {}

for imask in echam6_t63_ais_mask['mask'].keys():
    # imask = 'AIS'
    wisoaprt_daily_flatten[imask] = wisoaprt_daily.values[b_mask[imask] & (wisoaprt_daily.values >= 2e-8)]
    b_cell_area_flatten[imask] = b_cell_area[b_mask[imask] & (wisoaprt_daily.values >= 2e-8)]
    
    wisoaprt_daily_pd = pd.concat(
        [wisoaprt_daily_pd,
         pd.DataFrame(data={
             'Region': imask,
             'daily_pre': wisoaprt_daily_flatten[imask] * 3600 * 24,
             'weights': b_cell_area_flatten[imask], })],
        ignore_index=True,)
    
    print(imask)


# 6.4% of grid cells has daily precipitation

output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' daily precipitation histogram over AIS.png'

#-------- plot histogram

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

sns.kdeplot(
    data = wisoaprt_daily_pd,
    x= 'daily_pre',
    weights='weights',
    hue='Region',
    palette=['black', 'blue', 'red', 'magenta'],
    hue_order=['AIS', 'EAIS', 'WAIS', 'AP'],
    cut=0, clip=(0, 5),
    linewidth=1,
    common_norm=False,
    # bw_adjust=1,
)

ax.set_xticks(np.arange(0, 5 + 1e-4, 1))
ax.set_yticks(np.arange(0, 3 + 1e-4, 0.5))
ax.set_xlim(0, 5)

ax.set_xlabel('Daily precipitation [$mm \; day^{-1}$]')
ax.set_ylabel('Area-weighted PDF')

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.99)

fig.savefig(output_png)


'''
# stats.describe(wisoaprt_daily_flatten * 3600 * 24, )
# np.max(wisoaprt_daily.values[wisoaprt_daily.values < 2e-8])

#  & (wisoaprt_daily.values * 3600 * 24 < 0.05)
    # bins=np.arange(0, 0.05 + 1e-4, 0.002),
# ax.set_xticks(np.arange(0, 0.05 + 1e-4, 0.01))
# ax.set_xticklabels(np.arange(0, 0.05 + 1e-4, 0.01), size=8)
# ax.set_yticks(np.arange(0, 6 + 1e-4, 1))
# ax.set_yticklabels(np.arange(0, 6 + 1e-4, 1), size=8)


#-------- plot histogram

# plt_hist = plt.hist(
#     x=wisoaprt_daily_flatten * 3600 * 24,
#     weights=b_cell_area_flatten,
#     color=['lightgray', ],
#     bins=np.arange(0, 2.5 + 1e-4, 0.1),
#     density=True,
#     rwidth=1,
# )

# sns.histplot(
#     x=wisoaprt_daily_flatten * 3600 * 24,
#     weights=b_cell_area_flatten,
#     stat='density', bins=np.arange(0, 2.5 + 1e-4, 0.1), color=['lightgray', ],
#     # kde=True,
# )

b_ais_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['ais'][None, :, :], wisoaprt_daily.shape)
b_eais_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['eais'][None, :, :], wisoaprt_daily.shape)
b_wais_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['wais'][None, :, :], wisoaprt_daily.shape)
b_ap_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['ap'][None, :, :], wisoaprt_daily.shape)

# np.min(wisoaprt_daily_pd.daily_pre)
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

j = 0
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

djf_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_djf + 0,].copy() * 3600 * 24
jja_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_jja + 0,].copy() * 3600 * 24
# djf_pre.values[djf_pre.values < 2e-8 * 3600 * 24] = np.nan
# jja_pre.values[jja_pre.values < 2e-8 * 3600 * 24] = np.nan

plt1 = axs[0, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    djf_pre.sel(lat=slice(-60, -90)), 
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    jja_pre.sel(lat=slice(-60, -90)), 
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt3 = axs[0, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_djf + 0,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
axs[1, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_jja + 0,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf + 0,].values)[:10], transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja + 0,].values)[:10], transform=axs[1, 0].transAxes,
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
        djf_pre.sel(lat=slice(-60, -90)), 
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[1, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        jja_pre.sel(lat=slice(-60, -90)), 
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #---- plot daily pre_weighted_lon
    plt3 = axs[0, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_djf + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
    
    plt4 = axs[1, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_jja + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
    
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


# -----------------------------------------------------------------------------
# region plot daily pre and daily pre-weighted longitude


#-------------------------------- import pre_weighted_lon and ocean pre

i = 0
expid[i]

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

ocean_pre_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[i]] = pickle.load(f)


#-------------------------------- basic settings

itime_start_djf = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2038-12-01T23:52:30'))[0][0]
itime_end_son = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2039-12-01T23:52:30'))[0][0]

pltlevel = np.arange(0, 360 + 1e-4, 20)
pltticks = np.arange(0, 360 + 1e-4, 60)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel1 = np.arange(-90, 90 + 1e-4, 10)
pltticks1 = np.arange(-90, 90 + 1e-4, 20)
pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
pltcmp1 = cm.get_cmap('BrBG', len(pltlevel1)-1).reversed()

output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' daily precipitation and pre_weighted_lon.png'

#-------------------------------- plot
itime = 0

pltctr1 = np.array([0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

fig, ax = hemisphere_plot(northextent=-60)

plt_pre = ocean_pre_alltime[expid[i]]['daily'][itime_start_djf + itime,].copy() * 3600 * 24

plt2 = ax.contour(
    ocean_pre_alltime[expid[i]]['daily'].lon,
    ocean_pre_alltime[expid[i]]['daily'].lat.sel(lat=slice(-60, -90)),
    plt_pre.sel(lat=slice(-60, -90)),
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)


plt3 = ax.contour(
    ocean_pre_alltime[expid[i]]['daily'].lon,
    ocean_pre_alltime[expid[i]]['daily'].lat.sel(lat=slice(-60, -90)),
    plt_pre.sel(lat=slice(-60, -90)),
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

ax.pcolormesh(
    ocean_pre_alltime[expid[i]]['daily'].lon,
    ocean_pre_alltime[expid[i]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[i]]['daily'][itime_start_djf + itime,].sel(lat=slice(-60, -90)),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), )

plt.text(
    0, 0.95, str(ocean_pre_alltime[expid[i]]['daily'].time[itime_start_djf + itime,].values)[5:10], transform=ax.transAxes,
    ha='left', va='bottom', rotation='horizontal')

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.savefig(output_png)


'''
ocean_pre_alltime[expid[j]]['daily'].time[itime_start_djf].values
ocean_pre_alltime[expid[j]]['daily'].time[itimeend_son].values
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily pre and daily pre-weighted relative-longitude


#-------------------------------- import pre_weighted_lon and ocean pre

i = 0
expid[i]

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

ocean_pre_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[i]] = pickle.load(f)

lon = ocean_pre_alltime[expid[i]]['daily'].lon
lat = ocean_pre_alltime[expid[i]]['daily'].lat

lon_2d, lat_2d = np.meshgrid(lon, lat,)

#-------------------------------- basic settings

itime_start_djf = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2038-12-01T23:52:30'))[0][0]
itime_end_son = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2039-12-01T23:52:30'))[0][0]

pltlevel = np.arange(0, 360 + 1e-4, 20)
pltticks = np.arange(0, 360 + 1e-4, 60)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel1 = np.arange(-120, 120 + 1e-4, 15)
pltticks1 = np.arange(-120, 120 + 1e-4, 30)
pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
pltcmp1 = cm.get_cmap('BrBG', len(pltlevel1)-1)

output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' daily precipitation and relative pre_weighted_lon.png'

#-------------------------------- plot
itime = 1

plt_pre = ocean_pre_alltime[expid[i]]['daily'][itime_start_djf + itime,].copy() * 3600 * 24
pltctr1 = np.array([0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

fig, ax = hemisphere_plot(northextent=-60)

plt2 = ax.contour(
    lon,
    lat.sel(lat=slice(-60, -90)),
    plt_pre.sel(lat=slice(-60, -90)),
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)


plt3 = ax.contour(
    lon,
    lat.sel(lat=slice(-60, -90)),
    plt_pre.sel(lat=slice(-60, -90)),
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=True, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

plt_mesh = ax.pcolormesh(
    lon,
    lat.sel(lat=slice(-60, -90)),
    (calc_lon_diff(pre_weighted_lon[expid[i]]['daily'][itime_start_djf + itime,], lon_2d)).sel(lat=slice(-60, -90)),
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(), )

plt.text(
    0, 0.95, str(ocean_pre_alltime[expid[i]]['daily'].time[itime_start_djf + itime,].values)[5:10], transform=ax.transAxes,
    ha='left', va='bottom', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks1, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Precipitation-weighted open-oceanic\nrelative source longitude [$°$]', linespacing=1.5)
fig.savefig(output_png)


'''
#-------- check

# time indices
pre_weighted_lon[expid[i]]['daily'].time[itime_start_djf].values
ocean_pre_alltime[expid[i]]['daily'].time[itime_start_djf].values

pre_weighted_lon[expid[i]]['daily'].time[itime_end_son].values
ocean_pre_alltime[expid[i]]['daily'].time[itime_end_son].values
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate daily pre and daily pre-weighted relative-longitude


#-------------------------------- import pre_weighted_lon and ocean pre

i = 0
expid[i]

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

ocean_pre_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[i]] = pickle.load(f)

lon = ocean_pre_alltime[expid[i]]['daily'].lon
lat = ocean_pre_alltime[expid[i]]['daily'].lat

lon_2d, lat_2d = np.meshgrid(lon, lat,)

#-------------------------------- basic settings

itime_start_djf = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2038-12-01T23:52:30'))[0][0]
itime_end_son = np.where(ocean_pre_alltime[expid[i]]['daily'].time == np.datetime64('2039-12-01T23:52:30'))[0][0]

pltlevel1 = np.arange(-120, 120 + 1e-4, 15)
pltticks1 = np.arange(-120, 120 + 1e-4, 30)
pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
pltcmp1 = cm.get_cmap('BrBG', len(pltlevel1)-1)

output_mp4 = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0 ' + expid[i] + ' daily precipitation and relative pre_weighted_lon 1201_1130.mp4'

ctr_lev1 = np.array([0.5, ])
ctr_lev2 = np.array([1, 2, 4, 8, ])

#-------------------------------- plot

fig, ax = hemisphere_plot(northextent=-60)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks1, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Precipitation-weighted open-oceanic\nrelative source longitude [$°$]', linespacing=1.5)

plt_objs = []

def update_frames(itime):
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_data = ocean_pre_alltime[expid[i]]['daily'][itime_start_djf + itime,].copy() * 3600 * 24
    
    plt_ctr1 = ax.contour(
        lon,
        lat.sel(lat=slice(-60, -90)),
        plt_data.sel(lat=slice(-60, -90)),
        levels=ctr_lev1, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='dotted',
    )
    plt_ctr1.__class__ = mpl.contour.QuadContourSet
    plt_lab1 = ax.clabel(
        plt_ctr1, inline=True, colors='b', fmt=remove_trailing_zero,
        levels=ctr_lev1, inline_spacing=10, fontsize=7,)
    
    plt_ctr2 = ax.contour(
        lon,
        lat.sel(lat=slice(-60, -90)),
        plt_data.sel(lat=slice(-60, -90)),
        levels=ctr_lev2, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='solid',
    )
    plt_ctr2.__class__ = mpl.contour.QuadContourSet
    plt_lab2 = ax.clabel(
        plt_ctr2, inline=True, colors='b', fmt=remove_trailing_zero,
        levels=ctr_lev2, inline_spacing=10, fontsize=7,)
    
    plt_mesh = ax.pcolormesh(
        lon,
        lat.sel(lat=slice(-60, -90)),
        (calc_lon_diff(pre_weighted_lon[expid[i]]['daily'][itime_start_djf + itime,], lon_2d)).sel(lat=slice(-60, -90)),
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(), )
    
    plt_txt = plt.text(
        0, 0.95, str(ocean_pre_alltime[expid[i]]['daily'].time[itime_start_djf + itime,].values)[5:10], transform=ax.transAxes,
        ha='left', va='bottom', rotation='horizontal')
    
    plt_objs = plt_ctr1.collections + plt_ctr2.collections + \
        plt_lab1 + plt_lab2 + [plt_mesh, plt_txt, ]
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=366, interval=250, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)



'''
'''
# endregion
# -----------------------------------------------------------------------------



