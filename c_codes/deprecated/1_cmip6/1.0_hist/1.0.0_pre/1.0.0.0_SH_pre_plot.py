

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month_days,
    month,
    seasons,
)
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import hist am pre

hist_pre_dir = '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/pre/'
hist_pre_ds = ['HadGEM3-GC3.1-LL', 'AWI-ESM-1-1-LR', 'ERA5',
                     'MERRA2', 'JRA-55']

hist_am_pre = {}
hist_am_pre['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_pre_dir + \
    'pr_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_am.nc')
hist_am_pre['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_pre_dir + \
    'pr_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_am.nc')
hist_am_pre['ERA5'] = xr.open_dataset(hist_pre_dir + \
    'tp_ERA5_mon_sl_197901_201412_rg1_am.nc')
hist_am_pre['MERRA2'] = xr.open_dataset(hist_pre_dir + \
    'pre_MERRA2_198001_201412_rg1_am.nc')
hist_am_pre['JRA-55'] = xr.open_dataset(hist_pre_dir + \
    'pre_JRA-55_197901_201412_rg1_am.nc')




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist am pre

nrow = 3
ncol = 4

sh_pre_level = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
sh_pre_ticks = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01,300)))
sh_pre_norm = BoundaryNorm(sh_pre_level, ncolors=len(sh_pre_level))
sh_pre_cmp = cm.get_cmap('RdBu', len(sh_pre_level))

pre_rdif_level = np.arange(0, 2.01, 0.01)
pre_rdif_ticks = np.arange(0, 2.01, 0.2)
pre_rdif_norm = BoundaryNorm(pre_rdif_level, ncolors=len(pre_rdif_level))
pre_rdif_cmp = cm.get_cmap('BrBG', len(pre_rdif_level))


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5*ncol, 5.5*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.001, 'wspace': 0.08},)

for i in range(nrow):
    for j in range(ncol):
        if((i!=0) | (j !=0)):
            axs[i, j] = hemisphere_plot(northextent=-60, ax_org = axs[i, j])
        else:
            axs[i, j].axis('off')


axs[0, 1].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    hist_am_pre['ERA5'].tp.squeeze() *1000*365,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
axs[0, 1].text(
    0.5, -0.05, 'ERA5 1979-2014', transform=axs[0, 1].transAxes,
    ha='center', va='top')

axs[0, 2].pcolormesh(
    hist_am_pre['MERRA2'].lon, hist_am_pre['MERRA2'].lat,
    hist_am_pre['MERRA2'].PRECTOTCORR.squeeze() *86400*365,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
axs[0, 2].text(
    0.5, -0.05, 'MERRA2 1980-2014', transform=axs[0, 2].transAxes,
    ha='center', va='top')

axs[0, 3].pcolormesh(
    hist_am_pre['JRA-55'].lon, hist_am_pre['JRA-55'].lat,
    hist_am_pre['JRA-55'].var61.squeeze() *365,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
axs[0, 3].text(
    0.5, -0.05, 'JRA-55 1979-2014', transform=axs[0, 3].transAxes,
    ha='center', va='top')

axs[1, 0].pcolormesh(
    hist_am_pre['HadGEM3-GC3.1-LL'].lon, hist_am_pre['HadGEM3-GC3.1-LL'].lat,
    hist_am_pre['HadGEM3-GC3.1-LL'].pr.squeeze() *86400*365,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
axs[1, 0].text(
    0.5, -0.05, 'HadGEM3-GC3.1-LL 1979-2014', transform=axs[1, 0].transAxes,
    ha='center', va='top')

axs[2, 0].pcolormesh(
    hist_am_pre['AWI-ESM-1-1-LR'].lon, hist_am_pre['AWI-ESM-1-1-LR'].lat,
    hist_am_pre['AWI-ESM-1-1-LR'].pr.squeeze() *86400*365,
    norm=sh_pre_norm, cmap=sh_pre_cmp, transform=ccrs.PlateCarree(),)
axs[2, 0].text(
    0.5, -0.05, 'AWI-ESM-1-1-LR 1979-2014', transform=axs[2, 0].transAxes,
    ha='center', va='top')

# relative difference
axs[1, 1].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['HadGEM3-GC3.1-LL'].pr.squeeze() *86400*365) / (hist_am_pre['ERA5'].tp.squeeze() *1000*365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[1, 1].text(
    0.5, -0.05, 'HadGEM3-GC3.1-LL / ERA5', transform=axs[1, 1].transAxes,
    ha='center', va='top')

axs[1, 2].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['HadGEM3-GC3.1-LL'].pr.squeeze() *86400*365) / (hist_am_pre['MERRA2'].PRECTOTCORR.squeeze() *86400*365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[1, 2].text(
    0.5, -0.05, 'HadGEM3-GC3.1-LL / MERRA2', transform=axs[1, 2].transAxes,
    ha='center', va='top')

axs[1, 3].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['HadGEM3-GC3.1-LL'].pr.squeeze() * 86400*365) /
    (hist_am_pre['JRA-55'].var61.squeeze() * 365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[1, 3].text(
    0.5, -0.05, 'HadGEM3-GC3.1-LL / JRA-55', transform=axs[1, 3].transAxes,
    ha='center', va='top')

axs[2, 1].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['AWI-ESM-1-1-LR'].pr.squeeze() * 86400*365) /
    (hist_am_pre['ERA5'].tp.squeeze() * 1000*365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[2, 1].text(
    0.5, -0.05, 'AWI-ESM-1-1-LR / ERA5', transform=axs[2, 1].transAxes,
    ha='center', va='top')

axs[2, 2].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['AWI-ESM-1-1-LR'].pr.squeeze() * 86400*365) /
    (hist_am_pre['MERRA2'].PRECTOTCORR.squeeze() * 86400*365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[2, 2].text(
    0.5, -0.05, 'AWI-ESM-1-1-LR / MERRA2', transform=axs[2, 2].transAxes,
    ha='center', va='top')

axs[2, 3].pcolormesh(
    hist_am_pre['ERA5'].lon, hist_am_pre['ERA5'].lat,
    (hist_am_pre['AWI-ESM-1-1-LR'].pr.squeeze() * 86400*365) /
    (hist_am_pre['JRA-55'].var61.squeeze() * 365),
    norm=pre_rdif_norm, cmap=pre_rdif_cmp, transform=ccrs.PlateCarree(),)
axs[2, 3].text(
    0.5, -0.05, 'AWI-ESM-1-1-LR / JRA-55', transform=axs[2, 3].transAxes,
    ha='center', va='top')

# create color bar for pre
plt_pre_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_pre_norm, cmap=sh_pre_cmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.4), ticks=sh_pre_ticks)
plt_pre_cbar.ax.set_xlabel('Annual mean precipitation [$mm\;yr^{-1}$]')

# create color bar for pre_diff
plt_pre_diff_cbar = fig.colorbar(
    cm.ScalarMappable(norm=pre_rdif_norm, cmap=pre_rdif_cmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(1.1,-3.8),ticks=pre_rdif_ticks)
plt_pre_diff_cbar.ax.set_xlabel('Annual mean precipitation difference [-]')

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.15, top = 0.99)
# fig.savefig('figures/0_test/trial.png', dpi = 300)
fig.savefig('figures/2_cmip6/2.0_hist/2.0.0_pre/2.0.0.0_sh_pre_comparison.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import hist sm pre

hist_pre_dir = '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/pre/'

hist_sm_pre = {}
hist_sm_pre['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_pre_dir + \
    'pr_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_sm.nc')
hist_sm_pre['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_pre_dir + \
    'pr_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_sm.nc')
hist_sm_pre['ERA5'] = xr.open_dataset(hist_pre_dir + \
    'tp_ERA5_mon_sl_197901_201412_rg1_sm.nc')
hist_sm_pre['MERRA2'] = xr.open_dataset(hist_pre_dir + \
    'pre_MERRA2_198001_201412_rg1_sm.nc')
hist_sm_pre['JRA-55'] = xr.open_dataset(hist_pre_dir + \
    'pre_JRA-55_197901_201412_rg1_sm.nc')



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist sm pre

vertical_labels = ['ERA5', 'HadGEM3-GC3.1-LL / ERA5', 'AWI-ESM-1-1-LR / ERA5']
nrow = 3
ncol = 4

sh_pre_level = np.concatenate(
    (np.arange(0, 25, 0.25), np.arange(25, 400.01, 3.75)))
sh_pre_ticks = np.concatenate(
    (np.arange(0, 25, 5), np.arange(25, 400.01,75)))
sh_pre_norm = BoundaryNorm(sh_pre_level, ncolors=len(sh_pre_level))
sh_pre_cmp = cm.get_cmap('RdBu', len(sh_pre_level))

pre_rdif_level = np.arange(0, 2.01, 0.01)
pre_rdif_ticks = np.arange(0, 2.01, 0.2)
pre_rdif_norm = BoundaryNorm(pre_rdif_level, ncolors=len(pre_rdif_level))
pre_rdif_cmp = cm.get_cmap('BrBG', len(pre_rdif_level))


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5*ncol, 5*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.001},)


# plot framework and vertical labels
for i in range(nrow):
    for j in range(ncol):
        axs[i, j] = hemisphere_plot(northextent=-60, ax_org = axs[i, j])
    
    axs[i, 0].text(
        -0.05, 0.5, vertical_labels[i], transform=axs[i, 0].transAxes,
        rotation='vertical', ha='right', va='center')

# plot data and horizontal labels
for j in range(ncol):
    axs[0, j].text(
        0.5, 1.05, seasons[j], transform=axs[0, j].transAxes,
        ha='center', va='bottom')
    
    axs[0, j].pcolormesh(
        hist_sm_pre['ERA5'].lon, hist_sm_pre['ERA5'].lat,
        hist_sm_pre['ERA5'].tp[j, :, :] *1000*90,
        norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
    
    axs[1, j].pcolormesh(
        hist_sm_pre['ERA5'].lon, hist_sm_pre['ERA5'].lat,
        (hist_sm_pre['HadGEM3-GC3.1-LL'].pr[j, :, :] *86400*90) / (hist_sm_pre['ERA5'].tp[j, :, :] *1000*90),
        norm=pre_rdif_norm, cmap=pre_rdif_cmp,transform=ccrs.PlateCarree(),)
    
    axs[2, j].pcolormesh(
        hist_sm_pre['ERA5'].lon, hist_sm_pre['ERA5'].lat,
        (hist_sm_pre['AWI-ESM-1-1-LR'].pr[j, :, :] *86400*90) / (hist_sm_pre['ERA5'].tp[j, :, :] *1000*90),
        norm=pre_rdif_norm, cmap=pre_rdif_cmp,transform=ccrs.PlateCarree(),)

# create color bar for pre
plt_pre_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_pre_norm, cmap=sh_pre_cmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.35), ticks=sh_pre_ticks)
plt_pre_cbar.ax.set_xlabel('Seasonal mean precipitation [$mm\;sea^{-1}$]')

# create color bar for pre_diff
plt_pre_diff_cbar = fig.colorbar(
    cm.ScalarMappable(norm=pre_rdif_norm, cmap=pre_rdif_cmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(1.1,-3.8),ticks=pre_rdif_ticks)
plt_pre_diff_cbar.ax.set_xlabel('Seasonal mean precipitation difference [-]')

fig.subplots_adjust(left=0.05, right = 0.99, bottom = 0.15, top = 0.95)
# fig.savefig('figures/0_test/trial.png', dpi = 300)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.0_pre/2.0.0.1_sh_pre_sm_comparison.png')


'''
# check
fig, ax = hemisphere_plot(northextent=-60,)
ax.pcolormesh(
    hist_sm_pre['ERA5'].lon, hist_sm_pre['ERA5'].lat,
    hist_sm_pre['ERA5'].tp[1, :, :] *1000*90,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial.png', dpi = 300)

fig, ax = hemisphere_plot(northextent=-60,)
ax.pcolormesh(
    hist_sm_pre['ERA5'].lon, hist_sm_pre['ERA5'].lat,
    hist_sm_pre['HadGEM3-GC3.1-LL'].pr[1, :, :] *86400*90,
    norm=sh_pre_norm, cmap=sh_pre_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial0.png', dpi = 300)
'''
# endregion
# -----------------------------------------------------------------------------


