

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
# region import hist am+sm windplev

hist_windplev_dir = '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/wind_plev/'

hist_am_windplev = {}
hist_am_windplev['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_windplev_dir + 'ua_va_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_am.nc')
hist_am_windplev['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_windplev_dir + 'ua_va_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_am.nc')
hist_am_windplev['ERA5'] = xr.open_dataset(hist_windplev_dir + 'ua_va_ERA5_mon_pl1_197901_201412_rg1_am.nc')

hist_sm_windplev = {}
hist_sm_windplev['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_windplev_dir + 'ua_va_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_sm.nc')
hist_sm_windplev['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_windplev_dir + 'ua_va_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_sm.nc')
hist_sm_windplev['ERA5'] = xr.open_dataset(hist_windplev_dir + 'ua_va_ERA5_mon_pl1_197901_201412_rg1_sm.nc')

'''
hist_am_windplev['HadGEM3-GC3.1-LL'].ua.sel(plev=50000).squeeze()
hist_am_windplev['ERA5'].u.sel(level=500).squeeze()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist am+sm windplev

vertical_labels = [
    'ERA5', 'HadGEM3-GC3.1-LL - ERA5', 'AWI-ESM-1-1-LR - ERA5']
nrow = 3
ncol = 5

sh_windplev_level = np.arange(0, 27.01, 0.01)
sh_windplev_ticks = np.arange(0, 27.01, 3)
sh_windplev_norm = BoundaryNorm(sh_windplev_level, ncolors=len(sh_windplev_level))
sh_windplev_cmp = cm.get_cmap('plasma', len(sh_windplev_level))

windplev_adif_level = np.arange(-6, 6.01, 0.01)
windplev_adif_ticks = np.arange(-6, 6.01, 2)
windplev_adif_norm = BoundaryNorm(windplev_adif_level, ncolors=len(windplev_adif_level))
windplev_adif_cmp = cm.get_cmap('BrBG', len(windplev_adif_level))


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5*ncol, 5*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.001},)

# plot framework and vertical labels
for i in range(nrow):
    for j in range(ncol):
        axs[i, j] = hemisphere_plot(northextent=-45, ax_org=axs[i, j])
    
    axs[i, 0].text(
        -0.05, 0.5, vertical_labels[i], transform=axs[i, 0].transAxes,
        rotation='vertical', ha='right', va='center')

# plot data and horizontal labels
for k in range(len(seasons)):
    axs[0, k+1].text(
        0.5, 1.05, seasons[k], transform=axs[0, k+1].transAxes,
        ha='center', va='bottom')
    axs[0, k+1].pcolormesh(
        hist_sm_windplev['ERA5'].lon, hist_sm_windplev['ERA5'].lat,
        (hist_sm_windplev['ERA5'].u.sel(level=500)[k, :, :] ** 2 + hist_sm_windplev['ERA5'].v.sel(level=500)[k, :, :] ** 2) ** 0.5,
        norm=sh_windplev_norm, cmap=sh_windplev_cmp, transform=ccrs.PlateCarree(),)
    
    axs[1, k+1].pcolormesh(
        hist_sm_windplev['ERA5'].lon, hist_sm_windplev['ERA5'].lat,
        (hist_sm_windplev['HadGEM3-GC3.1-LL'].ua.sel(plev=50000)[k, :, :] ** 2 + hist_sm_windplev['HadGEM3-GC3.1-LL'].va.sel(plev=50000)[k, :, :]
         ** 2) ** 0.5 - (hist_sm_windplev['ERA5'].u.sel(level=500)[k, :, :] ** 2 + hist_sm_windplev['ERA5'].v.sel(level=500)[k, :, :] ** 2) ** 0.5,
        norm=windplev_adif_norm, cmap=windplev_adif_cmp, transform=ccrs.PlateCarree(),)
    
    axs[2, k+1].pcolormesh(
        hist_sm_windplev['ERA5'].lon, hist_sm_windplev['ERA5'].lat,
        (hist_sm_windplev['AWI-ESM-1-1-LR'].ua.sel(plev=50000)[k, :, :] ** 2 + hist_sm_windplev['AWI-ESM-1-1-LR'].va.sel(plev=50000)[k, :, :] **
         2) ** 0.5 - (hist_sm_windplev['ERA5'].u.sel(level=500)[k, :, :] ** 2 + hist_sm_windplev['ERA5'].v.sel(level=500)[k, :, :] ** 2) ** 0.5,
        norm=windplev_adif_norm, cmap=windplev_adif_cmp, transform=ccrs.PlateCarree(),)

axs[0, 0].text(
    0.5, 1.05, 'Ann', transform=axs[0, 0].transAxes,
    ha='center', va='bottom')

axs[0, 0].pcolormesh(
    hist_am_windplev['ERA5'].lon, hist_am_windplev['ERA5'].lat,
    (hist_am_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_am_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5,
    norm=sh_windplev_norm, cmap=sh_windplev_cmp, transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    hist_am_windplev['ERA5'].lon, hist_am_windplev['ERA5'].lat,
    (hist_am_windplev['HadGEM3-GC3.1-LL'].ua.sel(plev=50000).squeeze() ** 2 + hist_am_windplev['HadGEM3-GC3.1-LL'].va.sel(plev=50000).squeeze() ** 2) ** 0.5 - (hist_am_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_am_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5,
    norm=windplev_adif_norm, cmap=windplev_adif_cmp, transform=ccrs.PlateCarree(),)

axs[2, 0].pcolormesh(
    hist_am_windplev['ERA5'].lon, hist_am_windplev['ERA5'].lat,
    (hist_am_windplev['AWI-ESM-1-1-LR'].ua.sel(plev=50000).squeeze() ** 2 + hist_am_windplev['AWI-ESM-1-1-LR'].va.sel(plev=50000).squeeze() **
     2) ** 0.5 - (hist_am_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_am_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5,
    norm=windplev_adif_norm, cmap=windplev_adif_cmp, transform=ccrs.PlateCarree(),)

# create color bar for windplev
plt_windplev_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_windplev_norm, cmap=sh_windplev_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='max',
    anchor=(-0.2, -0.3), ticks=sh_windplev_ticks)
plt_windplev_cbar.ax.set_xlabel(
    'Annual/seasonal mean 500 hPa wind speed [$m\;s^{-1}$]')

# create color bar for windplev_diff
plt_windplev_diff_cbar = fig.colorbar(
    cm.ScalarMappable(norm=windplev_adif_norm, cmap=windplev_adif_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='both',
    anchor=(1.1, -3.8), ticks=windplev_adif_ticks)
plt_windplev_diff_cbar.ax.set_xlabel(
    'Annual/seasonal mean 500 hPa wind speed difference [$m\;s^{-1}$]')

fig.subplots_adjust(left=0.04, right=0.99, bottom=0.15, top=0.95)
# fig.savefig('figures/2_cmip6/2.0_hist/2.0.3_windplev/2.0.3.0_sh_windplev_am_sm_comparison.png')
# !!! wrong

'''
stats.describe(((hist_am_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_am_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5).sel(lat=slice(-90, -45)), axis=None)
stats.describe(((hist_sm_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_sm_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5).sel(lat=slice(-90, -45)), axis=None)

# check
fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_am_windplev['ERA5'].lon, hist_am_windplev['ERA5'].lat,
    (hist_am_windplev['ERA5'].u.sel(level=500).squeeze() ** 2 + hist_am_windplev['ERA5'].v.sel(level=500).squeeze() ** 2) ** 0.5,
    norm=sh_windplev_norm, cmap=sh_windplev_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial.png', dpi = 300)

fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_am_windplev['ERA5'].lon, hist_am_windplev['ERA5'].lat,
    (hist_am_windplev['HadGEM3-GC3.1-LL'].ua.sel(plev=50000).squeeze() ** 2 + hist_am_windplev['HadGEM3-GC3.1-LL'].va.sel(plev=50000).squeeze() ** 2) ** 0.5,
    norm=sh_windplev_norm, cmap=sh_windplev_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial0.png', dpi = 300)
'''
# endregion
# -----------------------------------------------------------------------------



