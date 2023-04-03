

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
# region import hist am+sm tas

hist_tas_dir = '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/tas/'
hist_tas_ds = ['HadGEM3-GC3.1-LL', 'AWI-ESM-1-1-LR', 'ERA5',]

hist_am_tas = {}
hist_am_tas['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_tas_dir + 'tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_am.nc')
hist_am_tas['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_tas_dir + 'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_am.nc')
hist_am_tas['ERA5'] = xr.open_dataset(hist_tas_dir + 'tas_ERA5_mon_sl_197901_201412_rg1_am.nc')


hist_sm_tas = {}
hist_sm_tas['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_tas_dir + 'tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_sm.nc')
hist_sm_tas['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_tas_dir + 'tas_Amon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_sm.nc')
hist_sm_tas['ERA5'] = xr.open_dataset(hist_tas_dir + 'tas_ERA5_mon_sl_197901_201412_rg1_sm.nc')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist am+sm tas

vertical_labels = ['ERA5', 'HadGEM3-GC3.1-LL - ERA5', 'AWI-ESM-1-1-LR - ERA5']
nrow = 3
ncol = 5

sh_tas_level = np.arange(210, 290.01, 0.2)
sh_tas_ticks = np.arange(210, 290.01, 10)
sh_tas_norm = BoundaryNorm(sh_tas_level, ncolors=len(sh_tas_level))
sh_tas_cmp = cm.get_cmap('viridis', len(sh_tas_level))

tas_adif_level = np.arange(-5, 5.01, 0.01)
tas_adif_ticks = np.arange(-5, 5.01, 1)
tas_adif_norm = BoundaryNorm(tas_adif_level, ncolors=len(tas_adif_level))
tas_adif_cmp = cm.get_cmap('PuOr', len(tas_adif_level)).reversed()


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
        hist_sm_tas['ERA5'].lon, hist_sm_tas['ERA5'].lat,
        hist_sm_tas['ERA5'].tas[k, :, :],
        norm=sh_tas_norm, cmap=sh_tas_cmp, transform=ccrs.PlateCarree(),)
    
    axs[1, k+1].pcolormesh(
        hist_sm_tas['ERA5'].lon, hist_sm_tas['ERA5'].lat,
        hist_sm_tas['HadGEM3-GC3.1-LL'].tas[k, :, :] - hist_sm_tas['ERA5'].tas[k, :, :],
        norm=tas_adif_norm, cmap=tas_adif_cmp, transform=ccrs.PlateCarree(),)
    
    axs[2, k+1].pcolormesh(
        hist_sm_tas['ERA5'].lon, hist_sm_tas['ERA5'].lat,
        hist_sm_tas['AWI-ESM-1-1-LR'].tas[k, :, :] - hist_sm_tas['ERA5'].tas[k, :, :],
        norm=tas_adif_norm, cmap=tas_adif_cmp, transform=ccrs.PlateCarree(),)

axs[0, 0].text(
    0.5, 1.05, 'Ann', transform=axs[0, 0].transAxes,
    ha='center', va='bottom')

axs[0, 0].pcolormesh(
    hist_am_tas['ERA5'].lon, hist_am_tas['ERA5'].lat,
    hist_am_tas['ERA5'].tas.squeeze(),
    norm=sh_tas_norm, cmap=sh_tas_cmp, transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    hist_am_tas['ERA5'].lon, hist_am_tas['ERA5'].lat,
    hist_am_tas['HadGEM3-GC3.1-LL'].tas.squeeze() - hist_am_tas['ERA5'].tas.squeeze(),
    norm=tas_adif_norm, cmap=tas_adif_cmp, transform=ccrs.PlateCarree(),)

axs[2, 0].pcolormesh(
    hist_am_tas['ERA5'].lon, hist_am_tas['ERA5'].lat,
    hist_am_tas['AWI-ESM-1-1-LR'].tas.squeeze() -
    hist_am_tas['ERA5'].tas.squeeze(),
    norm=tas_adif_norm, cmap=tas_adif_cmp, transform=ccrs.PlateCarree(),)

# create color bar for tas
plt_tas_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_tas_norm, cmap=sh_tas_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='both',
    anchor=(-0.2, -0.3), ticks=sh_tas_ticks)
plt_tas_cbar.ax.set_xlabel('Annual/seasonal mean 2m temperature [$K$]')

# create color bar for pre_diff
plt_pre_diff_cbar = fig.colorbar(
    cm.ScalarMappable(norm=tas_adif_norm, cmap=tas_adif_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='both',
    anchor=(1.1, -3.8), ticks=tas_adif_ticks)
plt_pre_diff_cbar.ax.set_xlabel(
    'Annual/seasonal mean 2m temperature difference [$K$]')

fig.subplots_adjust(left=0.04, right=0.99, bottom=0.15, top=0.95)
# fig.savefig('figures/0_test/trial.png', dpi = 300)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.1_tas/2.0.1.0_sh_tas_am_sm_comparison.png')


'''
stats.describe(hist_am_tas['ERA5'].tas.sel(lat=slice(-90, -45)), axis=None)
stats.describe(hist_sm_tas['ERA5'].tas.sel(lat=slice(-90, -45)), axis=None)

# check
fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_sm_tas['ERA5'].lon, hist_sm_tas['ERA5'].lat,
    hist_sm_tas['ERA5'].tas[0, :, :],
    norm=sh_tas_norm, cmap=sh_tas_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial.png', dpi = 300)

fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_sm_tas['ERA5'].lon, hist_sm_tas['ERA5'].lat,
    hist_sm_tas['HadGEM3-GC3.1-LL'].tas[0, :, :],
    norm=sh_tas_norm, cmap=sh_tas_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial0.png', dpi = 300)
'''





# endregion
# -----------------------------------------------------------------------------



