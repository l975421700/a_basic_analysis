

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
# region import hist am+sm siconc

hist_siconc_dir = '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/siconc/'
hist_siconc_ds = ['HadGEM3-GC3.1-LL', 'AWI-ESM-1-1-LR', 'NSIDC',]


hist_am_siconc = {}
hist_am_siconc['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_siconc_dir + 'siconc_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_am.nc')
hist_am_siconc['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_siconc_dir + 'siconc_SImon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_am.nc')
hist_am_siconc['NSIDC'] = xr.open_dataset(hist_siconc_dir + 'siconc_NSIDC_197901_201412_rg1_am.nc')


hist_sm_siconc = {}
hist_sm_siconc['HadGEM3-GC3.1-LL'] = xr.open_dataset(hist_siconc_dir + 'siconc_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_197901-201412_rg1_sm.nc')
hist_sm_siconc['AWI-ESM-1-1-LR'] = xr.open_dataset(hist_siconc_dir + 'siconc_SImon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_197901-201412_rg1_sm.nc')
hist_sm_siconc['NSIDC'] = xr.open_dataset(hist_siconc_dir + 'siconc_NSIDC_197901_201412_rg1_sm.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist am+sm siconc

vertical_labels = [
    'NSIDC', 'HadGEM3-GC3.1-LL - NSIDC', 'AWI-ESM-1-1-LR - NSIDC']
nrow = 3
ncol = 5

sh_siconc_level = np.arange(0, 100.01, 0.5)
sh_siconc_ticks = np.arange(0, 100.01, 20)
sh_siconc_norm = BoundaryNorm(sh_siconc_level, ncolors=len(sh_siconc_level))
sh_siconc_cmp = cm.get_cmap('Blues', len(sh_siconc_level))

siconc_adif_level = np.arange(-40, 40.01, 0.1)
siconc_adif_ticks = np.arange(-40, 40.01, 10)
siconc_adif_norm = BoundaryNorm(siconc_adif_level, ncolors=len(siconc_adif_level))
siconc_adif_cmp = cm.get_cmap('PuOr', len(siconc_adif_level))


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
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
        hist_sm_siconc['NSIDC'].siconc[k, :, :],
        norm=sh_siconc_norm, cmap=sh_siconc_cmp, transform=ccrs.PlateCarree(),)
    
    axs[1, k+1].pcolormesh(
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
        hist_sm_siconc['HadGEM3-GC3.1-LL'].siconc[k, :, :].values - hist_sm_siconc['NSIDC'].siconc[k, :, :].values,
        norm=siconc_adif_norm, cmap=siconc_adif_cmp, transform=ccrs.PlateCarree(),)
    
    axs[2, k+1].pcolormesh(
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
        hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
        hist_sm_siconc['AWI-ESM-1-1-LR'].siconc[k, :, :].values - hist_sm_siconc['NSIDC'].siconc[k, :, :].values,
        norm=siconc_adif_norm, cmap=siconc_adif_cmp, transform=ccrs.PlateCarree(),)

axs[0, 0].text(
    0.5, 1.05, 'Ann', transform=axs[0, 0].transAxes,
    ha='center', va='bottom')

axs[0, 0].pcolormesh(
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
    hist_am_siconc['NSIDC'].siconc.squeeze(),
    norm=sh_siconc_norm, cmap=sh_siconc_cmp, transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
    hist_am_siconc['HadGEM3-GC3.1-LL'].siconc.squeeze().values - hist_am_siconc['NSIDC'].siconc.squeeze().values,
    norm=siconc_adif_norm, cmap=siconc_adif_cmp, transform=ccrs.PlateCarree(),)

axs[2, 0].pcolormesh(
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lon,
    hist_sm_siconc['HadGEM3-GC3.1-LL'].lat,
    hist_am_siconc['AWI-ESM-1-1-LR'].siconc.squeeze().values -
    hist_am_siconc['NSIDC'].siconc.squeeze().values,
    norm=siconc_adif_norm, cmap=siconc_adif_cmp, transform=ccrs.PlateCarree(),)

# create color bar for siconc
plt_siconc_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_siconc_norm, cmap=sh_siconc_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='neither',
    anchor=(-0.2, -0.3), ticks=sh_siconc_ticks)
plt_siconc_cbar.ax.set_xlabel(
    'Annual/seasonal mean sea ice area fraction [$\%$]')

# create color bar for pre_diff
plt_pre_diff_cbar = fig.colorbar(
    cm.ScalarMappable(norm=siconc_adif_norm, cmap=siconc_adif_cmp), ax=axs,
    orientation="horizontal", shrink=0.5, aspect=40, extend='both',
    anchor=(1.1, -3.8), ticks=siconc_adif_ticks)
plt_pre_diff_cbar.ax.set_xlabel(
    'Annual/seasonal mean sea ice area fraction [$\%$]')

fig.subplots_adjust(left=0.04, right=0.99, bottom=0.15, top=0.95)
fig.savefig(
    'figures/2_cmip6/2.0_hist/2.0.2_siconc/2.0.2.0_sh_siconc_am_sm_comparison.png')


'''
stats.describe(hist_am_siconc['NSIDC'].siconc.sel(lat=slice(-90, -45)), axis=None)
stats.describe(hist_sm_siconc['NSIDC'].siconc.sel(lat=slice(-90, -45)), axis=None)

# check
fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_sm_siconc['NSIDC'].lon, hist_sm_siconc['NSIDC'].lat,
    hist_sm_siconc['NSIDC'].siconc[0, :, :],
    norm=sh_siconc_norm, cmap=sh_siconc_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial.png', dpi = 300)

fig, ax = hemisphere_plot(northextent=-45,)
ax.pcolormesh(
    hist_sm_siconc['NSIDC'].lon, hist_sm_siconc['NSIDC'].lat,
    hist_sm_siconc['HadGEM3-GC3.1-LL'].siconc[0, :, :],
    norm=sh_siconc_norm, cmap=sh_siconc_cmp,transform=ccrs.PlateCarree(),)
fig.savefig('figures/0_test/trial0.png', dpi = 300)
'''
# endregion
# -----------------------------------------------------------------------------



