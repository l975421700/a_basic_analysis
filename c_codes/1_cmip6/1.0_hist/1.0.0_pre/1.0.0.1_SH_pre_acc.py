

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
mpl.rcParams['figure.dpi'] = 300
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
# region

acc_pre = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/observations/products/Antarctic_Accumulation_Reconstructions/acc_recon_ERAI.nc')

sh_pre_level = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
sh_pre_ticks = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01,300)))
sh_pre_norm = BoundaryNorm(sh_pre_level, ncolors=len(sh_pre_level))
sh_pre_cmp = cm.get_cmap('RdBu', len(sh_pre_level))


fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([8.8, 9.8]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.04, fm_top=0.99,
    )

plt_cmp = ax.pcolormesh(
    acc_pre.lon, acc_pre.lat,
    acc_pre.recon_acc_bc[np.where(acc_pre.year >= 1979)[0], :, :].mean(axis=0),
    norm=sh_pre_norm, cmap=sh_pre_cmp, transform=ccrs.PlateCarree(),
)
plt_pre_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_pre_norm, cmap=sh_pre_cmp), ax=ax,
    orientation="horizontal",pad=0.02,shrink=0.9,aspect=40,extend='max',
    # anchor=(-0.2, -0.35), fraction=0.12, panchor=(0.5, 0),
    ticks=sh_pre_ticks)
plt_pre_cbar.ax.set_xlabel('Annual mean precipitation [$mm\;yr^{-1}$]\nAntarctic Accumulation Reconstructions, 1979-2000', linespacing=1.5)

fig.savefig('/home/users/qino/figures/0_test/trial.png',)


'''
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.99)
acc_pre.year[acc_pre.recon_acc_bc.sel(years=slice(0,21)).years]
acc_pre.year[acc_pre.recon_acc_bc[np.where(acc_pre.year >= 1979)[0], :, :].years]
'''
# endregion
# -----------------------------------------------------------------------------







