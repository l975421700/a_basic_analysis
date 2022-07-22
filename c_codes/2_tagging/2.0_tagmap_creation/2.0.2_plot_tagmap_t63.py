

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
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot tagmap_echam6_t63_0

tagmap_echam6_t63_0 = xr.open_dataset('bas_palaeoclim_qino/startdump/tagmap/tagmap_echam6_t63_0.nc')

nbasic_tags = 8
nocean_tags = 38
nland_tags = 9

# cmp for ocean basins
pltlevel_o = np.arange(1 - 0.5, nocean_tags + nland_tags +1.5, 1)
pltnorm_o = BoundaryNorm(pltlevel_o, ncolors=len(pltlevel_o), clip=False)
pltcmp_o = cm.get_cmap('PRGn', len(pltlevel_o))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

# plot ocean basins
for i in range(nocean_tags + nland_tags):
    # i = 0
    mask_data = tagmap_echam6_t63_0.tagmap.sel(level=i+nbasic_tags+1).values
    mask_data[mask_data == 0] = np.nan
    mask_data = mask_data * (i + 1)
    
    plt_cmp = ax.pcolormesh(
        tagmap_echam6_t63_0.lon, tagmap_echam6_t63_0.lat, mask_data,
        transform=ccrs.PlateCarree(), norm=pltnorm_o, cmap=pltcmp_o,)
    
    ax.contour(
        tagmap_echam6_t63_0.lon, tagmap_echam6_t63_0.lat,
        tagmap_echam6_t63_0.tagmap.sel(level=i+nbasic_tags+1).values,
        colors='black', levels=np.array([0.5]), linewidths=0.1,
        linestyles='dotted',)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm_o, cmap=pltcmp_o),
    ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=np.arange(1, nocean_tags + nland_tags +1, 5), extend="neither",)

cbar.ax.set_xlabel('Tagged regions', linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/3_tagging/3.0_tagmap_creation/3.0.2_division_of_ocean_basins_in_tagmap_echam6_t63_0.png')

'''
echam6_t63_slm = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm.nc')
# cmp for slm
pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

plt_cmp = ax.pcolormesh(
    echam6_t63_slm.lon, echam6_t63_slm.lat, echam6_t63_slm.slm,
    transform=ccrs.PlateCarree(), norm=pltnorm, cmap=pltcmp,)


# cmp for atlantic
pltlevel_a = np.arange(0, 15 + 8.01, 1)
pltnorm_a = BoundaryNorm(pltlevel_a, ncolors=len(pltlevel_a), clip=False)
pltcmp_a = cm.get_cmap('Greys', len(pltlevel_a))

# cmp for pacific
pltlevel_p = np.arange(0, 14 + 8.01, 1)
pltnorm_p = BoundaryNorm(pltlevel_p, ncolors=len(pltlevel_p), clip=False)
pltcmp_p = cm.get_cmap('Greens', len(pltlevel_p))

# cmp for indian ocean
pltlevel_i = np.arange(0, 9 + 8.01, 1)
pltnorm_i = BoundaryNorm(pltlevel_i, ncolors=len(pltlevel_i), clip=False)
pltcmp_i = cm.get_cmap('Purples', len(pltlevel_i))

# plot atlantic basins
for i in range(15):
    # i = 0
    mask_data = tagmap_echam6_t63_0.tagmap.sel(level=9+i).values
    mask_data[mask_data == 0] = np.nan
    mask_data = mask_data * (i + 1) + 4
    
    ax.pcolormesh(
        tagmap_echam6_t63_0.lon, tagmap_echam6_t63_0.lat,
        mask_data,
        transform=ccrs.PlateCarree(), norm=pltnorm_a, cmap=pltcmp_a,)

# plot atlantic basins
for i in range(14):
    # i = 0
    mask_data = tagmap_echam6_t63_0.tagmap.sel(level=9+15+i).values
    mask_data[mask_data == 0] = np.nan
    mask_data = mask_data * (i + 1) + 4
    
    ax.pcolormesh(
        tagmap_echam6_t63_0.lon, tagmap_echam6_t63_0.lat,
        mask_data,
        transform=ccrs.PlateCarree(), norm=pltnorm_p, cmap=pltcmp_p,)

# plot indian ocean basins
for i in range(9):
    # i = 0
    mask_data = tagmap_echam6_t63_0.tagmap.sel(level=9+15+14+i).values
    mask_data[mask_data == 0] = np.nan
    mask_data = mask_data * (i + 1) + 4
    
    ax.pcolormesh(
        tagmap_echam6_t63_0.lon, tagmap_echam6_t63_0.lat,
        mask_data,
        transform=ccrs.PlateCarree(), norm=pltnorm_i, cmap=pltcmp_i,)

'''
# endregion
# -----------------------------------------------------------------------------


