

# =============================================================================
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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.colors import ListedColormap

# self defined
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    plot_maxmin_points,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    regrid,
)

from a00_basic_analysis.b_module.namelist import (
    month_days,
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# region import and plot atmospheric grid cell area in HadGEM3-GC31-LL

# import area of each cell
areacella_hg3_ll = xr.open_dataset(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/areacella/gn/v20190709/areacella_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc',)

areacella_hg3_ll_300 = areacella_hg3_ll.areacella.values / (300 * 10**6)

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    areacella_hg3_ll.lon,
    areacella_hg3_ll.lat,
    areacella_hg3_ll_300,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend='neither')
cbar.ax.set_xlabel(
    'Atmospheric grid cell area [$300 \; km^{2}$]\nHadGEM3-GC31-LL',
    linespacing=1.5
)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    'figures/4_cmip6/4.0_HadGEM3-GC3.1/4.0.1_constants/4.0.1.7_global cell area in HadGEM3-GC31-LL.png',)



'''
stats.describe(areacella_hg3_ll.areacella.values, axis = None)
areacella_hg3_ll.areacella.values.sum()
areacella_hg3_ll_300.sum()
areacella_hg3_ll_300_rg.sum()
stats.describe(areacella_hg3_ll_300, axis = None)

# not valid
regridder = xe.Regridder(
    ds_in=areacella_hg3_ll, ds_out=xe.util.grid_global(1, 1),
    method='conservative')
areacella_hg3_ll_300_rg = regridder(areacella_hg3_ll_300)
'''
# endregion
# =============================================================================


# =============================================================================
# region

one_degree_grids = xe.util.grid_global(1, 1)
one_degree_grids_cdo = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo.nc')

one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')


areacella_hg3_ll.areacella.values.sum() / one_degree_grids_cdo_area.cell_area.sum().values

'''
# cdo -P 4 -remapcon,global_1 /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/areacella/gn/v20190709/areacella_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc bas_palaeoclim_qino/others/one_degree_grids_cdo.nc

# cdo gridarea bas_palaeoclim_qino/others/one_degree_grids_cdo.nc bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc
'''
# endregion
# =============================================================================

