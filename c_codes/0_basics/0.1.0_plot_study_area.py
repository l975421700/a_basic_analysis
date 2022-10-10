

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
from statsmodels.stats import multitest
import pycircstat as circ
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
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
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    cplot_lon180,
    cplot_wind_vectors,
    cplot_lon180_quiver,
    cplot_lon180_ctr,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

ais_imbie2 = gpd.read_file(
    'data_sources/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

bedmap_tif = xr.open_rasterio(
    'data_sources/products/bedmap2_tiff/bedmap2_surface.tif')
bedmap_height = (bedmap_tif.values.squeeze().copy()).astype(np.float32)
bedmap_height[bedmap_height == 32767] = np.nan
bedmap_transform = ccrs.epsg(3031)

echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)
echam6_t63_surface_height.values[echam6_t63_geosp.SLM.values == 0] = np.nan



'''
stats.describe(bedmap_height, axis=None, nan_policy='omit')

(echam6_t63_surface_height.values <= 1).sum()
(echam6_t63_surface_height.values <= 0).sum()
geopotential_to_height(echam6_t63_geosp.GEOSP)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS + height + ice cores + divisions

output_png = 'figures/1_study_area/1.0_AIS_height_icecores_division.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=4500, cm_interval1=250, cm_interval2=500, cmap='PuOr',
    reversed=False)

nrow = 1
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 8.8*nrow + 1]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.35, 'wspace': 0.35},)

axs[0] = hemisphere_plot(
    northextent=-60, ax_org = axs[0], plot_scalebar=True,
    add_grid_labels=True, l45label = False, loceanarcs = True,
    llatlabel = True)
axs[1] = hemisphere_plot(
    northextent=-60, ax_org = axs[1], plot_scalebar=True,
    add_grid_labels=True, l45label = False, loceanarcs = True)

ipanel=0
for jcol in range(ncol):
    plt.text(
        0, 1.12, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat,
                    axs[jcol], s=5)
    
    # plot AIS divisions
    plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='red', facecolor='none', linewidths=0.5, zorder=2)
    plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='blue', facecolor='none', linewidths=0.5, zorder=2)
    plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
        ax=axs[jcol], transform=ccrs.epsg(3031),
        edgecolor='m', facecolor='none', linewidths=0.5, zorder=2)
    
    ipanel += 1

for irow in range(major_ice_core_site.shape[0]):
    # irow = 0
    axs[1].text(
        major_ice_core_site.lon[irow], major_ice_core_site.lat[irow]+1,
        major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
        fontsize = 6, color='black')

axs[0].pcolormesh(
    bedmap_tif.x.values,
    bedmap_tif.y.values,
    bedmap_height,
    norm=pltnorm, cmap=pltcmp,transform=bedmap_transform,)

plt_mesh = axs[1].pcolormesh(
    echam6_t63_surface_height.lon,
    echam6_t63_surface_height.lat,
    echam6_t63_surface_height.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.12, 'Bedmap2', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.12,'ECHAM6 T63', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    anchor=(-0.3, 0.28),
    )
cbar.ax.set_xlabel('Antarctic surface height [$m$]', linespacing=2)

# axs[0]
patch1 = mpatches.Patch(
    edgecolor='blue', facecolor='none', label='EAIS', lw=0.5)
patch2 = mpatches.Patch(
    edgecolor='red', facecolor='none', label='WAIS', lw=0.5)
patch3 = mpatches.Patch(
    edgecolor='m', facecolor='none', label='AP', lw=0.5)
line1 = Line2D([0], [0], label='Atlantic sector', lw=2, linestyle='-',
               color='gray')
line2 = Line2D([0], [0], label='Indian sector', lw=2, linestyle='--',
               color='gray')
line3 = Line2D([0], [0], label='Pacific sector', lw=2, linestyle=':',
               color='gray')
plt.legend(handles=[patch1, patch2, patch3, line1, line2, line3],
           ncol=2, frameon=False, columnspacing=1,
           bbox_to_anchor=(1.1, -0.05),
           )

fig.subplots_adjust(left=0.08, right = 0.94, bottom = 0.23, top = 0.92)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


