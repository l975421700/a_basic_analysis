

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
# =============================================================================


# =============================================================================
# region import output: pi_final_qg_tag4

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = ['pi_final_qg_tag4', 'pi_final_qg_tag4_1y']

awi_esm_o = {}

for i in range(len(expid)):
    # i=1
    awi_esm_o[expid[i]] = {}
    
    ## echam
    awi_esm_o[expid[i]]['echam'] = {}
    awi_esm_o[expid[i]]['echam']['echam'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.nc')
    awi_esm_o[expid[i]]['echam']['echam_am'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.am.nc')
    awi_esm_o[expid[i]]['echam']['echam_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.ann.nc')
    
    ## wiso
    awi_esm_o[expid[i]]['wiso'] = {}
    awi_esm_o[expid[i]]['wiso']['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.nc')
    awi_esm_o[expid[i]]['wiso']['wiso_am'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.am.nc')
    awi_esm_o[expid[i]]['wiso']['wiso_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.ann.nc')

# endregion
# =============================================================================


# =============================================================================
# region check tagging water fraction


tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap4.nc')


# calculate the water tracer fraction
nregion = 3
tag_w_frac = {}

for i in range(len(expid)):
    # i=1
    tag_w_frac[expid[i]] = {}
    
    for j in range(nregion):
        # j=2
        tag_w_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprl[0, j+4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprc[0, j+4, :, :]) / (awi_esm_o[expid[i]]['echam']['echam_am'].aprl[0, :, :] + awi_esm_o[expid[i]]['echam']['echam_am'].aprc[0, :, :]) * 100


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('viridis', len(pltlevel)).reversed()


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.0_pi_final_qg_tag4_tag_w_frac_region_3.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4_1y',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.1_pi_final_qg_tag4_1y_tag_w_frac_region_3.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'] - tag_w_frac['pi_final_qg_tag4']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    'Fraction of precipitation from land surface in the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4 - pi_final_qg_tag4_1y',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag4-pi_final_qg_tag4_1y_tag_w_frac_region_3.png')

# endregion
# =============================================================================




