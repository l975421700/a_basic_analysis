

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

boundary_conditions = {}

pi_sst_file = 'startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc'
pi_sic_file = 'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc'
pi_wisosw_d_file = '/home/ollie/mwerner/model_input/ECHAM6-wiso/PI_ctrl/T63/T63_wisosw_d.nc'

pi_final_sst_file = 'startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final_t63.nc'
pi_final_sic_file = 'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc'
pi_final_wisosw_d_file = 'startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.pi_final_t63.nc'

lig_final_sst_file = 'startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final_t63.nc'
lig_final_sic_file = 'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63.nc'
lig_final_wisosw_d_file = 'startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63.nc'

lgm_50_sst_file = 'startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50_t63.nc'
lgm_50_sic_file = 'startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63.nc'
lgm_50_wisosw_d_file = 'startdump/model_input/lgm/lgm-50/wisosw_d.echam.2200_2229.lgm-50_t63.nc'

boundary_conditions['sst'] = {}
boundary_conditions['sst']['pi'] = xr.open_dataset(pi_sst_file)
boundary_conditions['sst']['pi_final'] = xr.open_dataset(pi_final_sst_file)
boundary_conditions['sst']['lig_final'] = xr.open_dataset(lig_final_sst_file)
boundary_conditions['sst']['lgm_50'] = xr.open_dataset(lgm_50_sst_file)

boundary_conditions['sic'] = {}
boundary_conditions['sic']['pi'] = xr.open_dataset(pi_sic_file)
boundary_conditions['sic']['pi_final'] = xr.open_dataset(pi_final_sic_file)
boundary_conditions['sic']['lig_final'] = xr.open_dataset(lig_final_sic_file)
boundary_conditions['sic']['lgm_50'] = xr.open_dataset(lgm_50_sic_file)

boundary_conditions['wisosw_d'] = {}
boundary_conditions['wisosw_d']['pi'] = xr.open_dataset(pi_wisosw_d_file)
boundary_conditions['wisosw_d']['pi_final'] = xr.open_dataset(pi_final_wisosw_d_file)
boundary_conditions['wisosw_d']['lig_final'] = xr.open_dataset(lig_final_wisosw_d_file)
boundary_conditions['wisosw_d']['lgm_50'] = xr.open_dataset(lgm_50_wisosw_d_file)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig SST

#-------- basic set
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.0.0 bc sst from pi, pi_final, and lig_final.png'

pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-5, 5 + 1e-4, 0.5)
pltticks2 = np.arange(-5, 5 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot
itime = 11

cbar_label1 = month[itime] + ' SST [$°C$]'
cbar_label2 = 'Differences in ' + month[itime] + ' SST [$°C$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi_final'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['lig_final'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi_final'].sst[itime] - boundary_conditions['sst']['pi'].sst[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['lig_final'].sst[itime] - boundary_conditions['sst']['pi_final'].sst[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi'].sst[itime] + boundary_conditions['sst']['lig_final'].sst[itime].values - boundary_conditions['sst']['pi_final'].sst[itime].values - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lig_final - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


#-------------------------------- animate

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt1 = axs[0, 0].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[0, 1].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi_final'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt3 = axs[0, 2].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['lig_final'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt4 = axs[1, 0].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi_final'].sst[itime] - boundary_conditions['sst']['pi'].sst[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt5 = axs[1, 1].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['lig_final'].sst[itime] - boundary_conditions['sst']['pi_final'].sst[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt6 = axs[1, 2].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi'].sst[itime] + boundary_conditions['sst']['lig_final'].sst[itime].values - boundary_conditions['sst']['pi_final'].sst[itime].values - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt7 = plt.text(
        0.5, -0.1, month[itime], transform=axs[1, 1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6, plt7]
    
    return(plt_objs)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lig_final - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar_label1 = 'SST [$°C$]'
cbar_label2 = 'Differences in SST [$°C$]'


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)

ani = animation.FuncAnimation(
    fig, update_frames, frames=12, interval=250, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.0.1 bc sst from pi, pi_final, and lig_final.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig sic

#-------- basic set
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.0.2 bc sic from pi, pi_final, and lig_final.png'

pltlevel = np.arange(0, 100 + 1e-4, 10)
pltticks = np.arange(0, 100 + 1e-4, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

pltlevel2 = np.arange(-50, 50 + 1e-4, 5)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot
itime = 11

cbar_label1 = month[itime] + ' sea ice concentration [$\%$]'
cbar_label2 = 'Differences in ' + month[itime] + ' sea ice concentration [$\%$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi_final'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['lig_final'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi_final'].sic[itime] - boundary_conditions['sic']['pi'].sic[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['lig_final'].sic[itime] - boundary_conditions['sic']['pi_final'].sic[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi'].sic[itime] + boundary_conditions['sic']['lig_final'].sic[itime].values - boundary_conditions['sic']['pi_final'].sic[itime].values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lig_final - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


#-------------------------------- animate

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt1 = axs[0, 0].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[0, 1].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi_final'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt3 = axs[0, 2].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['lig_final'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt4 = axs[1, 0].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi_final'].sic[itime] - boundary_conditions['sic']['pi'].sic[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt5 = axs[1, 1].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['lig_final'].sic[itime] - boundary_conditions['sic']['pi_final'].sic[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt6 = axs[1, 2].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi'].sic[itime] + boundary_conditions['sic']['lig_final'].sic[itime].values - boundary_conditions['sic']['pi_final'].sic[itime].values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt7 = plt.text(
        0.5, -0.1, month[itime], transform=axs[1, 1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6, plt7]
    
    return(plt_objs)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lig_final - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar_label1 = 'Sea ice concentration [$\%$]'
cbar_label2 = 'Differences in sea ice concentration [$\%$]'

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)

ani = animation.FuncAnimation(
    fig, update_frames, frames=12, interval=250, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.0.3 bc sic from pi, pi_final, and lig_final.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig wisosw_d

#-------- basic set

# water_isotope = '$\delta^{18}O$'
# ilev = 1
# pltlevel = np.concatenate(
#     (np.arange(-6, 0, 0.5), np.arange(0, 2.4 + 1e-4, 0.2)))
# pltticks = np.concatenate(
#     (np.arange(-6, 0, 1), np.arange(0, 2.4 + 1e-4, 0.4)))
# pltlevel2 = np.arange(-2, 2 + 1e-4, 0.25)
# pltticks2 = np.arange(-2, 2 + 1e-4, 0.5)

water_isotope = '$\delta D$'
ilev = 2
pltlevel = np.concatenate(
    (np.arange(-50, 0, 5), np.arange(0, 10 + 1e-4, 1)))
pltticks = np.concatenate(
    (np.arange(-50, 0, 10), np.arange(0, 10 + 1e-4, 2)))
pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
pltticks2 = np.arange(-20, 20 + 1e-4, 5)

#-------- plot configuration
mpl.rc('font', family='Times New Roman', size=10)
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.0.4.0 bc '+ water_isotope + ' from pi, pi_final, and lig_final.png'

pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1)

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot

cbar_label1 = water_isotope + ' of surface sea water [$‰$]'
cbar_label2 = 'Differences in ' + water_isotope + ' of surface sea water [$‰$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['lig_final'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev] - boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['lig_final'].wisosw_d[ilev] - boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev] + boundary_conditions['wisosw_d']['lig_final'].wisosw_d[ilev].values - boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev].values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lig_final - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi + lig_final - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


'''
wisosw_d = xr.open_dataset('/home/ollie/xshi/for_Janina/wisosw_d_pi_t63.nc')

pltlevel = np.concatenate(
    (np.arange(-50, 0, 5), np.arange(0, 10 + 1e-4, 1)))
pltticks = np.concatenate(
    (np.arange(-50, 0, 10), np.arange(0, 10 + 1e-4, 2)))
pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
pltticks2 = np.arange(-20, 20 + 1e-4, 5)

fig, ax = globe_plot()

plt_cmp = ax.pcolormesh(
    wisosw_d.wisosw_d.lon,
    wisosw_d.wisosw_d.lat,
    wisosw_d.wisosw_d[2],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
fig.savefig('figures/test.png')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create lig bias corrected boundary conditions

# wrong: ensure no negative SIC
# ! cdo -add /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final_t63_bias_corrected.nc

# ! cdo -add /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63_bias_corrected.nc

# ! cdo -add /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/mw/T63_wisosw_d.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.pi_final.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63_bias_corrected.nc


'''
#---------------- check
boundary_conditions['sst']['lig_final_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final_t63_bias_corrected.nc')
boundary_conditions['sic']['lig_final_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63_bias_corrected.nc')
boundary_conditions['wisosw_d']['lig_final_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63_bias_corrected.nc')

data1 = (boundary_conditions['sst']['pi'].sst.values + boundary_conditions['sst']['lig_final'].sst.values - boundary_conditions['sst']['pi_final'].sst.values)
data2 = boundary_conditions['sst']['lig_final_bias_corrected'].sst.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]


data1 = (boundary_conditions['sic']['pi'].sic.values + boundary_conditions['sic']['lig_final'].sic.values - boundary_conditions['sic']['pi_final'].sic.values)
data2 = boundary_conditions['sic']['lig_final_bias_corrected'].sic.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]


data1 = (boundary_conditions['wisosw_d']['pi'].wisosw_d.values + boundary_conditions['wisosw_d']['lig_final'].wisosw_d.values - boundary_conditions['wisosw_d']['pi_final'].wisosw_d.values)
data2 = boundary_conditions['wisosw_d']['lig_final_bias_corrected'].wisosw_d.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]

! cdo -add /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/mw/T63_wisosw_d.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63_bias_corrected_1.nc

import xarray as xr
data1 = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63_bias_corrected.nc')
data2 = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final_t63_bias_corrected_1.nc')

(data1.wisosw_d.values[np.isfinite(data1.wisosw_d.values)] == data2.wisosw_d.values[np.isfinite(data2.wisosw_d.values)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lgm SST

#-------- basic set
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.1.0 bc sst from pi, pi_final, and lgm_50.png'

pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-5, 5 + 1e-4, 0.5)
pltticks2 = np.arange(-5, 5 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot
itime = 11

cbar_label1 = month[itime] + ' SST [$°C$]'
cbar_label2 = 'Differences in ' + month[itime] + ' SST [$°C$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi_final'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['lgm_50'].sst[itime] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi_final'].sst[itime] - boundary_conditions['sst']['pi'].sst[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['lgm_50'].sst[itime] - boundary_conditions['sst']['pi_final'].sst[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['sst']['pi'].sst[itime].lon,
    boundary_conditions['sst']['pi'].sst[itime].lat,
    boundary_conditions['sst']['pi'].sst[itime] + boundary_conditions['sst']['lgm_50'].sst[itime].values - boundary_conditions['sst']['pi_final'].sst[itime].values - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50 - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lgm_50 - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


#-------------------------------- animate

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt1 = axs[0, 0].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[0, 1].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi_final'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt3 = axs[0, 2].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['lgm_50'].sst[itime] - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt4 = axs[1, 0].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi_final'].sst[itime] - boundary_conditions['sst']['pi'].sst[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt5 = axs[1, 1].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['lgm_50'].sst[itime] - boundary_conditions['sst']['pi_final'].sst[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt6 = axs[1, 2].pcolormesh(
        boundary_conditions['sst']['pi'].sst[itime].lon,
        boundary_conditions['sst']['pi'].sst[itime].lat,
        boundary_conditions['sst']['pi'].sst[itime] + boundary_conditions['sst']['lgm_50'].sst[itime].values - boundary_conditions['sst']['pi_final'].sst[itime].values - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt7 = plt.text(
        0.5, -0.1, month[itime], transform=axs[1, 1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6, plt7]
    
    return(plt_objs)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50 - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lgm_50 - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar_label1 = 'SST [$°C$]'
cbar_label2 = 'Differences in SST [$°C$]'


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)

ani = animation.FuncAnimation(
    fig, update_frames, frames=12, interval=250, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.1.1 bc sst from pi, pi_final, and lgm_50.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lgm sic

#-------- basic set
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.1.2 bc sic from pi, pi_final, and lgm_50.png'

pltlevel = np.arange(0, 100 + 1e-4, 10)
pltticks = np.arange(0, 100 + 1e-4, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

pltlevel2 = np.arange(-50, 50 + 1e-4, 5)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot
itime = 11

cbar_label1 = month[itime] + ' sea ice concentration [$\%$]'
cbar_label2 = 'Differences in ' + month[itime] + ' sea ice concentration [$\%$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi_final'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['lgm_50'].sic[itime],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi_final'].sic[itime] - boundary_conditions['sic']['pi'].sic[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['lgm_50'].sic[itime] - boundary_conditions['sic']['pi_final'].sic[itime].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['sic']['pi'].sic[itime].lon,
    boundary_conditions['sic']['pi'].sic[itime].lat,
    boundary_conditions['sic']['pi'].sic[itime] + boundary_conditions['sic']['lgm_50'].sic[itime].values - boundary_conditions['sic']['pi_final'].sic[itime].values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50 - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lgm_50 - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


#-------------------------------- animate

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt1 = axs[0, 0].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[0, 1].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi_final'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt3 = axs[0, 2].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['lgm_50'].sic[itime],
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt4 = axs[1, 0].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi_final'].sic[itime] - boundary_conditions['sic']['pi'].sic[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt5 = axs[1, 1].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['lgm_50'].sic[itime] - boundary_conditions['sic']['pi_final'].sic[itime].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    plt6 = axs[1, 2].pcolormesh(
        boundary_conditions['sic']['pi'].sic[itime].lon,
        boundary_conditions['sic']['pi'].sic[itime].lat,
        boundary_conditions['sic']['pi'].sic[itime] + boundary_conditions['sic']['lgm_50'].sic[itime].values - boundary_conditions['sic']['pi_final'].sic[itime].values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    plt7 = plt.text(
        0.5, -0.1, month[itime], transform=axs[1, 1].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6, plt7]
    
    return(plt_objs)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

plt.text(
    0.5, 1.05, 'AMIP pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - AMIP pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50 - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'AMIP pi + lgm_50 - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar_label1 = 'Sea ice concentration [$\%$]'
cbar_label2 = 'Differences in sea ice concentration [$\%$]'

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)

ani = animation.FuncAnimation(
    fig, update_frames, frames=12, interval=250, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.1.3 bc sic from pi, pi_final, and lgm_50.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lgm wisosw_d

#-------- basic set

water_isotope = '$\delta^{18}O$'
ilev = 1
pltlevel = np.concatenate(
    (np.arange(-6, 0, 0.5), np.arange(0, 2.4 + 1e-4, 0.2)))
pltticks = np.concatenate(
    (np.arange(-6, 0, 1), np.arange(0, 2.4 + 1e-4, 0.4)))
pltlevel2 = np.arange(-2, 2 + 1e-4, 0.25)
pltticks2 = np.arange(-2, 2 + 1e-4, 0.5)

# water_isotope = '$\delta D$'
# ilev = 2
# pltlevel = np.concatenate(
#     (np.arange(-50, 0, 5), np.arange(0, 10 + 1e-4, 1)))
# pltticks = np.concatenate(
#     (np.arange(-50, 0, 10), np.arange(0, 10 + 1e-4, 2)))
# pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
# pltticks2 = np.arange(-20, 20 + 1e-4, 5)

#-------- plot configuration
mpl.rc('font', family='Times New Roman', size=10)
output_png = 'figures/6_awi/6.1_echam6/6.1.5_boundary_conditions/6.1.5.1.4.0 bc '+ water_isotope + ' from pi, pi_final, and lgm_50.png'

pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1)

nrow = 2
ncol = 3
fm_bottom = 2 / (4.6*nrow + 2.5)


#-------------------------------- plot

cbar_label1 = water_isotope + ' of surface sea water [$‰$]'
cbar_label2 = 'Differences in ' + water_isotope + ' of surface sea water [$‰$]'

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

axs[0, 0].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['lgm_50'].wisosw_d[ilev],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1, 0].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev] - boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['lgm_50'].wisosw_d[ilev] - boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev].values,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lon,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev].lat,
    boundary_conditions['wisosw_d']['pi'].wisosw_d[ilev] + boundary_conditions['wisosw_d']['lgm_50'].wisosw_d[ilev].values - boundary_conditions['wisosw_d']['pi_final'].wisosw_d[ilev].values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'pi', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi_final', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'pi_final - pi', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'lgm_50 - pi_final', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'pi + lgm_50 - pi_final', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.9),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.95)
fig.savefig(output_png)


'''
wisosw_d = xr.open_dataset('/home/ollie/xshi/for_Janina/wisosw_d_lgm_t63.nc')

pltlevel = np.concatenate(
    (np.arange(-50, 0, 5), np.arange(0, 10 + 1e-4, 1)))
pltticks = np.concatenate(
    (np.arange(-50, 0, 10), np.arange(0, 10 + 1e-4, 2)))
pltlevel2 = np.arange(-20, 20 + 1e-4, 2.5)
pltticks2 = np.arange(-20, 20 + 1e-4, 5)

pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1)

fig, ax = globe_plot()

plt_cmp = ax.pcolormesh(
    wisosw_d.wisosw_d.lon,
    wisosw_d.wisosw_d.lat,
    wisosw_d.wisosw_d[2],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
fig.savefig('figures/test.png')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create lgm bias corrected boundary conditions

! cdo -add /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50_t63_bias_corrected.nc


! cdo -add /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63_bias_corrected.nc


! cdo -add /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/wisosw_d.echam.2200_2229.lgm-50_t63.nc -sub /work/ollie/qigao001/startdump/model_input/pi/mw/T63_wisosw_d.nc /work/ollie/qigao001/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.pi_final_t63.nc /work/ollie/qigao001/startdump/model_input/lgm/lgm-50/wisosw_d.echam.2200_2229.lgm-50_t63_bias_corrected.nc



'''

#---------------- check
boundary_conditions['sst']['lgm_50_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50_t63_bias_corrected.nc')
boundary_conditions['sic']['lgm_50_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63_bias_corrected.nc')
boundary_conditions['wisosw_d']['lgm_50_bias_corrected'] = xr.open_dataset('/work/ollie/qigao001/startdump/model_input/lgm/lgm-50/wisosw_d.echam.2200_2229.lgm-50_t63_bias_corrected.nc')

data1 = (boundary_conditions['sst']['pi'].sst.values + boundary_conditions['sst']['lgm_50'].sst.values - boundary_conditions['sst']['pi_final'].sst.values)
data2 = boundary_conditions['sst']['lgm_50_bias_corrected'].sst.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]


data1 = (boundary_conditions['sic']['pi'].sic.values + boundary_conditions['sic']['lgm_50'].sic.values - boundary_conditions['sic']['pi_final'].sic.values)
data2 = boundary_conditions['sic']['lgm_50_bias_corrected'].sic.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]


data1 = (boundary_conditions['wisosw_d']['pi'].wisosw_d.values + boundary_conditions['wisosw_d']['lgm_50'].wisosw_d.values - boundary_conditions['wisosw_d']['pi_final'].wisosw_d.values)
data2 = boundary_conditions['wisosw_d']['lgm_50_bias_corrected'].wisosw_d.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
test = data1[np.isfinite(data1)] - data2[np.isfinite(data2)]
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
data1[np.isfinite(data1)][wheremax]
data2[np.isfinite(data2)][wheremax]

'''
# endregion
# -----------------------------------------------------------------------------



