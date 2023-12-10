

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
    time_weighted_mean,
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
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

boundary_conditions = {}
boundary_conditions['sst'] = {}
boundary_conditions['sst']['pi'] = xr.open_dataset(
    'startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc')
boundary_conditions['sic'] = {}
boundary_conditions['sic']['pi'] = xr.open_dataset(
    'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')

lon = boundary_conditions['sst']['pi'].lon
lat = boundary_conditions['sst']['pi'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

# set land points as np.nan
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')

b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    boundary_conditions['sst']['pi'].sst.shape,
    )

boundary_conditions['sst']['pi'].sst.values[b_slm] = np.nan
boundary_conditions['sic']['pi'].sic.values[b_slm] = np.nan
boundary_conditions['sic']['pi'].sic.values[:] = \
    boundary_conditions['sic']['pi'].sic.clip(0, 100, keep_attrs=True).values

'''
# echam6_t63_slm = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_slm.nc')
    # echam6_t63_slm.SLM.values == 1,

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate am/sm sst

boundary_conditions['am_sst'] = {}
boundary_conditions['sm_sst'] = {}

for ikeys in boundary_conditions['sst'].keys():
    # ikeys = 'pi'
    print(ikeys)
    boundary_conditions['am_sst'][ikeys] = \
        time_weighted_mean(boundary_conditions['sst'][ikeys].sst)
    boundary_conditions['sm_sst'][ikeys] = \
        boundary_conditions['sst'][ikeys].sst.groupby(
            'time.season').map(time_weighted_mean)

boundary_conditions['am_sic'] = {}
boundary_conditions['sm_sic'] = {}

for ikeys in boundary_conditions['sic'].keys():
    # ikeys = 'pi'
    print(ikeys)
    boundary_conditions['am_sic'][ikeys] = \
        time_weighted_mean(boundary_conditions['sic'][ikeys].sic)
    boundary_conditions['sm_sic'][ikeys] = \
        boundary_conditions['sic'][ikeys].sic.groupby(
            'time.season').map(time_weighted_mean)

'''
#-------- check am calculation
ddd = time_weighted_mean(boundary_conditions['sst'][ikeys].sst)

ilat = 45
ilon = 90

np.average(
    boundary_conditions['sst'][ikeys].sst[:, ilat, ilon],
    weights = boundary_conditions['sst'][ikeys].sst[:, ilat, ilon].time.dt.days_in_month,
)

ddd[ilat, ilon].values

ccc = boundary_conditions['sst'][ikeys].sst.weighted(
    boundary_conditions['sst'][ikeys].sst.time.dt.days_in_month
    ).mean(dim='time', skipna=False)
(ddd == ccc).all()


#-------- check sm calculation
ddd = boundary_conditions['sst'][ikeys].sst.groupby(
            'time.season').map(time_weighted_mean)

iseason = 1
ilat = 48
ilon = 96
np.average(
    boundary_conditions['sst'][ikeys].sst[
        (iseason*3 - 1):(iseason*3+2), ilat, ilon],
    weights = boundary_conditions['sst'][ikeys].sst[
        (iseason*3 - 1):(iseason*3+2), ilat, ilon].time.dt.days_in_month
)

ddd[:, ilat, ilon].sel(season=seasons[iseason]).values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm SST globe


output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.0_sst/6.1.2.0 amip_pi pi_final lig_final lgm_50 sst am_sm.png'
cbar_label1 = 'SST [$°C$]'
cbar_label2 = 'Differences in SST [$°C$]'

pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 4
ncol = 5
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol], add_grid_labels=False)
        plt.text(
            0, 1.05, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='center', rotation='horizontal')
        ipanel += 1

# PI
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat,
    boundary_conditions['am_sst']['pi'] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    axs[0, iseason+1].pcolormesh(
        lon, lat,
        boundary_conditions['sm_sst']['pi'].sel(
            season=seasons[iseason]) - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    print(seasons[iseason])

ds_names_comp = ['pi', 'pi', 'pi_final', 'pi_final',]
for ids in np.arange(1, len(ds_names)):
    print(ds_names[ids] + ' vs. ' + ds_names_comp[ids])
    
    # am diff
    plt_mesh2 = axs[ids, 0].pcolormesh(
        lon, lat,
        boundary_conditions['am_sst'][ds_names[ids]] - \
            boundary_conditions['am_sst'][ds_names_comp[ids]].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    # sm diff
    for iseason in range(len(seasons)):
        axs[ids, iseason+1].pcolormesh(
            lon, lat,
            boundary_conditions['sm_sst'][ds_names[ids]].sel(
                season=seasons[iseason]) - \
                    boundary_conditions['sm_sst'][ds_names_comp[ids]].sel(
                        season=seasons[iseason]).values,
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        print(seasons[iseason])

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
for iseason in range(len(seasons)):
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[0, iseason + 1].transAxes,
        ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'AMIP PI', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'pi_final - AMIP PI', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'lig_final - pi_final', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'lgm-50 - pi_final', transform=axs[3, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=1.5)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.6),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=1.5)

fig.subplots_adjust(left=0.015, right = 0.99, bottom = fm_bottom * 0.8, top = 0.97)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm SST Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.0_sst/6.1.2.0 amip_pi pi_final lig_final lgm_50 sst am_sm Antarctica.png'
cbar_label1 = 'SST [$°C$]'
cbar_label2 = 'Differences in SST [$°C$]'

pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 4
ncol = 5
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.02, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-20, ax_org = axs[irow, jcol])
        cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# PI
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat,
    boundary_conditions['am_sst']['pi'] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    axs[0, iseason+1].pcolormesh(
        lon, lat,
        boundary_conditions['sm_sst']['pi'].sel(
            season=seasons[iseason]) - zerok,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    print(seasons[iseason])

ds_names_comp = ['pi', 'pi', 'pi_final', 'pi_final',]
for ids in np.arange(1, len(ds_names)):
    print(ds_names[ids] + ' vs. ' + ds_names_comp[ids])
    
    # am diff
    plt_mesh2 = axs[ids, 0].pcolormesh(
        lon, lat,
        boundary_conditions['am_sst'][ds_names[ids]] - \
            boundary_conditions['am_sst'][ds_names_comp[ids]].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    # sm diff
    for iseason in range(len(seasons)):
        axs[ids, iseason+1].pcolormesh(
            lon, lat,
            boundary_conditions['sm_sst'][ds_names[ids]].sel(
                season=seasons[iseason]) - \
                    boundary_conditions['sm_sst'][ds_names_comp[ids]].sel(
                        season=seasons[iseason]).values,
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        print(seasons[iseason])

plt.text(
    0.5, 1.05, 'Annual mean',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
for iseason in range(len(seasons)):
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[0, iseason + 1].transAxes,
        ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'AMIP PI', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'pi_final - AMIP PI', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'lig_final - pi_final', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'lgm-50 - pi_final', transform=axs[3, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=1.5)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=1.5)


fig.subplots_adjust(left=0.02, right = 0.995, bottom = fm_bottom, top = 0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am sst Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.0_sst/6.1.2.0 amip_pi sst am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=4, cmap='RdBu',)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    boundary_conditions['am_sst']['pi'] - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Sea surface temperature (SST) [$°C$]', linespacing=1.5,)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot DJF-JJA sst Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.0_sst/6.1.2.0 amip_pi sst DJF-JJA Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=6, cm_interval1=1, cm_interval2=1, cmap='Oranges',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    boundary_conditions['sm_sst']['pi'].sel(season='DJF') - \
        boundary_conditions['sm_sst']['pi'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('DJF - JJA SST [$°C$]', linespacing=1.5,)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am sic/sst/rh2m/wind10

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0

rh2m_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.rh2m_alltime.pkl', 'rb') as f:
    rh2m_alltime[expid[i]] = pickle.load(f)

wind10_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_alltime.pkl', 'rb') as f:
    wind10_alltime[expid[i]] = pickle.load(f)


output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2 am sic_sst_rh2m_wind10 Antarctica.png'

nrow = 1
ncol = 4

wspace = 0.01
hspace = 0.05
fm_left = 0.001
fm_bottom = hspace / nrow
fm_right = 0.999
fm_top = 0.98

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 7.8*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    )

northextents = [-50, -20, -20, -20]
ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(
            northextent=northextents[jcol], ax_org = axs[jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
        
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# am sic
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues',
    reversed=False)

# plt_mesh1 = axs[0].pcolormesh(
#     lon, lat,
#     boundary_conditions['am_sic']['pi'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, boundary_conditions['am_sic']['pi'], axs[0],
    pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=axs[0], aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.05,
    )
cbar.ax.set_xlabel('SIC [$\%$]', linespacing=1.5,)

# am sst
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)

# plt_mesh1 = axs[1].pcolormesh(
#     lon, lat,
#     boundary_conditions['am_sst']['pi'] - zerok,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, boundary_conditions['am_sst']['pi'] - zerok, axs[1],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=axs[1], aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.set_xlabel('SST [$°C$]', linespacing=1.5,)

# am rh2m
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=60, cm_max=110, cm_interval1=5, cm_interval2=10, cmap='cividis',
    reversed=False)

# plt_mesh1 = axs[2].pcolormesh(
#     lon, lat,
#     rh2m_alltime[expid[i]]['am'] * 100,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, rh2m_alltime[expid[i]]['am'] * 100, axs[2],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=axs[2], aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.set_xlabel('rh2m [$\%$]', linespacing=1.5,)


# am wind10
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=1, cm_max=13, cm_interval1=1, cm_interval2=2, cmap='magma_r')

# plt_mesh1 = axs[3].pcolormesh(
#     lon, lat,
#     wind10_alltime[expid[i]]['am'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh1 = plot_t63_contourf(
    lon, lat, wind10_alltime[expid[i]]['am'], axs[3],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=axs[3], aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.05,
    )
cbar.ax.set_xlabel('vel10 [$m\;s^{-1}$]',
                   linespacing=1.5,)

fig.subplots_adjust(
    left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
    wspace=wspace, hspace=hspace,
    )

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

