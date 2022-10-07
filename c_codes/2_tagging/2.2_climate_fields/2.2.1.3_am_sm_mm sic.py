

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
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec_num,
    month_dec,
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

ds_names = ['pi', 'pi_final', 'lig_final', 'lgm_50']

ds_files = [
    'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc',
    'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc',
    'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63.nc',
    'startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63.nc'
]

boundary_conditions = {}
boundary_conditions['sic'] = {}

for inames, ifiles in zip(ds_names, ds_files):
    boundary_conditions['sic'][inames] = xr.open_dataset(ifiles)
    
    print(inames + '\n' + ifiles)

lon = boundary_conditions['sic']['pi'].lon
lat = boundary_conditions['sic']['pi'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    boundary_conditions['sic']['pi'].sic.shape,)
boundary_conditions['sic']['pi'].sic.values[b_slm] = np.nan
boundary_conditions['sic']['pi'].sic.values[:] = \
    boundary_conditions['sic']['pi'].sic.clip(0, 100, keep_attrs=True).values

echam6_t63_cellarea = xr.open_dataset(
    'scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

'''
stats.describe(boundary_conditions['sic']['pi'].sic.values,
               axis=None, nan_policy='omit')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate sm/am sic


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
#-------- check

# sm
ikeys = 'pi'
ilon = 30
ilat = 82
boundary_conditions['sm_sic'][ikeys][2, ilat, ilon].values
np.average(boundary_conditions['sic'][ikeys].sic[2:5, ilat, ilon],
           weights = boundary_conditions['sic'][ikeys].sic[2:5, ilat, ilon].time.dt.days_in_month)

# am
ikeys = 'pi'
ilon = 40
ilat = 82
boundary_conditions['am_sic'][ikeys][ilat, ilon]
np.average(boundary_conditions['sic'][ikeys].sic[:, ilat, ilon],
           weights = boundary_conditions['sic'][ikeys].sic[:, ilat, ilon].time.dt.days_in_month)



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm sic Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.1_sic/6.1.2.1 amip_pi sic am_sm Antarctica.png'
cbar_label1 = 'SIC [$\%$]'
cbar_label2 = 'Differences in SIC [$\%$]'

pltlevel = np.arange(0, 100 + 1e-4, 10)
pltticks = np.arange(0, 100 + 1e-4, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

pltlevel2 = np.arange(-100, 100 + 1e-4, 10)
pltticks2 = np.arange(-100, 100 + 1e-4, 20)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(northextent=-45, ax_org = axs[irow, jcol])
            cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[irow, jcol])
            plt.text(
                0, 0.95, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='center', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat,
    boundary_conditions['am_sic']['pi'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh2 = axs[0, 1].pcolormesh(
    lon, lat,
    boundary_conditions['sm_sic']['pi'].sel(season='DJF') - boundary_conditions['sm_sic']['pi'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat,
    boundary_conditions['sm_sic']['pi'].sel(season='MAM') - boundary_conditions['sm_sic']['pi'].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

for jcol in range(ncol):
    #-------- sm
    axs[1, jcol].pcolormesh(
        lon, lat,
        boundary_conditions['sm_sic']['pi'].sel(
            season=seasons[jcol]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #-------- sm/am - 1
    axs[2, jcol].pcolormesh(
        lon, lat,
        boundary_conditions['sm_sic']['pi'].sel(season=seasons[jcol]) - boundary_conditions['am_sic']['pi'],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    print(seasons[jcol])


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

for jcol in range(ncol):
    plt.text(
        0.5, 1.05, seasons[jcol], transform=axs[1, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[jcol] + ' - Annual mean',
        transform=axs[2, jcol].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs, format=remove_trailing_zero_pos,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate monthly sea ice extent

seaice_extent = {}

for inames in ds_names:
    # inames = 'pi'
    seaice_extent[inames] = {}
    seaice_extent[inames]['mm'] = xr.DataArray(
        data = np.zeros(12),
        coords={'month': np.arange(1, 12+1e-4, 1)},
    )

    for imonth in range(12):
        # imonth = 0
        seaice_extent[inames]['mm'].values[imonth] = \
            (echam6_t63_cellarea.cell_area.values[
                (lat_2d < 0) & \
                    (boundary_conditions['sic'][inames].sic[imonth].values > 15)
            ]).sum() / 1e12
    
    print(inames)




'''
boundary_conditions['sic']['pi'].sic.to_netcdf('scratch/test/test.nc')
seaice_extent['pi']['mm']
seaice_extent['pi_final']['mm']
seaice_extent['lig_final']['mm']
seaice_extent['lgm_50']['mm']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sea ice extent

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.1_sic/6.1.2.1 amip_pi seaice_extent mm SH.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.set_ylim(0, 46)
ax.set_yticks(np.arange(0, 45 + 1e-4, 5))

plt_line1 = plt.plot(
    month, seaice_extent['pi']['mm'],
    lw=1, ls='-', c='grey', marker='o', ms = 2, label='AMIP PI')
plt_line2 = plt.plot(
    month, seaice_extent['pi_final']['mm'],
    lw=1, ls=':', c='m', marker='o', ms = 2, label='pi_final')
plt_line3 = plt.plot(
    month, seaice_extent['lig_final']['mm'],
    lw=1, ls=':', c='r', marker='o', ms = 2, label='lig_final')
plt_line4 = plt.plot(
    month, seaice_extent['lgm_50']['mm'],
    lw=1, ls=':', c='b', marker='o', ms = 2, label='lgm_50')

plt.legend()

ax.set_xlabel('Antarctic sea ice extent [$10^6 \; km^2$]')

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.99)
fig.savefig(output_png)


'''
# ax.set_xlabel('Daily precipitation [$mm \; day^{-1}$]')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am sic Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.1_sic/6.1.2.1 amip_pi sic am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    boundary_conditions['am_sic']['pi'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Sea ice concentration (SIC) [$\%$]', linespacing=1.5,)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------





