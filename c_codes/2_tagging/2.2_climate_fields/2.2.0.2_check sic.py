

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

ds_names = ['pi', 'pi_final', 'lig_final', 'lgm_50']

boundary_conditions = {}

pi_sic_file = 'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc'
pi_final_sic_file = 'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc'
lig_final_sic_file = 'startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63.nc'
lgm_50_sic_file = 'startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63.nc'

boundary_conditions['sic'] = {}
boundary_conditions['sic']['pi'] = xr.open_dataset(pi_sic_file)
boundary_conditions['sic']['pi_final'] = xr.open_dataset(pi_final_sic_file)
boundary_conditions['sic']['lig_final'] = xr.open_dataset(lig_final_sic_file)
boundary_conditions['sic']['lgm_50'] = xr.open_dataset(lgm_50_sic_file)

lon = boundary_conditions['sic']['pi'].lon
lat = boundary_conditions['sic']['pi'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]


# set land points as np.nan

esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')

b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    boundary_conditions['sic']['pi'].sic.shape,
    )

boundary_conditions['sic']['pi'].sic.values[b_slm] = np.nan

boundary_conditions['sic']['pi'].sic.values[:] = \
    boundary_conditions['sic']['pi'].sic.clip(0, 100, keep_attrs=True).values


'''
stats.describe(boundary_conditions['sic']['pi'].sic.values,
               axis=None, nan_policy='omit')

stats.describe(boundary_conditions['sic']['lgm_50'].sic.values,
               axis=None, nan_policy='omit')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate am/sm sic

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
ddd = time_weighted_mean(boundary_conditions['sic'][ikeys].sic)

ilat = 84
ilon = 91

np.average(
    boundary_conditions['sic'][ikeys].sic[:, ilat, ilon],
    weights = boundary_conditions['sic'][ikeys].sic[:, ilat, ilon].time.dt.days_in_month,
)

ddd[ilat, ilon].values

ccc = boundary_conditions['sic'][ikeys].sic.weighted(
    boundary_conditions['sic'][ikeys].sic.time.dt.days_in_month
    ).mean(dim='time', skipna=False)
(ddd == ccc).all()


#-------- check sm calculation
ddd = boundary_conditions['sic'][ikeys].sic.groupby(
            'time.season').map(time_weighted_mean)

iseason = 1
ilat = 84
ilon = 91
np.average(
    boundary_conditions['sic'][ikeys].sic[
        (iseason*3 - 1):(iseason*3+2), ilat, ilon],
    weights = boundary_conditions['sic'][ikeys].sic[
        (iseason*3 - 1):(iseason*3+2), ilat, ilon].time.dt.days_in_month
)

ddd[:, ilat, ilon].sel(season=seasons[iseason]).values
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/sm sic Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.1_sic/6.1.2.1 amip_pi pi_final lig_final lgm_50 sic am_sm Antarctica.png'
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
        axs[irow, jcol] = hemisphere_plot(northextent=-40, ax_org = axs[irow, jcol])
        cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# PI
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat,
    boundary_conditions['am_sic']['pi'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

for iseason in range(len(seasons)):
    axs[0, iseason+1].pcolormesh(
        lon, lat,
        boundary_conditions['sm_sic']['pi'].sel(
            season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    print(seasons[iseason])

ds_names_comp = ['pi', 'pi', 'pi_final', 'pi_final',]
for ids in np.arange(1, len(ds_names)):
    print(ds_names[ids] + ' vs. ' + ds_names_comp[ids])
    
    # am diff
    plt_mesh2 = axs[ids, 0].pcolormesh(
        lon, lat,
        boundary_conditions['am_sic'][ds_names[ids]] - \
            boundary_conditions['am_sic'][ds_names_comp[ids]].values,
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
    
    # sm diff
    for iseason in range(len(seasons)):
        axs[ids, iseason+1].pcolormesh(
            lon, lat,
            boundary_conditions['sm_sic'][ds_names[ids]].sel(
                season=seasons[iseason]) - \
                    boundary_conditions['sm_sic'][ds_names_comp[ids]].sel(
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
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=1.5)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(1.1,-4),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=1.5)


fig.subplots_adjust(left=0.02, right = 0.995, bottom = fm_bottom, top = 0.98)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------



