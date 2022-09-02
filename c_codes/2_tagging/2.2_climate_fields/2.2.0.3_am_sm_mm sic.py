

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

boundary_conditions = {}
pi_sic_file = 'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc'

boundary_conditions['sic'] = {}
boundary_conditions['sic']['pi'] = xr.open_dataset(pi_sic_file)

lon = boundary_conditions['sic']['pi'].lon
lat = boundary_conditions['sic']['pi'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]


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

seaice = {}
seaice['pi_alex'] = xr.open_dataset('startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')

echam6_t63_slm = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_slm.nc')

b_slm = np.broadcast_to(echam6_t63_slm.SLM.values == 1,
                        seaice['pi_alex'].sic.shape,)
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
stats.describe(seaice['pi_alex'].sic, axis=None, nan_policy='omit')
stats.describe(seaice['pi_alex_alltime']['mm'], axis=None, nan_policy='omit')

np.unique(echam6_t63_slm.SLM.values)

#-------- check
# sm
# seaice['pi_alex_alltime'] = {}

seaice['pi_alex_alltime']['mm'] = seaice['pi_alex'].sic.clip(0, 100, keep_attrs=True)
seaice['pi_alex_alltime']['mm'].values[b_slm] = np.nan


seaice['pi_alex_alltime']['sm'] = seaice['pi_alex_alltime']['mm'].groupby('time.season').map(time_weighted_mean)
seaice['pi_alex_alltime']['am'] = time_weighted_mean(seaice['pi_alex_alltime']['mm'])

ilon = 30
ilat = 82
seaice['pi_alex_alltime']['sm'][2, ilat, ilon]
seaice['pi_alex_alltime']['mm'][2:5, ilat, ilon].mean()
np.average(seaice['pi_alex_alltime']['mm'][2:5, ilat, ilon],
           weights = seaice['pi_alex_alltime']['mm'][2:5, ilat, ilon].time.dt.days_in_month)

# am
ilon = 40
ilat = 82
seaice['pi_alex_alltime']['am'][ilat, ilon]
seaice['pi_alex_alltime']['mm'][:, ilat, ilon].mean()
np.average(seaice['pi_alex_alltime']['mm'][:, ilat, ilon],
           weights = seaice['pi_alex_alltime']['mm'][:, ilat, ilon].time.dt.days_in_month)


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
# region plot mm sic Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.4_climate_fields/6.1.4.4 pi_alex sic mm Antarctica.png'
cbar_label2 = 'Differences in sea ice concentration [$\%$]'

pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-45, ax_org = axs[irow, jcol])

for jcol in range(ncol):
    for irow in range(nrow):
        # irow=0; jcol=0
        plt_mesh1 = axs[irow, jcol].pcolormesh(
            lon, lat,
            seaice['pi_alex_alltime']['mm'].sel(time=(seaice['pi_alex_alltime']['mm'].time.dt.month == month_dec_num[jcol*3+irow])).squeeze() - seaice['pi_alex_alltime']['am'],
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar2 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------



