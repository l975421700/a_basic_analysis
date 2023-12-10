

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp
from scipy.interpolate import UnivariateSpline

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

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
    regrid,
    find_ilat_ilon,
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
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

# HadCM3
with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime = pickle.load(f)

lon = hadcm3_output_regridded_alltime['PI']['SST']['am'].lon.values
lat = hadcm3_output_regridded_alltime['PI']['SST']['am'].lat.values

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime_2000.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime_2000 = pickle.load(f)

AMOC_xppfa_3000yrs = xr.open_dataset('/gws/nopw/j04/bas_palaeoclim/rahul/data/BAS_HADCM3_data/data_qingang/xppfa/29nov23/AMOC_xppfa_3000yrs.nc')

with open('scratch/share/from_rahul/data_qingang/hadcm3_output.pkl', 'rb') as f:
    hadcm3_output = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare 2k and 3k simulation

output_png = 'figures/7_lig/7.1_hadcm3/7.1.2.0 2k and 3k HadCM3 SST, SAT and SIC.png'

nrow = 3
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)
northextents = [-38, -60, -50,]

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.06, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=northextents[jcol], ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='top', rotation='horizontal')
        ipanel += 1

for jcol, ivar in enumerate(['SST', 'SAT', 'SIC', ]):
    # ivar = 'SST'
    print('#-------------------------------- ' + str(jcol) + ' ' + ivar)
    
    # print(stats.describe(hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'], axis=None, nan_policy='omit'))
    # print(stats.describe(abs(hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'] - hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am']), axis=None, nan_policy='omit'))
    
    if (ivar == 'SAT'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze() - zerok
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze() - zerok
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-50, cm_max=0, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
            cmap='RdBu',)
    elif (ivar == 'SIC'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze() * 100
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze() * 100
        plt_data1.values[plt_data1.values == 0] = np.nan
        plt_data2.values[plt_data2.values == 0] = np.nan
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1,
            cmap='PuOr', reversed=False,)
    elif (ivar == 'SST'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze()
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze()
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=20, cm_interval1=1, cm_interval2=2,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
            cmap='RdBu',)
    
    plt_data3 = plt_data1 - plt_data2
    ttest_fdr_res = ttest_fdr_control(
        hadcm3_output_regridded_alltime['LIG0.25'][ivar]['ann'],
        hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['ann'],)
    
    plt_data3.values[ttest_fdr_res == False] = np.nan
    
    # print(stats.describe(plt_data1.sel(y=slice(0, 52)), axis=None, nan_policy='omit'))
    # print(stats.describe(plt_data3.sel(y=slice(0, 52)), axis=None, nan_policy='omit'))
    
    axs[0, jcol].contourf(
        plt_data1.lon, plt_data1.lat, plt_data1,
        levels=pltlevel, extend='both',
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    axs[1, jcol].contourf(
        plt_data2.lon, plt_data2.lat, plt_data2,
        levels=pltlevel, extend='both',
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    axs[2, jcol].contourf(
        plt_data3.lon, plt_data3.lat, plt_data3,
        levels=pltlevel1, extend='both',
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree())
    
    for irow in range(nrow):
        if (ivar in ['SST', 'SIC']):
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
        elif (ivar == 'SAT'):
            axs[irow, jcol].add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

# # colorbar
# cbar1 = fig.colorbar(
#     plt_mesh1, ax=axs, aspect=20, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.4, ticks=pltticks1, extend='both',
#     anchor=(0.3, -0.8),
#     )
# cbar4 = fig.colorbar(
#     plt_mesh4, ax=axs, aspect=20, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.25, ticks=pltticks4, extend='both',
#     anchor=(1.12, -4.2),
#     )

# x label
plt.text(0.5, -0.08, 'Annual SST [$°C$]', transform=axs[2, 0].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Annual SAT [$°C$]', transform=axs[2, 1].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Annual SIC [$\%$]', transform=axs[2, 2].transAxes,
         ha='center', va='center', rotation='horizontal')

# y label
plt.text(-0.08, 0.5,
         '3000-year',
         transform=axs[0, 0].transAxes, ha='center', va='center',
         rotation='vertical')
plt.text(-0.08, 0.5,
         '2000-year',
         transform=axs[1, 0].transAxes, ha='center', va='center',
         rotation='vertical')
plt.text(-0.08, 0.5,
         'Diff. between 3000-year and 2000-year',
         transform=axs[2, 0].transAxes, ha='center', va='center',
         rotation='vertical')

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare 2k and 3k simulation NH

output_png = 'figures/7_lig/7.1_hadcm3/7.1.2.0 2k and 3k HadCM3 SST, SAT and SIC NH.png'

nrow = 3
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)
southextents = [38, 60, 50,]

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.NorthPolarStereo()},
    gridspec_kw={'hspace': 0.06, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            southextent=southextents[jcol], ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='left', va='top', rotation='horizontal')
        ipanel += 1

for jcol, ivar in enumerate(['SST', 'SAT', 'SIC', ]):
    # ivar = 'SAT'
    print('#-------------------------------- ' + str(jcol) + ' ' + ivar)
    
    # print(stats.describe(hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'], axis=None, nan_policy='omit'))
    # print(stats.describe(abs(hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'] - hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am']), axis=None, nan_policy='omit'))
    
    if (ivar == 'SAT'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze() - zerok
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze() - zerok
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-50, cm_max=0, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
            cmap='RdBu',)
    elif (ivar == 'SIC'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze() * 100
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze() * 100
        plt_data1.values[plt_data1.values == 0] = np.nan
        plt_data2.values[plt_data2.values == 0] = np.nan
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1,
            cmap='PuOr', reversed=False,)
    elif (ivar == 'SST'):
        plt_data1 = hadcm3_output_regridded_alltime['LIG0.25'][ivar]['am'].squeeze()
        plt_data2 = hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['am'].squeeze()
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=20, cm_interval1=1, cm_interval2=2,
            cmap='viridis',)
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1,
            cmap='RdBu',)
    
    plt_data3 = plt_data1 - plt_data2
    ttest_fdr_res = ttest_fdr_control(
        hadcm3_output_regridded_alltime['LIG0.25'][ivar]['ann'],
        hadcm3_output_regridded_alltime_2000['LIG0.25'][ivar]['ann'],)
    
    plt_data3.values[ttest_fdr_res == False] = np.nan
    
    # print(stats.describe(plt_data1.sel(y=slice(0, 52)), axis=None, nan_policy='omit'))
    # print(stats.describe(plt_data3.sel(y=slice(0, 52)), axis=None, nan_policy='omit'))
    
    axs[0, jcol].contourf(
        plt_data1.lon, plt_data1.lat, plt_data1,
        levels=pltlevel, extend='both',
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    axs[1, jcol].contourf(
        plt_data2.lon, plt_data2.lat, plt_data2,
        levels=pltlevel, extend='both',
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    axs[2, jcol].contourf(
        plt_data3.lon, plt_data3.lat, plt_data3,
        levels=pltlevel1, extend='both',
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree())
    
    for irow in range(nrow):
        if (ivar in ['SST', 'SIC']):
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
        elif (ivar == 'SAT'):
            axs[irow, jcol].add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

# # colorbar
# cbar1 = fig.colorbar(
#     plt_mesh1, ax=axs, aspect=20, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.4, ticks=pltticks1, extend='both',
#     anchor=(0.3, -0.8),
#     )
# cbar4 = fig.colorbar(
#     plt_mesh4, ax=axs, aspect=20, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.25, ticks=pltticks4, extend='both',
#     anchor=(1.12, -4.2),
#     )

# x label
plt.text(0.5, -0.08, 'Annual SST [$°C$]', transform=axs[2, 0].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Annual SAT [$°C$]', transform=axs[2, 1].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Annual SIC [$\%$]', transform=axs[2, 2].transAxes,
         ha='center', va='center', rotation='horizontal')

# y label
plt.text(-0.08, 0.5,
         '3000-year',
         transform=axs[0, 0].transAxes, ha='center', va='center',
         rotation='vertical')
plt.text(-0.08, 0.5,
         '2000-year',
         transform=axs[1, 0].transAxes, ha='center', va='center',
         rotation='vertical')
plt.text(-0.08, 0.5,
         'Diff. between 3000-year and 2000-year',
         transform=axs[2, 0].transAxes, ha='center', va='center',
         rotation='vertical')

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region diff betwen simulated site values

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values.pkl', 'rb') as f:
    hadcm3_output_site_values = pickle.load(f)

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_site_values_2000.pkl', 'rb') as f:
    hadcm3_output_site_values_2000 = pickle.load(f)

hadcm3_output_site_values['annual_sst']['JH']['sim_lig0.25Sv_pi'] - hadcm3_output_site_values_2000['annual_sst']['JH']['sim_lig0.25Sv_pi']



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AMOC and SO SST time series

xppfa_SST_cellarea = xr.open_dataset('scratch/share/from_rahul/data_qingang/xppfa/29nov23/xppfa_SST_cellarea.nc')

SO_40_60_SST = hadcm3_output['LIG0.25']['SST'].sel(lat=slice(-60, -40)).weighted(xppfa_SST_cellarea.cell_area.sel(lat=slice(-60, -40))).mean(dim=('lat', 'lon')).resample(time='1Y').mean(dim='time')

output_png = 'figures/7_lig/7.1_hadcm3/7.1.2.1 time series of AMOC and SO SST.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([10, 8]) / 2.54)

ax.plot(
    np.arange(0, 3000, 1),
    AMOC_xppfa_3000yrs.merid_Atlantic_ym_dpth.sel(time=slice('2754-06-01', '5754-06-02')),
    lw=0.2, c='tab:blue'
    )

ax.set_xlim(-100, 3100)
ax.set_ylim(0, 16)
ax.set_xticks(np.arange(0, 3000+1e-4, 500))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Modelling year')
ax.set_ylabel('AMOC [$Sv$]', c='tab:blue')

ax.grid(True, which='both',
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

ax2 = ax.twinx()
ax2.plot(
    np.arange(0, 3000, 1),
    SO_40_60_SST,
    lw=0.2, c = 'tab:orange'
)
ax2.set_ylim(8, 9.6)
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax2.set_ylabel('Annual averaged SST between 40-60$°\;S$ [$°C$]', c = 'tab:orange')

fig.subplots_adjust(left=0.14, right=0.86, bottom=0.15, top=0.97)
fig.savefig(output_png)




'''
# fit a spline
spl = UnivariateSpline(
    np.arange(0, 3000, 1),
    AMOC_xppfa_3000yrs.merid_Atlantic_ym_dpth.sel(time=slice('2754-06-01', '5754-06-02')),
    k=5)
xspl = np.linspace(0, 3000, 30000)
ax.plot(xspl, spl(xspl), lw=0.5,)

hadcm3_output_regridded_alltime['LIG0.25']['SST']['am'].to_netcdf('scratch/test/test0.nc')

hadcm3_output['LIG0.25']['SST'].sel(lat=slice(-60, -40)).mean(dim=('lat', 'lon')).resample(time='1Y').mean(dim='time')

'''
# endregion
# -----------------------------------------------------------------------------
