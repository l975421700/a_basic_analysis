

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

with open('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl', 'rb') as f:
    hadcm3_output_regridded_alltime = pickle.load(f)

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)

lon = hadcm3_output_regridded_alltime['PI']['SST']['am'].lon.values
lat = hadcm3_output_regridded_alltime['PI']['SST']['am'].lat.values

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)

with open('data_sources/LIG/HadISST1.1/HadISST_sic.pkl', 'rb') as f:
    HadISST['sic'] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot LIG-PI maps

output_png = 'figures/7_lig/7.1_hadcm3/7.1.0.0 LIG, LIG0.25_PI SST, SAT and SIC 1deg.png'

cbar_label1 = 'Annual SST [$°C$]'
cbar_label2 = 'Summer SST [$°C$]'
cbar_label3 = 'Annual SAT [$°C$]'
cbar_label4 = 'Sep SIC [$\%$]'

pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)
pltlevel3, pltticks3, pltnorm3, pltcmp3 = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)
pltlevel4, pltticks4, pltnorm4, pltcmp4 = plt_mesh_pars(
    cm_min=-70, cm_max=70, cm_interval1=10, cm_interval2=20, cmap='PuOr',
    reversed=False, asymmetric=False,)

max_size = 80
scale_size = 16

nrow = 2
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.06, 'wspace': 0.02},)

northextents = [-38, -38, -60, -50]
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
    
    #-------------------------------- 1st column annual SST
    # JH
    axs[irow, 0].scatter(
        x = lig_recs['JH']['SO_ann'].Longitude,
        y = lig_recs['JH']['SO_ann'].Latitude,
        c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
        s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
        lw=0.5, marker='s', edgecolors = 'black', zorder=2,
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)
    
    # EC SST
    axs[irow, 0].scatter(
        x = lig_recs['EC']['SO_ann'].Longitude,
        y = lig_recs['EC']['SO_ann'].Latitude,
        c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
        s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
        lw=0.5, marker='o', edgecolors = 'black', zorder=2,
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)
    
    # DC
    axs[irow, 0].scatter(
        x = lig_recs['DC']['annual_128'].Longitude,
        y = lig_recs['DC']['annual_128'].Latitude,
        c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
        s = max_size - scale_size * 1,
        lw=0.5, marker='v', edgecolors = 'black', zorder=2,
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)
    
    #-------------------------------- 2nd column summer SST
    # JH
    axs[irow, 1].scatter(
        x = lig_recs['JH']['SO_jfm'].Longitude,
        y = lig_recs['JH']['SO_jfm'].Latitude,
        c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
        s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
        lw=0.5, marker='s', edgecolors = 'black', zorder=2,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    # EC SST
    axs[irow, 1].scatter(
        x = lig_recs['EC']['SO_jfm'].Longitude,
        y = lig_recs['EC']['SO_jfm'].Latitude,
        c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
        s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
        lw=0.5, marker='o', edgecolors = 'black', zorder=2,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    # MC
    axs[irow, 1].scatter(
        x = lig_recs['MC']['interpolated'].Longitude,
        y = lig_recs['MC']['interpolated'].Latitude,
        c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
        s = max_size - scale_size * 1.09,
        lw=0.5, marker='^', edgecolors = 'black', zorder=2,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    # DC
    axs[irow, 1].scatter(
        x = lig_recs['DC']['JFM_128'].Longitude,
        y = lig_recs['DC']['JFM_128'].Latitude,
        c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
        s = max_size - scale_size * 1,
        lw=0.5, marker='v', edgecolors = 'black', zorder=2,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    #-------------------------------- 3rd column annual SAT
    axs[irow, 2].scatter(
        x = lig_recs['EC']['AIS_am'].Longitude,
        y = lig_recs['EC']['AIS_am'].Latitude,
        c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
        s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
        lw=0.5, marker='o', edgecolors = 'black', zorder=2,
        norm=pltnorm3, cmap=pltcmp3, transform=ccrs.PlateCarree(),)
    
    #-------------------------------- 4th column Sep SIC
    axs[irow, 3].scatter(
        x = lig_recs['MC']['interpolated'].Longitude,
        y = lig_recs['MC']['interpolated'].Latitude,
        c = lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'],
        s=64, lw=0.5, marker='^', edgecolors = 'black', zorder=2,
        norm=pltnorm4, cmap=pltcmp4, transform=ccrs.PlateCarree(),)

plt.text(0.5, -0.08, cbar_label1, transform=axs[1, 0].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, cbar_label2, transform=axs[1, 1].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, cbar_label3, transform=axs[1, 2].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, cbar_label4, transform=axs[1, 3].transAxes,
         ha='center', va='center', rotation='horizontal')

for irow, iperiod in zip(range(nrow), ['LIG', 'LIG0.25']):
    print('#-------------------------------- ' + str(irow) + ' ' + iperiod)
    
    #-------------------------------- 1st column annual SST
    ann_lig = hadcm3_output_regridded_alltime[iperiod]['SST']['ann'].values
    ann_pi  = hadcm3_output_regridded_alltime['PI']['SST']['ann'].values
    ttest_fdr_res = ttest_fdr_control(ann_lig, ann_pi,)
    
    am_diff = hadcm3_output_regridded_alltime[iperiod + '_PI']['SST']['am'].squeeze().values.copy()
    am_diff[ttest_fdr_res == False] = np.nan
    
    plt_mesh1 = axs[irow, 0].contourf(
        lon, lat, am_diff, levels=pltlevel1, extend='both',
        norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)
    
    #-------------------------------- 2nd column summer SST
    sea_lig = hadcm3_output_regridded_alltime[iperiod]['SST']['sea'][0::4].values
    sea_pi  = hadcm3_output_regridded_alltime['PI']['SST']['sea'][0::4].values
    ttest_fdr_res = ttest_fdr_control(sea_lig, sea_pi,)
    
    am_diff = hadcm3_output_regridded_alltime[iperiod + '_PI']['SST']['sm'].sel(time=3).values.copy()
    am_diff[ttest_fdr_res == False] = np.nan
    
    plt_mesh2 = axs[irow, 1].contourf(
        lon, lat, am_diff, levels=pltlevel2, extend='both',
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    
    #-------------------------------- 3rd column annual SAT
    ann_lig = hadcm3_output_regridded_alltime[iperiod]['SAT']['ann'].values
    ann_pi  = hadcm3_output_regridded_alltime['PI']['SAT']['ann'].values
    ttest_fdr_res = ttest_fdr_control(ann_lig, ann_pi,)
    
    am_diff = hadcm3_output_regridded_alltime[iperiod + '_PI']['SAT']['am'].squeeze().values.copy()
    am_diff[ttest_fdr_res == False] = np.nan
    
    plt_mesh3 = axs[irow, 2].contourf(
        lon, lat, am_diff, levels=pltlevel3, extend='both',
        norm=pltnorm3, cmap=pltcmp3, transform=ccrs.PlateCarree(),)
    axs[irow, 2].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
    
    #-------------------------------- 4th column Sep SIC
    ann_lig = hadcm3_output_regridded_alltime[iperiod]['SIC']['mon'][8::12].values
    ann_pi  = hadcm3_output_regridded_alltime['PI']['SIC']['mon'][8::12].values
    ttest_fdr_res = ttest_fdr_control(ann_lig, ann_pi,)
    
    am_diff = hadcm3_output_regridded_alltime[iperiod + '_PI']['SIC']['mm'].sel(time=9).values.copy()
    am_diff[ttest_fdr_res == False] = np.nan
    
    plt_mesh4 = axs[irow, 3].contourf(
        lon, lat, am_diff * 100, levels=pltlevel4, extend='both',
        norm=pltnorm4, cmap=pltcmp4, transform=ccrs.PlateCarree(),)
    
    sep_sic = hadcm3_output_regridded_alltime[iperiod]['SIC']['mm'].sel(time=9).values.copy()
    sep_sic[np.isnan(hadcm3_output_regridded_alltime['PI']['SST']['am'].squeeze()).values] = np.nan
    axs[irow, 3].contour(
        lon, lat, sep_sic,
        levels=[0.15], linestyles='dashed', linewidths=0.6, colors='k',
        transform=ccrs.PlateCarree(),)
    sep_sic = hadcm3_output_regridded_alltime['PI']['SIC']['mm'].sel(time=9).values.copy()
    sep_sic[np.isnan(hadcm3_output_regridded_alltime['PI']['SST']['am'].squeeze()).values] = np.nan
    axs[irow, 3].contour(
        lon, lat, sep_sic,
        levels=[0.15], linestyles='solid', linewidths=0.6, colors='k',
        transform=ccrs.PlateCarree(),)
    
    axs[irow, 3].add_feature(
        cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.25, ticks=pltticks1, extend='both',
    anchor=(0.3, -0.45),
    )
cbar4 = fig.colorbar(
    plt_mesh4, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.25, ticks=pltticks4, extend='both',
    anchor=(1.12, -4.2),
    )

plt.text(-0.08, 0.5,
         r'$\mathit{lig127k}$' + ' vs. ' + r'$\mathit{piControl}$',
         transform=axs[0, 0].transAxes, ha='center', va='center',
         rotation='vertical')

plt.text(-0.08, 0.5,
         r'$\mathit{lig127k\_0.25Sv}$' + ' vs. ' + r'$\mathit{piControl}$',
         transform=axs[1, 0].transAxes, ha='center', va='center',
         rotation='vertical')

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)




'''
# cbar2 = fig.colorbar(
#     plt_mesh2, ax=axs, aspect=40, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.25, ticks=pltticks2, extend='both',
#     location='bottom'
#     # anchor=(0.375, -0.4),
#     )
# cbar2.ax.set_xlabel(cbar_label2, linespacing=1.5)
# cbar3 = fig.colorbar(
#     plt_mesh3, ax=axs, aspect=40, format=remove_trailing_zero_pos,
#     orientation="horizontal", shrink=0.25, ticks=pltticks3, extend='both',
#     location='bottom'
#     # anchor=(0.625, -0.4),
#     )
# cbar3.ax.set_xlabel(cbar_label3, linespacing=1.5)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot PI-HadIsst maps

output_png = 'figures/7_lig/7.1_hadcm3/7.1.0.0 PI_HadISST SST and SIC 1deg.png'

cbar_label1 = 'Annual SST [$°C$]'
cbar_label2 = 'Summer SST [$°C$]'
cbar_label3 = 'Sep SIC [$\%$]'

pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='RdBu',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='RdBu',)
pltlevel3, pltticks3, pltnorm3, pltcmp3 = plt_mesh_pars(
    cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=40, cmap='PuOr',
    reversed=False, asymmetric=False,)

ncol = 3
fm_bottom = 2 / (5.8 + 2)

fig, axs = plt.subplots(
    1, ncol, figsize=np.array([5.8*ncol, 5.8 + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.06, 'wspace': 0.02},)

northextents = [-38, -38, -50]

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=northextents[jcol], ax_org = axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='top', rotation='horizontal')
    ipanel += 1

plt.text(0.5, -0.08, cbar_label1, transform=axs[0].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, cbar_label2, transform=axs[1].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, cbar_label3, transform=axs[2].transAxes,
         ha='center', va='center', rotation='horizontal')

#-------------------------------- 1st column annual SST
data1 = hadcm3_output_regridded_alltime['PI']['SST']['ann'].values
data2 = HadISST['sst']['1deg_alltime']['ann'].values
ttest_fdr_res = ttest_fdr_control(data1, data2,)

data_diff = hadcm3_output_regridded_alltime['PI']['SST']['am'].squeeze().values - HadISST['sst']['1deg_alltime']['am'].values
data_diff[ttest_fdr_res == False] = np.nan
plt_mesh1 = axs[0].contourf(
    lon, lat, data_diff, levels=pltlevel1, extend='both',
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)

#-------------------------------- 2nd column summer SST
data1 = hadcm3_output_regridded_alltime['PI']['SST']['sea'][0::4].values
data2 = HadISST['sst']['1deg_alltime']['sea'][0::4].values
ttest_fdr_res = ttest_fdr_control(data1, data2,)

data_diff = hadcm3_output_regridded_alltime['PI']['SST']['sm'].sel(time=3).values - HadISST['sst']['1deg_alltime']['sm'].sel(month=3).values
data_diff[ttest_fdr_res == False] = np.nan
axs[1].contourf(
    lon, lat, data_diff, levels=pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

#-------------------------------- 3rd column Sep SIC
data1 = hadcm3_output_regridded_alltime['PI']['SIC']['mon'][8::12].values * 100
data2 = HadISST['sic']['1deg_alltime']['mon'][8::12].values
ttest_fdr_res = ttest_fdr_control(data1, data2,)

data_diff = hadcm3_output_regridded_alltime['PI']['SIC']['mm'].sel(time=9).values * 100 - HadISST['sic']['1deg_alltime']['mm'].sel(month=9).values
data_diff[ttest_fdr_res == False] = np.nan
plt_mesh3 = axs[2].contourf(
    lon, lat, data_diff, levels=pltlevel3, extend='neither',
    norm=pltnorm3, cmap=pltcmp3, transform=ccrs.PlateCarree(),)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.35, ticks=pltticks1, extend='both',
    anchor=(0.18, -0.2),
    )
cbar3 = fig.colorbar(
    plt_mesh3, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.35, ticks=pltticks3, extend='neither',
    anchor=(1.15, -4.6),
    )

plt.text(-0.08, 0.5,
         r'$\mathit{piControl}$' + ' vs. HadISST1',
         transform=axs[0].transAxes, ha='center', va='center',
         rotation='vertical')

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------



