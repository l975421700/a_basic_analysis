

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

with open('scratch/share/from_rahul/data_qingang/hadcm3_128k_regridded_alltime.pkl', 'rb') as f:
    hadcm3_128k_regridded_alltime = pickle.load(f)

lon = hadcm3_128k_regridded_alltime['SST']['am'].lon
lat = hadcm3_128k_regridded_alltime['SST']['am'].lat

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the differences

output_png = 'figures/7_lig/7.1_hadcm3/7.1.1.0 HadCM3 128k_127k SST, SAT and SIC.png'

pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.25, cm_interval2=0.5, cmap='RdBu',)
pltlevel4, pltticks4, pltnorm4, pltcmp4 = plt_mesh_pars(
    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='PuOr',
    reversed=False, asymmetric=False,)

nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.06, 'wspace': 0.02},)

northextents = [-38, -38, -60, -50]
ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=northextents[jcol], ax_org = axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='top', rotation='horizontal')
    ipanel += 1

#---------------- plot annual SST differences
ttest_fdr_res = ttest_fdr_control(
    hadcm3_128k_regridded_alltime['SST']['ann'],
    hadcm3_output_regridded_alltime['LIG']['SST']['ann'],)
plt_data = (hadcm3_128k_regridded_alltime['SST']['am'].squeeze() - hadcm3_output_regridded_alltime['LIG']['SST']['am'].squeeze()).compute().values
plt_data[ttest_fdr_res == False] = np.nan

axs[0].contourf(
    lon, lat, plt_data,
    levels=pltlevel1, extend='both',
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(), zorder=1)
axs[0].add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

#---------------- plot summer SST differences
ttest_fdr_res = ttest_fdr_control(
    hadcm3_128k_regridded_alltime['SST']['sea'][::4],
    hadcm3_output_regridded_alltime['LIG']['SST']['sea'][::4],)
plt_data = (hadcm3_128k_regridded_alltime['SST']['sm'][0] - hadcm3_output_regridded_alltime['LIG']['SST']['sm'][0]).compute().values
plt_data[ttest_fdr_res == False] = np.nan

axs[1].contourf(
    lon, lat, plt_data,
    levels=pltlevel1, extend='both',
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(), zorder=1)
axs[1].add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

#---------------- plot annual SAT differences
ttest_fdr_res = ttest_fdr_control(
    hadcm3_128k_regridded_alltime['SAT']['ann'],
    hadcm3_output_regridded_alltime['LIG']['SAT']['ann'],)
plt_data = (hadcm3_128k_regridded_alltime['SAT']['am'].squeeze() - hadcm3_output_regridded_alltime['LIG']['SAT']['am'].squeeze()).compute().values
plt_data[ttest_fdr_res == False] = np.nan

plt_mesh1 = axs[2].contourf(
    lon, lat, plt_data,
    levels=pltlevel1, extend='both',
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(), zorder=1)
axs[2].add_feature(
    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

#---------------- plot September SIC differences
ttest_fdr_res = ttest_fdr_control(
    hadcm3_128k_regridded_alltime['SIC']['mon'][8::12],
    hadcm3_output_regridded_alltime['LIG']['SIC']['mon'][8::12],)
plt_data = (hadcm3_128k_regridded_alltime['SIC']['mm'][8] - hadcm3_output_regridded_alltime['LIG']['SIC']['mm'][8]).compute().values * 100
plt_data[ttest_fdr_res == False] = np.nan

plt_mesh4 = axs[3].contourf(
    lon, lat, plt_data,
    levels=pltlevel4, extend='both',
    norm=pltnorm4, cmap=pltcmp4, transform=ccrs.PlateCarree(), zorder=1)
axs[3].add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)


#---------------- plot color bar
cbar1 = fig.colorbar(
    plt_mesh1, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.4, ticks=pltticks1, extend='both',
    anchor=(0.25, -0.5),
    )
cbar4 = fig.colorbar(
    plt_mesh4, ax=axs, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.25, ticks=pltticks4, extend='both',
    anchor=(1.12, -4.2),
    )


plt.text(0.5, -0.08, 'Annual SST [$°C$]', transform=axs[0].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Summer SST [$°C$]', transform=axs[1].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Annual SAT [$°C$]', transform=axs[2].transAxes,
         ha='center', va='center', rotation='horizontal')
plt.text(0.5, -0.08, 'Sep SIC [$\%$]', transform=axs[3].transAxes,
         ha='center', va='center', rotation='horizontal')

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------



