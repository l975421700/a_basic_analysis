

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0


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
import cartopy.feature as cfeature
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
    regrid,
    mean_over_ais,
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
    cplot_ttest,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

acc_recon = xr.open_dataset('data_sources/products/Antarctic_Accumulation_Reconstructions/acc_recon_MERRA2.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
acc_recon
plt_err1 = (acc_recon.recon_acc_bc.sel(years=slice(100, 200)).std(dim='years', ddof=1)).compute()

plt_err2 = (acc_recon.recon_acc_err_bc.sel(years=slice(100, 200)) ** 2).mean(dim='years') ** 0.5

np.max(plt_err1 - plt_err2)

acc_recon_CFSR = xr.open_dataset('data_sources/products/Antarctic_Accumulation_Reconstructions/acc_recon_ERAI.nc')
acc_recon_ERAI = xr.open_dataset('data_sources/products/Antarctic_Accumulation_Reconstructions/acc_recon_ERAI.nc')
acc_recon_MERRA2 = xr.open_dataset('data_sources/products/Antarctic_Accumulation_Reconstructions/acc_recon_ERAI.nc')

#---- other input
with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am accumulation

#-------- get data
plt_data1 = (wisoaprt_alltime[expid[i]]['am'][0] + \
    wisoevap_alltime[expid[i]]['am'][0]).compute() * seconds_per_d
plt_data2 = acc_recon.recon_acc_bc.sel(
    years=slice(100, 200)).mean(dim='years') / 365
plt_data3 = (regrid(plt_data1) / regrid(plt_data2) - 1).compute() * 100
# plt_data1.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

plt_std1 = ((wisoaprt_alltime[expid[i]]['ann'][:, 0] + \
    wisoevap_alltime[expid[i]]['ann'][:, 0]).std(
        dim = 'time', ddof=1) * seconds_per_d / plt_data1 * 100).compute()
plt_std1.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
plt_std2 = (acc_recon.recon_acc_bc.sel(
    years=slice(100, 200)).std(
        dim='years', ddof=1) / 365 / plt_data2 * 100).compute()

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.2_pe/6.1.4.2 ' + expid[i] + ' Medley_rec accumulation am Antarctica.png'
cbar_label1 = 'Accumulation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences: (a)/(b) - 1 [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-100, 100 + 1e-4, 20)
pltticks2 = np.arange(-100, 100 + 1e-4, 40)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)

ctr_level = np.arange(0, 40 + 1e-4, 10)
ctr_color = 'r'

nrow = 1
ncol = 3
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

plt1 = plot_t63_contourf(
    plt_data1.lon, plt_data1.lat, plt_data1, axs[0],
    pltlevel, 'max', pltnorm, pltcmp, ccrs.PlateCarree(),)

axs[0].add_feature(
    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

axs[1].contourf(
    plt_data2.lon,
    plt_data2.lat,
    plt_data2,
    levels = pltlevel,extend='max',
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[2].contourf(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    levels = pltlevel2,extend='both',
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt_ctr1 = axs[0].contour(
    plt_std1.lon,
    plt_std1.lat,
    plt_std1,
    levels=ctr_level, colors = ctr_color, transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0].clabel(
    plt_ctr1, inline=1, colors=ctr_color, fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=8,)

plt_ctr2 = axs[1].contour(
    plt_std2.lon,
    plt_std2.lat,
    plt_std2,
    levels=ctr_level, colors = ctr_color, transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1].clabel(
    plt_ctr2, inline=1, colors=ctr_color, fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=8,)

plt.text(
    0.5, 1.05, 'Simulation', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Reconstruction', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(a)/(b) - 1', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(1.1,-2.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

h1, _ = plt_ctr1.legend_elements()
plt.legend(
    [h1[0]], ['Percentage [$\%$] of one standard deviation to the mean'],
    bbox_to_anchor = (0.5, -0.25), frameon=False)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


