

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i=0


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

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
from scipy.stats import pearsonr
from scipy.stats import linregress

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
import cartopy.feature as cfeature
from matplotlib.ticker import AutoMinorLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    ticks_labels,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
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
    plot_labels,
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

with open('scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.nc', 'rb') as f:
    ERA5_temp2_2013_2022_alltime = pickle.load(f)

temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)



'''
# temp2_bias = temp2_alltime[expid[i]]['am'] - (ERA5_temp2_2013_2022_alltime['am'] - zerok)

temp2_alltime[expid[i]]['am'].to_netcdf('scratch/test/test0.nc')
(ERA5_temp2_2013_2022_alltime['am'] - zerok).to_netcdf('scratch/test/test1.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am

plt_data1 = temp2_alltime[expid[i]]['ann'][-10:].mean(dim='time')
plt_data2 = (ERA5_temp2_2013_2022_alltime['am'] - zerok).compute()
plt_data3 = regrid(plt_data1) - regrid(plt_data2)

# stats.describe(plt_data2.sel(latitude=slice(-20, -90)).values.flatten())


#-------- plot configuration
output_png = 'figures/test/trial.png'
cbar_label1 = '2 m temperature [$°C$]'
cbar_label2 = 'Differences: (a) - (b) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-28, cm_max=28, cm_interval1=2, cm_interval2=4, cmap='RdBu',
    asymmetric=True,)

pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 1
ncol = 3
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-20, ax_org = axs[jcol])
    # cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

plt1 = plot_t63_contourf(
    plt_data1.lon,
    plt_data1.lat,
    plt_data1,
    axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

axs[1].contourf(
    plt_data2.longitude,
    plt_data2.latitude,
    plt_data2,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[2].contourf(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    levels = pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'ERA5', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(a) - (b)', transform=axs[2].transAxes,
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

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


