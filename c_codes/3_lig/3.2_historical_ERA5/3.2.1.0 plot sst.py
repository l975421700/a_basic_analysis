

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
from scipy.stats import pearsonr

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


with open('scratch/cmip6/hist/tos/hist_tos_regrid_alltime.pkl', 'rb') as f:
    hist_tos_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/hist/tos/esacci_sst_alltime.pkl', 'rb') as f:
    esacci_sst_alltime = pickle.load(f)

with open('scratch/cmip6/hist/tos/hist_tos_regrid.pkl', 'rb') as f:
    hist_tos_regrid = pickle.load(f)

lon = hist_tos_regrid['ACCESS-ESM1-5'].lon.values
lat = hist_tos_regrid['ACCESS-ESM1-5'].lat.values
models = list(hist_tos_regrid_alltime.keys())

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot hist - esacci

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 hist-esacci am sst multiple models 1deg.png'
cbar_label = r'$\mathit{historical}$' + ' vs. ESACCI2.1 Annual SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='RdBu',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        # plot insignificant diff
        ann_hist = hist_tos_regrid_alltime[model]['ann'].values
        ann_esacci = esacci_sst_alltime['ann'].values
        ttest_fdr_res = ttest_fdr_control(ann_hist, ann_esacci,)
        
        # plot diff
        am_data = hist_tos_regrid_alltime[model]['am'].values[0] - esacci_sst_alltime['am'].values[0]
        am_data[ttest_fdr_res == False] = np.nan
        
        plt_mesh = axs[irow, jcol].contourf(
            lon, lat, am_data, levels=pltlevel, extend='both',
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


