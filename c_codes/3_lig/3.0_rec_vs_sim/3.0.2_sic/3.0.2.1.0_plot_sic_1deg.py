

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

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)


with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)


lon = lig_pi_sic_regrid_alltime['ACCESS-ESM1-5']['am'].lon.values
lat = lig_pi_sic_regrid_alltime['ACCESS-ESM1-5']['am'].lat.values

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',]

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi sep sic

output_png = 'figures/7_lig/7.0_sim_rec/7.0.1_sic/7.0.1.1 lig-pi sep sic multiple models 1deg.png'
cbar_label = 'LIG Sep SIC anomalies [$\%$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-70, cm_max=20, cm_interval1=10, cm_interval2=10, cmap='PuOr',
    reversed=False, asymmetric=True,)

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
        
        # MC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['MC']['interpolated'].Longitude,
            y = lig_recs['MC']['interpolated'].Latitude,
            c = lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'],
            s=16, lw=0.5, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        sep_lig_pi = lig_pi_sic_regrid_alltime[model]['mm'][8].values
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, sep_lig_pi,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        rmse = {}
        for irec in ['MC']:
            # irec = 'MC'
            print(irec)
            
            rmse[irec] = int(np.round(SO_sep_sic_site_values[irec].loc[
                SO_sep_sic_site_values[irec].Model == model
                ]['sim_rec_sep_sic_lig_pi'].mean(), 0))
        
        plt.text(
            0.5, 1.05,
            model + ': ' + str(rmse['MC']),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_scatter, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------




