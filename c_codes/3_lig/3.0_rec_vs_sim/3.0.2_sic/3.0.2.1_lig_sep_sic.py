

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

with open('scratch/cmip6/lig/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)

models=sorted(lig_sic_regrid_alltime.keys())

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

with open('scratch/cmip6/lig/obs_sim_lig_so_sic_mc.pkl', 'rb') as f:
    obs_sim_lig_so_sic_mc = pickle.load(f)


'''

chadwick_interp.sic_sep
np.round(np.sqrt(np.nanmean((chadwick_interp.sic_sep)**2)), 0)



with open('scratch/cmip6/lig/amip_pi_sic_regrid.pkl', 'rb') as f:
    amip_pi_sic_regrid = pickle.load(f)


with open('scratch/cmip6/lig/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'rb') as f:
    pi_sic_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)
with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)

chadwick2021 = pd.read_csv(
    'data_sources/LIG/Chadwick-etal_2021.tab', sep='\t', header=0, skiprows=43)
indices = [10, 31, 45, 62, 75, 90, 106, 127, 140]


chadwick2021.Event.iloc[
    np.where(np.floor(
        chadwick2021['Age [ka BP] (Age model, EDC3 (EPICA Ice Do...)']
        ) == 127)]

# 7 cores
with open('scratch/cmip6/lig/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig Sep sic

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.1_sic/7.0.1.0 lig sic sep multiple models.png'
cbar_label = 'LIG September SIC [$\%$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues',
    reversed=False)

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
            northextent=-50, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = chadwick_interp.lon,
            y = chadwick_interp.lat,
            c = chadwick_interp.sic_sep,
            s=10, lw=0.3, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        # model = 'NorESM2-LM'
        # model = 'HadGEM3-GC31-LL'
        print(model)
        
        if (model != 'NorESM2-LM'):
            lon = lig_sic[model].lon.values
            lat = lig_sic[model].lat.values
            plt_data = lig_sic_alltime[model]['mm'].sel(month=9).values
            if (model == 'HadGEM3-GC31-LL'):
                plt_data *= 100
        else:
            lon = lig_sic_regrid_alltime[model]['am'].lon.values
            lat = lig_sic_regrid_alltime[model]['am'].lat.values
            plt_data = lig_sic_regrid_alltime[model]['mm'].sel(month=9).values
        
        if not (lon.shape == plt_data.shape):
            lon = lon.transpose()
            lat = lat.transpose()
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, plt_data,
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, model + ': ' + \
                str(np.round(np.sqrt(np.nanmean((obs_sim_lig_so_sic_mc[
                    (obs_sim_lig_so_sic_mc.models == model)
                    ].sim_obs_lig)**2)), 0)),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='neither',
    anchor=(0.5, -0.3),
    )
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)




'''
nrow = 3
ncol = 4
for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        print(
            model + ': ' + \
                str(np.round(np.sqrt(np.nanmean((obs_sim_lig_so_sic_mc[
                    (obs_sim_lig_so_sic_mc.models == model)
                    ].sim_obs_lig)**2)), 0)))



        if (np.isnan(lon).sum() == 0):
        else:
            axs[irow, jcol].contourf(
                lon, lat, plt_data,
                norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

'''
# endregion
# -----------------------------------------------------------------------------





