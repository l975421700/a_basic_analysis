

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


with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'rb') as f:
    pi_tas_alltime = pickle.load(f)

models=sorted(lig_tas_alltime.keys())


#-------- import EC reconstruction
ec_sst_rec = {}
# 47 cores
ec_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name='Capron et al. 2017', header=0, skiprows=12, nrows=47,
    usecols=['Station', 'Latitude', 'Longitude', 'Area', 'Type',
             '127 ka Median PIAn [°C]', '127 ka 2s PIAn [°C]'])

ec_sst_rec['AIS_am'] = ec_sst_rec['original'].loc[
    ec_sst_rec['original']['Area']=='Antarctica',]

with open('scratch/cmip6/lig/obs_sim_lig_pi_ais_tas.pkl', 'rb') as f:
    obs_sim_lig_pi_ais_tas = pickle.load(f)


'''
with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi am sst


output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.2_tas/7.0.2.0 lig-pi tas am multiple models.png'

cbar_label = 'LIG - PI annual mean SAT [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-3.5, cm_max=3.5, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG',)

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
            northextent=-60, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = ec_sst_rec['AIS_am'].Longitude,
            y = ec_sst_rec['AIS_am'].Latitude,
            c = ec_sst_rec['AIS_am']['127 ka Median PIAn [°C]'],
            s=10, lw=0.3, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        # model = 'ACCESS-ESM1-5'
        # model = 'HadGEM3-GC31-LL'
        # model = 'CNRM-CM6-1'
        # model = 'AWI-ESM-1-1-LR'
        print(model)
        
        plt_data = lig_tas_alltime[model]['am'].values - \
            pi_tas_alltime[model]['am'].values
        
        ann_data_lig = lig_tas_alltime[model]['ann']
        ann_data_pi  = pi_tas_alltime[model]['ann']
        
        ttest_fdr_res = ttest_fdr_control(ann_data_lig, ann_data_pi,)
        
        lon = pi_tas_alltime[model]['am'].lon
        lat = pi_tas_alltime[model]['am'].lat
        
        if not (lon.shape == plt_data.shape):
            lon = lon.transpose()
            lat = lat.transpose()
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, plt_data,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        axs[irow, jcol].scatter(
            x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
            s=0.3, c='k', marker='.', edgecolors='none',
            transform=ccrs.PlateCarree(),
            )
        
        rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_ais_tas[
            obs_sim_lig_pi_ais_tas.models == model
            ].sim_obs_lig_pi)**2))
        
        plt.text(
            0.5, 1.05,
            model + ': ' + str(np.round(rmse, 1)),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
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
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        
        rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_ais_tas[
            obs_sim_lig_pi_ais_tas.models == model
            ].sim_obs_lig_pi)**2))
        
        print(model + ': ' + str(np.round(rmse, 1)))



model = 'HadGEM3-GC31-LL'
regrid(lig_sst_alltime[model]['am'], ds_out = pi_sst[model]).to_netcdf('test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


