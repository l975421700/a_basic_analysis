

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

with open('scratch/cmip6/lig/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'rb') as f:
    amip_pi_sst_regrid = pickle.load(f)

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

models=sorted(pi_sst_regrid_alltime.keys())


'''
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)


# 7 cores

#-------- import EC reconstruction
ec_sst_rec = {}
# 47 cores
ec_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name='Capron et al. 2017', header=0, skiprows=12, nrows=47,
    usecols=['Station', 'Latitude', 'Longitude', 'Area', 'Type',
             '127 ka Median PIAn [°C]', '127 ka 2s PIAn [°C]'])

# 2 cores
ec_sst_rec['SO_ann'] = ec_sst_rec['original'].loc[
    (ec_sst_rec['original']['Area']=='Southern Ocean') & \
        (ec_sst_rec['original']['Type']=='Annual SST'),]
# 15 cores
ec_sst_rec['SO_djf'] = ec_sst_rec['original'].loc[
    (ec_sst_rec['original']['Area']=='Southern Ocean') & \
        (ec_sst_rec['original']['Type']=='Summer SST'),]

#-------- import JH reconstruction
jh_sst_rec = {}
# 37 cores
jh_sst_rec['original'] = pd.read_excel(
    'data_sources/LIG/mmc1.xlsx',
    sheet_name=' Hoffman et al. 2017', header=0, skiprows=14, nrows=37,)
# 12 cores
jh_sst_rec['SO_ann'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']],
        ]

jh_sst_rec['SO_djf'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']],
        ]

with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst.pkl', 'rb') as f:
    obs_sim_lig_pi_so_sst = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi anomalies compared to amip pi am sst


output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 pi-amip_pi sst am multiple models.png'
cbar_label = 'PI - AMIP_PI annual mean SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

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
            northextent=-30, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1


for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'AWI-ESM-1-1-LR'
        print(model)
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            pi_sst_regrid_alltime[model]['am'].lon,
            pi_sst_regrid_alltime[model]['am'].lat,
            pi_sst_regrid_alltime[model]['am'].values - \
                amip_pi_sst_regrid['am'].values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        diff = pi_sst_regrid_alltime[model]['am'].values - \
            amip_pi_sst_regrid['am'].values
        
        mask = {}
        mask['SO'] = pi_sst_regrid_alltime[model]['am'].lat <= -40
        mask['Atlantic'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= -70) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 20))
        mask['Indian'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= 20) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 140))
        mask['Pacific'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            ((pi_sst_regrid_alltime[model]['am'].lon >= 140) | \
                (pi_sst_regrid_alltime[model]['am'].lon < -70)))
        
        diff_reg = {}
        diff_reg['SO'] = diff[mask['SO']]
        diff_reg['Atlantic'] = diff[mask['Atlantic']]
        diff_reg['Indian'] = diff[mask['Indian']]
        diff_reg['Pacific'] = diff[mask['Pacific']]
        
        area_reg = {}
        area_reg['SO'] = cdo_area1deg.cell_area.values[mask['SO']]
        area_reg['Atlantic'] = cdo_area1deg.cell_area.values[mask['Atlantic']]
        area_reg['Indian'] = cdo_area1deg.cell_area.values[mask['Indian']]
        area_reg['Pacific'] = cdo_area1deg.cell_area.values[mask['Pacific']]
        
        rmse_reg = {}
        rmse_reg['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['SO']),
                mask=np.isnan(np.square(diff_reg['SO']))),
            weights=area_reg['SO']))
        rmse_reg['Atlantic'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Atlantic']),
                mask=np.isnan(np.square(diff_reg['Atlantic']))),
            weights=area_reg['Atlantic']))
        rmse_reg['Indian'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Indian']),
                mask=np.isnan(np.square(diff_reg['Indian']))),
            weights=area_reg['Indian']))
        rmse_reg['Pacific'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Pacific']),
                mask=np.isnan(np.square(diff_reg['Pacific']))),
            weights=area_reg['Pacific']))
        
        plt.text(
            0.5, 1.05,
            model + ': ' + \
                r"$\bf{" + str(np.round(rmse_reg['SO'], 1)) + "}$, " + \
                    str(np.round(rmse_reg['Atlantic'], 1)) + '/' + \
                        str(np.round(rmse_reg['Indian'], 1)) + '/' + \
                            str(np.round(rmse_reg['Pacific'], 1)),
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
        model = models[jcol + ncol * irow]
        # model = 'AWI-ESM-1-1-LR'
        print(model)
        
        diff = pi_sst_regrid_alltime[model]['am'].values - \
            amip_pi_sst_regrid['am'].values
        
        mask = {}
        mask['SO'] = pi_sst_regrid_alltime[model]['am'].lat <= -40
        mask['Atlantic'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= -70) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 20))
        mask['Indian'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= 20) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 140))
        mask['Pacific'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            ((pi_sst_regrid_alltime[model]['am'].lon >= 140) | \
                (pi_sst_regrid_alltime[model]['am'].lon < -70)))
        
        diff_reg = {}
        diff_reg['SO'] = diff[mask['SO']]
        diff_reg['Atlantic'] = diff[mask['Atlantic']]
        diff_reg['Indian'] = diff[mask['Indian']]
        diff_reg['Pacific'] = diff[mask['Pacific']]
        
        area_reg = {}
        area_reg['SO'] = cdo_area1deg.cell_area.values[mask['SO']]
        area_reg['Atlantic'] = cdo_area1deg.cell_area.values[mask['Atlantic']]
        area_reg['Indian'] = cdo_area1deg.cell_area.values[mask['Indian']]
        area_reg['Pacific'] = cdo_area1deg.cell_area.values[mask['Pacific']]
        
        rmse_reg = {}
        rmse_reg['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['SO']),
                mask=np.isnan(np.square(diff_reg['SO']))),
            weights=area_reg['SO']))
        rmse_reg['Atlantic'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Atlantic']),
                mask=np.isnan(np.square(diff_reg['Atlantic']))),
            weights=area_reg['Atlantic']))
        rmse_reg['Indian'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Indian']),
                mask=np.isnan(np.square(diff_reg['Indian']))),
            weights=area_reg['Indian']))
        rmse_reg['Pacific'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Pacific']),
                mask=np.isnan(np.square(diff_reg['Pacific']))),
            weights=area_reg['Pacific']))
        
        print(
            model + ': ' + \
                str(np.round(rmse_reg['SO'], 1)) + ", " + \
                    str(np.round(rmse_reg['Atlantic'], 1)) + '/' + \
                        str(np.round(rmse_reg['Indian'], 1)) + '/' + \
                            str(np.round(rmse_reg['Pacific'], 1)))




model = 'GISS-E2-1-G'

# check mask creation
        (mask['SO'].values == (mask['Atlantic'].values | mask['Indian'].values | mask['Pacific'].values)).all()
        (mask['SO'].values.sum() == (mask['Atlantic'].values.sum() + mask['Indian'].values.sum() + mask['Pacific'].values.sum()))

AWI-ESM-1-1-LR: 1.45, 1.33/1.48/1.50

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi anomalies compared to amip pi DJF sst


output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 pi-amip_pi sst DJF multiple models.png'
cbar_label = 'PI - AMIP_PI summer SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

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
            northextent=-30, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1


for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        print(model)
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            pi_sst_regrid_alltime[model]['am'].lon,
            pi_sst_regrid_alltime[model]['am'].lat,
            pi_sst_regrid_alltime[model]['sm'].sel(season='DJF').values - \
                amip_pi_sst_regrid['sm'].sel(season='DJF').values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        diff = pi_sst_regrid_alltime[model]['sm'].sel(season='DJF').values - \
            amip_pi_sst_regrid['sm'].sel(season='DJF').values
        
        mask = {}
        mask['SO'] = pi_sst_regrid_alltime[model]['am'].lat <= -40
        mask['Atlantic'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= -70) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 20))
        mask['Indian'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            (pi_sst_regrid_alltime[model]['am'].lon >= 20) & \
                (pi_sst_regrid_alltime[model]['am'].lon < 140))
        mask['Pacific'] = ((pi_sst_regrid_alltime[model]['am'].lat <= -40) & \
            ((pi_sst_regrid_alltime[model]['am'].lon >= 140) | \
                (pi_sst_regrid_alltime[model]['am'].lon < -70)))
        
        diff_reg = {}
        diff_reg['SO'] = diff[mask['SO']]
        diff_reg['Atlantic'] = diff[mask['Atlantic']]
        diff_reg['Indian'] = diff[mask['Indian']]
        diff_reg['Pacific'] = diff[mask['Pacific']]
        
        area_reg = {}
        area_reg['SO'] = cdo_area1deg.cell_area.values[mask['SO']]
        area_reg['Atlantic'] = cdo_area1deg.cell_area.values[mask['Atlantic']]
        area_reg['Indian'] = cdo_area1deg.cell_area.values[mask['Indian']]
        area_reg['Pacific'] = cdo_area1deg.cell_area.values[mask['Pacific']]
        
        rmse_reg = {}
        rmse_reg['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['SO']),
                mask=np.isnan(np.square(diff_reg['SO']))),
            weights=area_reg['SO']))
        rmse_reg['Atlantic'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Atlantic']),
                mask=np.isnan(np.square(diff_reg['Atlantic']))),
            weights=area_reg['Atlantic']))
        rmse_reg['Indian'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Indian']),
                mask=np.isnan(np.square(diff_reg['Indian']))),
            weights=area_reg['Indian']))
        rmse_reg['Pacific'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff_reg['Pacific']),
                mask=np.isnan(np.square(diff_reg['Pacific']))),
            weights=area_reg['Pacific']))
        
        plt.text(
            0.5, 1.05,
            model + ': ' + \
                r"$\bf{" + str(np.round(rmse_reg['SO'], 1)) + "}$, " + \
                    str(np.round(rmse_reg['Atlantic'], 1)) + '/' + \
                        str(np.round(rmse_reg['Indian'], 1)) + '/' + \
                            str(np.round(rmse_reg['Pacific'], 1)),
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


# endregion
# -----------------------------------------------------------------------------


