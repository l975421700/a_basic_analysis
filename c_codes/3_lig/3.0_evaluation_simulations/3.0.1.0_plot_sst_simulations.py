

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


with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

models=sorted(lig_sst_alltime.keys())

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

# 1 core
ec_sst_rec['NH_ann'] = ec_sst_rec['original'].loc[
    ((ec_sst_rec['original']['Area']=='Norwegian Sea') | \
        (ec_sst_rec['original']['Area']=='North Atlantic') | \
            (ec_sst_rec['original']['Area']=='Labrador Sea')) & \
                (ec_sst_rec['original']['Type']=='Annual SST'),]
# 23 cores
ec_sst_rec['NH_sum'] = ec_sst_rec['original'].loc[
    ((ec_sst_rec['original']['Area']=='Norwegian Sea') | \
        (ec_sst_rec['original']['Area']=='North Atlantic') | \
            (ec_sst_rec['original']['Area']=='Labrador Sea')) & \
                (ec_sst_rec['original']['Type']=='Summer SST'),]
# 1 core
ec_sst_rec['GrIS_am'] = ec_sst_rec['original'].loc[
    (ec_sst_rec['original']['Area']=='Greenland'),]


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

# 9 cores
jh_sst_rec['NH_ann'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='North Atlantic') & \
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']], ]
# 9 cores
jh_sst_rec['NH_sum'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='North Atlantic') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']], ]

with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst.pkl', 'rb') as f:
    obs_sim_lig_pi_so_sst = pickle.load(f)

with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

with open('scratch/cmip6/lig/obs_sim_lig_pi_so_sst_mc.pkl', 'rb') as f:
    obs_sim_lig_pi_so_sst_mc = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi am sst

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst am multiple models.png'
cbar_label = 'LIG annual mean SST anomalies [$°C$]'

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
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = jh_sst_rec['SO_ann'].Longitude,
            y = jh_sst_rec['SO_ann'].Latitude,
            c = jh_sst_rec['SO_ann']['127 ka SST anomaly (°C)'],
            s=10, lw=0.3, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        axs[irow, jcol].scatter(
            x = ec_sst_rec['SO_ann'].Longitude,
            y = ec_sst_rec['SO_ann'].Latitude,
            c = ec_sst_rec['SO_ann']['127 ka Median PIAn [°C]'],
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
        
        if (model != 'HadGEM3-GC31-LL'):
            plt_data = lig_sst_alltime[model]['am'].values - \
                pi_sst_alltime[model]['am'].values
        elif (model == 'HadGEM3-GC31-LL'):
            plt_data = regrid(
                lig_sst_alltime[model]['am'], ds_out = pi_sst[model]).values - \
                pi_sst_alltime[model]['am'].values
        
        ann_data_lig = lig_sst_alltime[model]['ann']
        ann_data_pi  = pi_sst_alltime[model]['ann']
        
        if (model == 'HadGEM3-GC31-LL'):
            ann_data_lig = regrid(ann_data_lig, ds_out = pi_sst[model])
        
        ttest_fdr_res = ttest_fdr_control(ann_data_lig, ann_data_pi,)
        
        lon = pi_sst[model].lon
        lat = pi_sst[model].lat
        
        if (model != 'AWI-ESM-1-1-LR'):
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
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].scatter(
                x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
                s=0.3, c='k', marker='.', edgecolors='none',
                transform=ccrs.PlateCarree(),
                )
        
        ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
                (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
                (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        
        plt.text(
            0.5, 1.05, model + ': ' + \
                str(np.round(ec_rmse, 1)) + ', ' + \
                    str(np.round(jh_rmse, 1)),
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
        
        ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
                (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
                (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        
        print(
            model + ': ' + \
                str(np.round(ec_rmse, 1)) + ', ' + \
                    str(np.round(jh_rmse, 1)), )


model = 'HadGEM3-GC31-LL'
regrid(lig_sst_alltime[model]['am'], ds_out = pi_sst[model]).to_netcdf('test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi DJF sst

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst djf multiple models.png'
cbar_label = 'LIG summer SST anomalies [$°C$]'

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
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = jh_sst_rec['SO_djf'].Longitude,
            y = jh_sst_rec['SO_djf'].Latitude,
            c = jh_sst_rec['SO_djf']['127 ka SST anomaly (°C)'],
            s=10, lw=0.3, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        axs[irow, jcol].scatter(
            x = ec_sst_rec['SO_djf'].Longitude,
            y = ec_sst_rec['SO_djf'].Latitude,
            c = ec_sst_rec['SO_djf']['127 ka Median PIAn [°C]'],
            s=10, lw=0.3, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        axs[irow, jcol].scatter(
            x = chadwick_interp.lon,
            y = chadwick_interp.lat,
            c = chadwick_interp.sst_sum,
            s=10, lw=0.3, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        # model = 'ACCESS-ESM1-5'
        # model = 'HadGEM3-GC31-LL'
        # model = 'CNRM-CM6-1'
        # model = 'AWI-ESM-1-1-LR'
        print(model)
        
        if (model != 'HadGEM3-GC31-LL'):
            plt_data = lig_sst_alltime[model]['sm'].sel(season='DJF').values - \
                pi_sst_alltime[model]['sm'].sel(season='DJF').values
        elif (model == 'HadGEM3-GC31-LL'):
            # model = 'HadGEM3-GC31-LL'
            plt_data = regrid(
                lig_sst_alltime[model]['sm'].sel(season='DJF'),
                ds_out = pi_sst[model]).values - \
                pi_sst_alltime[model]['sm'].sel(season='DJF').values
        
        djf_data_lig = lig_sst_alltime[model]['sea'][3::4]
        djf_data_pi  = pi_sst_alltime[model]['sea'][3::4]
        
        if (model == 'HadGEM3-GC31-LL'):
            djf_data_lig = regrid(djf_data_lig, ds_out = pi_sst[model])
        
        ttest_fdr_res = ttest_fdr_control(djf_data_lig, djf_data_pi,)
        
        lon = pi_sst[model].lon
        lat = pi_sst[model].lat
        
        if (model != 'AWI-ESM-1-1-LR'):
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
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].scatter(
                x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
                s=0.3, c='k', marker='.', edgecolors='none',
                transform=ccrs.PlateCarree(),
                )
        
        ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
                (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
                (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        mc_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst_mc[
            (obs_sim_lig_pi_so_sst_mc.models == model)
            ].sim_obs_lig_pi)**2))
        
        plt.text(
            0.5, 1.05, model + ': ' + \
                str(np.round(ec_rmse, 1)) + ', ' + \
                    str(np.round(jh_rmse, 1)) + ', ' + \
                        str(np.round(mc_rmse, 1)),
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
        
        ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
                (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
            (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
                (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
                    (obs_sim_lig_pi_so_sst.models == model)
                    ].sim_obs_lig_pi)**2))
        mc_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst_mc[
            (obs_sim_lig_pi_so_sst_mc.models == model)
            ].sim_obs_lig_pi)**2))
        
        print(
            model + ': ' + \
                str(np.round(ec_rmse, 1)) + ', ' + \
                    str(np.round(jh_rmse, 1)) + ', ' + \
                        str(np.round(mc_rmse, 1)), )


model = 'HadGEM3-GC31-LL'
regrid(lig_sst_alltime[model]['am'], ds_out = pi_sst[model]).to_netcdf('test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate area weighted LIG-PI SO am/DJF SST

with open('scratch/cmip6/lig/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

models=sorted(lig_sst_regrid_alltime.keys())

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

lat = lig_sst_regrid_alltime['ACCESS-ESM1-5']['am'].lat.values
mask_so = (lat < -40)

print('#---------------- Am SST')
for model in models:
    # model = 'ACCESS-ESM1-5'
    # model = 'HadGEM3-GC31-LL'
    # model = 'AWI-ESM-1-1-LR'
    print(model)
    
    lig_sst_so_mean = np.ma.average(
        np.ma.MaskedArray(
            lig_sst_regrid_alltime[model]['am'].values[mask_so],
            mask=np.isnan(lig_sst_regrid_alltime[model]['am'].values[mask_so])),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    pi_sst_so_mean = np.ma.average(
        np.ma.MaskedArray(
            pi_sst_regrid_alltime[model]['am'].values[mask_so],
            mask=np.isnan(pi_sst_regrid_alltime[model]['am'].values[mask_so])),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    # LIG ann
    b_mask_so = np.broadcast_to(
        mask_so, lig_sst_regrid_alltime[model]['ann'].shape)
    b_cell_area = np.broadcast_to(
        cdo_area1deg.cell_area, lig_sst_regrid_alltime[model]['ann'].shape)
    lig_sst_so_ann = np.ma.average(
        np.ma.MaskedArray(
            lig_sst_regrid_alltime[model]['ann'].copy().where(
                b_mask_so, np.nan),
            mask = np.isnan(lig_sst_regrid_alltime[model]['ann'].copy().where(
                b_mask_so, np.nan)),
            ),
        weights = b_cell_area,
        axis=(1, 2),
    )
    
    # PI ann
    b_mask_so = np.broadcast_to(
        mask_so, pi_sst_regrid_alltime[model]['ann'].shape)
    b_cell_area = np.broadcast_to(
        cdo_area1deg.cell_area, pi_sst_regrid_alltime[model]['ann'].shape)
    pi_sst_so_ann = np.ma.average(
        np.ma.MaskedArray(
            pi_sst_regrid_alltime[model]['ann'].copy().where(
                b_mask_so, np.nan),
            mask = np.isnan(pi_sst_regrid_alltime[model]['ann'].copy().where(
                b_mask_so, np.nan)),
            ),
        weights = b_cell_area,
        axis=(1, 2),
    )
    
    if ((lig_sst_so_mean - lig_sst_so_ann.mean()) / lig_sst_so_mean > 1e-5):
        print('Large difference between am and mean(ann)')
    if ((pi_sst_so_mean - pi_sst_so_ann.mean()) / pi_sst_so_mean > 1e-5):
        print('Large difference between am and mean(ann)')
    
    am_diff = lig_sst_so_mean - pi_sst_so_mean
    ann_std = (lig_sst_so_ann.std(ddof=1)**2 + pi_sst_so_ann.std(ddof=1)**2)**0.5
    
    print(str(np.round(am_diff, 1)) + '±' + str(np.round(ann_std, 1)))


print('#---------------- DJF SST')
for model in models:
    # model = 'ACCESS-ESM1-5'
    print(model)
    
    lig_sst_so_mean = np.ma.average(
        np.ma.MaskedArray(
            lig_sst_regrid_alltime[model][
                'sm'].sel(season='DJF').values[mask_so],
            mask=np.isnan(lig_sst_regrid_alltime[model][
                'sm'].sel(season='DJF').values[mask_so])),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    pi_sst_so_mean = np.ma.average(
        np.ma.MaskedArray(
            pi_sst_regrid_alltime[model][
                'sm'].sel(season='DJF').values[mask_so],
            mask=np.isnan(pi_sst_regrid_alltime[model][
                'sm'].sel(season='DJF').values[mask_so])),
        weights=cdo_area1deg.cell_area.values[mask_so])
    
    # LIG sea
    b_mask_so = np.broadcast_to(
        mask_so,
        lig_sst_regrid_alltime[model]['sea'][3::4].shape)
    b_cell_area = np.broadcast_to(
        cdo_area1deg.cell_area,
        lig_sst_regrid_alltime[model]['sea'][3::4].shape)
    lig_sst_so_sea = np.ma.average(
        np.ma.MaskedArray(
            lig_sst_regrid_alltime[model]['sea'][3::4].copy(
                ).where(b_mask_so, np.nan),
            mask = np.isnan(lig_sst_regrid_alltime[model]['sea'][3::4].copy(
                ).where(b_mask_so, np.nan)),
            ),
        weights = b_cell_area,
        axis=(1, 2),
    )
    
    # PI sea
    b_mask_so = np.broadcast_to(
        mask_so,
        pi_sst_regrid_alltime[model]['sea'][3::4].shape)
    b_cell_area = np.broadcast_to(
        cdo_area1deg.cell_area,
        pi_sst_regrid_alltime[model]['sea'][3::4].shape)
    pi_sst_so_sea = np.ma.average(
        np.ma.MaskedArray(
            pi_sst_regrid_alltime[model]['sea'][3::4].copy(
                ).where(b_mask_so, np.nan),
            mask = np.isnan(pi_sst_regrid_alltime[model]['sea'][3::4].copy(
                ).where(b_mask_so, np.nan)),
            ),
        weights = b_cell_area,
        axis=(1, 2),
    )
    
    if ((lig_sst_so_mean - lig_sst_so_sea.mean()) / lig_sst_so_mean > 1e-5):
        print('Large difference between am and mean(ann)')
    if ((pi_sst_so_mean - pi_sst_so_sea.mean()) / pi_sst_so_mean > 1e-5):
        print('Large difference between am and mean(ann)')
    
    sm_diff = lig_sst_so_mean - pi_sst_so_mean
    sea_std = (lig_sst_so_sea.std(ddof=1)**2 + pi_sst_so_sea.std(ddof=1)**2)**0.5
    
    print(str(np.round(sm_diff, 1)) + '±' + str(np.round(sea_std, 1)))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi am sst NH

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst am multiple models NH.png'
cbar_label = 'LIG annual mean SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.NorthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            southextent=38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = jh_sst_rec['NH_ann'].Longitude,
            y = jh_sst_rec['NH_ann'].Latitude,
            c = jh_sst_rec['NH_ann']['127 ka SST anomaly (°C)'],
            s=10, lw=0.3, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        axs[irow, jcol].scatter(
            x = ec_sst_rec['NH_ann'].Longitude,
            y = ec_sst_rec['NH_ann'].Latitude,
            c = ec_sst_rec['NH_ann']['127 ka Median PIAn [°C]'],
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
        
        if (model != 'HadGEM3-GC31-LL'):
            plt_data = lig_sst_alltime[model]['am'].values - \
                pi_sst_alltime[model]['am'].values
        elif (model == 'HadGEM3-GC31-LL'):
            plt_data = regrid(
                lig_sst_alltime[model]['am'], ds_out = pi_sst[model]).values - \
                pi_sst_alltime[model]['am'].values
        
        ann_data_lig = lig_sst_alltime[model]['ann']
        ann_data_pi  = pi_sst_alltime[model]['ann']
        
        if (model == 'HadGEM3-GC31-LL'):
            ann_data_lig = regrid(ann_data_lig, ds_out = pi_sst[model])
        
        ttest_fdr_res = ttest_fdr_control(ann_data_lig, ann_data_pi,)
        
        lon = pi_sst[model].lon
        lat = pi_sst[model].lat
        
        if (model != 'AWI-ESM-1-1-LR'):
            if not (lon.shape == plt_data.shape):
                lon = lon.transpose()
                lat = lat.transpose()
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            # axs[irow, jcol].scatter(
            #     x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
            #     s=0.3, c='k', marker='.', edgecolors='none',
            #     transform=ccrs.PlateCarree(),
            #     )
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            # axs[irow, jcol].scatter(
            #     x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
            #     s=0.3, c='k', marker='.', edgecolors='none',
            #     transform=ccrs.PlateCarree(),
            #     )
        
        # ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
        #     (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
        #         (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
        #             (obs_sim_lig_pi_so_sst.models == model)
        #             ].sim_obs_lig_pi)**2))
        # jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
        #     (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
        #         (obs_sim_lig_pi_so_sst.types == 'Annual SST') & \
        #             (obs_sim_lig_pi_so_sst.models == model)
        #             ].sim_obs_lig_pi)**2))
        
        # plt.text(
        #     0.5, 1.05, model + ': ' + \
        #         str(np.round(ec_rmse, 1)) + ', ' + \
        #             str(np.round(jh_rmse, 1)),
        #     transform=axs[irow, jcol].transAxes,
        #     ha='center', va='center', rotation='horizontal')
        
        plt.text(
            0.5, 1.05, model,
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
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi summer sst NH

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst summer multiple models NH.png'
cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.NorthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            southextent=38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        axs[irow, jcol].scatter(
            x = jh_sst_rec['NH_sum'].Longitude,
            y = jh_sst_rec['NH_sum'].Latitude,
            c = jh_sst_rec['NH_sum']['127 ka SST anomaly (°C)'],
            s=10, lw=0.3, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        axs[irow, jcol].scatter(
            x = ec_sst_rec['NH_sum'].Longitude,
            y = ec_sst_rec['NH_sum'].Latitude,
            c = ec_sst_rec['NH_sum']['127 ka Median PIAn [°C]'],
            s=10, lw=0.3, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        # model = 'ACCESS-ESM1-5'
        # model = 'HadGEM3-GC31-LL'
        # model = 'CNRM-CM6-1'
        # model = 'AWI-ESM-1-1-LR'
        print(model)
        
        if (model != 'HadGEM3-GC31-LL'):
            plt_data = lig_sst_alltime[model]['sm'].sel(season='JJA').values - \
                pi_sst_alltime[model]['sm'].sel(season='JJA').values
        elif (model == 'HadGEM3-GC31-LL'):
            # model = 'HadGEM3-GC31-LL'
            plt_data = regrid(
                lig_sst_alltime[model]['sm'].sel(season='JJA'),
                ds_out = pi_sst[model]).values - \
                pi_sst_alltime[model]['sm'].sel(season='JJA').values
        
        # djf_data_lig = lig_sst_alltime[model]['sea'][3::4]
        # djf_data_pi  = pi_sst_alltime[model]['sea'][3::4]
        
        # if (model == 'HadGEM3-GC31-LL'):
        #     djf_data_lig = regrid(djf_data_lig, ds_out = pi_sst[model])
        
        # ttest_fdr_res = ttest_fdr_control(djf_data_lig, djf_data_pi,)
        
        lon = pi_sst[model].lon
        lat = pi_sst[model].lat
        
        if (model != 'AWI-ESM-1-1-LR'):
            if not (lon.shape == plt_data.shape):
                lon = lon.transpose()
                lat = lat.transpose()
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            # axs[irow, jcol].scatter(
            #     x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
            #     s=0.3, c='k', marker='.', edgecolors='none',
            #     transform=ccrs.PlateCarree(),
            #     )
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            # axs[irow, jcol].scatter(
            #     x=lon.values[ttest_fdr_res], y=lat.values[ttest_fdr_res],
            #     s=0.3, c='k', marker='.', edgecolors='none',
            #     transform=ccrs.PlateCarree(),
            #     )
        
        # ec_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
        #     (obs_sim_lig_pi_so_sst.datasets == 'EC') & \
        #         (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
        #             (obs_sim_lig_pi_so_sst.models == model)
        #             ].sim_obs_lig_pi)**2))
        # jh_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst[
        #     (obs_sim_lig_pi_so_sst.datasets == 'JH') & \
        #         (obs_sim_lig_pi_so_sst.types == 'Summer SST') & \
        #             (obs_sim_lig_pi_so_sst.models == model)
        #             ].sim_obs_lig_pi)**2))
        # mc_rmse = np.sqrt(np.nanmean((obs_sim_lig_pi_so_sst_mc[
        #     (obs_sim_lig_pi_so_sst_mc.models == model)
        #     ].sim_obs_lig_pi)**2))
        
        # plt.text(
        #     0.5, 1.05, model + ': ' + \
        #         str(np.round(ec_rmse, 1)) + ', ' + \
        #             str(np.round(jh_rmse, 1)) + ', ' + \
        #                 str(np.round(mc_rmse, 1)),
        #     transform=axs[irow, jcol].transAxes,
        #     ha='center', va='center', rotation='horizontal')
        
        plt.text(
            0.5, 1.05, model,
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
'''
# endregion
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# region plot lig am sst

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig sst am multiple models.png'
# output_png = 'figures/0_test/trial.png'
cbar_label = 'LIG annual mean SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=2, cmap='RdBu',)
# ctrlevel = np.array([0.5, 1, 1.5, 2])

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
        # axs[irow, jcol].add_feature(
        #     cfeature.LAND, zorder=2, facecolor='white')
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        # model = 'GISS-E2-1-G'
        # model = 'ACCESS-ESM1-5'
        # model = 'HadGEM3-GC31-LL'
        # model = 'CNRM-CM6-1'
        print(model)
        
        plt_data = lig_sst_alltime[model]['am'].values
        # plt_data2 = lig_sst_alltime[model]['ann'].std(
        #     dim='time', skipna=True, ddof=1).values
        
        if (model != 'AWI-ESM-1-1-LR'):
            lon = lig_sst[model].lon
            lat = lig_sst[model].lat
            
            if not (lon.shape == plt_data.shape):
                lon = lon.transpose()
                lat = lat.transpose()
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plt_data,
                norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
            # plt_ctr = axs[irow, jcol].contour(
            #     lon, lat, plt_data2,
            #     levels=ctrlevel, colors = 'm', transform=ccrs.PlateCarree(),
            #     linewidths=0.5, linestyles='solid',)
            # axs[irow, jcol].clabel(
            #     plt_ctr, inline=1, colors='m', fmt=remove_trailing_zero,
            #     levels=ctrlevel, inline_spacing=10, fontsize=6,)
            
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            # plt_ctr = axs[irow, jcol].tricontour(
            #     tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
            #     plt_data2,
            #     levels=ctrlevel, colors = 'm', transform=ccrs.PlateCarree(),
            #     linewidths=0.5, linestyles='solid',)
            # axs[irow, jcol].clabel(
            #     plt_ctr, inline=1, colors='m', fmt=remove_trailing_zero,
            #     levels=ctrlevel, inline_spacing=10, fontsize=6,)
        
        plt.text(
            0.5, 1.05, model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.3),
    )
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
stats.describe(plt_data[np.isfinite(plt_data)])
stats.describe(plt_data2[np.isfinite(plt_data2)])

lig_sst_alltime['HadGEM3-GC31-LL']['am'].values
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi am sst

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 pi sst am multiple models.png'
cbar_label = 'PI annual mean SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=2, cmap='RdBu',)

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
        print(model)
        
        plt_data = pi_sst_alltime[model]['am'].values
        
        if (model != 'AWI-ESM-1-1-LR'):
            lon = pi_sst[model].lon
            lat = pi_sst[model].lat
            if not (lon.shape == plt_data.shape):
                lon = lon.transpose()
                lat = lat.transpose()
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plt_data,
                norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        elif (model == 'AWI-ESM-1-1-LR'):
            # model = 'AWI-ESM-1-1-LR'
            tri2plot = mesh2plot(
                meshdir='startdump/fesom2/mesh/CORE2_final/',
                abg=[50, 15, -90], usepickle=False)
            axs[irow, jcol].tripcolor(
                tri2plot['x2'], tri2plot['y2'], tri2plot['elem2plot'],
                plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.3),
    )
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


