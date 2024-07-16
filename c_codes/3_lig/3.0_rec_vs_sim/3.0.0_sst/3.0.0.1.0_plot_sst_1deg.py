

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

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)


with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

lon = lig_pi_sst_regrid_alltime['ACCESS-ESM1-5']['am'].lon.values
lat = lig_pi_sst_regrid_alltime['ACCESS-ESM1-5']['am'].lat.values

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',]

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)

HadISST = {}
with open('data_sources/LIG/HadISST1.1/HadISST_sst.pkl', 'rb') as f:
    HadISST['sst'] = pickle.load(f)

mask = {}
mask['SO'] = lat <= -40
mask['Atlantic'] = ((lat <= -40) & (lon >= -70) & (lon < 20))
mask['Indian'] = ((lat <= -40) & (lon >= 20) & (lon < 140))
mask['Pacific'] = ((lat <= -40) & ((lon >= 140) | (lon < -70)))

cdo_area1deg = xr.open_dataset('scratch/others/one_degree_grids_cdo_area.nc')

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi-hadisst am sst

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 pi-hadisst am sst multiple models 1deg.png'
cbar_label = r'$\mathit{piControl}$' + ' vs. HadISST1 Annual SST [$°C$]'

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
        ann_pi = pi_sst_regrid_alltime[model]['ann'].values
        ann_hadisst = HadISST['sst']['1deg_alltime']['ann'].values
        ttest_fdr_res = ttest_fdr_control(ann_pi, ann_hadisst,)
        
        # plot diff
        am_data = pi_sst_regrid_alltime[model]['am'].values[0] - \
            HadISST['sst']['1deg_alltime']['am'].values
        
        am_data[ttest_fdr_res == False] = np.nan
        
        plt_mesh = axs[irow, jcol].contourf(
            lon, lat, am_data, levels=pltlevel, extend='both',
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(am_data)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(am_data)],
        #     s=0.5, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),)
        
        # # calculate RMSE
        # am_data = pi_sst_regrid_alltime[model]['am'].values[0] - \
        #     HadISST['sst']['1deg_alltime']['am'].values
        # diff = {}
        # diff['SO'] = am_data[mask['SO']]
        # diff['Atlantic'] = am_data[mask['Atlantic']]
        # diff['Indian'] = am_data[mask['Indian']]
        # diff['Pacific'] = am_data[mask['Pacific']]
        
        # area = {}
        # area['SO'] = cdo_area1deg.cell_area.values[mask['SO']]
        # area['Atlantic'] = cdo_area1deg.cell_area.values[mask['Atlantic']]
        # area['Indian'] = cdo_area1deg.cell_area.values[mask['Indian']]
        # area['Pacific'] = cdo_area1deg.cell_area.values[mask['Pacific']]
        
        # rmse = {}
        # rmse['SO'] = np.sqrt(np.ma.average(
        #     np.ma.MaskedArray(
        #         np.square(diff['SO']), mask=np.isnan(diff['SO'])),
        #     weights=area['SO']))
        # rmse['Atlantic'] = np.sqrt(np.ma.average(
        #     np.ma.MaskedArray(
        #         np.square(diff['Atlantic']), mask=np.isnan(diff['Atlantic'])),
        #     weights=area['Atlantic']))
        # rmse['Indian'] = np.sqrt(np.ma.average(
        #     np.ma.MaskedArray(
        #         np.square(diff['Indian']), mask=np.isnan(diff['Indian'])),
        #     weights=area['Indian']))
        # rmse['Pacific'] = np.sqrt(np.ma.average(
        #     np.ma.MaskedArray(
        #         np.square(diff['Pacific']), mask=np.isnan(diff['Pacific'])),
        #     weights=area['Pacific']))
        
        plt.text(
            0.5, 1.05,
            model,
            # model + ' (' + \
            #     str(np.round(rmse['SO'], 1)) + ', ' + \
            #         str(np.round(rmse['Atlantic'], 1)) + ', ' + \
            #             str(np.round(rmse['Indian'], 1)) + ', ' + \
            #                 str(np.round(rmse['Pacific'], 1)) + ')',
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
#-------------------------------- check

mask
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi-hadisst summer sst

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 pi-hadisst summer sst multiple models 1deg.png'
cbar_label = r'$\mathit{piControl}$' + ' vs. HadISST1 Summer SST [$°C$]'

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
        sea_pi = pi_sst_regrid_alltime[model]['sea'][::4].values
        sea_hadisst = HadISST['sst']['1deg_alltime']['sea'][::4].values
        ttest_fdr_res = ttest_fdr_control(sea_pi, sea_hadisst,)
        
        # plot diff
        sm_data = pi_sst_regrid_alltime[model]['sm'][0].values - \
            HadISST['sst']['1deg_alltime']['sm'][0].values
        sm_data[ttest_fdr_res == False] = np.nan
        
        plt_mesh = axs[irow, jcol].contourf(
            lon, lat, sm_data, levels=pltlevel, extend='both',
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(sm_data)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(sm_data)],
        #     s=0.5, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),)
        
        # calculate RMSE
        sm_data = pi_sst_regrid_alltime[model]['sm'][0].values - \
            HadISST['sst']['1deg_alltime']['sm'][0].values
        diff = {}
        diff['SO'] = sm_data[mask['SO']]
        diff['Atlantic'] = sm_data[mask['Atlantic']]
        diff['Indian'] = sm_data[mask['Indian']]
        diff['Pacific'] = sm_data[mask['Pacific']]
        
        area = {}
        area['SO'] = cdo_area1deg.cell_area.values[mask['SO']]
        area['Atlantic'] = cdo_area1deg.cell_area.values[mask['Atlantic']]
        area['Indian'] = cdo_area1deg.cell_area.values[mask['Indian']]
        area['Pacific'] = cdo_area1deg.cell_area.values[mask['Pacific']]
        
        rmse = {}
        rmse['SO'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['SO']), mask=np.isnan(diff['SO'])),
            weights=area['SO']))
        rmse['Atlantic'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['Atlantic']), mask=np.isnan(diff['Atlantic'])),
            weights=area['Atlantic']))
        rmse['Indian'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['Indian']), mask=np.isnan(diff['Indian'])),
            weights=area['Indian']))
        rmse['Pacific'] = np.sqrt(np.ma.average(
            np.ma.MaskedArray(
                np.square(diff['Pacific']), mask=np.isnan(diff['Pacific'])),
            weights=area['Pacific']))
        
        plt.text(
            0.5, 1.05,
            model + ' (' + \
                str(np.round(rmse['SO'], 1)) + ', ' + \
                    str(np.round(rmse['Atlantic'], 1)) + ', ' + \
                        str(np.round(rmse['Indian'], 1)) + ', ' + \
                            str(np.round(rmse['Pacific'], 1)) + ')',
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
#-------------------------------- check

mask
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate and plot Taylor Diagram

import skill_metrics as sm
from skill_metrics import taylor_diagram

ccoef = np.array([])
crmsd = np.array([])
sdev = np.array([])

for imodel in pi_sst_regrid_alltime.keys():
    # imodel = 'AWI-ESM-1-1-LR'
    print('#-------------------------------- ' + imodel)
    
    pred_data = pi_sst_regrid_alltime[imodel]['am'].squeeze().values[mask['SO']]
    ref_data  = HadISST['sst']['1deg_alltime']['am'].values[mask['SO']]
    subset = np.isfinite(pred_data) & np.isfinite(ref_data)
    
    taylor_stats1 = sm.taylor_statistics(pred_data[subset], ref_data[subset],)
    
    if (imodel == 'ACCESS-ESM1-5'):
        ccoef = np.append(ccoef, taylor_stats1['ccoef'][0])
        crmsd = np.append(crmsd, taylor_stats1['crmsd'][0])
        sdev  = np.append(sdev, taylor_stats1['sdev'][0])
    
    ccoef = np.append(ccoef, taylor_stats1['ccoef'][1])
    crmsd = np.append(crmsd, taylor_stats1['crmsd'][1])
    sdev  = np.append(sdev, taylor_stats1['sdev'][1])


labels = ['HadISST1'] + list(pi_sst_regrid_alltime.keys())

plt.close()
sm.taylor_diagram(
    sdev, crmsd, ccoef, markerLabel = labels, markerLegend = 'on',
    styleOBS = '-', colOBS = 'r', markerobs = 'o',
    markerSize = 4, titleOBS = 'HadISST1',
    # markersymbol='x',
    # markerLabelColor = 'r', markerColor = 'r',
    # tickRMS = [0.0, 1.0, 2.0, 3.0], titleRMS = 'on',
    # tickRMSangle = 115, showlabelsRMS = 'on',
    )

plt.savefig('figures/test/test.png')




'''

    # print(taylor_stats1['ccoef'][1])
    # print(taylor_stats1['crmsd'][1])
    print(taylor_stats1['sdev'][1])



imodel = 'ACCESS-ESM1-5'
pred_data = pi_sst_regrid_alltime[imodel]['am'].squeeze().values[mask['SO']]
ref_data  = HadISST['sst']['1deg_alltime']['am'].values[mask['SO']]
subset = np.isfinite(pred_data) & np.isfinite(ref_data)
taylor_stats1 = sm.taylor_statistics(pred_data[subset], ref_data[subset],)



    
    
    # print(pearsonr(pred_data[subset], ref_data[subset])[0])
    
    
    
    am_data = pi_sst_regrid_alltime[imodel]['am'].values[0] - \
            HadISST['sst']['1deg_alltime']['am'].values
    rmse = np.sqrt(np.ma.average(np.ma.MaskedArray(np.square(
        am_data[mask['SO']]), mask=np.isnan(am_data[mask['SO']]))))
    # rmse = np.sqrt(np.nanmean(np.square(am_data[mask['SO']])))
    print(rmse)
    rmse = np.sqrt(np.nanmean(np.square(pred_data[subset] - ref_data[subset])))
    print(rmse)
'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region plot lig-pi am sst _no_rec

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 lig-pi am sst multiple models 1deg_no_rec.png'
cbar_label = r'$\mathit{lig127k}$' + ' vs. ' + r'$\mathit{piControl}$' + ' Annual SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

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
        
        # JH
        axs[irow, jcol].scatter(
            x = lig_recs['JH']['SO_ann'].Longitude,
            y = lig_recs['JH']['SO_ann'].Latitude,
            c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
            s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
            lw=0.5, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # EC SST
        axs[irow, jcol].scatter(
            x = lig_recs['EC']['SO_ann'].Longitude,
            y = lig_recs['EC']['SO_ann'].Latitude,
            c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
            s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
            lw=0.5, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # DC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['DC']['annual_128'].Longitude,
            y = lig_recs['DC']['annual_128'].Latitude,
            c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
            s = max_size - scale_size * 1,
            lw=0.5, marker='v', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        # plot insignificant diff
        ann_lig = lig_sst_regrid_alltime[model]['ann'].values
        ann_pi = pi_sst_regrid_alltime[model]['ann'].values
        ttest_fdr_res = ttest_fdr_control(ann_lig, ann_pi,)
        
        # plot diff
        am_lig_pi = lig_pi_sst_regrid_alltime[model]['am'][0].values.copy()
        am_lig_pi[ttest_fdr_res == False] = np.nan
        plt_mesh = axs[irow, jcol].contourf(
            lon, lat, am_lig_pi, levels=pltlevel, extend='both',
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(am_lig_pi)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(am_lig_pi)],
        #     s=0.5, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),)
        
        # # calculate mean error
        # am_lig_pi = lig_pi_sst_regrid_alltime[model]['am'][0].values.copy()
        # diff = {}
        # area = {}
        # mean_diff = {}
        
        # for iregion in ['SO', 'Atlantic', 'Indian', 'Pacific']:
        #     diff[iregion] = am_lig_pi[mask[iregion]]
        #     area[iregion] = cdo_area1deg.cell_area.values[mask[iregion]]
        #     mean_diff[iregion] = np.ma.average(
        #         np.ma.MaskedArray(diff[iregion], mask=np.isnan(diff[iregion])),
        #         weights=area[iregion],)
        
        plt.text(
            0.5, 1.05,
            model,
            # model + ' (' + \
            #     str(np.round(mean_diff['SO'], 1)) + ', ' + \
            #         str(np.round(mean_diff['Atlantic'], 1)) + ', ' + \
            #             str(np.round(mean_diff['Indian'], 1)) + ', ' + \
            #                 str(np.round(mean_diff['Pacific'], 1)) + ')',
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi summer sst _no_rec

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 lig-pi summer sst multiple models 1deg_no_rec.png'
cbar_label = r'$\mathit{lig127k}$' + ' vs. ' + r'$\mathit{piControl}$' + ' Summer SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

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
        
        # JH
        axs[irow, jcol].scatter(
            x = lig_recs['JH']['SO_jfm'].Longitude,
            y = lig_recs['JH']['SO_jfm'].Latitude,
            c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
            s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
            lw=0.5, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # EC SST
        axs[irow, jcol].scatter(
            x = lig_recs['EC']['SO_jfm'].Longitude,
            y = lig_recs['EC']['SO_jfm'].Latitude,
            c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
            s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
            lw=0.5, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # MC
        axs[irow, jcol].scatter(
            x = lig_recs['MC']['interpolated'].Longitude,
            y = lig_recs['MC']['interpolated'].Latitude,
            c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
            s = max_size - scale_size * 1.09,
            lw=0.5, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # DC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['DC']['JFM_128'].Longitude,
            y = lig_recs['DC']['JFM_128'].Latitude,
            c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
            s = max_size - scale_size * 1,
            lw=0.5, marker='v', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        # plot insignificant diff
        sea_lig = lig_sst_regrid_alltime[model]['sea'][0::4].values
        sea_pi = pi_sst_regrid_alltime[model]['sea'][0::4].values
        ttest_fdr_res = ttest_fdr_control(sea_lig, sea_pi,)
        
        # plot diff
        sm_lig_pi = lig_pi_sst_regrid_alltime[model]['sm'][0].values.copy()
        sm_lig_pi[ttest_fdr_res == False] = np.nan
        
        plt_mesh = axs[irow, jcol].contourf(
            lon, lat, sm_lig_pi, levels=pltlevel, extend='both',
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(sm_lig_pi)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(sm_lig_pi)],
        #     s=0.5, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),)
        
        # # calculate mean error
        # sm_lig_pi = lig_pi_sst_regrid_alltime[model]['sm'][0].values.copy()
        # diff = {}
        # area = {}
        # mean_diff = {}
        
        # for iregion in ['SO', 'Atlantic', 'Indian', 'Pacific']:
        #     diff[iregion] = sm_lig_pi[mask[iregion]]
        #     area[iregion] = cdo_area1deg.cell_area.values[mask[iregion]]
        #     mean_diff[iregion] = np.ma.average(
        #         np.ma.MaskedArray(diff[iregion], mask=np.isnan(diff[iregion])),
        #         weights=area[iregion],)
        
        plt.text(
            0.5, 1.05,
            model,
            # model + ' (' + \
            #     str(np.round(mean_diff['SO'], 1)) + ', ' + \
            #         str(np.round(mean_diff['Atlantic'], 1)) + ', ' + \
            #             str(np.round(mean_diff['Indian'], 1)) + ', ' + \
            #                 str(np.round(mean_diff['Pacific'], 1)) + ')',
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


