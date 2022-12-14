

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
# 4 cores
ec_sst_rec['AIS_am'] = ec_sst_rec['original'].loc[
    ec_sst_rec['original']['Area']=='Antarctica',]

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
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']], ]
# 7 cores
jh_sst_rec['SO_djf'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='Southern Ocean') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']], ]
# 9 cores
jh_sst_rec['NH_ann'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='North Atlantic') & \
        ['Annual SST' in string for string in jh_sst_rec['original']['Type']], ]
# 9 cores
jh_sst_rec['NH_sum'] = jh_sst_rec['original'].loc[
    (jh_sst_rec['original']['Region']=='North Atlantic') & \
        ['Summer SST' in string for string in jh_sst_rec['original']['Type']], ]



with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

lig_datasets = pd.read_excel(
    'data_sources/LIG/lig_datasets.xlsx', header=0, nrows=49,)

'''
# 14 cores for am
# 22 cores for djf
# 36 cores in total
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of am sst


output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst am reconstruction.png'
cbar_label = 'LIG annual mean SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

fig, ax = hemisphere_plot(northextent=-38)

ax.scatter(
    x = jh_sst_rec['SO_ann'].Longitude,
    y = jh_sst_rec['SO_ann'].Latitude,
    c = jh_sst_rec['SO_ann']['127 ka SST anomaly (°C)'],
    # s=10,
    s = 16 - 2.5 * jh_sst_rec['SO_ann']['127 ka 2σ (°C)'],
    lw=0.3, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = ec_sst_rec['SO_ann'].Longitude,
    y = ec_sst_rec['SO_ann'].Latitude,
    c = ec_sst_rec['SO_ann']['127 ka Median PIAn [°C]'],
    # s=10,
    s = 16 - 2.5 * ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = ec_sst_rec['AIS_am'].Longitude,
    y = ec_sst_rec['AIS_am'].Latitude,
    c = ec_sst_rec['AIS_am']['127 ka Median PIAn [°C]'],
    # s=10,
    s = 16 - 2.5 * ec_sst_rec['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='s', s=16 - 2.5 * 1, lw=0.3, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=16 - 2.5 * 2, lw=0.3, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=16 - 2.5 * 3, lw=0.3, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=16 - 2.5 * 4, lw=0.3, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, fontsize=8)
fig.savefig(output_png)


'''
np.max(jh_sst_rec['SO_ann']['127 ka 2σ (°C)']) - np.min(jh_sst_rec['SO_ann']['127 ka 2σ (°C)'])
np.max(ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]']) - np.min(ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]'])

sns.scatterplot(
    x = lig_datasets.Latitude, y = lig_datasets.Longitude,
    size = lig_datasets['two-sigma errors [°C]'],
    style = lig_datasets['Dataset'],
    transform=ccrs.PlateCarree(),
    )


, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of djf sst

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst djf reconstruction.png'
cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

fig, ax = hemisphere_plot(northextent=-38)

ax.scatter(
    x = jh_sst_rec['SO_djf'].Longitude,
    y = jh_sst_rec['SO_djf'].Latitude,
    c = jh_sst_rec['SO_djf']['127 ka SST anomaly (°C)'],
    # s=10,
    s = 16 - 2.5 * jh_sst_rec['SO_djf']['127 ka 2σ (°C)'],
    lw=0.3, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = ec_sst_rec['SO_djf'].Longitude,
    y = ec_sst_rec['SO_djf'].Latitude,
    c = ec_sst_rec['SO_djf']['127 ka Median PIAn [°C]'],
    # s=10,
    s = 16 - 2.5 * ec_sst_rec['SO_djf']['127 ka 2s PIAn [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = chadwick_interp.lon,
    y = chadwick_interp.lat,
    c = chadwick_interp.sst_sum,
    s=10, lw=0.3, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=16 - 2.5 * 1, lw=0.3, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=16 - 2.5 * 2, lw=0.3, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=16 - 2.5 * 3, lw=0.3, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=16 - 2.5 * 4, lw=0.3, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.savefig(output_png)


'''
, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations

output_png = 'figures/test/trial.pdf'
fig, ax = hemisphere_plot(northextent=-38, loceanarcs=True)

# output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.3_site_names/7.0.3 am sst site names.pdf'
# for isite in range(len(jh_sst_rec['SO_ann'].Longitude)):
#     ax.text(
#         jh_sst_rec['SO_ann'].Longitude.iloc[isite],
#         jh_sst_rec['SO_ann'].Latitude.iloc[isite],
#         jh_sst_rec['SO_ann'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     jh_sst_rec['SO_ann'].Longitude, jh_sst_rec['SO_ann'].Latitude, ax, s=10,
#     marker='s', )

# for isite in range(len(ec_sst_rec['SO_ann'].Longitude)):
#     ax.text(
#         ec_sst_rec['SO_ann'].Longitude.iloc[isite],
#         ec_sst_rec['SO_ann'].Latitude.iloc[isite],
#         ec_sst_rec['SO_ann'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     ec_sst_rec['SO_ann'].Longitude, ec_sst_rec['SO_ann'].Latitude, ax, s=10,
#     marker='o', )

# output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.3_site_names/7.0.3 summer sst site names.pdf'
# for isite in range(len(jh_sst_rec['SO_djf'].Longitude)):
#     ax.text(
#         jh_sst_rec['SO_djf'].Longitude.iloc[isite],
#         jh_sst_rec['SO_djf'].Latitude.iloc[isite],
#         jh_sst_rec['SO_djf'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     jh_sst_rec['SO_djf'].Longitude, jh_sst_rec['SO_djf'].Latitude, ax, s=10,
#     marker='s', )

# for isite in range(len(ec_sst_rec['SO_djf'].Longitude)):
#     ax.text(
#         ec_sst_rec['SO_djf'].Longitude.iloc[isite],
#         ec_sst_rec['SO_djf'].Latitude.iloc[isite],
#         ec_sst_rec['SO_djf'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     ec_sst_rec['SO_djf'].Longitude, ec_sst_rec['SO_djf'].Latitude, ax, s=10,
#     marker='o', )

for isite in range(len(chadwick_interp.lon)):
    ax.text(
        chadwick_interp.lon.iloc[isite],
        chadwick_interp.lat.iloc[isite],
        chadwick_interp.sites.iloc[isite],
        transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
cplot_ice_cores(
    chadwick_interp.lon, chadwick_interp.lat, ax, s=10,
    marker='^', )


fig.savefig(output_png)


'''
# lig_datasets_jh = lig_datasets.loc[
#     lig_datasets.Dataset == 'Hoffman et al. (2017)']
# for isite in range(len(lig_datasets_jh.Longitude)):
#     ax.text(
#         lig_datasets_jh.Longitude.iloc[isite],
#         lig_datasets_jh.Latitude.iloc[isite],
#         lig_datasets_jh.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_jh.Longitude, lig_datasets_jh.Latitude, ax, s=10,
#     marker='s', )

# lig_datasets_mc = lig_datasets.loc[
#     lig_datasets.Dataset == 'Chadwick et al. (2021)']
# for isite in range(len(lig_datasets_mc.Longitude)):
#     ax.text(
#         lig_datasets_mc.Longitude.iloc[isite],
#         lig_datasets_mc.Latitude.iloc[isite],
#         lig_datasets_mc.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_mc.Longitude, lig_datasets_mc.Latitude, ax, s=10,
#     marker='^', )

# lig_datasets_ec = lig_datasets.loc[
#     lig_datasets.Dataset == 'Capron et al. (2017)']
# for isite in range(len(lig_datasets_ec.Longitude)):
#     ax.text(
#         lig_datasets_ec.Longitude.iloc[isite],
#         lig_datasets_ec.Latitude.iloc[isite],
#         lig_datasets_ec.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_ec.Longitude, lig_datasets_ec.Latitude, ax, s=10,
#     marker='o', )

double station entries: ODP-1089, MD88-770
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of sep sic

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.1_sic/7.0.1.0 lig sic sep reconstruction.png'
cbar_label = 'LIG Sep SIC [$\%$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-50)

ax.scatter(
    x = chadwick_interp.lon,
    y = chadwick_interp.lat,
    c = chadwick_interp.sic_sep,
    s=10, lw=0.3, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
# cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.savefig(output_png)


'''
, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region analyse HadISST_sst

HadISST_sst = xr.open_dataset('data_sources/products/HadISST/HadISST_sst.nc')
HadISST_sst.sst.values[HadISST_sst.sst.values == -1000] = np.nan

HadISST_sst_alltime = mon_sea_ann(var_monthly=HadISST_sst.sst)



HadISST_sst_changes = HadISST_sst_alltime['ann'].isel(
    time=slice(-31, -1)).mean(dim='time') - \
    HadISST_sst_alltime['ann'].isel(
        time=slice(0, 30)).mean(dim='time')

HadISST_sst_changes.to_netcdf('scratch/test/test0.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of am sst NH


output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst am reconstruction NH.png'
cbar_label = 'LIG annual mean SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

fig, ax = hemisphere_plot(southextent=38, lw=0.1)

size_scale = 1.5

ax.scatter(
    x = jh_sst_rec['NH_ann'].Longitude,
    y = jh_sst_rec['NH_ann'].Latitude,
    c = jh_sst_rec['NH_ann']['127 ka SST anomaly (°C)'],
    s = 16 - size_scale * jh_sst_rec['NH_ann']['127 ka 2σ (°C)'],
    lw=0.3, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = ec_sst_rec['NH_ann'].Longitude,
    y = ec_sst_rec['NH_ann'].Latitude,
    c = ec_sst_rec['NH_ann']['127 ka Median PIAn [°C]'],
    s = 16 - size_scale * ec_sst_rec['NH_ann']['127 ka 2s PIAn [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='s', s=16 - size_scale * 1, lw=0.3, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=16 - size_scale * 2, lw=0.3, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=16 - size_scale * 3, lw=0.3, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=16 - size_scale * 4, lw=0.3, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, fontsize=8)
fig.savefig(output_png)


'''
# ax.scatter(
#     x = ec_sst_rec['GrIS_am'].Longitude,
#     y = ec_sst_rec['GrIS_am'].Latitude,
#     c = ec_sst_rec['GrIS_am']['127 ka Median PIAn [°C]'],
#     s = 16 - 2.5 * ec_sst_rec['GrIS_am']['127 ka 2s PIAn [°C]'],
#     lw=0.3, marker='o', edgecolors = 'black', zorder=2,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of summer sst NH

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 lig-pi sst summer reconstruction NH.png'
cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

fig, ax = hemisphere_plot(southextent=38, lw=0.1)

size_scale = 1.5

ax.scatter(
    x = jh_sst_rec['NH_sum'].Longitude,
    y = jh_sst_rec['NH_sum'].Latitude,
    c = jh_sst_rec['NH_sum']['127 ka SST anomaly (°C)'],
    s = 16 - size_scale * jh_sst_rec['NH_sum']['127 ka 2σ (°C)'],
    lw=0.3, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

ax.scatter(
    x = ec_sst_rec['NH_sum'].Longitude,
    y = ec_sst_rec['NH_sum'].Latitude,
    c = ec_sst_rec['NH_sum']['127 ka Median PIAn [°C]'],
    s = 16 - size_scale * ec_sst_rec['NH_sum']['127 ka 2s PIAn [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 1, lw=0.3, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 2, lw=0.3, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 3, lw=0.3, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 4, lw=0.3, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.savefig(output_png)


'''
, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------



