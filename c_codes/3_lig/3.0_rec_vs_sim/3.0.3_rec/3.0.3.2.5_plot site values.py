

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
sys.path.append('/home/users/qino')
os.chdir('/home/users/qino')

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
    regional_plot,
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
    marker_recs,
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

lig_datasets = pd.read_excel('data_sources/LIG/lig_datasets.xlsx', header=0,)

lig_datasubsets = {}
lig_datasubsets['annual_sst'] = lig_datasets.loc[
    lig_datasets.Type == 'Annual SST']
lig_datasubsets['summer_sst'] = lig_datasets.loc[
    lig_datasets.Type == 'Summer SST']
lig_datasubsets['annual_sat'] = lig_datasets.loc[
    lig_datasets.Type == 'Annual SAT']


with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/tas/tas_regrid_alltime_ens_stats.pkl', 'rb') as f:
    tas_regrid_alltime_ens_stats = pickle.load(f)

lon = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lon
lat = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lat

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - Antarctica ccrs.PlateCarree am sst/sat

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_SO.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[-180, 180, -90, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 30,
    figsize = np.array([30, 7.2]) / 2.54,
    central_longitude = 180,)

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.3), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.4, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.16, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

# fig.subplots_adjust(left=0.03, right=0.999, bottom=0.12, top=0.999)

fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - NZ am sst/sat

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_zoomNZ.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[140, 190, -55, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 10, ymajortick_int = 5,
    figsize = np.array([11.5, 7.2]) / 2.54,
    central_longitude = 180,)

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (-0.1, -0.3), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.16, format=remove_trailing_zero_pos,
    anchor=(1., 1),
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

# fig.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.98)

fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - zoomIndian

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_zoomIndian.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[75, 100, -50, -40]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    xminortick_int = 2.5, yminortick_int = 2.5,
    figsize = np.array([11.5, 6.4]) / 2.54,)

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (-0.1, -0.3), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.16, format=remove_trailing_zero_pos,
    anchor=(1, 1),
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

# fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.98)

fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - Antarctica ccrs.PlateCarree summer sst

cbar_label = 'LIG summer SST anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_SO.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[-180, 180, -90, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 30,
    figsize = np.array([30, 7.2]) / 2.54,
    central_longitude = 180,)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.3), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.4, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.16, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

# fig.subplots_adjust(left=0.03, right=0.999, bottom=0.12, top=0.999)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - zoom_indian summer sst

cbar_label = 'LIG summer SST anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_zoomIndian.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[0, 15, -60, -35]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    figsize = np.array([4, 6.2]) / 2.54,
    central_longitude = 0,)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (-0.5, -0.55), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1.4, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.16, format=remove_trailing_zero_pos,
    anchor=(0.7, 1),
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, size=8)

# fig.subplots_adjust(left=0.22, right=0.9, bottom=0.18, top=0.98)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - zoom_indian2 summer sst

cbar_label = 'LIG summer SST anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_zoomIndian2.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[70, 100, -60, -40]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    figsize = np.array([8.4, 7.2]) / 2.54,)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.2, -0.52), handletextpad=0.05, columnspacing=0.3,)


cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2, format=remove_trailing_zero_pos,
    anchor=(0.7, 1),
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, size=8)

# fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.98)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - zoom_NZ summer sst

cbar_label = 'LIG summer SST anomalies [$°C$]'
output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_zoomNZ.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 40
scale_size = 8

extent=[165, 180, -50, -35]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    xminortick_int = 5, yminortick_int = 5,
    figsize = np.array([5.4, 6.2]) / 2.54,)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0, -0.55), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

# fig.subplots_adjust(left=0.16, right=0.95, bottom=0.16, top=0.98)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot site values - Atlantic sector


#-------------------------------- annual SST

extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 6.6]) / 2.54)

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.2, -0.5), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_atlantic.png'
fig.savefig(output_png)


#-------------------------------- Summer SST


extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 6.6]) / 2.54)

cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.2, -0.5), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_atlantic.png'
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - Indian sector

#-------------------------------- annual SST

extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 6.6]) / 2.54)

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.3, -0.5), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=40,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_indian.png'
fig.savefig(output_png)


#-------------------------------- Summer SST


extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 6.6]) / 2.54)

cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.3, -0.5), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=40,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_indian.png'
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site values - Pacific sector


#-------------------------------- annual SST

extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 6.4]) / 2.54,
    central_longitude = 180)

cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.3, -0.525), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of annual sst_sat reconstructions_pacific.png'
fig.savefig(output_png)


#-------------------------------- Summer SST

extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 6.4]) / 2.54,
    central_longitude = 180)

cbar_label = 'LIG summer SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1.09,
    lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.3, -0.525), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.14, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3 anomalies of summer sst reconstructions_pacific.png'
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

