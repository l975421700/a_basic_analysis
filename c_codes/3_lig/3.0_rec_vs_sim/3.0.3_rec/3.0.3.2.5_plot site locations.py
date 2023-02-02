

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
# -----------------------------------------------------------------------------
# region plot site locations - Antarctic plot

#-------------------------------- annual SST

# datasets: symbol

fig, ax = hemisphere_plot(
    northextent=-38,
    figsize=np.array([5.8, 6.8]) / 2.54, fm_bottom=0.12,
    loceanarcs=True,
    )

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.08, 'Site locations of\nLIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, linespacing=1.5)

'''
for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6,
    )
'''

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions.pdf'
fig.savefig(output_png)




#-------------------------------- Summer SST

fig, ax = hemisphere_plot(
    northextent=-38,
    figsize=np.array([5.8, 6.8]) / 2.54, fm_bottom=0.12,
    loceanarcs=True,
    )

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.08, 'Site locations of\nLIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, linespacing=1.5)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions.pdf'
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations - Atlantic sector

symbol_size = 25

#-------------------------------- annual SST

extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 5.6]) / 2.54)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.16, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6,
    )

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_atlantic.pdf'
fig.savefig(output_png)


#-------------------------------- Summer SST


extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 5.6]) / 2.54)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.16, 'Site locations of LIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_atlantic.pdf'
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations - Indian sector

symbol_size = 25

#-------------------------------- annual SST
extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 5.8]) / 2.54)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.16, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6,
    )

fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_indian.pdf'
fig.savefig(output_png)


#-------------------------------- Summer SST


extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 5.8]) / 2.54)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.16, 'Site locations of LIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_indian.pdf'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations - Pacific sector

symbol_size = 25


#-------------------------------- annual SST

extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 5.6]) / 2.54,
    central_longitude = 180)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.16, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6,
    )

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_pacific.pdf'
fig.savefig(output_png)


#-------------------------------- Summer SST

extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 5.6]) / 2.54,
    central_longitude = 180)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = 20,
    lw=0.5, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)
plt.text(
    0.5, -0.16, 'Site locations of LIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=6, clip_on=True,
    )

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.99)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_pacific.pdf'
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations - Antarctica ccrs.PlateCarree

symbol_size = 30
linewidth = 1
fontsize = 12

#-------------------------------- annual SST

extent=[-180, 180, -90, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 30,
    figsize = np.array([30, 6.2]) / 2.54,
    central_longitude = 180,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.15, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=fontsize,)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize,
    )

fig.subplots_adjust(left=0.03, right=0.999, bottom=0.12, top=0.999)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_SO.pdf'
fig.savefig(output_png)




#-------------------------------- Summer SST


extent=[-180, 180, -90, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 30,
    figsize = np.array([30, 6.2]) / 2.54,
    central_longitude = 180,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.15, 'Site locations of LIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=fontsize,)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

fig.subplots_adjust(left=0.03, right=0.999, bottom=0.12, top=0.999)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_SO.pdf'
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations - zoomin

symbol_size = 30
linewidth = 1
fontsize = 12

#-------------------------------- annual SST - NZ

extent=[140, 190, -55, -30]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 10, ymajortick_int = 5,
    figsize = np.array([11.5, 6.2]) / 2.54,
    central_longitude = 180,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.15, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=fontsize,)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize,
    )

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.98)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_NZ.pdf'
fig.savefig(output_png)


#-------------------------------- annual SST - zoomIndian

extent=[75, 100, -50, -40]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    xminortick_int = 2.5, yminortick_int = 2.5,
    figsize = np.array([11.5, 5.4]) / 2.54,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.18, 'Site locations of LIG annual SST/SAT reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=fontsize,)

for itext in range(len(lig_datasubsets['annual_sst']['No.'])):
    ax.text(
        lig_datasubsets['annual_sst']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sst']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

for itext in range(len(lig_datasubsets['annual_sat']['No.'])):
    ax.text(
        lig_datasubsets['annual_sat']['Longitude'].iloc[itext],
        lig_datasubsets['annual_sat']['Latitude'].iloc[itext],
        lig_datasubsets['annual_sat']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize,
    )

fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.98)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of annual sst_sat reconstructions_zoomIndian.pdf'
fig.savefig(output_png)



#-------------------------------- summer SST - zoom_indian

extent=[0, 15, -60, -35]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    figsize = np.array([4, 6.2]) / 2.54,
    central_longitude = 0,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.4, -0.2, 'Site locations of\nLIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=8,)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

fig.subplots_adjust(left=0.22, right=0.9, bottom=0.18, top=0.98)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_zoomIndian.pdf'
fig.savefig(output_png)




#-------------------------------- summer SST - zoom_indian2

extent=[70, 100, -60, -40]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    figsize = np.array([8.4, 6.2]) / 2.54,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.18, 'Site locations of LIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=10,)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.98)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_zoomIndian2.pdf'
fig.savefig(output_png)




#-------------------------------- summer SST - zoom_NZ

extent=[165, 180, -50, -35]
fig, ax = regional_plot(
    extent=extent,
    xmajortick_int = 5, ymajortick_int = 5,
    xminortick_int = 5, yminortick_int = 5,
    figsize = np.array([5.4, 6.2]) / 2.54,)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_jfm'].Longitude,
    y = lig_recs['JH']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='s', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# EC
ax.scatter(
    x = lig_recs['EC']['SO_jfm'].Longitude,
    y = lig_recs['EC']['SO_jfm'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='o', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# MC
ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='^', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

# DC
ax.scatter(
    x = lig_recs['DC']['JFM_128'].Longitude,
    y = lig_recs['DC']['JFM_128'].Latitude,
    c = 'white', s = symbol_size,
    lw=linewidth, marker='v', edgecolors = 'blue', zorder=2, alpha=0.75,
    transform=ccrs.PlateCarree(),)

plt.text(
    0.5, -0.2, 'Site locations of\nLIG summer SST reconstructions',
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=10,)

for itext in range(len(lig_datasubsets['summer_sst']['No.'])):
    ax.text(
        lig_datasubsets['summer_sst']['Longitude'].iloc[itext],
        lig_datasubsets['summer_sst']['Latitude'].iloc[itext],
        lig_datasubsets['summer_sst']['No.'].iloc[itext],
        horizontalalignment='center', verticalalignment='center',
        transform=ccrs.PlateCarree(), size=fontsize, clip_on=True,
    )

fig.subplots_adjust(left=0.16, right=0.95, bottom=0.16, top=0.98)

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.1_rec_site_locations/7.0.3.1 locations of summer sst reconstructions_zoomNZ.pdf'
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


