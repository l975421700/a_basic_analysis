

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

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/tas/tas_regrid_alltime_ens_stats.pkl', 'rb') as f:
    tas_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

lon = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lon
lat = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lat


'''
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


# 14 cores for am
# 22 cores for djf
# 36 cores in total
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of am sst/sat


output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec am sst lig-pi.png'
cbar_label = 'Annual SST and SAT [$°C$]\nPMIP4'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

fig, ax = hemisphere_plot(northextent=-38,)

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


cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
fig.savefig(output_png)


'''
sns.scatterplot(
    x = lig_datasets.Latitude, y = lig_datasets.Longitude,
    size = lig_datasets['two-sigma errors [°C]'],
    style = lig_datasets['Dataset'],
    transform=ccrs.PlateCarree(),
    )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of djf sst


output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec jfm sst lig-pi.png'
cbar_label = 'Summer SST [$°C$]\nPMIP4'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

fig, ax = hemisphere_plot(northextent=-38,)

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
    s = max_size - scale_size * 1,
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

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.savefig(output_png)


'''
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
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)


, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of sep sic


plt_data = sic_regrid_alltime_ens_stats['lig_pi']['mm']['mean'][8].values

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

plt_data[np.isnan(sst_regrid_alltime_ens_stats['lig_pi']['sm']['mean'][0].values)] = np.nan

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec sep sic lig-pi.png'
cbar_label = 'Sep SIC [$\%$]\nPMIP4'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-70, cm_max=20, cm_interval1=10, cm_interval2=10, cmap='PuOr',
    reversed=False, asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-50,)

ax.pcolormesh(
    lon,
    lat,
    plt_data,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

plt_scatter = ax.scatter(
    x = lig_recs['MC']['interpolated'].Longitude,
    y = lig_recs['MC']['interpolated'].Latitude,
    c = lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'],
    s=60, lw=0.5, marker='^', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
fig.savefig(output_png)


'''
, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


