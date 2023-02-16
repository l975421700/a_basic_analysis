

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


pmip3_lig_sim = {}
pmip3_lig_sim['annual_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sst_c.nc')
pmip3_lig_sim['summer_sst'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sstdjf_c.nc')

pmip3_lig_sim['annual_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfc_c.nc')
pmip3_lig_sim['summer_sat'] = xr.open_dataset('data_sources/LIG/Supp_Info_PMIP3/netcdf_data_for_ensemble/LIG_ensemble_sfcdjf_c.nc')

longitude = pmip3_lig_sim['annual_sst'].longitude
latitude = pmip3_lig_sim['annual_sst'].latitude


'''
# lon = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lon
# lat = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lat
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of am sst/sat


output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec am sst lig-pi_pmip3.png'
cbar_label = 'Annual SST and SAT [$°C$]\nPMIP3'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

fig, ax = hemisphere_plot(northextent=-38,)

ax.pcolormesh(
    longitude,
    latitude,
    pmip3_lig_sim['annual_sst'].sst,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    longitude,
    latitude,
    pmip3_lig_sim['annual_sat'].sfc,
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
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot reconstructions of djf sst


output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec jfm sst lig-pi_pmip3.png'
cbar_label = 'Summer SST [$°C$]\nPMIP3'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

fig, ax = hemisphere_plot(northextent=-38,)

ax.pcolormesh(
    longitude,
    latitude,
    pmip3_lig_sim['summer_sst'].sstdjf,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

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
fig.savefig(output_png)


'''
, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


