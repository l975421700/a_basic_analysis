

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
    # usecols=['Station', 'Latitude', 'Longitude', 'Area', 'Type',
    #          '127 ka Median PIAn [°C]', '127 ka 2s PIAn [°C]'],
    )

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


#-------- import dc reconstructions
# 30 cores
dc_sst_rec = {}
dc_sst_rec['original'] = pd.read_csv(
    'data_sources/LIG/Chandler-Langebroek_2021_SST-anom.tab',
    sep='\t', header=0, skiprows=76,
    )
dc_sst_rec['summary'] = pd.read_csv(
    'data_sources/LIG/Chandler-Langebroek_2021_SST-anom-stats.tab',
    sep='\t', header=0, skiprows=25,
    )
# 25 cores
dc_sst_rec['ann'] = dc_sst_rec['original'].loc[
    (dc_sst_rec['original']['Season (A annual, S summer (months JFM))'] == 'A')
]
# 26 cores
dc_sst_rec['sum'] = dc_sst_rec['original'].loc[
    (dc_sst_rec['original']['Season (A annual, S summer (months JFM))'] == 'S')
]

dc_sst_rec['ann_124'] = dc_sst_rec['ann'].loc[dc_sst_rec['ann']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 124]
dc_sst_rec['ann_126'] = dc_sst_rec['ann'].loc[dc_sst_rec['ann']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 126]
dc_sst_rec['ann_128'] = dc_sst_rec['ann'].loc[dc_sst_rec['ann']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 128]
dc_sst_rec['ann_130'] = dc_sst_rec['ann'].loc[dc_sst_rec['ann']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 130]

dc_sst_rec['sum_124'] = dc_sst_rec['sum'].loc[dc_sst_rec['sum']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 124]
dc_sst_rec['sum_126'] = dc_sst_rec['sum'].loc[dc_sst_rec['sum']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 126]
dc_sst_rec['sum_128'] = dc_sst_rec['sum'].loc[dc_sst_rec['sum']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 128]
dc_sst_rec['sum_130'] = dc_sst_rec['sum'].loc[dc_sst_rec['sum']['Age [ka BP] (Chronology follows Lisiecki a...)'] == 130]





'''
ec_sst_rec['SO_ann']['127-130 ka diff [°C]']
ec_sst_rec['SO_ann']['127-130 ka diff 2s [°C]']

ec_sst_rec['SO_djf']['127-130 ka diff [°C]']
ec_sst_rec['SO_djf']['127-130 ka diff 2s [°C]']

ec_sst_rec['AIS_am']['127-130 ka diff [°C]']
ec_sst_rec['AIS_am']['127-130 ka diff 2s [°C]']

len(dc_sst_rec['sum']['Event'].unique())
dc_sst_rec['sum_130']['Season (A annual, S summer (months JFM))']

#---------------- check correspondence

dc_sst_rec['summary'].loc[dc_sst_rec['summary']['Age [ka BP]'] == 126][
    'NOBS [#] (Number of records, annual SST)']
dc_sst_rec['summary'].loc[dc_sst_rec['summary']['Age [ka BP]'] == 126][
    'SST anomaly [°C] (Mean annual SST anomaly, rela...)']

dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'].count()
np.nanmean(dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'])

dc_sst_rec['summary'].loc[dc_sst_rec['summary']['Age [ka BP]'] == 126][
    'NOBS [#] (Number of records, summer (JF...)']
dc_sst_rec['summary'].loc[dc_sst_rec['summary']['Age [ka BP]'] == 126][
    'SST anomaly [°C] (Mean summer (JFM) SST anomaly...)']

dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'].count()
np.nanmean(dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'])



np.nanmean(dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'])
np.nanmean(dc_sst_rec['ann_128']['SST anomaly [°C] (Anomaly relative to the World...)'])

np.nanmean(dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'])
np.nanmean(dc_sst_rec['sum_128']['SST anomaly [°C] (Anomaly relative to the World...)'])

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region diff 127 - 130

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 127-130 ka sst reconstruction.png'
cbar_label = '127-130 ka SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG',)

size_scale = 4

fig, ax = hemisphere_plot(northextent=-38)

# ax.scatter(
#     x = jh_sst_rec['SO_ann'].Longitude,
#     y = jh_sst_rec['SO_ann'].Latitude,
#     c = jh_sst_rec['SO_ann']['127 ka SST anomaly (°C)'],
#     # s=10,
#     s = 16 - 2.5 * jh_sst_rec['SO_ann']['127 ka 2σ (°C)'],
#     lw=0.3, marker='s', edgecolors = 'black', zorder=2,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


# EC, SO, am
ax.scatter(
    x = ec_sst_rec['SO_ann'].Longitude,
    y = ec_sst_rec['SO_ann'].Latitude,
    c = ec_sst_rec['SO_ann']['127-130 ka diff [°C]'],
    s = 16 - size_scale * ec_sst_rec['SO_ann']['127-130 ka diff 2s [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC, AIS, am
ax.scatter(
    x = ec_sst_rec['AIS_am'].Longitude,
    y = ec_sst_rec['AIS_am'].Latitude,
    c = ec_sst_rec['AIS_am']['127-130 ka diff [°C]'],
    # s=10,
    s = 16 - size_scale * ec_sst_rec['AIS_am']['127-130 ka diff 2s [°C]'],
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC, SO, djf
ax.scatter(
    x = ec_sst_rec['SO_djf'].Longitude,
    y = ec_sst_rec['SO_djf'].Latitude,
    c = ec_sst_rec['SO_djf']['127-130 ka diff [°C]'],
    s = 16 - size_scale * ec_sst_rec['SO_djf']['127-130 ka diff 2s [°C]'],
    lw=0.3, marker='o', edgecolors = 'red', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 0.5, lw=0.3, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 1, lw=0.3, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 1.5, lw=0.3, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='o', s=16 - size_scale * 2, lw=0.3, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['0.5', '1', '1.5', '2 $°C$'], ncol=4, frameon=False,
    loc = (0.01, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, fontsize=8)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region DC diff 126 - 128

anndiff = dc_sst_rec['ann_128']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values
sumdiff = dc_sst_rec['sum_128']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values

print(np.nanmean(anndiff))
print(np.nanmean(sumdiff))

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 128-126 ka sst reconstruction dc.png'
cbar_label = '128-126 ka SST anomalies [$°C$]'


anndiff = dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['ann_130']['SST anomaly [°C] (Anomaly relative to the World...)'].values
sumdiff = dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['sum_130']['SST anomaly [°C] (Anomaly relative to the World...)'].values

print(np.nanmean(anndiff))
print(np.nanmean(sumdiff))

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 126-130 ka sst reconstruction dc.png'
cbar_label = '126-130 ka SST anomalies [$°C$]'


anndiff = dc_sst_rec['ann_128']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['ann_130']['SST anomaly [°C] (Anomaly relative to the World...)'].values
sumdiff = dc_sst_rec['sum_128']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['sum_130']['SST anomaly [°C] (Anomaly relative to the World...)'].values

print(np.nanmean(anndiff))
print(np.nanmean(sumdiff))

output_png = 'figures/7_lig/7.0_boundary_conditions/7.0.0_sst/7.0.0.0 128-130 ka sst reconstruction dc.png'
cbar_label = '128-130 ka SST anomalies [$°C$]'


pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG',)

fig, ax = hemisphere_plot(northextent=-38)


# DC, SO, am
ax.scatter(
    x = dc_sst_rec['ann_126'].Longitude,
    y = dc_sst_rec['ann_126'].Latitude,
    c = anndiff,
    s = 10,
    lw=0.3, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC, SO, djf
ax.scatter(
    x = dc_sst_rec['sum_126'].Longitude,
    y = dc_sst_rec['sum_126'].Latitude,
    c = sumdiff,
    s = 10,
    lw=0.3, marker='o', edgecolors = 'red', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# l1 = plt.scatter(
#     [],[], c='white', marker='o', s=16 - size_scale * 0.5, lw=0.3, edgecolors = 'black',)
# l2 = plt.scatter(
#     [],[], c='white', marker='o', s=16 - size_scale * 1, lw=0.3, edgecolors = 'black',)
# l3 = plt.scatter(
#     [],[], c='white', marker='o', s=16 - size_scale * 1.5, lw=0.3, edgecolors = 'black',)
# l4 = plt.scatter(
#     [],[], c='white', marker='o', s=16 - size_scale * 2, lw=0.3, edgecolors = 'black',)
# plt.legend(
#     [l1, l2, l3, l4,], ['0.5', '1', '1.5', '2 $°C$'], ncol=4, frameon=False,
#     loc = (0.01, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5, fontsize=8)
fig.savefig(output_png)




'''
# On average, SO is warmer in 126 than in 124/130

anndiff = dc_sst_rec['ann_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['ann_124']['SST anomaly [°C] (Anomaly relative to the World...)'].values
sumdiff = dc_sst_rec['sum_126']['SST anomaly [°C] (Anomaly relative to the World...)'].values - dc_sst_rec['sum_124']['SST anomaly [°C] (Anomaly relative to the World...)'].values



print((dc_sst_rec['ann_126']['Site'].values == dc_sst_rec['ann_128']['Site'].values).all())

print((dc_sst_rec['sum_126']['Site'].values == dc_sst_rec['sum_128']['Site'].values).all())
'''
# endregion
# -----------------------------------------------------------------------------
