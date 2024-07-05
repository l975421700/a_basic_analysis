

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
from sklearn.metrics import mean_squared_error

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
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
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


# import reconstructions

lig_recs = {}

with open('scratch/cmip6/lig/rec/lig_recs_dc.pkl', 'rb') as f:
    lig_recs['DC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_ec.pkl', 'rb') as f:
    lig_recs['EC'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_jh.pkl', 'rb') as f:
    lig_recs['JH'] = pickle.load(f)

with open('scratch/cmip6/lig/rec/lig_recs_mc.pkl', 'rb') as f:
    lig_recs['MC'] = pickle.load(f)


# import model site values

with open('scratch/cmip6/lig/sst/SO_ann_sst_site_values.pkl', 'rb') as f:
    SO_ann_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sst/SO_jfm_sst_site_values.pkl', 'rb') as f:
    SO_jfm_sst_site_values = pickle.load(f)

with open('scratch/cmip6/lig/tas/AIS_ann_tas_site_values.pkl', 'rb') as f:
    AIS_ann_tas_site_values = pickle.load(f)

with open('scratch/cmip6/lig/sic/SO_sep_sic_site_values.pkl', 'rb') as f:
    SO_sep_sic_site_values = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot summer sst against latitude


symbol_size = 60
linewidth = 1
alpha = 0.75

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

ax.scatter(
    lig_recs['DC']['JFM_128']['Latitude'],
    lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
    marker=marker_recs['DC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

ax.scatter(
    lig_recs['EC']['SO_jfm']['Latitude'],
    lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
    marker=marker_recs['EC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

ax.scatter(
    lig_recs['JH']['SO_jfm']['Latitude'],
    lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
    marker=marker_recs['JH'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

ax.scatter(
    lig_recs['MC']['interpolated']['Latitude'],
    lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
    marker=marker_recs['MC'],
    s=symbol_size, c='white', edgecolors='k', lw=linewidth, alpha=alpha,
    )

ax.set_xlabel('Latitude [$° \; S$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Summer SST anomalies [$°C$]')
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.97)
fig.savefig('figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.3_rec_site_values/7.0.3.3.0 Summer SST anomalies vs. latitude.png')





'''

lig_recs['EC']['SO_ann']['Latitude']
lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]']

lig_recs['JH']['SO_ann']['Latitude']
lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)']


lig_recs['DC']['annual_128']['Latitude']
lig_recs['DC']['annual_128']['sst_anom_hadisst_ann']




recs = ['EC', 'JH', 'DC', 'MC',]

# annual_sst
lig_datasets = {}
lig_anom_name = {}
lig_datasets['annual_sst'] = {
    'EC': lig_recs['EC']['SO_ann'],
    'JH': lig_recs['JH']['SO_ann'],
    'DC': lig_recs['DC']['annual_128'],
    'MC': None,}
lig_anom_name['annual_sst'] = {
    'EC': '127 ka Median PIAn [°C]',
    'JH': '127 ka SST anomaly (°C)',
    'DC': 'sst_anom_hadisst_ann',
    'MC': None,}


for irec in ['EC', 'JH', 'DC',]:
    # irec = 'EC'
    print(irec)
    
    



lig_datasets['annual_sst'][irec]['Latitude']
lig_datasets['annual_sst'][irec][lig_anom_name['annual_sst'][irec]]

marker_recs[irec]

'''
# endregion
# -----------------------------------------------------------------------------
