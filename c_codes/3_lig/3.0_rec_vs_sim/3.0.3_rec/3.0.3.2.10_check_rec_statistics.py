

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
    find_ilat_ilon_general,
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
# region Number of cores

# annual SST
print(lig_recs['EC']['SO_ann'].shape)
print(lig_recs['JH']['SO_ann'].shape)
print(lig_recs['DC']['annual_128'].shape)

# summer SST
print(lig_recs['EC']['SO_jfm'].shape)
print(lig_recs['JH']['SO_jfm'].shape)
print(lig_recs['DC']['JFM_128'].shape)
print(lig_recs['MC']['interpolated'].shape)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Averaged SST/SAT/SIC anomalies

#---- SO annual SST
# EC
lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]']

# JH
mean_value = np.mean(lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'])
std_values = np.std(lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

# DC
mean_value = np.mean(lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'])
std_values = np.std(lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'], ddof = 1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


#---- SO summer SST
# EC
mean_value = np.mean(lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'])
std_values = np.std(lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

# JH
mean_value = np.mean(lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'])
std_values = np.std(lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

# DC
mean_value = np.mean(lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'])
std_values = np.std(lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

# MC
mean_value = np.mean(lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'])
std_values = np.std(lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#---- AIS SAT
mean_value = np.mean(lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'])
std_values = np.std(lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))

#---- Sep SIC
mean_value = np.mean(lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'])
std_values = np.std(lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'], ddof=1)
print(str(np.round(mean_value, 1)) + ' ± ' + str(np.round(std_values, 1)))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check annual SST

all_rec = np.concatenate((
    lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'].values,
    lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'].values,
    lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'].values,
))
np.min(all_rec)
np.max(all_rec)


# EC
lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]']
lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]']

# DC
lig_recs['DC']['annual_128']['sst_anom_hadisst_ann']

# JH
lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)']
lig_recs['JH']['SO_ann']['127 ka 2σ (°C)']

neg_indices = np.where(lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'] < 0)

for iind in neg_indices[0]:
    # iind = neg_indices[0][0]
    print(iind)
    
    station = lig_recs['JH']['SO_ann']['Station'].iloc[iind]
    reconstruction = np.round(
        lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'].iloc[iind],
        1)
    uncertainty = np.round(
        lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'].iloc[iind],
        1)
    
    print(station + ': ' + str(reconstruction) + '±' + str(uncertainty))


all_sim = np.concatenate((
    SO_ann_sst_site_values['EC'].groupby(['Station']).mean()[
        ['sim_ann_sst_lig_pi']],
    SO_ann_sst_site_values['JH'].groupby(['Station']).mean()[
        ['sim_ann_sst_lig_pi']],
    SO_ann_sst_site_values['DC'].groupby(['Station']).mean()[
        ['sim_ann_sst_lig_pi']],
))
np.min(all_sim)
np.max(all_sim)

(all_sim <= -0.05).sum()







# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check annual SAT

lig_recs['EC']['AIS_am']['Station']
lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]']
lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]']

AIS_ann_tas_site_values['EC'].groupby(['Station']).mean()['sim_ann_tas_lig_pi']

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check summer SST

all_rec = np.concatenate((
    lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'].values,
    lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'].values,
    lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'].values,
    lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'].values,
))

np.min(all_rec)
np.max(all_rec)


lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'].values
lig_recs['JH']['SO_jfm']['Station']

lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'].values
lig_recs['DC']['JFM_128']['Station']

(all_rec <= 0).sum()


neg_indices = np.where(lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'] < 0)

for iind in neg_indices[0]:
    # iind = neg_indices[0][0]
    print(iind)
    
    station = lig_recs['EC']['SO_jfm']['Station'].iloc[iind]
    reconstruction = np.round(
        lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'].iloc[iind],
        1)
    uncertainty = np.round(
        lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'].iloc[iind],
        1)
    
    print(station + ': ' + str(reconstruction) + '±' + str(uncertainty))

neg_indices = np.where(lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'] < 0)

for iind in neg_indices[0]:
    # iind = neg_indices[0][0]
    print(iind)
    
    station = lig_recs['JH']['SO_jfm']['Station'].iloc[iind]
    reconstruction = np.round(
        lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'].iloc[iind],
        1)
    uncertainty = np.round(
        lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'].iloc[iind],
        1)
    
    print(station + ': ' + str(reconstruction) + '±' + str(uncertainty))


lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'].values
lig_recs['MC']['interpolated']['Station']

all_sim = np.concatenate((
    SO_jfm_sst_site_values['EC'].groupby(['Station']).mean()[
        ['sim_jfm_sst_lig_pi']],
    SO_jfm_sst_site_values['JH'].groupby(['Station']).mean()[
        ['sim_jfm_sst_lig_pi']],
    SO_jfm_sst_site_values['DC'].groupby(['Station']).mean()[
        ['sim_jfm_sst_lig_pi']],
    SO_jfm_sst_site_values['MC'].groupby(['Station']).mean()[
        ['sim_jfm_sst_lig_pi']],
))
np.min(all_sim)
np.max(all_sim)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check Sep SIC

lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'].mean()
lig_recs['MC']['interpolated']['sic_anom_hadisst_sep']


SO_sep_sic_site_values['MC'].groupby(['Station']).mean()['sim_rec_sep_sic_lig_pi']
SO_sep_sic_site_values['MC'].groupby(['Station']).mean()['sim_sep_sic_lig_pi']

# SO_sep_sic_site_values['MC'].groupby(['Station']).mean()['sim_sep_sic_lig_pi'].mean()




# endregion
# -----------------------------------------------------------------------------


