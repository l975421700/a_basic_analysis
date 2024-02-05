

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
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
    mean_over_ais,
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
    cplot_ttest,
    xr_par_cor,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get cleaned data


#---- import data

isotopes_EDC_800kyr = pd.read_csv(
    'data_sources/ice_core_records/water_isotopes_EDC_0_800ka/EPICA-DOME-C-ice-core_dD_d180.tab',
    sep='\t', header=0, skiprows=26,)

isotopes_EDC_800kyr = isotopes_EDC_800kyr.rename(columns={
    'Depth ice/snow [m]': 'depth',
    'δD H2O [‰ SMOW]': 'dD',
    'δ18O H2O [‰ SMOW]': 'dO18',
})

dD_EDC_AICC = pd.read_csv(
    'data_sources/ice_core_records/AICC_dD_EDC_0_800ka/EDC_d2H_AICC2012_chron.tab',
    sep='\t', header=0, skiprows=21,)

dD_EDC_AICC = dD_EDC_AICC.rename(columns={
    'Depth ice/snow [m]': 'depth',
    'Age [ka BP]': 'age',
}).drop(columns="δD [‰ SMOW]")


#---- merge to get age info

isotopes_EDC_800kyr_AICC = \
    isotopes_EDC_800kyr.merge(dD_EDC_AICC, how='left', on='depth')

isotopes_EDC_800kyr_AICC = isotopes_EDC_800kyr_AICC.dropna(ignore_index=True)


#---- get d_ln and d_ex

ln_dD   = 1000 * np.log(1 + isotopes_EDC_800kyr_AICC['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + isotopes_EDC_800kyr_AICC['dO18'] / 1000)

isotopes_EDC_800kyr_AICC['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

isotopes_EDC_800kyr_AICC['d_excess'] = isotopes_EDC_800kyr_AICC['dD'] - 8 * isotopes_EDC_800kyr_AICC['dO18']

with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'wb') as f:
    pickle.dump(isotopes_EDC_800kyr_AICC, f)




'''
#-------------------------------- check
with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'rb') as f:
    isotopes_EDC_800kyr_AICC = pickle.load(f)

isotopes_EDC_800kyr = pd.read_csv(
    'data_sources/ice_core_records/water_isotopes_EDC_0_800ka/EPICA-DOME-C-ice-core_dD_d180.tab',
    sep='\t', header=0, skiprows=26,)

isotopes_EDC_800kyr = isotopes_EDC_800kyr.rename(columns={
    'Depth ice/snow [m]': 'depth',
    'δD H2O [‰ SMOW]': 'dD',
    'δ18O H2O [‰ SMOW]': 'dO18',
})

dD_EDC_AICC = pd.read_csv(
    'data_sources/ice_core_records/AICC_dD_EDC_0_800ka/EDC_d2H_AICC2012_chron.tab',
    sep='\t', header=0, skiprows=21,)

dD_EDC_AICC = dD_EDC_AICC.rename(columns={
    'Depth ice/snow [m]': 'depth',
    'Age [ka BP]': 'age',
}).drop(columns="δD [‰ SMOW]")


test = isotopes_EDC_800kyr.merge(dD_EDC_AICC, how='inner', on='depth')

(isotopes_EDC_800kyr_AICC['dD'] == test['dD']).all()
(isotopes_EDC_800kyr_AICC['d_excess'] == (test['dD'] - 8 * test['dO18'])).all()
(isotopes_EDC_800kyr_AICC['d_ln'] == (
    (1000 * np.log(1 + test['dD'] / 1000)) - 8.47 * (1000 * np.log(1 + test['dO18'] / 1000)) + 0.0285 * ((1000 * np.log(1 + test['dO18'] / 1000)) ** 2))).all()


#-------------------------------- check merge
#---- left
(isotopes_EDC_800kyr_AICC['depth'] == isotopes_EDC_800kyr['depth']).all()

(isotopes_EDC_800kyr_AICC['age'][np.isfinite(isotopes_EDC_800kyr_AICC['age'])].values == dD_EDC_AICC['age'].values).all()

'''
# endregion
# -----------------------------------------------------------------------------


