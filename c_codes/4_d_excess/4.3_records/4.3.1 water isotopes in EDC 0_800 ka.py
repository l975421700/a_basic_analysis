

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
# region import data

isotopes_EDC_800kyr = pd.read_csv(
    'data_sources/ice_core_records/water_isotopes_EDC_0_800ka/EPICA-DOME-C-ice-core_dD_d180.tab',
    sep='\t', header=0, skiprows=26,)

dD_EDC_AICC = pd.read_csv(
    'data_sources/ice_core_records/AICC_dD_EDC_0_800ka/EDC_d2H_AICC2012_chron.tab',
    sep='\t', header=0, skiprows=21,)


'''
np.mean(isotopes_EDC_800kyr['δD H2O [‰ SMOW]'])
np.mean(isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'])
np.mean(isotopes_EDC_800kyr['d_ln_EDC_800kyr'])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_ln and age

isotopes_EDC_800kyr['d_ln_EDC_800kyr'] = \
    (np.log(1 + isotopes_EDC_800kyr['δD H2O [‰ SMOW]'] / 1000) - \
        8.47 * np.log(1 + isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'] / 1000) + \
            0.0285 * (np.log(1 + isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'] / 1000)) ** 2) * 1000

isotopes_EDC_800kyr['d_excess_EDC_800kyr'] = \
    isotopes_EDC_800kyr['δD H2O [‰ SMOW]'] - \
        8 * isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]']

isotopes_EDC_800kyr['wrong_d_ln_EDC_800kyr'] = \
    1000 * np.log(1 + isotopes_EDC_800kyr['δD H2O [‰ SMOW]'] / 1000) - \
        8.47 * 1000 * np.log(1 + isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'] / 1000) + \
            0.0285 * (1000 * np.log(1 + isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'] / 1000)) ** 2

isotopes_EDC_800kyr_AICC = \
    isotopes_EDC_800kyr.merge(
        dD_EDC_AICC, how='left', on='Depth ice/snow [m]')

isotopes_EDC_800kyr_AICC = isotopes_EDC_800kyr_AICC.dropna(
    ignore_index=True).drop(columns="δD [‰ SMOW]")

with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'wb') as f:
        pickle.dump(isotopes_EDC_800kyr_AICC, f)


'''
# import data
with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'rb') as f:
        isotopes_EDC_800kyr_AICC = pickle.load(f)


np.log(10)
np.log(np.e)

#-------------------------------- check merge
#---- left
(isotopes_EDC_800kyr_AICC['Depth ice/snow [m]'] == isotopes_EDC_800kyr['Depth ice/snow [m]']).all()
(isotopes_EDC_800kyr_AICC['δD H2O [‰ SMOW]'] == isotopes_EDC_800kyr['δD H2O [‰ SMOW]']).all()
(isotopes_EDC_800kyr_AICC['δ18O H2O [‰ SMOW]'] == isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]']).all()
(isotopes_EDC_800kyr_AICC['d_ln_EDC_800kyr'] == isotopes_EDC_800kyr['d_ln_EDC_800kyr']).all()

indices = np.where(isotopes_EDC_800kyr_AICC['δ18O H2O [‰ SMOW]'] != isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'])
isotopes_EDC_800kyr_AICC['δ18O H2O [‰ SMOW]'][indices[0]]
isotopes_EDC_800kyr['δ18O H2O [‰ SMOW]'][indices[0]]

indices = np.where(isotopes_EDC_800kyr_AICC['d_ln_EDC_800kyr'] != isotopes_EDC_800kyr['d_ln_EDC_800kyr'])
isotopes_EDC_800kyr_AICC['d_ln_EDC_800kyr'][indices[0]]
isotopes_EDC_800kyr['d_ln_EDC_800kyr'][indices[0]]

#---- inner
isotopes_EDC_800kyr_AICC = \
    isotopes_EDC_800kyr.merge(
        dD_EDC_AICC, how='inner', on='Depth ice/snow [m]')
(isotopes_EDC_800kyr_AICC['Depth ice/snow [m]'] == dD_EDC_AICC['Depth ice/snow [m]']).all()
(isotopes_EDC_800kyr_AICC['Age [ka BP]'] == dD_EDC_AICC['Age [ka BP]']).all()
(isotopes_EDC_800kyr_AICC['δD [‰ SMOW]'] == dD_EDC_AICC['δD [‰ SMOW]']).all()

#---- dropna after left merge
(isotopes_EDC_800kyr_AICC['Depth ice/snow [m]'] == dD_EDC_AICC['Depth ice/snow [m]']).all()
(isotopes_EDC_800kyr_AICC['Age [ka BP]'] == dD_EDC_AICC['Age [ka BP]']).all()
(isotopes_EDC_800kyr_AICC['δD [‰ SMOW]'] == dD_EDC_AICC['δD [‰ SMOW]']).all()

for ikey in isotopes_EDC_800kyr_AICC.keys():
    print(ikey)
    print((np.isnan(isotopes_EDC_800kyr_AICC[ikey])).sum())

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d18O, dD, and d_ln

#---------------- settings

#-------- isotopes

#---- d18O
ivar = 'd18O'
icolnames = 'δ18O H2O [‰ SMOW]'
ylabel = '$\delta^{18}O$ [‰ SMOW]'

#---- dD
ivar = 'dD'
icolnames = 'δD H2O [‰ SMOW]'
ylabel = '$\delta D$ [‰ SMOW]'

#---- d_ln
ivar = 'd_ln'
icolnames = 'd_ln_EDC_800kyr'
ylabel = '$d_{ln}$ [‰ SMOW]'

#---- d_excess
ivar = 'd_excess'
icolnames = 'd_excess_EDC_800kyr'
ylabel = 'd-excess [‰ SMOW]'

#---- wrong_d_ln
ivar = 'wrong_d_ln'
icolnames = 'wrong_d_ln_EDC_800kyr'
ylabel = '$d_{ln}$ [‰ SMOW]'

#-------- figures

#----
xaxis_max = 800
xaxis_interval = 100

#----
xaxis_max = 140
xaxis_interval = 10

xaxis_max = 0.3
xaxis_interval = 0.02

#---------------- plot

output_png = 'figures/8_d-excess/8.0_records/8.0.1_ice cores/8.0.1.1 EDC ' + ivar + ' of past ' + str(xaxis_max) + ' kyr on AICC.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    isotopes_EDC_800kyr_AICC['Age [ka BP]'].values,
    isotopes_EDC_800kyr_AICC[icolnames].values,
    c='k', lw=0.3, ls='-')

ax.set_ylabel(ylabel)
ax.set_ylim(-75, -55)
# ax.set_yticks(np.arange(-10, 4 + 1e-4, 2))

ax.set_xlabel('Age before 1950 [kyr]')
ax.set_xlim(0, xaxis_max)
ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))

# ax.spines[['right', 'top']].set_visible(False)

fig.subplots_adjust(left=0.1, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


# endregion
# -----------------------------------------------------------------------------
