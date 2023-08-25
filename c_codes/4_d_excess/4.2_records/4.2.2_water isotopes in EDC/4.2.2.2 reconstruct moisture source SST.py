

expid = [
    'pi_600_5.0',
    ]


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
    seconds_per_d,
    plot_labels,
    plot_labels_no_unit,
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

with open('data_sources/ice_core_records/isotopes_EDC_800kyr_AICC.pkl',
          'rb') as f:
    isotopes_EDC_800kyr_AICC = pickle.load(f)

# remove the anomalous spike

isotopes_EDC_800kyr_AICC = isotopes_EDC_800kyr_AICC.drop(
    index=np.argmin(isotopes_EDC_800kyr_AICC['d_ln'])
    ).reset_index(drop=True)



'''
isotopes_EDC_800kyr_AICC.iloc[np.argmin(isotopes_EDC_800kyr_AICC['d_ln'])]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region slope for temperature reconstructions


expid_slope = {}
expid_slope['pi_600_5.0'] = {
    'EDC': [0.3082, 0.263, 0.354],
}


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region reconstruct moisture source SST

i = 0
icores = 'EDC'

d_ln_2k = np.mean(isotopes_EDC_800kyr_AICC['d_ln'][
    isotopes_EDC_800kyr_AICC['age'] <= 0.15])

delta_d_ln = isotopes_EDC_800kyr_AICC['d_ln'] - d_ln_2k


tem_rec = {}
tem_rec[expid[i]] = {}
tem_rec[expid[i]][icores] = {}

tem_rec[expid[i]][icores]['delta_T_source'] = {}

tem_rec[expid[i]][icores]['delta_T_source']['mean'] = \
    expid_slope[expid[i]][icores][0] * delta_d_ln
tem_rec[expid[i]][icores]['delta_T_source']['low'] = \
    expid_slope[expid[i]][icores][1] * delta_d_ln
tem_rec[expid[i]][icores]['delta_T_source']['high'] = \
    expid_slope[expid[i]][icores][2] * delta_d_ln


(isotopes_EDC_800kyr_AICC['age'] <= 2).sum()
(isotopes_EDC_800kyr_AICC['age'] <= 0.15).sum()


'''
np.mean(isotopes_EDC_800kyr_AICC['d_ln'][isotopes_EDC_800kyr_AICC['age'] <= 0.15])
np.mean(isotopes_EDC_800kyr_AICC['d_ln'][isotopes_EDC_800kyr_AICC['age'] <= 2])
np.mean(delta_d_ln[isotopes_EDC_800kyr_AICC['age'] <= 2])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot moisture source SST

xaxis_max = 800
xaxis_interval = 100
# xaxis_max = 140
# xaxis_interval = 10

output_png = 'figures/8_d-excess/8.0_records/8.0.2_reconstructions/8.0.2.0 ' + expid[i] + ' ' + icores + ' T_source reconstructions of past ' + str(xaxis_max) + ' kyr on AICC.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([16, 6]) / 2.54)

ax.plot(
    isotopes_EDC_800kyr_AICC['age'].values,
    tem_rec[expid[i]][icores]['delta_T_source']['mean'].values,
    c='k', lw=0.3, ls='-')

ax.fill_between(
    isotopes_EDC_800kyr_AICC['age'].values,
    tem_rec[expid[i]][icores]['delta_T_source']['low'].values,
    tem_rec[expid[i]][icores]['delta_T_source']['high'].values,
    # alpha=0.2,
)

ax.set_ylabel(plot_labels['sst'])
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.set_xlabel('Age before 1950 [kyr]')
ax.set_xlim(0, xaxis_max)
ax.set_xticks(np.arange(0, xaxis_max + 1e-4, xaxis_interval))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# ax.spines[['right', 'top']].set_visible(False)
ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
fig.savefig(output_png, dpi=600)


'''
np.max(abs(tem_rec[expid[i]][icores]['delta_T_source']['low'].values - tem_rec[expid[i]][icores]['delta_T_source']['high'].values))
'''
# endregion
# -----------------------------------------------------------------------------
