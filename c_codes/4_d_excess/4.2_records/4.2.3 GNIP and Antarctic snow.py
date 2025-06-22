

# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

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
from scipy.stats import linregress

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


# region import data

GNIP = pd.read_excel(
    'data_sources/products/GNIP/2021-12-01_GNIP_Snapshot.xlsx',
    header=0,)

GNIP = GNIP[['Site', 'O18', 'H2', 'Precipitation']].dropna(subset=('O18', 'H2', 'Precipitation'), how='any').reset_index(drop=True)

counts = GNIP['Site'].value_counts()
valid_entries = counts[counts >= 24].index
GNIP = GNIP[GNIP['Site'].isin(valid_entries)].reset_index(drop=True)


def weighted_average(group, value_column, weight_column):
    return np.average(group[value_column], weights=group[weight_column])

GNIP_weighted = GNIP.groupby('Site').apply(
    lambda group: pd.Series({
        'd18O': weighted_average(group, 'O18', 'Precipitation'),
        'dD': weighted_average(group, 'H2', 'Precipitation')
    })).reset_index()


GNIP_weighted['d_xs'] = GNIP_weighted['dD'] - 8 * GNIP_weighted['d18O']

GNIP_weighted['ln_dD'] = 1000 * np.log(1 + GNIP_weighted['dD'] / 1000)
GNIP_weighted['ln_d18O'] = 1000 * np.log(1 + GNIP_weighted['d18O'] / 1000)
GNIP_weighted['d_ln'] = GNIP_weighted['ln_dD'] - 8.47 * GNIP_weighted['ln_d18O'] + 0.0285 * (GNIP_weighted['ln_d18O'] ** 2)




Antarctic_snow_isotopes = pd.read_csv(
    'data_sources/ice_core_records/Antarctic_snow_isotopic_composition/Antarctic_snow_isotopic_composition_DB.tab',
    sep='\t', header=0, skiprows=97,)

Antarctic_snow_isotopes = Antarctic_snow_isotopes.rename(columns={
    'δD [‰ SMOW] (Calculated average/mean values)': 'dD',
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)': 'd18O',
})[['dD', 'd18O']].dropna(how='any').reset_index(drop=True)

Antarctic_snow_isotopes['d_xs'] = Antarctic_snow_isotopes['dD'] - 8 * Antarctic_snow_isotopes['d18O']

Antarctic_snow_isotopes['ln_dD'] = 1000 * np.log(1 + Antarctic_snow_isotopes['dD'] / 1000)
Antarctic_snow_isotopes['ln_d18O'] = 1000 * np.log(1 + Antarctic_snow_isotopes['d18O'] / 1000)
Antarctic_snow_isotopes['d_ln'] = Antarctic_snow_isotopes['ln_dD'] - 8.47 * Antarctic_snow_isotopes['ln_d18O'] + 0.0285 * (Antarctic_snow_isotopes['ln_d18O'] ** 2)



'''
np.average(GNIP[GNIP['Site'] == 'ZUNYI']['O18'], weights=GNIP[GNIP['Site'] == 'ZUNYI']['Precipitation'])
np.average(GNIP[GNIP['Site'] == 'ZUNYI']['H2'], weights=GNIP[GNIP['Site'] == 'ZUNYI']['Precipitation'])

GNIP.columns
Antarctic_snow_isotopes.columns
'''
# endregion


# region plot data

xdata = np.arange(-65, 5, 0.1)
ydata1 = 8 * xdata + 10
ydata2 = -0.0285 * xdata**2 + 8.47 * xdata + 13.3

output_png = 'figures/test/test.png'
fig, axs = plt.subplots(2, 2, figsize=np.array([6.6 * 2, 6.6 * 2]) / 2.54,
                        sharex=True, )

marker1 = 'o'
marker2 = 'x'
lw=0.2
size=8

plt1 = axs[0, 0].scatter(
    GNIP_weighted['d18O'].values, GNIP_weighted['dD'].values,
    label='GNIP precipitation',
    s=size, fc='none', ec='k', marker=marker1, lw=lw, zorder=2)
plt2 = axs[0, 0].scatter(
    Antarctic_snow_isotopes['d18O'].values,
    Antarctic_snow_isotopes['dD'].values,
    label='Antarctic snow',
    s=size, fc='none', ec='k', marker=marker2, lw=lw, zorder=2)
plt3 = axs[0, 0].plot(xdata, ydata1, c='k', lw=lw*2, label='GMWL')

axs[0, 0].legend(loc='upper left', frameon=False, handlelength=1,
                 handletextpad=0.4, borderaxespad=0.2,)

axs[0, 1].scatter(
    GNIP_weighted['ln_d18O'].values, GNIP_weighted['ln_dD'].values,
    label='GNIP precipitation',
    s=size, fc='none', ec='k', marker=marker1, lw=lw, zorder=2)
axs[0, 1].scatter(
    Antarctic_snow_isotopes['ln_d18O'].values,
    Antarctic_snow_isotopes['ln_dD'].values,
    label='Antarctic snow',
    s=size, fc='none', ec='k', marker=marker2, lw=lw, zorder=2)
plt4 = axs[0, 1].plot(xdata, ydata2, c='k', lw=lw*2, label='GMWC')

axs[0, 1].legend(loc='upper left', frameon=False, handlelength=1,
                 handletextpad=0.4, borderaxespad=0.2,)

axs[1, 0].scatter(GNIP_weighted['d18O'].values,
                  GNIP_weighted['d_xs'].values,
                  s=size, fc='none', ec='k', marker=marker1, lw=lw, zorder=2)
axs[1, 0].scatter(Antarctic_snow_isotopes['d18O'].values,
                  Antarctic_snow_isotopes['d_xs'].values,
                  s=size, fc='none', ec='k', marker=marker2, lw=lw, zorder=2)

axs[1, 1].scatter(GNIP_weighted['ln_d18O'].values,
                  GNIP_weighted['d_ln'].values,
                  s=size, fc='none', ec='k', marker=marker1, lw=lw, zorder=2)
axs[1, 1].scatter(Antarctic_snow_isotopes['ln_d18O'].values,
                  Antarctic_snow_isotopes['d_ln'].values,
                  s=size, fc='none', ec='k', marker=marker2, lw=lw, zorder=2)

axs[0, 0].set_ylim(-700, 50)
axs[0, 1].set_ylim(-700, 50)
axs[1, 0].set_ylim(-20, 40)
axs[1, 1].set_ylim(-20, 40)

axs[1, 0].set_xlim(-65, 5)
axs[1, 1].set_xlim(-65, 5)

# plt.text(0.5, 1.25, 'Global Meteoric Water Line',
#          transform=axs[0, 0].transAxes, ha='center', va='center', weight='bold')
# plt.text(0.5, 1.25, 'Global Meteoric Water Curve',
#          transform=axs[0, 1].transAxes, ha='center', va='center', weight='bold')
axs[0, 0].set_ylabel(r'$\delta D$ [$‰$]', )
axs[0, 1].set_ylabel(r'$ln(\delta D+1)$ [$‰$]', )
axs[1, 0].set_ylabel(r'$d_{xs}$ [$‰$]', )
axs[1, 1].set_ylabel(r'$d_{ln}$ [$‰$]', )

axs[1, 0].set_xlabel(r'$\delta^{18}O$ [$‰$]', )
axs[1, 1].set_xlabel(r'$ln(\delta^{18}O+1)$ [$‰$]', )

ipanel=0
for irow in range(2):
    for jcol in range(2):
        plt.text(
            -0.25, 1.05, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes)
        axs[irow, jcol].grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        ipanel += 1

fig.subplots_adjust(
    left=0.12, right=0.99, bottom=0.1, top=0.96, wspace=0.3, hspace=0.2)
fig.savefig(output_png)

# endregion

