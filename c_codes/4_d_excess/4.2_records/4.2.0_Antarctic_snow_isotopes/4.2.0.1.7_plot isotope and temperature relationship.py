

# -----------------------------------------------------------------------------
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
from scipy.stats import pearsonr
import statsmodels.api as sm
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
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region import data

Antarctic_snow_isotopes = pd.read_csv(
    'data_sources/ice_core_records/Antarctic_snow_isotopic_composition/Antarctic_snow_isotopic_composition_DB.tab',
    sep='\t', header=0, skiprows=97,)

Antarctic_snow_isotopes = Antarctic_snow_isotopes.rename(columns={
    'Latitude': 'lat',
    'Longitude': 'lon',
    't [°C]': 'temperature',
    'Acc rate [cm/a] (Calculated)': 'accumulation',
    'δD [‰ SMOW] (Calculated average/mean values)': 'dD',
    'δD std dev [±]': 'dD_std',
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)': 'dO18',
    'δ18O std dev [±]': 'dO18_std',
    'd xs [‰] (Calculated average/mean values)': 'd_excess',
    'd xs std dev [±] (Calculated)': 'd_excess_std',
})

Antarctic_snow_isotopes = Antarctic_snow_isotopes[[
    'lat', 'lon', 'temperature', 'accumulation', 'dD', 'dD_std', 'dO18', 'dO18_std', 'd_excess', 'd_excess_std',
]]



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot data

subset = np.isfinite(Antarctic_snow_isotopes['temperature']) & np.isfinite(Antarctic_snow_isotopes['dD'])

pearsonr(Antarctic_snow_isotopes['temperature'][subset], Antarctic_snow_isotopes['dD'][subset])

ols_fit = sm.OLS(
    Antarctic_snow_isotopes['dD'][subset].values,
    sm.add_constant(Antarctic_snow_isotopes['temperature'][subset].values),
    ).fit()

params = ols_fit.params
rsquared = ols_fit.rsquared
predicted_y = ols_fit.params[0] + ols_fit.params[1] * Antarctic_snow_isotopes['temperature'][subset].values
RMSE = np.sqrt(np.average(np.square(predicted_y - Antarctic_snow_isotopes['dD'][subset].values)))

output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0 VM08 isotopes and temperature.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    Antarctic_snow_isotopes['temperature'][subset].values,
    Antarctic_snow_isotopes['dD'][subset].values,
    s=12, lw=0.5, facecolors='white', edgecolors='k',)
ax.axline((0, ols_fit.params[0]), slope = ols_fit.params[1], lw=0.5, color='k')

eq_text = '$y = ' + str(np.round(params[1], 2)) + 'x' + str(np.round(params[0], 2)) + '$\n$R^2 = ' + str(np.round(rsquared, 2)) + '$'

plt.text(
    0.05, 0.95, eq_text,
    transform=ax.transAxes, linespacing=2,
    va='top', ha='left',)

ax.set_xlabel('Temperature [$°C$]',)
ax.set_xlim(-60, 0)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('$\delta D$ [$‰$]',)
ax.set_ylim(-500, 0)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')

fig.subplots_adjust(
    left=0.18, right=0.98, bottom=0.16, top=0.98)
fig.savefig(output_png)


'''
subset = np.isfinite(Antarctic_snow_isotopes['temperature']) & np.isfinite(Antarctic_snow_isotopes['dO18'])
pearsonr(Antarctic_snow_isotopes['temperature'][subset], Antarctic_snow_isotopes['dO18'][subset])
'''
# endregion
# -----------------------------------------------------------------------------


