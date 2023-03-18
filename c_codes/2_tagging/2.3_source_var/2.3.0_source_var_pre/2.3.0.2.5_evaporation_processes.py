

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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
from scipy.stats import linregress
from scipy.stats import pearsonr

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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate dimensionless E

from metpy.calc import saturation_mixing_ratio
from metpy.calc import specific_humidity_from_mixing_ratio
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units



from metpy.units import units
def calculate_dimensionless_E(
    wind10 = 15 * units('m/s'),
    rh2m = 0.8,
    sst = 15 * units.degC,
    temp2m = 15 * units.degC,
    pressure = 1000 * units.hPa,
    ):
    '''
    
    '''
    
    from metpy.calc import saturation_mixing_ratio
    from metpy.calc import specific_humidity_from_mixing_ratio
    from metpy.calc import relative_humidity_from_specific_humidity
    
    # saturation mixing ratio at the surface
    w_s_surface = saturation_mixing_ratio(pressure, sst)
    # specific humidity at the surface
    q_s_surface = specific_humidity_from_mixing_ratio(w_s_surface)
    
    # saturation mixing ratio at 2m
    w_s_2m = saturation_mixing_ratio(pressure, temp2m)
    
    q2m = rh2m * w_s_2m / (1 + rh2m * w_s_2m)
    
    E_unitless = (wind10 * (q_s_surface - q2m)).magnitude
    
    return(E_unitless)


E_orginal = calculate_dimensionless_E()

wind10 = np.arange(0, 30+1e-4, 1) * units('m/s')
E_wind10 = calculate_dimensionless_E(wind10 = wind10) / E_orginal

sst = np.arange(0, 30+1e-4, 1) * units.degC
E_sst = calculate_dimensionless_E(sst = sst) / E_orginal

rh2m = np.arange(0.4, 1.2+1e-4, 0.05)
E_rh2m = calculate_dimensionless_E(rh2m = rh2m) / E_orginal




'''
pressure = 1000 * units.hPa
wind10 = 15 * units('m/s')
# rh2m = 0.8
rh2m = np.arange(0.4, 1.2+1e-4, 0.05)
sst = 15 * units.degC
temp2m = sst

# saturation mixing ratio at the surface
w_s_surface = saturation_mixing_ratio(pressure, sst)
# specific humidity at the surface
q_s_surface = specific_humidity_from_mixing_ratio(w_s_surface)

# saturation mixing ratio at 2m
w_s_2m = saturation_mixing_ratio(pressure, temp2m)

q2m = (rh2m * w_s_2m) / (1 + rh2m * w_s_2m)

E_unitless = wind10 * (q_s_surface - q2m)


pressure = 1000 * units.hPa
temperature = np.arange(0, 25+1e-4, 1) * units.degC
w_s = saturation_mixing_ratio(pressure, temperature).to('g/kg')

relative_humidity_from_specific_humidity(
    pressure, temp2m, q2m.magnitude
)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot E versus wind10, sst, and rh2m


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_evaporation/6.1.3.6.0 sensitivity of evaporation to wind10.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6]) / 2.54)

sns.scatterplot(
    wind10,
    E_wind10,)


linearfit = linregress(
    x = wind10,
    y = E_wind10,)
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5,
          c='r')
plt.text(
    0.55, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, linespacing=1.5)

ax.axhline(1, color='k', linestyle='--', linewidth=0.5)
ax.axvline(15, color='k', linestyle='--', linewidth=0.5)

ax.set_xlim(7, 17)
ax.set_xlabel('wind10 [$m \; s^{-1}$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylabel('Dimensionless evaporation [$-$]')
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.98)
plt.savefig(output_png)
plt.close()




output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_evaporation/6.1.3.6.0 sensitivity of evaporation to sst.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6]) / 2.54)

sns.scatterplot(
    sst,
    E_sst,)


linearfit = linregress(
    x = sst,
    y = E_sst,)
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5,
          c='r')
plt.text(
    0.55, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, linespacing=1.5)

ax.axhline(1, color='k', linestyle='--', linewidth=0.5)
ax.axvline(15, color='k', linestyle='--', linewidth=0.5)

ax.set_xlim(2, 24)
ax.set_xlabel('SST [$°C$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylim(-3, 9)
ax.set_yticks(np.arange(-2, 8+1e-4, 2))
ax.set_ylabel('Dimensionless evaporation [$-$]')
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(1))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.98)
plt.savefig(output_png)
plt.close()




output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_evaporation/6.1.3.6.0 sensitivity of evaporation to rh2m.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([6.6, 6]) / 2.54)

sns.scatterplot(
    rh2m * 100,
    E_rh2m,)

linearfit = linregress(
    x = rh2m * 100,
    y = E_rh2m,)
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5,
          c='r')
plt.text(
    0.05, 0.05,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, linespacing=1.5)

ax.axhline(1, color='k', linestyle='--', linewidth=0.5)
ax.axvline(80, color='k', linestyle='--', linewidth=0.5)

ax.set_xlim(70, 87.5)
# ax.set_xlim(35, 125)
# ax.set_xticks(np.arange(40, 120+1e-4, 20))
ax.set_xlabel('rh2m [$\%$]')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_ylim(-1.2, 3.2)
ax.set_yticks(np.arange(-1, 3+1e-4, 1))
ax.set_ylabel('Dimensionless evaporation [$-$]')
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.98)
plt.savefig(output_png)
plt.close()








'''
# [0, 2]
sns.scatterplot(
    wind10,
    E_wind10,
)
plt.savefig('figures/test/trial.png')
plt.close()

# [-2.3, 8.7]
sns.scatterplot(
    sst,
    E_sst,
)
plt.savefig('figures/test/trial.png')
plt.close()

# [3.0, -1]
sns.scatterplot(
    rh2m * 100,
    E_rh2m,
)
plt.savefig('figures/test/trial.png')
plt.close()

# [-3, 9]
'''
# endregion
# -----------------------------------------------------------------------------
