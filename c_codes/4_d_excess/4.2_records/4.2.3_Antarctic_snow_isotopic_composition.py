

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

# len(np.unique(Antarctic_snow_isotopes['Sample label']))

'''
['Latitude', 'Longitude', 'Sample label',
'δD [‰ SMOW] (Calculated average/mean values)',
'δ18O H2O [‰ SMOW] (Calculated average/mean values)',
'd xs [‰] (Calculated average/mean values)',
]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot delta 018 versus delta D

delta_018 = Antarctic_snow_isotopes[
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)'].copy()
delta_D = Antarctic_snow_isotopes[
    'δD [‰ SMOW] (Calculated average/mean values)'].copy()

subset = np.isfinite(delta_018) & np.isfinite(delta_D)
delta_018 = delta_018[subset]
delta_D = delta_D[subset]

linearfit = linregress(
    x = delta_018,
    y = delta_D,)

ax = sns.scatterplot(
    x = delta_018,
    y = delta_D,)
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5)
ax.axline((0, 10), slope = 8, lw=0.5, color='k')
plt.savefig('figures/test/test.png')
plt.close()


'''
len(delta_018)
np.isfinite(delta_018).sum()

len(delta_D)
np.isfinite(delta_D).sum()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region spatial distribution of delta_D and delta 018

output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0 Antarctic snow isotopes, delta_D, Masson-Delmotte et al., 2008.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-450, cm_max=-100, cm_interval1=25, cm_interval2=50,
    cmap='viridis', reversed=True)

fig, ax = hemisphere_plot(northextent=-60)

plt_scatter = ax.scatter(
    Antarctic_snow_isotopes['Longitude'],
    Antarctic_snow_isotopes['Latitude'],
    s=8,
    c=Antarctic_snow_isotopes['δD [‰ SMOW] (Calculated average/mean values)'],
    edgecolors='k', linewidths=0.1,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('$\delta D$ [‰ SMOW]\nMasson-Delmotte et al. (2008)', linespacing=1.5)
fig.savefig(output_png)


output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0 Antarctic snow isotopes, delta_O18, Masson-Delmotte et al., 2008.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=-20, cm_interval1=5, cm_interval2=5,
    cmap='viridis', reversed=True)

fig, ax = hemisphere_plot(northextent=-60)

plt_scatter = ax.scatter(
    Antarctic_snow_isotopes['Longitude'],
    Antarctic_snow_isotopes['Latitude'],
    s=8,
    c=Antarctic_snow_isotopes['δ18O H2O [‰ SMOW] (Calculated average/mean values)'],
    edgecolors='k', linewidths=0.1,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('$\delta^{18}O$ [‰ SMOW]\nMasson-Delmotte et al. (2008)', linespacing=1.5)
fig.savefig(output_png)


'''
np.nanmax(Antarctic_snow_isotopes['δD [‰ SMOW] (Calculated average/mean values)'])
np.nanmin(Antarctic_snow_isotopes['δD [‰ SMOW] (Calculated average/mean values)'])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region spatial distribution of deteurium excess

ln_dD = 1000 * np.log(1 + Antarctic_snow_isotopes['δD [‰ SMOW] (Calculated average/mean values)'] / 1000)
ln_d18O = 1000 * np.log(1 + Antarctic_snow_isotopes['δ18O H2O [‰ SMOW] (Calculated average/mean values)'] / 1000)

d_ln = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.0 Antarctic snow isotopes, d_ln, Masson-Delmotte et al., 2008.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10,
    cmap='viridis', reversed=False)

fig, ax = hemisphere_plot(northextent=-60)

plt_scatter = ax.scatter(
    Antarctic_snow_isotopes['Longitude'],
    Antarctic_snow_isotopes['Latitude'],
    s=8,
    c=d_ln,
    edgecolors='k', linewidths=0.1,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('$d_{ln}$ [‰ SMOW]\nMasson-Delmotte et al. (2008)', linespacing=1.5)
fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


