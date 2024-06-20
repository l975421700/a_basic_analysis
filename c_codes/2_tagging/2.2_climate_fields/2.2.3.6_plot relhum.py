

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['nudged_703_6.0_k52',]
i = 0

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
# region import data

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_relhum = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_relhum_plev.nc'))

exp_org_o[expid[i]]['relhum'] = xr.open_mfdataset(filenames_relhum,)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann relhum

relhum_alltime = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['relhum'].relhum)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.relhum_alltime.pkl', 'wb') as f:
    pickle.dump(relhum_alltime, f)


'''
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.relhum_alltime.pkl', 'rb') as f:
    relhum_alltime = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot SH/NH am relhum

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.relhum_alltime.pkl', 'rb') as f:
    relhum_alltime = pickle.load(f)

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, reversed= False,
    cmap='Purples',
    )

# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.10 ' + expid[i] + ' zm am relhum SH.png'
output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.10 ' + expid[i] + ' zm am relhum NH.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)

plt_mesh = ax.contourf(
    # relhum_alltime['am'].lat.sel(lat=slice(3, -90)),
    relhum_alltime['am'].lat.sel(lat=slice(90, -3)),
    relhum_alltime['am'].plev.sel(plev=slice(1e+5, 2e+4)) / 100,
    # (relhum_alltime['am'].mean(dim='lon') * 100).sel(lat=slice(3, -90), plev=slice(1e+5, 2e+4)),
    (relhum_alltime['am'].mean(dim='lon') * 100).sel(lat=slice(90, -3), plev=slice(1e+5, 2e+4)),
    norm=pltnorm, cmap=pltcmp, levels=pltlevel, extend='neither',
    )

ax.set_xticks(np.arange(90, -90 - 1e-4, -10))
# ax.set_xlim(0, -88.57)
ax.set_xlim(90, 0)
ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))

ax.invert_yaxis()
ax.set_ylim(1000, 200)
ax.set_yticks(np.arange(1000, 200 - 1e-4, -100))
ax.set_ylabel('Pressure [$hPa$]')

ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=25, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    extend='neither', pad=0.1, fraction=0.04, anchor=(0.4, -1),)
cbar.ax.set_xlabel('Relative humidity [$\%$]',)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
fig.savefig(output_png)


'''

'''
# endregion
# -----------------------------------------------------------------------------



