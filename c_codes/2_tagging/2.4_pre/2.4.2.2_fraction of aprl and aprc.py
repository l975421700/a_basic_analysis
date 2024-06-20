

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['nudged_703_6.0_k52',]
i = 0

ifile_start = 0
ifile_end   = 528

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

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(
    filenames_echam[ifile_start:ifile_end],)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann aprc and aprl

aprl_alltime = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['echam'].aprl)
aprc_alltime = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['echam'].aprc)
aprt_alltime = mon_sea_ann(var_monthly=(exp_org_o[expid[i]]['echam'].aprl + exp_org_o[expid[i]]['echam'].aprc.values))

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprl_alltime.pkl', 'wb') as f:
    pickle.dump(aprl_alltime, f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprc_alltime.pkl', 'wb') as f:
    pickle.dump(aprc_alltime, f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_alltime.pkl', 'wb') as f:
    pickle.dump(aprt_alltime, f)


'''


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprl_alltime.pkl', 'rb') as f:
    aprl_alltime = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprc_alltime.pkl', 'rb') as f:
    aprc_alltime = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate and plot fraction of aprl and aprc

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprl_alltime.pkl', 'rb') as f:
    aprl_alltime = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprc_alltime.pkl', 'rb') as f:
    aprc_alltime = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_alltime.pkl', 'rb') as f:
    aprt_alltime = pickle.load(f)


pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, reversed= False,
    cmap='Blues',
    # cmap='Oranges',
    )

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.10 ' + expid[i] + ' fraction of aprl.png'
# output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.10 ' + expid[i] + ' fraction of aprc.png'

cbar_label = 'Fraction of large scale precipitation [%]'
# cbar_label = 'Fraction of convective precipitation [%]'

fig, ax = globe_plot(
    add_grid_labels=False, figsize=np.array([8.8, 6]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.1, fm_top=0.99,)

plt1 = plot_t63_contourf(
    aprl_alltime['am'].lon,
    aprl_alltime['am'].lat,
    (aprl_alltime['am'] / aprt_alltime['am']) * 100,
    # (aprc_alltime['am'] / aprt_alltime['am']) * 100,
    ax, pltlevel, 'neither', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.7, ticks=pltticks,
    pad=0.05, fraction=0.12,)

cbar.ax.tick_params(length=0.5, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2, size=9)
fig.savefig(output_png)


'''
# np.min((aprl_alltime['am'] + aprc_alltime['am']) / aprt_alltime['am'])
'''
# endregion
# -----------------------------------------------------------------------------

