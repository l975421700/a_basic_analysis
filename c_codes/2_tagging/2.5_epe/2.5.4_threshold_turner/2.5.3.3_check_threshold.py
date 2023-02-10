

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
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
    remove_trailing_zero_pos_abs,
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
    ten_sites_names,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
    calc_lon_diff_np,
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
# region fraction of precipitation below 0.02 mm/day

wisoaprt_masked_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_st.pkl',
    'rb') as f:
    wisoaprt_masked_st[expid[i]] = pickle.load(f)

np.min(wisoaprt_masked_st[expid[i]]['frc']['1%']['am'])
# 0.99324927

wisoaprt_masked = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked.pkl',
    'rb') as f:
    wisoaprt_masked[expid[i]] = pickle.load(f)

np.min(wisoaprt_masked[expid[i]]['frc']['1%']['am'])
# 0.86385676


ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' aprt_frc am below threshold 0.02.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    0, 14, 2, 2, cmap='Blues', reversed=False)

fig, ax = hemisphere_plot(northextent=-60)
cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
plt_cmp = ax.pcolormesh(
    wisoaprt_masked[expid[i]]['frc']['1%']['am'].lon,
    wisoaprt_masked[expid[i]]['frc']['1%']['am'].lat,
    100 - wisoaprt_masked[expid[i]]['frc']['1%']['am'] * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of daily precipitation amount\nbelow the threshold 0.02 $mm \; day^{-1}$  [$\%$]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------
