

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
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
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

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

d_ln_alltime = {}
regression_sst_d_AIS = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.regression_sst_d_AIS.pkl', 'rb') as f:
        regression_sst_d_AIS[expid[i]] = pickle.load(f)

lon = regression_sst_d_AIS[expid[i]]['d_ln']['ann']['RMSE'].lon
lat = regression_sst_d_AIS[expid[i]]['d_ln']['ann']['RMSE'].lat


source_var = ['sst', ]
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_sst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source sst vs. slope

i = 0
ivar = 'sst'
iisotope = 'd_ln'

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

pearsonr(
    pre_weighted_var[expid[i]][ivar]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
    regression_sst_d_AIS[expid[i]][iisotope]['ann no am']['slope'].values[echam6_t63_ais_mask['mask']['AIS']],
)

pearsonr(
    d_ln_alltime[expid[i]]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
    # pre_weighted_var[expid[i]][ivar]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
    regression_sst_d_AIS[expid[i]][iisotope]['ann no am']['slope'].values[echam6_t63_ais_mask['mask']['AIS']],
)

scatter_size = 6

output_png = 'figures/test/test.png'

# fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 5]) / 2.54)
fig = plt.figure(figsize=np.array([4.4, 5]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

ax.scatter_density(
    d_ln_alltime[expid[i]]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
    # pre_weighted_var[expid[i]][ivar]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
    regression_sst_d_AIS[expid[i]][iisotope]['ann no am']['slope'].values[echam6_t63_ais_mask['mask']['AIS']],
    cmap=white_viridis)
# ax.scatter(
#     pre_weighted_var[expid[i]][ivar]['am'].values[echam6_t63_ais_mask['mask']['AIS']],
#     regression_sst_d_AIS[expid[i]][iisotope]['ann no am']['slope'].values[echam6_t63_ais_mask['mask']['AIS']],
#     s=scatter_size, lw=0.1,
#     # c=color_var, norm=pltnorm, cmap=pltcmp,
#     facecolors='white', edgecolors='k',
#     )
ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(
    left=0.32, right=0.95, bottom=0.22, top=0.95)
fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region source sst vs. dln ann no am over AIS

i = 0
ivar = 'sst'
iisotope = 'd_ln'
ialltime = 'ann no am'

b_ais_mask = np.broadcast_to(
    echam6_t63_ais_mask['mask']['AIS'],
    d_ln_alltime[expid[i]][ialltime].shape,
)

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

output_png = 'figures/test/test.png'

fig = plt.figure(figsize=np.array([4.4, 5]) / 2.54,)
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

ax.scatter_density(
    d_ln_alltime[expid[i]][ialltime].values[b_ais_mask],
    pre_weighted_var[expid[i]][ivar][ialltime].values[b_ais_mask],
    cmap=white_viridis)
ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(
    left=0.32, right=0.95, bottom=0.22, top=0.95)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


