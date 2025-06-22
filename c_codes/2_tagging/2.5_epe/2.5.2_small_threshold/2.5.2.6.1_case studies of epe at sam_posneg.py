

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=240GB


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_705_6.0',
    'nudged_703_6.0_k52',
    ]
i = 0

pos_case_time = '2021-10-21'
neg_case_time = '2022-02-10'


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
from scipy.stats import circmean

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
# region import data

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# get q, dD/d_ln, source lat/lon/sst, just for 2021/2022

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)

dD_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
    dD_q_alltime[expid[i]] = pickle.load(f)

d_ln_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_alltime.pkl', 'rb') as f:
    d_ln_q_alltime[expid[i]] = pickle.load(f)

q_weighted_lat = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lat.pkl',
          'rb') as f:
    q_weighted_lat[expid[i]] = pickle.load(f)

q_weighted_sst = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_sst.pkl',
          'rb') as f:
    q_weighted_sst[expid[i]] = pickle.load(f)

q_weighted_lon = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lon.pkl',
          'rb') as f:
    q_weighted_lon[expid[i]] = pickle.load(f)


'''
wiso_q_plev_alltime[expid[i]]['q16o']['daily'].time
dD_q_alltime[expid[i]]['daily'].time
d_ln_q_alltime[expid[i]]['daily'].time
q_weighted_lat[expid[i]]['daily'].time
q_weighted_sst[expid[i]]['daily'].time
q_weighted_lon[expid[i]]['daily'].time
'''
# endregion
# -----------------------------------------------------------------------------


# region plot the meridional cross-section


var = 'lon'
if var == 'q':
    plt_data1 = wiso_q_plev_alltime[expid[i]]['q16o']['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest') * 1000
    plt_data2 = wiso_q_plev_alltime[expid[i]]['q16o']['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest') * 1000
    plt_data3 = (plt_data1 - plt_data2) / plt_data2 * 100
    cbar_label1 = 'Specific humidity [$g \; kg^{-1}$]'
    cbar_label2 = 'Difference: (a)/(b) - 1 [%]'
    pltlevel = np.array([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10])
    pltticks = np.array([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10])
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
    pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=40, cmap='BrBG',
        reversed=False)
    extend = 'max'
    extend1 = 'both'
elif var == 'dD':
    plt_data1 = dD_q_alltime[expid[i]]['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data2 = dD_q_alltime[expid[i]]['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data3 = plt_data1 - plt_data2
    cbar_label1 = '$\delta D$ [‰]'
    cbar_label2 = 'Difference: (a) - (b) [‰]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-550, cm_max=-50, cm_interval1=50, cm_interval2=100,
        cmap='viridis', reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=40, cmap='BrBG',
        reversed=False)
    extend = 'both'
    extend1 = 'both'
elif var == 'd_ln':
    plt_data1 = d_ln_q_alltime[expid[i]]['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data2 = d_ln_q_alltime[expid[i]]['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data3 = plt_data1 - plt_data2
    cbar_label1 = '$d_{ln}$ [‰]'
    cbar_label2 = 'Difference: (a) - (b) [‰]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=0, cm_max=40, cm_interval1=4, cm_interval2=4,
        cmap='viridis', reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-20, cm_max=20, cm_interval1=4, cm_interval2=4, cmap='BrBG',
        reversed=False)
    extend = 'both'
    extend1 = 'both'
elif var == 'lat':
    plt_data1 = q_weighted_lat[expid[i]]['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data2 = q_weighted_lat[expid[i]]['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data3 = plt_data1 - plt_data2
    cbar_label1 = 'Source latitude [$°\;S$]'
    cbar_label2 = 'Difference: (a) - (b) [$°$]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-55, cm_max=-10, cm_interval1=5, cm_interval2=5, cmap='viridis',
        reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2, cmap='BrBG',
        reversed=False)
    extend = 'both'
    extend1 = 'both'
elif var == 'sst':
    plt_data1 = q_weighted_sst[expid[i]]['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data2 = q_weighted_sst[expid[i]]['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest')
    plt_data3 = plt_data1 - plt_data2
    cbar_label1 = 'Source SST [$°C$]'
    cbar_label2 = 'Difference: (a) - (b) [$°C$]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis',
        reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-8, cm_max=8, cm_interval1=2, cm_interval2=2, cmap='BrBG',
        reversed=False)
    extend = 'both'
    extend1 = 'both'
elif var == 'lon':
    plt_data1 = calc_lon_diff(q_weighted_lon[expid[i]]['daily'].sel(time=pos_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest'), t63_sites_indices['EDC']['lon'])
    plt_data2 = calc_lon_diff(q_weighted_lon[expid[i]]['daily'].sel(time=neg_case_time, lon=t63_sites_indices['EDC']['lon'], method='nearest'), t63_sites_indices['EDC']['lon'])
    plt_data3 = calc_lon_diff(plt_data1, plt_data2)
    cbar_label1 = 'Relative source longitude [$°$]'
    cbar_label2 = 'Difference: (a) - (b) [$°$]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-180, cm_max=180, cm_interval1=30, cm_interval2=60,
        cmap='twilight_shifted',)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',
        reversed=False)
    extend = 'neither'
    extend1 = 'both'

opng=f'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.4_posnegSAM EPE {pos_case_time} {neg_case_time} {var} meridional cross section across EDC.png'
nrow = 1
ncol = 3
fm_bottom = 2.4 / (5.8*nrow + 2.4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 5.8*nrow + 2.4]) / 2.54,
    gridspec_kw={'hspace': 0.02, 'wspace': 0.05}, sharey=True)

plt1 = axs[0].pcolormesh(
    plt_data1.lat, plt_data1.plev/100, plt_data1,
    norm=pltnorm, cmap=pltcmp, )
plt2 = axs[1].pcolormesh(
    plt_data2.lat, plt_data2.plev/100, plt_data2,
    norm=pltnorm, cmap=pltcmp, )
plt3 = axs[2].pcolormesh(
    plt_data3.lat, plt_data3.plev/100, plt_data3,
    norm=pltnorm1, cmap=pltcmp1, )

for jcol in range(ncol):
    axs[jcol].set_xticks(np.arange(0, -90 - 1e-4, -10))
    axs[jcol].set_xlim(0, -88.57)
    axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    
    axs[jcol].invert_yaxis()
    axs[jcol].set_ylim(1000, 200)
    axs[jcol].set_yticks(np.arange(1000, 200 - 1e-4, -100))
    
    axs[jcol].grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--', zorder=9)

plt.text(
    0.5, 1.05, f'(a) HP during SAM+ at Dome C on {pos_case_time}',
    transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, f'(b) HP during SAM- at Dome C on {neg_case_time}',
    transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c) Difference between (a) and (b)',
    transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

axs[0].set_ylabel('Pressure [$hPa$]')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40, extend=extend,
    anchor=(0.1, 0.4), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1,)
if var=='lat':
    cbar1.ax.set_xticklabels(
        [remove_trailing_zero(x) for x in np.negative(pltticks)])

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.35,aspect=25, extend=extend1,
    anchor=(1.15,-3.8),ticks=pltticks1)
cbar2.ax.set_xlabel(cbar_label2,)

fig.subplots_adjust(left=0.06, right=0.99, bottom=fm_bottom, top=0.92)
fig.savefig(opng)





# endregion



