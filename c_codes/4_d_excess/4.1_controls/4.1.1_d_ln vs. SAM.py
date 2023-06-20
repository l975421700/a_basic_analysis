

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]
# i = 0


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

sam_mon = {}
d_ln_alltime = {}
b_sam_mon = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    sam_mon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    b_sam_mon[expid[i]], _ = xr.broadcast(
        sam_mon[expid[i]].sam,
        d_ln_alltime[expid[i]]['mon'])

lon = d_ln_alltime[expid[i]]['am'].lon
lat = d_ln_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
(sam_mon[expid[0]].sam.values == sam_mon[expid[3]].sam.values).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Corr. d_ln & SAM

cor_sam_d_ln = {}
# cor_sam_d_ln_p = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    cor_sam_d_ln[expid[i]] = xr.corr(
        b_sam_mon[expid[i]],
        d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'],
        dim='time').compute()
    
    # cor_sam_d_ln_p[expid[i]] = xs.pearson_r_eff_p_value(
    #     b_sam_mon[expid[i]],
    #     d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
    #         d_ln_alltime[expid[i]]['mm'],
    #     dim='time').values
    
    # cor_sam_d_ln[expid[i]].values[cor_sam_d_ln_p[expid[i]] > 0.05] = np.nan


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.6, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-7] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.0 pi_600_3 corr. sam_d_ln mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat, cor_sam_d_ln[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)


cbar = fig.colorbar(
    plt1, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: SAM & $d_{ln}$', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region SAM+ vs. SAM- d_ln

sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_mon[expid[0]].sam > sam_mon[expid[0]].sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_mon[expid[0]].sam < (-1 * sam_mon[expid[0]].sam.std(ddof = 1))

sam_posneg_d_ln = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    sam_posneg_d_ln[expid[i]] = {}
    
    # sam_posneg_d_ln[expid[i]]['pos'] = \
    #     d_ln_alltime[expid[i]]['mon'][sam_posneg_ind['pos']]
    # sam_posneg_d_ln[expid[i]]['neg'] = \
    #     d_ln_alltime[expid[i]]['mon'][sam_posneg_ind['neg']]
    
    sam_posneg_d_ln[expid[i]]['pos'] = \
        (d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'])[sam_posneg_ind['pos']]
    sam_posneg_d_ln[expid[i]]['neg'] = \
        (d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
            d_ln_alltime[expid[i]]['mm'])[sam_posneg_ind['neg']]
    
    sam_posneg_d_ln[expid[i]]['pos_mean'] = \
        sam_posneg_d_ln[expid[i]]['pos'].mean(dim='time')
    sam_posneg_d_ln[expid[i]]['neg_mean'] = \
        sam_posneg_d_ln[expid[i]]['neg'].mean(dim='time')


#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-8, cm_max=8, cm_interval1=1, cm_interval2=2,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-5] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.1 pi_600_3 sam_posneg_d_ln mon.png'

nrow = 1
ncol = 4

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1
    
    plt.text(
        0.5, 1.08, column_names[jcol],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    
    # ttest_fdr_res = ttest_fdr_control(
    #     sam_posneg_d_ln[expid[jcol]]['pos'],
    #     sam_posneg_d_ln[expid[jcol]]['neg'],)
    d_ln_diff = (sam_posneg_d_ln[expid[jcol]]['pos_mean'] - \
        sam_posneg_d_ln[expid[jcol]]['neg_mean']).compute()
    # d_ln_diff.values[ttest_fdr_res == False] = np.nan
    
    # plot corr.
    plt1 = plot_t63_contourf(
        lon, lat,
        d_ln_diff,
        axs[jcol], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)


cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.6, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Differences in $d_{ln}$ [‰] between SAM+ and SAM-', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)




'''
#-------------------------------- check

len(sam_mon[expid[0]].sam)
sam_posneg_ind['pos'].sum()
sam_posneg_ind['neg'].sum()




d_ln_alltime[expid[i]]['mon'].groupby('time.month') - \
    d_ln_alltime[expid[i]]['mm']
'''
# endregion
# -----------------------------------------------------------------------------
