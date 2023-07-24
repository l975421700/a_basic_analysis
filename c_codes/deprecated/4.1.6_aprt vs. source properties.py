

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
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

#---- import wisoaprt

wisoaprt_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoaprt_alltime[expid[0]]['am'].lon
lat = wisoaprt_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

#---- import precipitation sources

source_var = ['latitude', 'SST', 'rh2m', 'wind10']
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
(sam_mon[expid[0]].sam.values == sam_mon[expid[3]].sam.values).all()

'''
# endregion
# -----------------------------------------------------------------------------

ialltime = 'daily'
ialltime = 'mon'
ialltime = 'ann'

ivar = 'latitude'
ivar = 'SST'
ivar = 'rh2m'
ivar = 'wind10'

# -----------------------------------------------------------------------------
# region Corr. aprt & source properties

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.0 pi_600_3 ' + ialltime + ' corr. aprt and ' + ivar + '.png'

cor_aprt_ivar = {}
cor_aprt_ivar_p = {}

if (ialltime == 'daily'):
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        cor_aprt_ivar[expid[i]] = xr.corr(
            wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1),
            pre_weighted_var[expid[i]][ivar]['daily'],
            dim='time').compute()
    
elif (ialltime == 'mon'):
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        cor_aprt_ivar[expid[i]] = xr.corr(
            wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).groupby('time.month') - \
                wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1),
            pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
                pre_weighted_var[expid[i]][ivar]['mm'],
            dim='time').compute()
        
        # cor_aprt_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        #     wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).groupby('time.month') - \
        #         wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1),
        #     pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
        #         pre_weighted_var[expid[i]][ivar]['mm'],
        #     dim='time').values
        
        # cor_aprt_ivar[expid[i]].values[
        #     cor_aprt_ivar_p[expid[i]] > 0.05] = np.nan
    
elif (ialltime == 'ann'):
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        cor_aprt_ivar[expid[i]] = xr.corr(
            (wisoaprt_alltime[expid[i]]['ann'].sel(wisotype=1) - \
                wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1)).compute(),
            (pre_weighted_var[expid[i]][ivar]['ann'] - \
                pre_weighted_var[expid[i]][ivar]['am']).compute(),
            dim='time').compute()
        
        # cor_aprt_ivar_p[expid[i]] = xs.pearson_r_eff_p_value(
        #     wisoaprt_alltime[expid[i]]['ann'].sel(wisotype=1) - \
        #         wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1),
        #     pre_weighted_var[expid[i]][ivar]['ann'] - \
        #         pre_weighted_var[expid[i]][ivar]['am'],
        #     dim='time').values
        
        # cor_aprt_ivar[expid[i]].values[
        #     cor_aprt_ivar_p[expid[i]] > 0.05] = np.nan

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

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
        lon, lat, cor_aprt_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Correlation: source '+ivar+' & precipitation', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


ialltime = 'ann'
ialltime = 'mon'

ivar = 'latitude'
ivar = 'rh2m'
ivar = 'wind10'

control_var = 'SST'

# -----------------------------------------------------------------------------
# region Partial Corr. aprt & source properties, given source SST

output_png = 'figures/8_d-excess/8.1_controls/8.1.1_pre_sources/8.1.1.1 pi_600_3 ' + ialltime + ' partial corr. aprt and ' + ivar + ' controlling ' + control_var + '.png'


par_cor_aprt_ivar = {}
par_cor_aprt_ivar_p = {}

if (ialltime == 'ann'):
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        par_cor_aprt_ivar[expid[i]] = xr.apply_ufunc(
            xr_par_cor,
            wisoaprt_alltime[expid[i]]['ann'].sel(wisotype=1),
            pre_weighted_var[expid[i]][ivar]['ann'],
            pre_weighted_var[expid[i]][control_var]['ann'],
            input_core_dims=[["time"], ["time"], ["time"]],
            kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
        )
        
        # par_cor_aprt_ivar_p[expid[i]] = xr.apply_ufunc(
        #     xr_par_cor,
        #     wisoaprt_alltime[expid[i]]['ann'].sel(wisotype=1),
        #     pre_weighted_var[expid[i]][ivar]['ann'],
        #     pre_weighted_var[expid[i]][control_var]['ann'],
        #     input_core_dims=[["time"], ["time"], ["time"]],
        #     kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
        # )
elif (ialltime == 'mon'):
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        par_cor_aprt_ivar[expid[i]] = xr.apply_ufunc(
            xr_par_cor,
            wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).groupby('time.month') - \
                wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1),
            pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
                pre_weighted_var[expid[i]][ivar]['mm'],
            pre_weighted_var[expid[i]][control_var]['mon'].groupby('time.month') - \
                pre_weighted_var[expid[i]][control_var]['mm'],
            input_core_dims=[["time"], ["time"], ["time"]],
            kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
        )
        
        # par_cor_aprt_ivar_p[expid[i]] = xr.apply_ufunc(
        #     xr_par_cor,
        #     wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).groupby('time.month') - \
        #         wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1),
        #     pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - \
        #         pre_weighted_var[expid[i]][ivar]['mm'],
        #     pre_weighted_var[expid[i]][control_var]['mon'].groupby('time.month') - \
        #         pre_weighted_var[expid[i]][control_var]['mm'],
        #     input_core_dims=[["time"], ["time"], ["time"]],
        #     kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
        # )

#---------------- plot

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.1, cm_interval2=0.1,
    cmap='PuOr', asymmetric=False, reversed=True)
pltticks[-9] = 0

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

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
        lon, lat, par_cor_aprt_ivar[expid[jcol]], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=50,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('Partial correlation: source '+ivar+' & precipitation, controlling ' + control_var, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)




'''
#-------------------------------- check individual grid

import pingouin as pg
i = 2
ilat = 30
ilon = 60

x = aprt_alltime[expid[i]]['ann'][:, ilat, ilon]
y = pre_weighted_var[expid[i]][ivar]['ann'][:, ilat, ilon]
covar = pre_weighted_var[expid[i]][control_var]['ann'][:, ilat, ilon]

xr_par_cor(x, y, covar, output = 'r')
par_cor_aprt_ivar[expid[i]][ilat, ilon]

xr_par_cor(x, y, covar, output = 'p')
par_cor_aprt_ivar_p[expid[i]][ilat, ilon]

'''
# endregion
# -----------------------------------------------------------------------------






