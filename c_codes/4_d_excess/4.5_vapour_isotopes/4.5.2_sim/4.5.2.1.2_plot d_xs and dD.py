
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import pearsonr
from scipy.stats import linregress
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath
from metpy.calc import pressure_to_height_std
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
    remove_trailing_zero_pos_abs,
    ticks_labels,
    hemisphere_conic_plot,
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
    plot_labels,
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

dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

d_excess_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)

q_sfc_weighted_var = {}
q_sfc_weighted_var[expid[i]] = {}

for src_var in ['lat', ]:
    print('#--------------------------------' + src_var)
    src_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + src_var + '.pkl'
    print(src_file)
    
    with open(src_file, 'rb') as f:
        q_sfc_weighted_var[expid[i]][src_var] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am and sm data

imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)

for ialltime in ['DJF', 'am']:
    print('#-------------------------------- ' + ialltime)
    
    for idelta in ['dD', 'd18O']:
        print('#---------------- ' + idelta)
        
        for idex in ['d_ln', 'd_xs']:
            print('#-------- ' + idex)
            
            if ((ialltime == 'DJF') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['sm'].sel(season='DJF').values[mask_high]
            elif ((ialltime == 'DJF') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['sm'].sel(season='DJF').values[mask_high]
            elif ((ialltime == 'am') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['am'].values[mask_high]
            elif ((ialltime == 'am') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['am'].values[mask_high]
            
            if ((ialltime == 'DJF') & (idex == 'd_ln')):
                ydata = d_ln_q_sfc_alltime[expid[i]]['sm'].sel(season='DJF').values[mask_high]
            elif ((ialltime == 'DJF') & (idex == 'd_xs')):
                ydata = d_excess_q_sfc_alltime[expid[i]]['sm'].sel(season='DJF').values[mask_high]
            elif ((ialltime == 'am') & (idex == 'd_ln')):
                ydata = d_ln_q_sfc_alltime[expid[i]]['am'].values[mask_high]
            elif ((ialltime == 'am') & (idex == 'd_xs')):
                ydata = d_excess_q_sfc_alltime[expid[i]]['am'].values[mask_high]
            
            output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.2 ' + expid[i] + ' ' + ialltime + ' ' + idelta + ' vs. ' + idex + ' over Antarctic Plateau.png'
            
            fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
            
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            sns.scatterplot(x=xdata, y=ydata, s=12, marker="x",)
            
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlabel(ialltime + ' ' + plot_labels[idelta] + ' over Antarctic Plateau', labelpad=6)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel(ialltime + ' ' + plot_labels[idex] + ' over Antarctic Plateau', labelpad=6)
            
            ax.grid(True, which='both',
                    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
            fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
            fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily data

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)

b_mask_high_all = np.broadcast_to(mask_high, dD_q_sfc_alltime[expid[i]]['daily'].shape)
b_mask_high_DJF = np.broadcast_to(mask_high, dD_q_sfc_alltime[expid[i]]['daily'][dD_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].shape)

for ialltime in ['DJF', 'Ann']:
    # ialltime = 'am'
    print('#-------------------------------- ' + ialltime)
    
    for idelta in ['dD', 'd18O']:
        # idelta = 'dD'
        print('#---------------- ' + idelta)
        
        for idex in ['d_ln', 'd_xs']:
            # idex = 'd_ln'
            print('#-------- ' + idex)
            
            if ((ialltime == 'DJF') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['daily'][dD_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                xmin = -640
                xmax = -250
            elif ((ialltime == 'DJF') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['daily'][dO18_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                xmin = -85
                xmax = -30
            elif ((ialltime == 'Ann') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                xmin = -1000
                xmax = -240
            elif ((ialltime == 'Ann') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                xmin = -130
                xmax = -30
            
            if ((ialltime == 'DJF') & (idex == 'd_ln')):
                ydata = d_ln_q_sfc_alltime[expid[i]]['daily'][d_ln_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                ymin = -20
                ymax = 50
            elif ((ialltime == 'DJF') & (idex == 'd_xs')):
                ydata = d_excess_q_sfc_alltime[expid[i]]['daily'][d_excess_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                ymin = -10
                ymax = 60
            elif ((ialltime == 'Ann') & (idex == 'd_ln')):
                ydata = d_ln_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                ymin = -150
                ymax = 50
            elif ((ialltime == 'Ann') & (idex == 'd_xs')):
                ydata = d_excess_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                ymin = -20
                ymax = 150
            
            output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.2 ' + expid[i] + ' daily_values in ' + ialltime + ' ' + idelta + ' vs. ' + idex + ' over Antarctic Plateau.png'
            
            fig = plt.figure(figsize=np.array([8.8, 8.8]) / 2.54,)
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            ax.scatter_density(xdata, ydata, cmap=white_viridis)
            # sns.scatterplot(x=xdata, y=ydata, s=12, marker="x",)
            
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlabel('Daily ' + plot_labels[idelta] + ' over Antarctic Plateau: ' + ialltime, labelpad=6)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel('Daily ' + plot_labels[idex] + ' over Antarctic Plateau: ' + ialltime, labelpad=6)
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
            ax.grid(True, which='both',
                    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
            fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
            fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot daily dD and source latitude

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)

b_mask_high_all = np.broadcast_to(mask_high, dD_q_sfc_alltime[expid[i]]['daily'].shape)
b_mask_high_DJF = np.broadcast_to(mask_high, dD_q_sfc_alltime[expid[i]]['daily'][dD_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].shape)

for ialltime in ['DJF', 'Ann']:
    # ialltime = 'am'
    print('#-------------------------------- ' + ialltime)
    
    for idelta in ['dD', 'd18O']:
        # idelta = 'dD'
        print('#---------------- ' + idelta)
        
        for ivar in ['lat']:
            # idex = 'd_ln'
            print('#-------- ' + ivar)
            
            if ((ialltime == 'DJF') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['daily'][dD_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                xmin = -640
                xmax = -250
            elif ((ialltime == 'DJF') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['daily'][dO18_q_sfc_alltime[expid[i]]['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                xmin = -85
                xmax = -30
            elif ((ialltime == 'Ann') & (idelta == 'dD')):
                xdata = dD_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                xmin = -1000
                xmax = -240
            elif ((ialltime == 'Ann') & (idelta == 'd18O')):
                xdata = dO18_q_sfc_alltime[expid[i]]['daily'].values[b_mask_high_all].copy()
                xmin = -130
                xmax = -30
            
            if ((ialltime == 'DJF') & (ivar == 'lat')):
                ydata = q_sfc_weighted_var[expid[i]]['lat']['daily'][q_sfc_weighted_var[expid[i]]['lat']['daily'].time.dt.season == 'DJF'].values[b_mask_high_DJF].copy()
                ymin = -70
                ymax = -30
            elif ((ialltime == 'Ann') & (ivar == 'lat')):
                ydata = q_sfc_weighted_var[expid[i]]['lat']['daily'].values[b_mask_high_all].copy()
                ymin = -70
                ymax = -30
            
            output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0.2 ' + expid[i] + ' daily_values in ' + ialltime + ' ' + idelta + ' vs. ' + ivar + ' over Antarctic Plateau.png'
            
            # fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
            fig = plt.figure(figsize=np.array([8.8, 8.8]) / 2.54,)
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            
            subset = (np.isfinite(xdata) & np.isfinite(ydata))
            xdata = xdata[subset]
            ydata = ydata[subset]
            
            ax.scatter_density(xdata, ydata, cmap=white_viridis)
            # sns.scatterplot(x=xdata, y=ydata, s=12, marker="x",)
            
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlabel('Daily ' + plot_labels[idelta] + ' over Antarctic Plateau: ' + ialltime, labelpad=6)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel('Daily ' + plot_labels[ivar] + ' over Antarctic Plateau: ' + ialltime, labelpad=6)
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
            ax.grid(True, which='both',
                    linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
            fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
            fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region lowest model height

! cdo -selname,aps /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_5.0/unknown/hist_700_5.0_187001.01_g3b_1m.nc aps
! cdo -selname,geosp /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_5.0/unknown/hist_700_5.0_187001.01_echam.nc geosp
! cdo -selname,q /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_5.0/unknown/hist_700_5.0_187001.01_gl_1m.nc q
! cdo -sp2gp -selname,st /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_5.0/unknown/hist_700_5.0_187001.01_sp_1m.nc t
! cdo -gheight -merge aps geosp q t geop


echam6_t63_geop = xr.open_dataset('scratch/test/geop')
klev_height = echam6_t63_geop.zh.squeeze().sel(lev=47)

from metpy.calc import geopotential_to_height
from metpy.units import units
echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

stats.describe(klev_height.values - echam6_t63_surface_height.values, axis=None)
# 24 - 34 m

# echam6_t63_geop.zh.squeeze().sel(lev=47).to_netcdf('scratch/test/test.nc')
(klev_height - echam6_t63_surface_height.values).to_netcdf('albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_5.0/analysis/echam/lowest_model_height.nc')


# endregion
# -----------------------------------------------------------------------------
