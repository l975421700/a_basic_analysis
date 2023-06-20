

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

d_excess_alltime = {}
d_ln_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)

lon = d_excess_alltime[expid[i]]['am'].lon
lat = d_excess_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

'''
d_excess_alltime[expid[i]]['am']

d_ln_alltime[expid[i]]['am']

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d-excess

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=50, cm_interval1=2.5, cm_interval2=5,
    cmap='viridis', reversed=True)

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.0 pi_600_3 d_excess am.png'

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
        lon, lat, d_excess_alltime[expid[jcol]]['am'], axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('d-excess [‰]', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d_ln

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10,
    cmap='viridis', reversed=False)

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.0 pi_600_3 d_ln am.png'

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
        lon, lat.sel(lat=slice(-57, -90)),
        (d_ln_alltime[expid[jcol]]['am']).sel(lat=slice(-57, -90)),
        axs[jcol],
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt1, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel('$d_{ln}$ [‰]', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot d_ln differences

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=10, cm_max=20, cm_interval1=1, cm_interval2=2,
    cmap='viridis', reversed=False)

pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20,
    cmap='PiYG', reversed=False)

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

output_png = 'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.0 pi_600_3 d_ln and differences am.png'

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

# plot
plt1 = plot_t63_contourf(
    lon, lat.sel(lat=slice(-57, -90)),
    (d_ln_alltime[expid[0]]['am']).sel(lat=slice(-57, -90)),
    axs[0], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

for jcol in np.arange(1, 4, 1):
    print(jcol)
    
    plt2 = plot_t63_contourf(
        lon, lat.sel(lat=slice(-57, -90)),
        ((d_ln_alltime[expid[jcol]]['am'] - d_ln_alltime[expid[0]]['am'])).sel(lat=slice(-57, -90)),
        axs[jcol], pltlevel2, 'both', pltnorm2, pltcmp2, ccrs.PlateCarree(),)

for jcol in range(ncol):
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar1 = fig.colorbar(
    plt1, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(-0.2, 0.35), format=remove_trailing_zero_pos,
    )
cbar1.ax.set_xlabel('$d_{ln}$ [‰]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt2, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks2, extend='both',
    anchor=(1.2, -3.5), format=remove_trailing_zero_pos,
    )
cbar2.ax.set_xlabel('Differences in $d_{ln}$ [‰] (*) vs. (a)', linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
fig.savefig(output_png)




# check diff
for jcol in np.arange(0, 4, 1):
    print('#-------- ' + expid[jcol])
    
    diff = (d_ln_alltime[expid[jcol]]['am'] - d_ln_alltime[expid[0]]['am']).values[echam6_t63_ais_mask['mask']['AIS']]
    print(np.round(np.min(diff), 1))
    print(np.round(np.max(diff), 1))

'''
#-------- pi_600_5.0
0.0
0.0
#-------- pi_601_5.1
2.5
3.8
#-------- pi_602_5.2
-5.9
-3.9
#-------- pi_603_5.3
-4.6
71.6
'''
# endregion
# -----------------------------------------------------------------------------

