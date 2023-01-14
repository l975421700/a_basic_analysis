

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
from haversine import haversine

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
    find_ilat_ilon,
    find_ilat_ilon_general,
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_rh2m = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.pkl', 'rb') as f:
    pre_weighted_rh2m[expid[i]] = pickle.load(f)

lon = pre_weighted_rh2m[expid[i]]['am'].lon.values
lat = pre_weighted_rh2m[expid[i]]['am'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)


pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

rh2m_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.rh2m_alltime.pkl', 'rb') as f:
    rh2m_alltime[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get rh2m at sources


rh2m_sources = {}
rh2m_sources[expid[i]] = {}
rh2m_sources[expid[i]]['am'] = pre_weighted_rh2m[expid[i]]['am'].copy()
rh2m_sources[expid[i]]['am'] = rh2m_sources[expid[i]]['am'].rename('rh2m_sources')

rh2m_sources[expid[i]]['am'].values[:] = 0

for ilat in range(len(lat)):
    # ilat = 87
    # print('#---------------- ' + str(ilat))
    
    for ilon in range(len(lon)):
        # ilon = 23
        # print('#-------- ' + str(ilon))
        
        try:
            source_lat = pre_weighted_lat[expid[i]]['am'][ilat, ilon].values
            source_lon = pre_weighted_lon[expid[i]]['am'][ilat, ilon].values
            
            sources_ind = find_ilat_ilon(source_lat, source_lon, lat, lon)
            # sources_ind = find_ilat_ilon_general(
            #     source_lat, source_lon, lat_2d, lon_2d)
            
            rh2m_sources[expid[i]]['am'].values[ilat, ilon] = \
                rh2m_alltime[expid[i]]['am'][sources_ind[0], sources_ind[1]].values
        except:
            print('no source lat/lon')
            rh2m_sources[expid[i]]['am'].values[ilat, ilon] = np.nan




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source rh2m and rh2m at sources


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_rh2m/6.1.3.3 ' + expid[i] + ' rh2m sources differences.png'

nrow = 1
ncol = 3

wspace = 0.02
hspace = 0.12
fm_left = 0.02
fm_bottom = hspace / nrow
fm_right = 0.98
fm_top = 0.98

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 7.8*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    )

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(
            northextent=-60, ax_org = axs[jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
        
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# source rh2m

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=75, cm_max=83, cm_interval1=1, cm_interval2=1, cmap='PRGn',
    reversed=False)

plt_mesh = axs[0].pcolormesh(
    lon,
    lat,
    pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, ax=axs[0], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source rh2m [$\%$]', linespacing=1.5,)


# rh2m at sources

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=10-2, cm_max=11.5-2, cm_interval1=0.25, cm_interval2=0.25,
#     cmap='PiYG',)

plt_mesh = axs[1].pcolormesh(
    lon,
    lat,
    rh2m_sources[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, ax=axs[1], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('rh2m at source lat/lon [$\%$]', linespacing=1.5,)


# source rh2m - rh2m at sources

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG',
    reversed=False, asymmetric=True,)

plt_mesh = axs[2].pcolormesh(
    lon,
    lat,
    pre_weighted_rh2m[expid[i]]['am'] - rh2m_sources[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, ax=axs[2], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Differences (a - b) [$\%$]', linespacing=1.5,)

fig.subplots_adjust(
    left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
    wspace=wspace, hspace=hspace,
    )

fig.savefig(output_png)




# (pre_weighted_rh2m[expid[i]]['am'] - rh2m_sources[expid[i]]['am']).to_netcdf('scratch/test/test.nc')

# endregion
# -----------------------------------------------------------------------------
