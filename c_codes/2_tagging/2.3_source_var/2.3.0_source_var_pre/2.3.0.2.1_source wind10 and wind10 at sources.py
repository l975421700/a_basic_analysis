

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
from scipy.stats import pearsonr
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
from matplotlib.ticker import AutoMinorLocator
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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_wind10 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.pkl', 'rb') as f:
    pre_weighted_wind10[expid[i]] = pickle.load(f)

lon = pre_weighted_wind10[expid[i]]['am'].lon.values
lat = pre_weighted_wind10[expid[i]]['am'].lat.values
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

wind10_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_alltime.pkl', 'rb') as f:
    wind10_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wind10 at sources - am


wind10_sources = {}
wind10_sources[expid[i]] = {}
wind10_sources[expid[i]]['am'] = pre_weighted_wind10[expid[i]]['am'].copy()
wind10_sources[expid[i]]['am'] = wind10_sources[expid[i]]['am'].rename('wind10_sources')

wind10_sources[expid[i]]['am'].values[:] = 0

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
            
            wind10_sources[expid[i]]['am'].values[ilat, ilon] = \
                wind10_alltime[expid[i]]['am'][sources_ind[0], sources_ind[1]].values
        except:
            print('no source lat/lon')
            wind10_sources[expid[i]]['am'].values[ilat, ilon] = np.nan




'''
        
        #---- check
        distance = haversine(
            [source_lat, source_lon],
            [lat[sources_ind[0]], lon[sources_ind[1]]],
            normalize=True,)
        if (distance > 130):
            print(distance)
        
        distance = haversine(
            [source_lat, source_lon],
            [lat_2d[sources_ind[0], sources_ind[1]],
             lon_2d[sources_ind[0], sources_ind[1]]],
            normalize=True,)
        
        if (distance > 130):
            print(distance)



sources_ind = find_ilat_ilon(source_lat, source_lon, lat, lon)
sources_ind = find_ilat_ilon_general(
    source_lat, source_lon, lat_2d, lon_2d)
distance = haversine(
    [source_lat, source_lon],
    [lat[sources_ind[0]], lon[sources_ind[1]]],
    normalize=True,)
if (distance > 130):
    print(distance)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source wind10 and wind10 at sources - am


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.4_wind10/6.1.3.4 ' + expid[i] + ' wind10 sources differences.png'

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

# source wind10

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=7, cm_max=12, cm_interval1=0.5, cm_interval2=0.5, cmap='magma_r',)

plt_data = pre_weighted_wind10[expid[i]]['am']
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

# plt_mesh = axs[0].pcolormesh(
#     lon,
#     lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, plt_data, axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[0].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[0], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source vel10 [$m \; s^{-1}$]', linespacing=1.5,)


# wind10 at sources

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=10-2, cm_max=11.5-2, cm_interval1=0.25, cm_interval2=0.25,
#     cmap='PiYG',)

plt_data = wind10_sources[expid[i]]['am']
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

# plt_mesh = axs[1].pcolormesh(
#     lon,
#     lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, plt_data, axs[1],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[1].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[1], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('vel10 at source lat & lon [$m \; s^{-1}$]', linespacing=1.5,)


# source wind10 - wind10 at sources

plt_data = pre_weighted_wind10[expid[i]]['am'] - wind10_sources[expid[i]]['am']
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=4, cm_interval1=0.5, cm_interval2=0.5, cmap='PuOr_r',
    reversed=False, asymmetric=True,)

# plt_mesh = axs[2].pcolormesh(
#     lon,
#     lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, plt_data, axs[2],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[2].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[2], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Differences: (a) - (b) [$m \; s^{-1}$]', linespacing=1.5,)

fig.subplots_adjust(
    left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
    wspace=wspace, hspace=hspace,
    )

fig.savefig(output_png)



# (pre_weighted_wind10[expid[i]]['am'] - wind10_sources[expid[i]]['am']).to_netcdf('scratch/test/test.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region AIS differences

diff = pre_weighted_wind10[expid[i]]['am'] - wind10_sources[expid[i]]['am']

imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]
np.average(
    diff.values[mask],
    weights = echam6_t63_cellarea.cell_area.values[mask],
)



'''
np.min(diff.values[echam6_t63_ais_mask['mask']['AIS']])
np.max(diff.values[echam6_t63_ais_mask['mask']['AIS']])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source wind10 against wind10

imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]
pearsonr(
    wind10_sources[expid[i]]['am'].values[mask],
    pre_weighted_wind10[expid[i]]['am'].values[mask],)
# 0.53

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.6_var_correlation/6.1.3.6.0_wind10/6.1.3.6.0 ' + expid[i] + ' correlation wind10_wind10_at_sources am_AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

sns.scatterplot(
    wind10_sources[expid[i]]['am'].values[mask],
    pre_weighted_wind10[expid[i]]['am'].values[mask],
)

linearfit = linregress(
    x = wind10_sources[expid[i]]['am'].values[mask],
    y = pre_weighted_wind10[expid[i]]['am'].values[mask],
    )
ax.axline((0, linearfit.intercept), slope = linearfit.slope, lw=0.5,
          c='r')
plt.text(
    0.05, 0.85,
    '$y = $' + str(np.round(linearfit.slope, 1)) + '$x + $' + \
        str(np.round(linearfit.intercept, 1)) + \
            '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)),
        transform=ax.transAxes, linespacing=1.5)

ax.set_xlim(
    np.min(wind10_sources[expid[i]]['am'].values[mask]) - 0.5,
    np.max(wind10_sources[expid[i]]['am'].values[mask]) + 0.5,
    )
ax.set_xlabel('Wind10 at source locations[$m\;s^{-1}$]')

# ax.set_ylim(
#     np.min(pre_weighted_wind10[expid[i]]['am'].values[mask]) - 0.1,
#     np.max(pre_weighted_wind10[expid[i]]['am'].values[mask]) + 0.1,
#     )
ax.set_ylim(10, 11.4,)
ax.set_ylabel('Source wind10 [$m\;s^{-1}$]')

ax.grid(
    True, which='both',
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.98)
plt.savefig(output_png)
plt.close()



'''
imask = 'AIS'
mask = echam6_t63_ais_mask['mask'][imask]

pearsonr(
    pre_weighted_wind10[expid[i]]['am'].values[mask],
    wind10_sources[expid[i]]['am'].values[mask],
)
# statistic=0.5330805490996126, pvalue=2.9670112814612486e-123

sns.scatterplot(
    pre_weighted_wind10[expid[i]]['am'].values[mask],
    wind10_sources[expid[i]]['am'].values[mask],
)
plt.savefig('figures/test/trial.png')
plt.close()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wind10 at sources - sm


wind10_sources = {}
wind10_sources[expid[i]] = {}
wind10_sources[expid[i]]['sm'] = pre_weighted_wind10[expid[i]]['sm'].copy()
wind10_sources[expid[i]]['sm'] = wind10_sources[expid[i]]['sm'].rename('wind10_sources')
wind10_sources[expid[i]]['sm'].values[:] = 0

for iseason in ['DJF', 'MAM', 'JJA', 'SON']:
    # iseason = 'DJF'
    print('#---------------- ' + iseason)
    
    for ilat in range(len(lat)):
        # ilat = 87
        # print('#---------------- ' + str(ilat))
        
        for ilon in range(len(lon)):
            # ilon = 23
            # print('#-------- ' + str(ilon))
            
            try:
                source_lat = pre_weighted_lat[expid[i]]['sm'].sel(
                    season = iseason)[ilat, ilon].values
                source_lon = pre_weighted_lon[expid[i]]['sm'].sel(
                    season = iseason)[ilat, ilon].values
                sources_ind = find_ilat_ilon(source_lat, source_lon, lat, lon)
                
                wind10_sources[expid[i]]['sm'].sel(
                    season = iseason).values[ilat, ilon] = \
                    wind10_alltime[expid[i]]['sm'].sel(
                    season = iseason)[sources_ind[0], sources_ind[1]].values
            except:
                print('no source lat/lon')
                wind10_sources[expid[i]]['sm'].sel(
                    season = iseason).values[ilat, ilon] = \
                    np.nan



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source wind10 and wind10 at sources - sm

nrow = 1
ncol = 3

wspace = 0.02
hspace = 0.12
fm_left = 0.02
fm_bottom = hspace / nrow
fm_right = 0.98
fm_top = 0.98

for iseason in ['DJF', 'MAM', 'JJA', 'SON']:
    # iseason = 'DJF'
    print(iseason)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.4_wind10/6.1.3.4 ' + expid[i] + ' wind10 sources differences_' + \
        iseason + '.png'

    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([5.8*ncol, 7.8*nrow]) / 2.54,
        subplot_kw={'projection': ccrs.SouthPolarStereo()},)
    
    ipanel=0
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
        
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
    
    #---------------- plot source wind10
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=7, cm_max=12, cm_interval1=0.5, cm_interval2=0.5, cmap='PiYG',)
    
    plt_data = pre_weighted_wind10[expid[i]]['sm'].sel(season=iseason)
    plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
    plt_mesh = axs[0].pcolormesh(
        lon,
        lat,
        plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt_mesh, ax=axs[0], aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.05,)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel(
        iseason + ' Source wind10 [$m \; s^{-1}$]',
        linespacing=1.5,)
    
    #---------------- plot wind10 at sources
    plt_data = wind10_sources[expid[i]]['sm'].sel(season=iseason)
    plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
    plt_mesh = axs[1].pcolormesh(
        lon,
        lat,
        plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt_mesh, ax=axs[1], aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.05,)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel(
        'wind10 at source lat & lon [$m \; s^{-1}$]', linespacing=1.5,)
    
    #---------------- plot source wind10 - wind10 at sources
    
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=0, cm_max=4, cm_interval1=0.5, cm_interval2=0.5, cmap='Purples',
        reversed=False, asymmetric=False,)
    
    plt_data = pre_weighted_wind10[expid[i]]['sm'].sel(season=iseason) - \
        wind10_sources[expid[i]]['sm'].sel(season=iseason)
    plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
    plt_mesh = axs[2].pcolormesh(
        lon,
        lat,
        plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt_mesh, ax=axs[2], aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.05,
        )
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel(
        'Differences: (a) - (b) [$m \; s^{-1}$]', linespacing=1.5,)
    
    fig.subplots_adjust(
        left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
        wspace=wspace, hspace=hspace,)
    fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region

for iseason in ['DJF', 'MAM', 'JJA', 'SON']:
    # iseason = 'DJF'
    print(iseason)
    
    diff = pre_weighted_wind10[expid[i]]['sm'].sel(season=iseason) - \
        wind10_sources[expid[i]]['sm'].sel(season=iseason)
    
    imask = 'AIS'
    mask = echam6_t63_ais_mask['mask'][imask]
    
    ave_diff = np.average(
        diff.values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],)
    print(np.round(ave_diff, 1))


'''
DJF
2.9
MAM
1.6
JJA
1.3
SON
2.2
'''
# endregion
# -----------------------------------------------------------------------------



