

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress
import pingouin as pg

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
from scipy.stats import pearsonr
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
    hemisphere_conic_plot,
    ticks_labels,
    plot_maxmin_points,
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
    plot_labels_no_unit,
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

d_ln_q_sfc_alltime = {}
# d_excess_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    # with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    #     d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

source_var = ['lat', 'lon', 'sst', 'RHsst', 'distance']
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        prefix + '.q_sfc_weighted_lon.pkl',
        prefix + '.q_sfc_weighted_sst.pkl',
        prefix + '.q_sfc_weighted_RHsst.pkl',
        prefix + '.q_sfc_transport_distance.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)

corr_sources_isotopes_q_sfc = {}
par_corr_sources_isotopes_q_sfc={}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
        par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

lon = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_ln']['mon']['r'].lat

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.4,
    cmap='PuOr', asymmetric=False, reversed=True)

RHsst_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'rb') as f:
    RHsst_alltime[expid[i]] = pickle.load(f)

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

daily_uv_ml_k_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.daily_uv_ml_k_alltime.pkl', 'rb') as f:
    daily_uv_ml_k_alltime[expid[i]] = pickle.load(f)

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


#-------------------------------- Daily negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- Daily positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])


#-------------------------------- Annual negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_min = np.min(annual_par_corr)
where_annual_min = np.where(annual_par_corr == annual_min)
# print(annual_min)
# print(annual_par_corr[where_annual_min[0][0], where_annual_min[1][0]])

annual_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lon.values
annual_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lat.values

annual_min_ilon = np.where(lon == annual_min_lon)[0][0]
annual_min_ilat = np.where(lat == annual_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_min_ilat, annual_min_ilon])


#-------------------------------- Annual positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_max = np.max(annual_par_corr)
where_annual_max = np.where(annual_par_corr == annual_max)
# print(annual_max)
# print(annual_par_corr[where_annual_max[0][0], where_annual_max[1][0]])

annual_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lon.values
annual_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lat.values

annual_max_ilon = np.where(lon == annual_max_lon)[0][0]
annual_max_ilat = np.where(lat == annual_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_max_ilat, annual_max_ilon])




daily_pos_lon = 30
daily_pos_lat = -40
daily_pos_ilon = np.argmin(abs(lon.values - daily_pos_lon))
daily_pos_ilat = np.argmin(abs(lat.values - daily_pos_lat))

daily_neg_lon = 30
daily_neg_lat = -50
daily_neg_ilon = np.argmin(abs(lon.values - daily_neg_lon))
daily_neg_ilat = np.argmin(abs(lat.values - daily_neg_lat))


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot wind and mslp

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11759
idatalength = 30

wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][itimestart].values
wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][itimestart].values
pres   = psl_zh[expid[i]]['psl'][ialltime][itimestart] / 100

pres_interval = 5
pres_interval1 = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)


#-------------------------------- plot

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp at ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + '.png'

fig, ax = hemisphere_conic_plot(
    lat_min=-75, lat_max=-15, lon_min=0, lon_max=90,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -70, lon_min_tick = 0,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8.8, 7.2]) / 2.54,)

# cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12, lw=1,
#                 edgecolors = 'tab:blue')
cplot_ice_cores(lon=daily_neg_lon, lat=daily_neg_lat, ax=ax, s=12, lw=1,
                edgecolors = 'k')
cplot_ice_cores(lon=daily_pos_lon, lat=daily_pos_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:blue')

#-------- plot mslp

plt_pres = ax.contour(
    lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
    pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
    colors='b', levels=pres_interval1, linewidths=0.2,
    transform=ccrs.PlateCarree(),)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),
    )

plot_maxmin_points(
    lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
    pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
    ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
    pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
    ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)


#-------- plot winds
iarrow = 1
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.072, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

# cplot_ice_cores(
#     lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][itimestart].sel(lon=daily_min_lon, lat=daily_min_lat, method='nearest').values,
#     lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][itimestart].sel(lon=daily_min_lon, lat=daily_min_lat, method='nearest').values,
#     ax=ax, s=12, edgecolors = 'tab:blue', alpha=0.3,)
cplot_ice_cores(
    lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][itimestart].sel(lon=daily_neg_lon, lat=daily_neg_lat, method='nearest').values,
    lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][itimestart].sel(lon=daily_neg_lon, lat=daily_neg_lat, method='nearest').values,
    ax=ax, s=12, edgecolors = 'k', alpha=0.3,)
cplot_ice_cores(
    lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][itimestart].sel(lon=daily_pos_lon, lat=daily_pos_lat, method='nearest').values,
    lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][itimestart].sel(lon=daily_pos_lon, lat=daily_pos_lat, method='nearest').values,
    ax=ax, s=12, edgecolors = 'tab:blue', alpha=0.3,)

plt.text(
    0.5, -0.14,
    str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10],
    ha='center', va='top', transform=ax.transAxes,
)
fig.savefig(output_png)




'''
# ticklabel = ticks_labels(10, 75, -70, -15, 10, 10,)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate wind and mslp

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11730
idatalength = 30

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + '.mp4'

fig, ax = hemisphere_conic_plot(
    lat_min=-75, lat_max=-15, lon_min=0, lon_max=90,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -70, lon_min_tick = 0,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8.8, 7.2]) / 2.54,)

cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:blue')
cplot_ice_cores(lon=daily_neg_lon, lat=daily_neg_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:orange')
cplot_ice_cores(lon=daily_pos_lon, lat=daily_pos_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:red')

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    print('#---------------- ' + str(iday))
    
    wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][iday].values
    wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][iday].values
    pres   = psl_zh[expid[i]]['psl'][ialltime][iday] / 100
    
    pres_interval = 5
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    #-------- plot mslp
    
    plt_pres = ax.contour(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        colors='b', levels=pres_interval1, linewidths=0.2,
        transform=ccrs.PlateCarree(),)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    hpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
    lpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)
    
    #-------- plot winds
    iarrow = 1
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1,
        transform=ccrs.PlateCarree(),)
    
    timetext = plt.text(
        0.5, -0.14,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext] + hpre + lpre)
    print(str(iday) + '/' + str(itimestart + idatalength - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.072, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=1000)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate wind and mslp and moisture sources

#-------------------------------- settings

ialltime = 'daily'
# itimestart = 11730
itimestart = 15730
idatalength = 30

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' and moisture sources.mp4'

fig, ax = hemisphere_conic_plot(
    lat_min=-75, lat_max=-15, lon_min=0, lon_max=90,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -70, lon_min_tick = 0,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8.8, 7.2]) / 2.54,)

cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:blue')
cplot_ice_cores(lon=daily_neg_lon, lat=daily_neg_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:orange')
cplot_ice_cores(lon=daily_pos_lon, lat=daily_pos_lat, ax=ax, s=12, lw=1,
                edgecolors = 'tab:red')

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    # iday = itimestart
    print('#---------------- ' + str(iday))
    
    wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][iday].values
    wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][iday].values
    pres   = psl_zh[expid[i]]['psl'][ialltime][iday] / 100
    
    pres_interval = 5
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    #-------- plot mslp
    
    plt_pres = ax.contour(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        colors='b', levels=pres_interval1, linewidths=0.2,
        transform=ccrs.PlateCarree(),)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    hpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
    lpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 90)), lat.sel(lat=slice(-15, -75)),
        pres.sel(lon=slice(0, 90), lat=slice(-15, -75)),
        ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)
    
    #-------- plot winds
    iarrow = 1
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1,
        transform=ccrs.PlateCarree(),)
    
    timetext = plt.text(
        0.5, -0.14,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    plt_scatter1 = cplot_ice_cores(
        lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][iday].sel(lon=daily_min_lon, lat=daily_min_lat, method='nearest').values,
        lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][iday].sel(lon=daily_min_lon, lat=daily_min_lat, method='nearest').values,
        ax=ax, s=12, edgecolors = 'tab:blue', alpha=0.3,)
    plt_scatter2 = cplot_ice_cores(
        lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][iday].sel(lon=daily_neg_lon, lat=daily_neg_lat, method='nearest').values,
        lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][iday].sel(lon=daily_neg_lon, lat=daily_neg_lat, method='nearest').values,
        ax=ax, s=12, edgecolors = 'tab:orange', alpha=0.3,)
    plt_scatter3 = cplot_ice_cores(
        lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][iday].sel(lon=daily_pos_lon, lat=daily_pos_lat, method='nearest').values,
        lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][iday].sel(lon=daily_pos_lon, lat=daily_pos_lat, method='nearest').values,
        ax=ax, s=12, edgecolors = 'tab:red', alpha=0.3,)
    
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext, plt_scatter1, plt_scatter2, plt_scatter3] + hpre + lpre)
    print(str(iday) + '/' + str(itimestart + idatalength - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.072, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=1000)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate SST

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11730
idatalength = 30

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=28, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' local SST from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + '.mp4'

fig, ax = hemisphere_conic_plot(
    lat_min=-75, lat_max=-15, lon_min=0, lon_max=90,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -70, lon_min_tick = 0,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8.8, 7.2]) / 2.54,)

cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12,)

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    print('#---------------- ' + str(iday))
    
    plt_mesh1 = plot_t63_contourf(
        lon, lat, tsw_alltime[expid[i]][ialltime][iday], ax,
        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
    plt_mesh1.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_mesh1.collections
    
    timetext = plt.text(
        0.5, -0.32,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    ims.append(add_arts + [timetext])
    print(str(iday) + '/' + str(itimestart + idatalength - 1))

ax.add_feature(
    cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02,
    )
cbar.ax.set_xlabel('Daily SST [$°C$]',)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=250)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot wind and mslp at daily_max

# print(daily_max_lon)
# print(daily_max_lat)

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11730
idatalength = 30

wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][itimestart].values
wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][itimestart].values
pres   = psl_zh[expid[i]]['psl'][ialltime][itimestart] / 100

pres_interval = 5
pres_interval1 = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)


#-------------------------------- plot

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp at ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' daily_max.png'

fig, ax = hemisphere_conic_plot(
    lat_min=-60, lat_max=0, lon_min=250, lon_max=310,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -60, lon_min_tick = 250,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8, 8]) / 2.54,)

cplot_ice_cores(lon=daily_max_lon, lat=daily_max_lat, ax=ax, s=12, lw=1)


#-------- plot mslp

plt_pres = ax.contour(
    lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
    pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
    colors='b', levels=pres_interval1, linewidths=0.2,
    transform=ccrs.PlateCarree(),)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),
    )

plot_maxmin_points(
    lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
    pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
    ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
    pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
    ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)


#-------- plot winds
iarrow = 1
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.08, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

cplot_ice_cores(
    lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][itimestart].sel(lon=daily_max_lon, lat=daily_max_lat, method='nearest').values,
    lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][itimestart].sel(lon=daily_max_lon, lat=daily_max_lat, method='nearest').values,
    ax=ax, s=12, edgecolors = 'k', alpha=0.3)

plt.text(
    0.5, -0.14,
    str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10],
    ha='center', va='top', transform=ax.transAxes,
)
fig.savefig(output_png)




'''
# ticklabel = ticks_labels(10, 75, -70, -15, 10, 10,)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate wind and mslp at daily_max

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11730
idatalength = 30

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' daily_max.mp4'

fig, ax = hemisphere_conic_plot(
    lat_min=-60, lat_max=0, lon_min=250, lon_max=310,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -60, lon_min_tick = 250,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8, 8]) / 2.54,)

cplot_ice_cores(lon=daily_max_lon, lat=daily_max_lat, ax=ax, s=12,)

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    print('#---------------- ' + str(iday))
    
    wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][iday].values
    wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][iday].values
    pres   = psl_zh[expid[i]]['psl'][ialltime][iday] / 100
    
    pres_interval = 5
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    #-------- plot mslp
    
    plt_pres = ax.contour(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        colors='b', levels=pres_interval1, linewidths=0.2,
        transform=ccrs.PlateCarree(),)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    hpre = plot_maxmin_points(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
    lpre = plot_maxmin_points(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)
    
    #-------- plot winds
    iarrow = 1
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1,
        transform=ccrs.PlateCarree(),)
    
    timetext = plt.text(
        0.5, -0.14,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext] + hpre + lpre)
    print(str(iday) + '/' + str(itimestart + idatalength - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.08, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=1000)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate wind and mslp at daily_max and moiture sources

#-------------------------------- settings

ialltime = 'daily'
# itimestart = 11730
itimestart = 15730
idatalength = 30

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' daily_max moisture sources.mp4'

fig, ax = hemisphere_conic_plot(
    lat_min=-60, lat_max=0, lon_min=250, lon_max=310,
    lon_interval=10, lat_interval=10,
    lat_min_tick = -60, lon_min_tick = 250,
    add_grid_labels=True, gl_xlabelsize=0, gl_ylabelsize=8,
    fm_left=0.06, fm_right=0.94, fm_bottom=0.14, fm_top=0.98,
    figsize=np.array([8, 8]) / 2.54,)

cplot_ice_cores(lon=daily_max_lon, lat=daily_max_lat, ax=ax, s=12, lw=1)

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    print('#---------------- ' + str(iday))
    
    wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][iday].values
    wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][iday].values
    pres   = psl_zh[expid[i]]['psl'][ialltime][iday] / 100
    
    pres_interval = 5
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    #-------- plot mslp
    
    plt_pres = ax.contour(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        colors='b', levels=pres_interval1, linewidths=0.2,
        transform=ccrs.PlateCarree(),)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    hpre = plot_maxmin_points(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
    lpre = plot_maxmin_points(
        lon.sel(lon=slice(250, 310)), lat.sel(lat=slice(0, -60)),
        pres.sel(lon=slice(250, 310), lat=slice(0, -60)),
        ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)
    
    #-------- plot winds
    iarrow = 1
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1,
        transform=ccrs.PlateCarree(),)
    
    timetext = plt.text(
        0.5, -0.14,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    plt_scatter = cplot_ice_cores(
        lon=q_sfc_weighted_var[expid[i]]['lon'][ialltime][iday].sel(lon=daily_max_lon, lat=daily_max_lat, method='nearest').values,
        lat=q_sfc_weighted_var[expid[i]]['lat'][ialltime][iday].sel(lon=daily_max_lon, lat=daily_max_lat, method='nearest').values,
        ax=ax, s=12, edgecolors = 'k', alpha=0.3)
    
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext, plt_scatter] + hpre + lpre)
    print(str(iday) + '/' + str(itimestart + idatalength - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.16),)

ax.quiverkey(plt_quiver, X=0.75, Y=-0.08, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=1000)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot wind and mslp SH

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11735
idatalength = 30

wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][itimestart].values
wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][itimestart].values
pres   = psl_zh[expid[i]]['psl'][ialltime][itimestart] / 100

pres_interval = 5
pres_interval1 = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)


#-------------------------------- plot

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp at ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' SH.png'

fig, ax = hemisphere_plot(northextent=0, figsize=np.array([10, 11]) / 2.54,
                          fm_bottom=0.07,)

cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12,)
cplot_ice_cores(lon=daily_max_lon, lat=daily_max_lat, ax=ax, s=12,)

#-------- plot mslp

plt_pres = ax.contour(
    lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
    pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
    colors='b', levels=pres_interval1, linewidths=0.2,
    transform=ccrs.PlateCarree(),)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.08),
    )

plot_maxmin_points(
    lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
    pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
    ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
    pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
    ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)


#-------- plot winds
iarrow = 2
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=1.5, headlength=2.5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.7, Y=-0.024, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

plt.text(
    0.5, -0.06,
    str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10],
    ha='center', va='top', transform=ax.transAxes,
)
fig.savefig(output_png)




'''
# ticklabel = ticks_labels(10, 75, -70, -15, 10, 10,)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate wind and mslp SH

#-------------------------------- settings

ialltime = 'daily'
itimestart = 11730
idatalength = 30

output_mp4 = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.6 ' + expid[i] + ' ' + ialltime + ' uv and mslp from ' + str(psl_zh[expid[i]]['psl'][ialltime][itimestart].time.values)[:10] + ' SH.mp4'

fig, ax = hemisphere_plot(northextent=0, figsize=np.array([10, 11]) / 2.54,
                          fm_bottom=0.07,)

cplot_ice_cores(lon=daily_min_lon, lat=daily_min_lat, ax=ax, s=12,)
cplot_ice_cores(lon=daily_max_lon, lat=daily_max_lat, ax=ax, s=12,)

ims = []

for iday in np.arange(itimestart, itimestart + idatalength, 1):
    print('#---------------- ' + str(iday))
    
    wind_u = daily_uv_ml_k_alltime[expid[i]]['u'][ialltime][iday].values
    wind_v = daily_uv_ml_k_alltime[expid[i]]['v'][ialltime][iday].values
    pres   = psl_zh[expid[i]]['psl'][ialltime][iday] / 100
    
    pres_interval = 5
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    #-------- plot mslp
    
    plt_pres = ax.contour(
        lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
        pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
        colors='b', levels=pres_interval1, linewidths=0.2,
        transform=ccrs.PlateCarree(),)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    hpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
        pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
        ax, 'max', 5, symbol='H', color='b', transform=ccrs.PlateCarree(),)
    lpre = plot_maxmin_points(
        lon.sel(lon=slice(0, 360)), lat.sel(lat=slice(0, -90)),
        pres.sel(lon=slice(0, 360), lat=slice(0, -90)),
        ax, 'min', 5, symbol='L', color='r', transform=ccrs.PlateCarree(),)
    
    #-------- plot winds
    iarrow = 2
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=1.5, headlength=2.5, alpha=1,
        transform=ccrs.PlateCarree(),)
    
    timetext = plt.text(
        0.5, -0.06,
        str(psl_zh[expid[i]]['psl'][ialltime][iday].time.values)[:10],
        ha='center', va='top', transform=ax.transAxes,)
    
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext] + hpre + lpre)
    print(str(iday) + '/' + str(itimestart + idatalength - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    handlelength=1, bbox_to_anchor=(0.35, -0.08),)

ax.quiverkey(plt_quiver, X=0.7, Y=-0.024, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=1000)
ani.save(
    output_mp4,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)



# endregion
# -----------------------------------------------------------------------------


