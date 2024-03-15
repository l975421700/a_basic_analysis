

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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

# self defined
from a_basic_analysis.b_module.mapplot import (
    remove_trailing_zero,
    remove_trailing_zero_pos,
    hemisphere_conic_plot,
    hemisphere_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
    expid_colours,
    expid_labels,
    zerok,
    plot_labels_only_unit,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
    rainbow_text,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

MC16_Dome_C_1d_sim = {}
for i in range(len(expid)):
    print('#-------------------------------- ' + expid[i])
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.MC16_Dome_C_1d_sim.pkl', 'rb') as f:
        MC16_Dome_C_1d_sim[expid[i]] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')
isite = 'EDC'
site_lat = ten_sites_loc[ten_sites_loc['Site'] == isite]['lat'][0]
site_lon = ten_sites_loc[ten_sites_loc['Site'] == isite]['lon'][0]

with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

ERA5_daily_q_2013_2022 = xr.open_dataset('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc', chunks={'time': 720})
ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})


'''
# check
for ivar in ['dD_sim', 'd18O_sim', 'd_ln_sim', 'd_xs_sim', 'q_sim']:
    print((MC16_Dome_C_1d_sim['nudged_717_6.0_I03_2yr'][ivar] == MC16_Dome_C_1d_sim['nudged_713_6.0_2yr'][ivar]).all())

echam6_t63_geosp = xr.open_dataset(exp_odir + expid[i] + '/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)
MC16_1d_height = find_gridvalue_at_site(
    MC16_Dome_C_1d_sim[expid[i]]['lat'].values[0],
    MC16_Dome_C_1d_sim[expid[i]]['lon'].values[0],
    echam6_t63_surface_height.lat.values,
    echam6_t63_surface_height.lon.values,
    echam6_t63_surface_height.values,
)
print('Height of Dome C in T63 ECHAM6: ' + str(np.round(MC16_1d_height, 1)))


#---------------- check correlation
for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 't_3m', 'q']:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    subset = np.isfinite(MC16_Dome_C_1d_sim[expid[i]][var_name]) & np.isfinite(MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'])
    
    print(np.round(pearsonr(MC16_Dome_C_1d_sim[expid[i]][var_name][subset], MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'][subset], ).statistic ** 2, 3))

q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)
MC16_1d_oo2q = find_multi_gridvalue_at_site_time(
    MC16_Dome_C_1d_sim[expid[i]]['time'],
    MC16_Dome_C_1d_sim[expid[i]]['lat'],
    MC16_Dome_C_1d_sim[expid[i]]['lon'],
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].time.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['daily'].sel(geo_regions='Open Ocean').values,
    )
print(stats.describe(MC16_1d_oo2q))
# 80%
print(find_gridvalue_at_site(
    MC16_Dome_C_1d_sim[expid[i]]['lat'][0],
    MC16_Dome_C_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='Open Ocean').values,
    ))
# 83.7%
print(find_gridvalue_at_site(
    MC16_Dome_C_1d_sim[expid[i]]['lat'][0],
    MC16_Dome_C_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='AIS').values,
    ))
# 5.4%
print(find_gridvalue_at_site(
    MC16_Dome_C_1d_sim[expid[i]]['lat'][0],
    MC16_Dome_C_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='Land excl. AIS').values,
    ))
# 7.1%
print(find_gridvalue_at_site(
    MC16_Dome_C_1d_sim[expid[i]]['lat'][0],
    MC16_Dome_C_1d_sim[expid[i]]['lon'][0],
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lat.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].lon.values,
    q_geo7_sfc_frc_alltime[expid[i]]['am'].sel(geo_regions='SH seaice').values,
    ))
# 3.7%


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region time series - only one model

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']:
    # var_name = 'q'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.1 ' + expid[i] + ' MC16 time series of observed and simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 5.5]) / 2.54)
    
    xdata = MC16_Dome_C_1d_sim[expid[i]]['time'].values
    ydata = MC16_Dome_C_1d_sim[expid[i]][var_name].values
    ydata_sim = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].values
    
    if (var_name == 'q'):
        ydata = ydata * 1000
        ydata_sim = ydata_sim * 1000
    
    RMSE = np.sqrt(np.average(np.square(ydata - ydata_sim)))
    rsquared = pearsonr(ydata, ydata_sim).statistic ** 2
    
    ax.plot(xdata, ydata, 'o', ls='-', ms=2, lw=0.5,
            c='k', label='Observation',)
    ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5,
            c=expid_colours[expid[i]],
            label=expid_labels[expid[i]],)
    
    #  + \
    #                 ': $R^2 = $' + str(np.round(rsquared, 2)) +\
    #                     ', $RMSE = $' + str(np.round(RMSE, 1))
    
    # hourly_y = MC16_Dome_C['1h'][var_name].values
    # if (var_name == 'q'):
    #     hourly_y = hourly_y * 1000
    # ax.plot(
    #     MC16_Dome_C['1h']['time'].values,
    #     hourly_y,
    #     ls='-', lw=0.2, label='Hourly Obs.',)
    
    ax.set_xticks(xdata[::4])
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.set_xlabel('Date', labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[var_name], labelpad=6)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    # if (var_name == 't_3m'):
    #     ax.legend(handlelength=1.5, loc='upper right')
    # else:
    #     ax.legend().set_visible(False)
    ax.legend().set_visible(False)
    
    ax.set_xlabel(
        '$R^2 = $' + str(np.round(rsquared, 2)) + \
            ', $RMSE = $' + str(np.round(RMSE, 1)) + plot_labels_only_unit[var_name],
            color=expid_colours[expid[i]],
            labelpad=6)
    
    # ax.legend(
    #     handlelength=1, loc=(-0.16, -0.35),
    #     framealpha=0.25, ncol=2, columnspacing=1, fontsize=9)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.36, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot - only one model

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']:
    # var_name = 'q', 't_3m'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.0 ' + expid[i] + ' MC16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    xdata = MC16_Dome_C_1d_sim[expid[i]][var_name]
    ydata = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim']
    subset = (np.isfinite(xdata) & np.isfinite(ydata))
    xdata = xdata[subset]
    ydata = ydata[subset]
    
    if (var_name == 'q'):
        xdata = xdata * 1000
        ydata = ydata * 1000
    
    RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
    
    sns.scatterplot(
        x=xdata, y=ydata,
        s=12,
        # marker="o",
    )
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    linearfit = linregress(x = xdata, y = ydata,)
    ax.axline(
        (0, linearfit.intercept), slope = linearfit.slope, lw=1,)
    
    if (linearfit.intercept >= 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x + $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, 1))
    if (linearfit.intercept < 0):
        eq_text = '$y = $' + \
            str(np.round(linearfit.slope, 2)) + '$x $' + \
                str(np.round(linearfit.intercept, 1)) + \
                    '\n$R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                        '\n$RMSE = $' + str(np.round(RMSE, 1))
    
    plt.text(
        0.65, 0.05, eq_text,
        transform=ax.transAxes, fontsize=10, ha='left', va='bottom')
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate daily observations and simulations

#-------------------------------- import data

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

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)
temp2_alltime[expid[i]]['daily']['time'] = temp2_alltime[expid[i]]['daily']['time'].dt.floor('D').rename('time')

lon = d_ln_q_sfc_alltime[expid[i]]['am'].lon
lat = d_ln_q_sfc_alltime[expid[i]]['am'].lat


#-------------------------------- animate data

itime_start = np.datetime64('2014-12-25')
itime_end   = np.datetime64('2015-01-17')

north_extent = -60

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']:
    # var_name = 'dD'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']
    print('#-------------------------------- ' + var_name)
    
    if (var_name == 'dD'):
        var = dD_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-500, cm_max=-150, cm_interval1=25, cm_interval2=50,
            cmap='viridis', reversed=False)
    elif (var_name == 'd18O'):
        var = dO18_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-70, cm_max=-15, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_xs'):
        var = d_excess_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=50, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    elif (var_name == 'd_ln'):
        var = d_ln_q_sfc_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=50, cm_interval1=5, cm_interval2=10,
            cmap='viridis', reversed=False)
    elif (var_name == 'q'): #g/kg
        var = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'].sel(
            lev=47, lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end)) * 1000
        pltlevel = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6,])
        pltticks = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = cm.get_cmap('viridis', len(pltlevel)-1)
    elif ((var_name == 'temp2') | (var_name == 't_3m')):
        var = temp2_alltime[expid[i]]['daily'].sel(lat=slice(north_extent + 2, -90), time=slice(itime_start, itime_end))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-35, cm_max=0, cm_interval1=2.5, cm_interval2=5,
            cmap='viridis', reversed=False)
    
    # b_mask = np.broadcast_to(echam6_t63_ais_mask['mask']['AIS'][-17:, :], var.shape)
    # print(stats.describe(var.values[b_mask], axis=None, nan_policy='omit'))
    # print(stats.describe(MC16_Dome_C_1d_sim[expid[i]][var_name], axis=None, nan_policy='omit'))
    
    itime_start_idx = np.argmin(abs(var.time.values - itime_start))
    itime_end_idx = np.argmin(abs(var.time.values - itime_end))
    plt_data = var[itime_start_idx:itime_end_idx].compute().copy()
    
    start_time = str(plt_data.time.values[0])[:10]
    end_time   = str(plt_data.time.values[-1])[:10]
    # print('Start time: ' + start_time)
    # print('End time:   ' + end_time)
    
    output_mp4 = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.2 ' + expid[i] + ' MC16 daily_sfc ' + var_name + ' ' + start_time + ' to ' + end_time + '.mp4'
    
    fig, ax = hemisphere_plot(northextent=north_extent, fm_top=0.92,)
    # cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
    
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
        format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.15,
        )
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('Daily surface ' + plot_labels[var_name], linespacing=1.5)
    
    plt_objs = []
    
    def update_frames(itime):
        # itime = 0
        global plt_objs
        for plt_obj in plt_objs:
            plt_obj.remove()
        plt_objs = []
        
        plt_mesh = plot_t63_contourf(
            plt_data.lon, plt_data.lat, plt_data[itime], ax,
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
        
        plt_ocean = ax.add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        scatter_values = MC16_Dome_C_1d_sim[expid[i]][var_name][itime].copy()
        if (var_name == 'q'):
            scatter_values = scatter_values * 1000
        plt_scatter = ax.scatter(
            MC16_Dome_C_1d_sim[expid[i]]['lon'][itime],
            MC16_Dome_C_1d_sim[expid[i]]['lat'][itime],
            c=scatter_values,
            marker='o', edgecolors='k', lw=0.5,
            cmap=pltcmp, norm=pltnorm, transform=ccrs.PlateCarree(),
        )
        
        plt_txt = plt.text(
            0.5, 1, str(plt_data.time[itime].values)[:10],
            transform=ax.transAxes,
            ha='center', va='bottom', rotation='horizontal')
        
        plt_objs = plt_mesh.collections + [plt_txt, plt_ocean, plt_scatter,]
        return(plt_objs)
    
    ani = animation.FuncAnimation(
        fig, update_frames, frames=itime_end_idx - itime_start_idx,
        interval=500, blit=False)
    
    ani.save(
        output_mp4,
        progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region time series multiple models

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln',]:
    # var_name = 'q'
    # ['dD', 'd18O', 'd_xs', 'd_ln', 'q']
    print('#-------- ' + var_name)
    
    # output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.1 nudged_712_9 MC16 time series of observed and simulated daily ' + var_name + '.png'
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.1 nudged_712_9 MC16 time series of observed and simulated daily ' + var_name + ' No RMSE.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([10, 9.8]) / 2.54)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = MC16_Dome_C_1d_sim[expid[i]]['time'].values
        ydata = MC16_Dome_C_1d_sim[expid[i]][var_name].values
        ydata_sim = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].values
        
        if (var_name == 'q'):
            ydata = ydata * 1000
            ydata_sim = ydata_sim * 1000
        
        if (i == 0):
            ax.plot(xdata, ydata, 'o', ls='-', ms=2, lw=0.5,
                    c='k', label='Observation',)
        
        RMSE = np.sqrt(np.average(np.square(ydata - ydata_sim)))
        rsquared = pearsonr(ydata, ydata_sim).statistic ** 2
        
        # ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5,
        #         c=expid_colours[expid[i]],
        #         label=expid_labels[expid[i]] + \
        #             ': $RMSE = $' + str(np.round(RMSE, 1)),)
        ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5,
                c=expid_colours[expid[i]],
                label=expid_labels[expid[i]])
    
    # hourly_y = MC16_Dome_C['1h'][var_name].values
    # if (var_name == 'q'):
    #     hourly_y = hourly_y * 1000
    # ax.plot(
    #     MC16_Dome_C['1h']['time'].values,
    #     hourly_y,
    #     ls='-', lw=0.2, label='Hourly Obs.',)
    
    ax.set_xticks(xdata[::4])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.set_xlabel('Date', labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[var_name], labelpad=6)
    plt.xticks(rotation=30, ha='right')
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    ax.legend(
        handlelength=1, loc=(-0.2, -0.66),
        framealpha=0.25, ncol=2, columnspacing=1,)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.4, top=0.98)
    fig.savefig(output_png)




# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Q-Q plot multiple models


for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'q', 't_3m'
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.0 nudged_712_9 MC16 observed vs. simulated daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
    
    for i in range(len(expid)):
        print(str(i) + ': ' + expid[i])
        
        xdata = MC16_Dome_C_1d_sim[expid[i]][var_name].copy()
        ydata = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].copy()
        subset = (np.isfinite(xdata) & np.isfinite(ydata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        
        if (var_name == 'q'):
            xdata = xdata * 1000
            ydata = ydata * 1000
        
        RMSE = np.sqrt(np.average(np.square(xdata - ydata)))
        
        ax.scatter(
            xdata, ydata,
            s=6, lw=0.4, alpha=0.5,
            facecolors='white',
            edgecolors=expid_colours[expid[i]],
            )
        
        linearfit = linregress(x = xdata, y = ydata,)
        ax.axline(
            (0, linearfit.intercept), slope = linearfit.slope,
            lw=1, alpha=0.5,
            color=expid_colours[expid[i]],
            )
        
        if (linearfit.intercept >= 0):
            eq_text = expid_labels[expid[i]] + ':       $y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x + $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        if (linearfit.intercept < 0):
            eq_text = expid_labels[expid[i]] + ':       $y = $' + \
                str(np.round(linearfit.slope, 2)) + '$x $' + \
                    str(np.round(linearfit.intercept, 1)) + \
                        ', $R^2 = $' + str(np.round(linearfit.rvalue**2, 2)) +\
                            ', $RMSE = $' + str(np.round(RMSE, 1))
        
        plt.text(
            0.05, 0.95 - 0.06*i, eq_text,
            transform=ax.transAxes, fontsize=6,
            color=expid_colours[expid[i]],
            ha='left')
    
    xylim = np.concatenate((np.array(ax.get_xlim()), np.array(ax.get_ylim())))
    xylim_min = np.min(xylim)
    xylim_max = np.max(xylim)
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    
    ax.axline((0, 0), slope = 1, lw=1, color='grey', alpha=0.5)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Observed '  + plot_labels[var_name], labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel('Simulated ' + plot_labels[var_name], labelpad=6)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region time series - one model and ERA5

for var_name in ['q',]:
    # var_name = 'q'
    # ['q', 't_3m']
    print('#-------- ' + var_name)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.1_MC16/8.3.0.1.1 ' + expid[i] + ' MC16 time series of observed, simulated, and ERA5 daily ' + var_name + '.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 5.5]) / 2.54)
    
    xdata = MC16_Dome_C_1d_sim[expid[i]]['time'].values
    ydata = MC16_Dome_C_1d_sim[expid[i]][var_name].values
    ydata_sim = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].values
    
    if (var_name == 'q'):
        ydata = ydata * 1000
        ydata_sim = ydata_sim * 1000
    
    RMSE = np.sqrt(np.average(np.square(ydata - ydata_sim)))
    rsquared = pearsonr(ydata, ydata_sim).statistic ** 2
    
    ax.plot(xdata, ydata, 'o', ls='-', ms=2, lw=0.5,
            c='k', label='Observation',)
    ax.plot(xdata, ydata_sim, 'o', ls='-', ms=2, lw=0.5,
            c=expid_colours[expid[i]],
            label=expid_labels[expid[i]],)
    
    #  + \
    #                 ': $R^2 = $' + str(np.round(rsquared, 2)) +\
    #                     ', $RMSE = $' + str(np.round(RMSE, 1))
    
    # hourly_y = MC16_Dome_C['1h'][var_name].values
    # if (var_name == 'q'):
    #     hourly_y = hourly_y * 1000
    # ax.plot(
    #     MC16_Dome_C['1h']['time'].values,
    #     hourly_y,
    #     ls='-', lw=0.2, label='Hourly Obs.',)
    
    if (var_name == 'q'):
        ERA5_data   = ERA5_daily_q_2013_2022.q.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2014-12-25', '2015-01-16')).values
    elif (var_name == 't_3m'):
        ERA5_data   = ERA5_daily_temp2_2013_2022.t2m.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2014-12-25', '2015-01-16')).values - zerok
    
    ax.plot(xdata, ERA5_data,
            'o', ls='-', ms=2, lw=0.5, c='tab:pink', label='ERA5')
    
    RMSE2 = np.sqrt(np.average(np.square(ydata-ERA5_data)))
    rsquared2 = pearsonr(ydata, ERA5_data).statistic ** 2
    
    ax.set_xticks(xdata[::4])
    plt.xticks(rotation=30, ha='right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.set_xlabel('Date', labelpad=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylabel(plot_labels[var_name], labelpad=6)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    if (var_name == 'q'):
        ax.set_ylim(0.19, 0.54)
        ax.legend(loc='upper right', handlelength=1, handletextpad=0.5,
                  labelspacing=0.4,)
    else:
        ax.legend().set_visible(False)
    # ax.legend().set_visible(False)
    
    if (var_name == 'q'):
        round_digit = 3
    else:
        round_digit = 1
    
    rainbow_text(
        -0.22, -0.54,
        ['$R^2 = $' + str(np.round(rsquared, 2)) + ', $RMSE = $' + str(np.round(RMSE, round_digit)) + plot_labels_only_unit[var_name],
         '; ',
         '$R^2 = $' + str(np.round(rsquared2, 2)) + ', $RMSE = $' + str(np.round(RMSE2, round_digit)) + plot_labels_only_unit[var_name],
         ],
        [expid_colours[expid[i]], 'k', 'tab:pink'],
        ax,
    )
    # ax.set_xlabel(
    #     '$R^2 = $' + str(np.round(rsquared, 2)) + \
    #         ', $RMSE = $' + str(np.round(RMSE, 1)),
    #         color=expid_colours[expid[i]],
    #         labelpad=6)
    
    # ax.legend(
    #     handlelength=1, loc=(-0.16, -0.35),
    #     framealpha=0.25, ncol=2, columnspacing=1, fontsize=9)
    
    ax.grid(True, which='both',
            linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.36, top=0.98)
    fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check statistics

var_name = 'd_ln' # ['dD', 'd18O', 'd_xs', 'd_ln', 'q', 't_3m']

xdata = MC16_Dome_C_1d_sim[expid[i]]['time'].values
ydata = MC16_Dome_C_1d_sim[expid[i]][var_name].values
ydata_sim = MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'].values

if (var_name == 'q'):
    ERA5_data   = ERA5_daily_q_2013_2022.q.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2014-12-25', '2015-01-16')).values
elif (var_name == 't_3m'):
    ERA5_data   = ERA5_daily_temp2_2013_2022.t2m.sel(latitude=site_lat, longitude=site_lon, method='nearest').sel(time=slice('2014-12-25', '2015-01-16')).values - zerok


#-------------------------------- check d_ln

np.std(ydata, ddof=1)
np.std(ydata_sim, ddof=1)

np.min(ydata_sim - ydata)
np.max(ydata_sim - ydata)

pearsonr(ydata, ydata_sim).statistic ** 2
np.sqrt(np.average(np.square(ydata - ydata_sim)))



#-------------------------------- check d18O
pearsonr(ydata, ydata_sim).statistic ** 2
np.sqrt(np.average(np.square(ydata - ydata_sim)))


#-------------------------------- check dD
np.min(ydata_sim - ydata)
np.max(ydata_sim - ydata)
np.average(
    MC16_Dome_C_1d_sim[expid[i]]['dD'].values,
    weights=MC16_Dome_C_1d_sim[expid[i]]['q'].values,
)
np.average(
    MC16_Dome_C_1d_sim[expid[i]]['dD_sim'].values,
    weights=MC16_Dome_C_1d_sim[expid[i]]['q_sim'].values,
)

pearsonr(ydata, ydata_sim).statistic ** 2
np.sqrt(np.average(np.square(ydata - ydata_sim)))


#-------------------------------- check q

np.min(ydata_sim - ydata)
np.max(ydata_sim - ydata)


if (var_name == 'q'):
    ydata = ydata * 1000
    ydata_sim = ydata_sim * 1000

pearsonr(ydata, ydata_sim).statistic ** 2
np.sqrt(np.average(np.square(ydata - ydata_sim)))


#-------------------------------- check t_3m

np.min(ydata_sim - ydata)
np.max(ydata_sim - ydata)

pearsonr(ydata, ERA5_data).statistic ** 2
np.sqrt(np.average(np.square(ydata - ERA5_data)))

# endregion
# -----------------------------------------------------------------------------
