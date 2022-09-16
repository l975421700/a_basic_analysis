

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
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
    cplot_wind_vectors,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

uv_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl',
    'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)

lon = uv_plev[expid[i]]['u']['am'].lon
lat = uv_plev[expid[i]]['u']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot 850 hPa wind am

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.2_winds/6.1.2.2 ' + expid[i] + ' am 850hPa wind Antarctica.png'

fig, ax = hemisphere_plot(northextent=-30, figsize=np.array([5.8, 6.2]) / 2.54,
                          fm_bottom=0.1,
                          )
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

iarrow = 3
plt_quiver = ax.quiver(
    lon[::iarrow], lat[::iarrow],
    uv_plev[expid[i]]['u']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    uv_plev[expid[i]]['v']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    color='blue', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)


ax.quiverkey(plt_quiver, X=0.2, Y=-0.07, U=10,
             label='10 $m \; s^{-1}$    850 $hPa$ wind',
             labelpos='E', labelsep=0.05,)

fig.savefig(output_png)

'''
plt_quiver = ax.quiver(
    lon[::iarrow], lat[::iarrow],
    uv_plev[expid[i]]['u']['sm'].sel(season='JJA', plev=85000).values[::iarrow, ::iarrow],
    uv_plev[expid[i]]['v']['sm'].sel(season='JJA', plev=85000).values[::iarrow, ::iarrow],
    color='red', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot 850 hPa wind am/sm Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.2_winds/6.1.2.2 ' + expid[i] + ' 850hPa wind am_sm Antarctica.png'


nrow = 3
ncol = 4
fm_bottom = 0 / (5.8*nrow + 0)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(
                northextent=-30, ax_org = axs[irow, jcol])
            cplot_ice_cores(
                major_ice_core_site.lon, major_ice_core_site.lat,
                axs[irow, jcol])
            plt.text(
                0, 0.95, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='center', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_quiver = cplot_wind_vectors(
    lon, lat,
    uv_plev[expid[i]]['u']['am'].sel(plev=85000).values,
    uv_plev[expid[i]]['v']['am'].sel(plev=85000).values,
    axs[0, 0],
    )

for iseason in range(len(seasons)):
    #-------- sm
    cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['sm'].sel(
            season=seasons[iseason], plev=85000).values,
        uv_plev[expid[i]]['v']['sm'].sel(
            season=seasons[iseason], plev=85000).values,
        axs[1, iseason],
    )
    #-------- sm - am
    cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['sm'].sel(
            season=seasons[iseason], plev=85000).values - \
            uv_plev[expid[i]]['u']['am'].sel(plev=85000).values,
        uv_plev[expid[i]]['v']['sm'].sel(
            season=seasons[iseason], plev=85000).values - \
            uv_plev[expid[i]]['v']['am'].sel(plev=85000).values,
        axs[2, iseason], color='red', scale=200,
        )
    
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

#-------- DJF - JJA
plt_quiver1 = cplot_wind_vectors(
    lon, lat,
    uv_plev[expid[i]]['u']['sm'].sel(season='DJF', plev=85000).values - \
        uv_plev[expid[i]]['u']['sm'].sel(season='JJA', plev=85000).values,
    uv_plev[expid[i]]['v']['sm'].sel(season='DJF', plev=85000).values - \
        uv_plev[expid[i]]['v']['sm'].sel(season='JJA', plev=85000).values,
    axs[0, 1], color='red', scale=200,
    )

#-------- MAM - SON
cplot_wind_vectors(
    lon, lat,
    uv_plev[expid[i]]['u']['sm'].sel(season='MAM', plev=85000).values - \
        uv_plev[expid[i]]['u']['sm'].sel(season='SON', plev=85000).values,
    uv_plev[expid[i]]['v']['sm'].sel(season='MAM', plev=85000).values - \
        uv_plev[expid[i]]['v']['sm'].sel(season='SON', plev=85000).values,
    axs[0, 2], color='red', scale=200,
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

axs[0, 0].quiverkey(plt_quiver, X=3.5, Y=0.6, U=10,
             label='10 $m \; s^{-1}$    850 $hPa$ wind',
             labelpos='E', labelsep=0.05,)
axs[0, 0].quiverkey(plt_quiver1, X=2.2, Y=0.4, U=10,
             label='10 $m \; s^{-1}$    850 $hPa$ wind differences',
             labelpos='E', labelsep=0.05,)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''

iarrow = 3
plt_quiver = ax.quiver(
    lon[::iarrow], lat[::iarrow],
    uv_plev[expid[i]]['u']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    uv_plev[expid[i]]['v']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    color='blue', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot 850 hPa wind DJF-JJA Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.2_winds/6.1.2.2 ' + expid[i] + ' 850hPa wind DJF-JJA Antarctica.png'

fig, ax = hemisphere_plot(northextent=-30, figsize=np.array([5.8, 6.2]) / 2.54,
                          fm_bottom=0.1,
                          )
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_quiver = cplot_wind_vectors(
    lon, lat,
    uv_plev[expid[i]]['u']['sm'].sel(season='DJF', plev=85000).values - \
        uv_plev[expid[i]]['u']['sm'].sel(season='JJA', plev=85000).values,
    uv_plev[expid[i]]['v']['sm'].sel(season='DJF', plev=85000).values - \
        uv_plev[expid[i]]['v']['sm'].sel(season='JJA', plev=85000).values,
    ax, color='red', scale=200,
    )

ax.quiverkey(plt_quiver, X=0.05, Y=-0.07, U=10,
             label='10 $m \; s^{-1}$    850 $hPa$ wind differences',
             labelpos='E', labelsep=0.05,)

fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate troposphere wind DJF-JJA Antarctica

plevs = uv_plev[expid[i]]['u']['am'].plev
plev200 = (plevs >= 20000).sum()

output_mp4 = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.2_winds/6.1.2.2 ' + expid[i] + ' tropospheric wind DJF-JJA Antarctica.mp4'

fig, ax = hemisphere_plot(northextent=-30, figsize=np.array([5.8, 6.2]) / 2.54,
                          fm_bottom=0.1,
                          )
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_objs = []
def update_frames(ilev):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_quiver = cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['sm'].sel(
            season='DJF', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['u']['sm'].sel(
                season='JJA', plev=plevs[ilev]).values,
        uv_plev[expid[i]]['v']['sm'].sel(
            season='DJF', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['v']['sm'].sel(
                season='JJA', plev=plevs[ilev]).values,
        ax, color='red', scale=200,
        )
    
    plt_quiverkey = ax.quiverkey(
        plt_quiver, X=0.03, Y=-0.07, U=10,
        label='10 $m \; s^{-1}$ ' + str(int(plevs[ilev]/100)) + ' $hPa$ wind differences',
        labelpos='E', labelsep=0.05,)
    
    plt_objs = [plt_quiver, plt_quiverkey]
    
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=plev200.values, interval=1000, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)



'''
import xarray as xr
ncfile1 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_echam.nc')
ncfile1.tropo.sel(lat=slice(-30, -90)).mean() / 100
# 200hPa
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate troposphere wind am/sm Antarctica

plevs = uv_plev[expid[i]]['u']['am'].plev
plev200 = (plevs >= 20000).sum()

output_mp4 = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.2_winds/6.1.2.2 ' + expid[i] + ' tropospheric wind am_sm Antarctica.mp4'

nrow = 3
ncol = 4
fm_bottom = 0 / (5.8*nrow + 0)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol != 3)):
            axs[irow, jcol] = hemisphere_plot(
                northextent=-30, ax_org = axs[irow, jcol])
            cplot_ice_cores(
                major_ice_core_site.lon, major_ice_core_site.lat,
                axs[irow, jcol])
            plt.text(
                0, 0.95, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='center', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM - SON', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

for iseason in range(len(seasons)):
    plt.text(
        0.5, 1.05, seasons[iseason], transform=axs[1, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')
    plt.text(
        0.5, 1.05, seasons[iseason] + ' - Annual mean',
        transform=axs[2, iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)


plt_objs = []
def update_frames(ilev):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    #-------- Am
    
    plt_quiver = cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['am'].sel(plev=plevs[ilev]).values,
        uv_plev[expid[i]]['v']['am'].sel(plev=plevs[ilev]).values,
        axs[0, 0],
        )
    plt_objs += [plt_quiver]
    
    for iseason in range(len(seasons)):
        #-------- sm
        plt_quiver = cplot_wind_vectors(
            lon, lat,
            uv_plev[expid[i]]['u']['sm'].sel(
                season=seasons[iseason], plev=plevs[ilev]).values,
            uv_plev[expid[i]]['v']['sm'].sel(
                season=seasons[iseason], plev=plevs[ilev]).values,
            axs[1, iseason],
        )
        plt_objs += [plt_quiver]
        
        #-------- sm - am
        plt_quiver1 = cplot_wind_vectors(
            lon, lat,
            uv_plev[expid[i]]['u']['sm'].sel(
                season=seasons[iseason], plev=plevs[ilev]).values - \
                uv_plev[expid[i]]['u']['am'].sel(plev=plevs[ilev]).values,
            uv_plev[expid[i]]['v']['sm'].sel(
                season=seasons[iseason], plev=plevs[ilev]).values - \
                uv_plev[expid[i]]['v']['am'].sel(plev=plevs[ilev]).values,
            axs[2, iseason], color='red', scale=200,
            )
        plt_objs += [plt_quiver1]
    
    #-------- DJF - JJA
    plt_quiver1 = cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['sm'].sel(season='DJF', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['u']['sm'].sel(season='JJA', plev=plevs[ilev]).values,
        uv_plev[expid[i]]['v']['sm'].sel(season='DJF', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['v']['sm'].sel(season='JJA', plev=plevs[ilev]).values,
        axs[0, 1], color='red', scale=200,
        )
    plt_objs += [plt_quiver1]
    
    #-------- MAM - SON
    plt_quiver1 = cplot_wind_vectors(
        lon, lat,
        uv_plev[expid[i]]['u']['sm'].sel(season='MAM', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['u']['sm'].sel(season='SON', plev=plevs[ilev]).values,
        uv_plev[expid[i]]['v']['sm'].sel(season='MAM', plev=plevs[ilev]).values - \
            uv_plev[expid[i]]['v']['sm'].sel(season='SON', plev=plevs[ilev]).values,
        axs[0, 2], color='red', scale=200,
        )
    plt_objs += [plt_quiver1]
    
    plt_quiverkey = axs[0, 0].quiverkey(
        plt_quiver, X=0.1, Y=1.8, U=10,
        label='10 $m \; s^{-1}$ ' + str(int(plevs[ilev]/100)) + ' $hPa$ wind',
        labelpos='E', labelsep=0.05,)
    plt_objs += [plt_quiverkey]
    
    plt_quiverkey1 = axs[0, 0].quiverkey(
        plt_quiver1, X=1.1, Y=0.4, U=10,
        label='10 $m \; s^{-1}$ ' + str(int(plevs[ilev]/100)) + ' $hPa$ wind differences',
        labelpos='E', labelsep=0.05,)
    plt_objs += [plt_quiverkey1]
    
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames, frames=plev200.values, interval=1000, blit=False)

ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)


# endregion
# -----------------------------------------------------------------------------

