

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
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

lon = pre_weighted_rh2m[expid[i]]['am'].lon
lat = pre_weighted_rh2m[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_rh2m am + am aprt


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_rh2m/6.1.3.3 ' + expid[i] + ' pre_weighted_rh2m am Antarctica + am aprt.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=75, cm_max=83, cm_interval1=1, cm_interval2=1, cmap='PRGn',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# plot am aprt
pltctr1 = np.array([0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, ])
plt_data = wisoaprt_alltime[expid[i]]['am'][0] * seconds_per_d

plt2 = ax.contour(
    lon, lat.sel(lat=slice(-50, -90)),
    plt_data.sel(lat=slice(-50, -90)),
    levels=pltctr1, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=6,)

plt3 = ax.contour(
    lon, lat.sel(lat=slice(-50, -90)),
    plt_data.sel(lat=slice(-50, -90)),
    levels=pltctr2, colors = 'blue', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='blue', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=5, fontsize=6,)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
# cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source rh2m [$\%$]', linespacing=2)
fig.savefig(output_png, dpi=1200)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_rh2m am_sm_5

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_rh2m/6.1.3.3 ' + expid[i] + ' pre_weighted_rh2m am_sm_5 Antarctica.png'
cbar_label1 = 'Source rh2m [$\%$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=73, cm_max=85, cm_interval1=1, cm_interval2=1, cmap='PRGn',
    reversed=False)
ctr_level = np.array([1, 2, 3, ])

nrow = 1
ncol = 5
fm_right = 2 / (5.8*ncol + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol + 2, 5.8*nrow+0.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(
        northextent=-60, ax_org = axs[jcol],
        l45label = False, loceanarcs = False)
    cplot_ice_cores(
        ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])

#-------- Am
plt_mesh1 = axs[0].pcolormesh(
    lon, lat,
    pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_ctr1 = axs[0].contour(
    lon, lat.sel(lat=slice(-60, -90)),
    pre_weighted_rh2m[expid[i]]['ann'].std(
        dim='time', skipna=True, ddof=1).sel(lat=slice(-60, -90)),
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',)
axs[0].clabel(
    plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=6,)
plt.text(
    0.5, 1.04, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

#-------- sm
for iseason in range(len(seasons)):
    axs[1 + iseason].pcolormesh(
        lon, lat,
        pre_weighted_rh2m[expid[i]]['sm'].sel(season=seasons[iseason]),
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt_ctr = axs[1 + iseason].contour(
        lon, lat.sel(lat=slice(-60, -90)),
        pre_weighted_rh2m[expid[i]]['sea'].sel(
            time=(pre_weighted_rh2m[expid[i]]['sea'].time.dt.month == \
                seasons_last_num[iseason])
            ).std(dim='time', skipna=True, ddof=1).sel(lat=slice(-60, -90)),
        levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
        linewidths=0.5, linestyles='solid',
    )
    axs[1 + iseason].clabel(
        plt_ctr, inline=1, colors='b', fmt=remove_trailing_zero,
        levels=ctr_level, inline_spacing=10, fontsize=6,)
    plt.text(
        0.5, 1.04, seasons[iseason], transform=axs[1 + iseason].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=20,extend='both', ticks=pltticks,
    anchor=(1.5, 0.5))
cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator(1))
cbar1.ax.set_ylabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 1-fm_right, bottom = 0, top = 0.94)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pre_weighted_rh2m DJF-JJA


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_rh2m/6.1.3.3 ' + expid[i] + ' pre_weighted_rh2m DJF-JJA Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=8, cm_interval1=1, cm_interval2=1, cmap='BrBG')
# pltcmp = pplt.Colormap('DryWet', samples=len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    pre_weighted_rh2m[expid[i]]['sm'].sel(season='DJF') - \
        pre_weighted_rh2m[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_rh2m[expid[i]]['sea'][3::4,],
    pre_weighted_rh2m[expid[i]]['sea'][1::4,],)
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('DJF - JJA source rh2m [$\%$]', linespacing=2)
fig.savefig(output_png, dpi=1200)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region cross check pre_weighted_rh2m am

#-------- import data
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.3_rh2m/6.1.3.3 ' + expid[i] + ' pre_weighted_rh2m am cross_check.png'
file_dir = 'output/echam-6.3.05p2-wiso/pi/'
pre_weighted_var_files = [
    'pi_m_404_4.7/analysis/echam/pi_m_404_4.7.pre_weighted_rh2m_am.nc',
    'pi_m_408_4.8/analysis/echam/pi_m_408_4.8.pre_weighted_rh2m_am.nc',
]

pre_weighted_var = {}
pre_weighted_var['am_lowres'] = xr.open_dataset(
    file_dir + pre_weighted_var_files[0])
pre_weighted_var['am_highres'] = xr.open_dataset(
    file_dir + pre_weighted_var_files[1])

pre_weighted_rh2m = {}
with open(exp_odir + expid[i] + '/analysis/echam/source_var_short/' + expid[i] + '.pre_weighted_rh2m.pkl', 'rb') as f:
    pre_weighted_rh2m[expid[i]] = pickle.load(f)

#-------------------------------- plot
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=60, cm_max=90, cm_interval1=2, cm_interval2=6, cmap='PRGn',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='bwr',)

cbar_label1 = 'Source rh2m [$\%$]'
cbar_label2 = 'Differences in source rh2m [$\%$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

# plot am values
plt_mesh1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plot am norm - lowres
plt_mesh2 = axs[1].pcolormesh(
    lon, lat, pre_weighted_var['am_lowres'].pre_weighted_rh2m_am * 100 - \
        pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# plot am norm - highres
plt_mesh2 = axs[2].pcolormesh(
    lon, lat, pre_weighted_var['am_highres'].pre_weighted_rh2m_am * 100 - \
        pre_weighted_rh2m[expid[i]]['am'],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Scaling approach',
    transform=axs[0].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Binning (5$\%$ rh2m bins) vs scaling approach',
    transform=axs[1].transAxes, ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Binning (2$\%$ rh2m bins) vs scaling approach',
    transform=axs[2].transAxes, ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)
cbar2.ax.yaxis.set_minor_locator(AutoMinorLocator(1))

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)
cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator(1))

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)





'''

'''
# endregion
# -----------------------------------------------------------------------------


