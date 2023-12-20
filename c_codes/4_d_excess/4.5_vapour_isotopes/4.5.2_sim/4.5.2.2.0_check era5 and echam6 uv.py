

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
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
import metpy.calc as mpcalc

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
from metpy.units import units
from windrose import WindroseAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    regrid,
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

exp_org_o = {}
exp_org_o[expid[i]] = {}

exp_org_o[expid[i]]['uv'] = xr.open_mfdataset(sorted(glob.glob('albedo_scratch/output/echam-6.3.05p2-wiso/pi/' + expid[i] + '/outdata/echam/' + expid[i] + '_??????.daily_uv.nc')))

ERA5_daily_v10_2013_2022 = xr.open_dataset('scratch/ERA5/wind10/ERA5_daily_v10_2013_2022.nc')
ERA5_daily_u10_2013_2022 = xr.open_dataset('scratch/ERA5/wind10/ERA5_daily_u10_2013_2022.nc')

with open('data_sources/AWS/Climantartide_6577cf0dbf2f3/AWS_Dome_C.pkl', 'rb') as f:
    AWS_Dome_C = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


# ERA5_hourly_v10_2013_2022 = xr.open_dataset('scratch/ERA5/wind10/ERA5_hourly_v10_2013_2022.nc')
# ERA5_hourly_u10_2013_2022 = xr.open_dataset('scratch/ERA5/wind10/ERA5_hourly_u10_2013_2022.nc')

'''
exp_org_o[expid[i]]['era5_uv'] = xr.open_mfdataset(sorted(glob.glob('/albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/' + expid[i] + '/forcing/echam/era5_uv_??????.nc'))[-120:-90])

exp_org_o[expid[i]]['uv']

xr.open_dataset('/albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/forcing/echam/era5_uv_197903.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot windrose

isite = 'EDC'

for idataset in [expid[i], 'ERA5_daily', 'ERA5_hourly', 'AWS_hourly', 'AWS_daily',]:
    # expid[i], 'ERA5_daily', 'ERA5_hourly',
    # idataset = 'AWS_daily'
    print('#-------------------------------- ' + idataset)
    
    output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.1 ' + idataset + ' windrose at ' + isite + '.png'
    
    if (idataset == expid[i]):
        wind_u = exp_org_o[expid[i]]['uv']['u'].sel(
            lev=47, time=slice('2013-01-01T23:52:30', '2022-12-31T23:52:30')).sel(
                lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
                lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
                method='nearest',
            ).values
        wind_v = exp_org_o[expid[i]]['uv']['v'].sel(
            lev=47, time=slice('2013-01-01T23:52:30', '2022-12-31T23:52:30')).sel(
                lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
                lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
                method='nearest',
            ).values
    elif (idataset == 'ERA5_daily'):
        wind_u = ERA5_daily_u10_2013_2022.u10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).values
        wind_v = ERA5_daily_v10_2013_2022.v10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).values
    elif (idataset == 'ERA5_hourly'):
        wind_u = ERA5_hourly_u10_2013_2022.u10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).values
        wind_v = ERA5_hourly_v10_2013_2022.v10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).values
    elif (idataset == 'AWS_hourly'):
        wind_strength = AWS_Dome_C['1h']['Vel'].values[(AWS_Dome_C['1h']['time'].dt.year >= 2013)]
        wind_direction = AWS_Dome_C['1h']['Dir'].values[(AWS_Dome_C['1h']['time'].dt.year >= 2013)] + 180
        wind_direction[wind_direction > 360] -= 360
        
        wind_direction = wind_direction[wind_strength>0]
        wind_strength = wind_strength[wind_strength>0]
    elif (idataset == 'AWS_daily'):
        wind_u, wind_v = mpcalc.wind_components(AWS_Dome_C['1h']['Vel'].values * units('m/s'), AWS_Dome_C['1h']['Dir'].values * units.deg)
        AWS_Dome_C['1h']['wind_u'] = wind_u.magnitude
        AWS_Dome_C['1h']['wind_v'] = wind_v.magnitude
        AWS_Dome_C['1d'] = AWS_Dome_C['1h'].resample('1d', on='time').mean().reset_index()
        
        wind_u = AWS_Dome_C['1d']['wind_u'].values
        wind_v = AWS_Dome_C['1d']['wind_v'].values
    
    if (idataset != 'AWS_hourly'):
        wind_strength = (wind_u**2 + wind_v**2)**0.5
        wind_direction = mpcalc.wind_direction(
            u=wind_u * units('m/s'), v=wind_v * units('m/s'), convention='to'
            ).magnitude
    
    if (idataset == 'AWS_daily'):
        wind_direction = wind_direction[wind_strength>0]
        wind_strength = wind_strength[wind_strength>0]
    
    # stats.describe(wind_strength)
    
    windbins = np.arange(0, 12.1, 2, dtype = 'int32')
    fig, ax = plt.subplots(
        figsize=np.array([8.8, 7]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree()},)
    ax.set_extent([0, 1, 0, 1])
    
    windrose_ax = inset_axes(
        ax, width=2.2, height=2.2, loc=10, bbox_to_anchor = (0.45, 0.5),
        bbox_transform=ax.transData, axes_class=WindroseAxes
    )
    
    windrose_ax.bar(
        wind_direction, wind_strength, normed=True,
        opening=1, edgecolor=None, nsector=36,
        bins=windbins,
        cmap=cm.get_cmap('viridis', len(windbins)),
        label='Wind velocity [$m \; s^{-1}$]',
        )
    
    windrose_legend = windrose_ax.legend(
        loc=(1.02, 0.15),
        decimal_places=0, ncol = 1,
        borderpad=0.1,
        labelspacing=0.5, handlelength=1.2, handletextpad = 0.6,
        fancybox=False,
        fontsize=8,
        frameon=False,
        title='Wind velocity [$m \; s^{-1}$]', title_fontsize = 8,
        labels=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b'],
    )
    windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)
    
    for lh in windrose_legend.legendHandles:
        lh.set_edgecolor(None)
    
    windrose_ax.tick_params(axis='x', which='major', pad=0)
    
    windrose_ax.set_yticks(np.arange(1, 8.1, step=1))
    windrose_ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, '8%'])
    
    ax.axis('off')
    
    # ax.text(0.1, 0.1, 'Mean wind direction: 79° + 180°')
    fig.subplots_adjust(left=0.01, right=0.8, bottom=0.2, top=0.8)
    fig.savefig(output_png,)




# fig.savefig('figures/00_test/trial1.png', dpi=300)

'''
(AWS_Dome_C['1h']['Dir'].values == 0).sum()
(AWS_Dome_C['1h']['Vel'].values == 0).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot time series

isite = 'EDC'

time = ERA5_daily_u10_2013_2022.u10.time.sel(time=slice('2014-12-25', '2015-01-16'))

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.1 time series of wind quivers at ' + isite + ' 2014-12-25 to 2015-01-16.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 6.6]) / 2.54)

for irow, idataset in zip(range(3), [expid[i], 'ERA5_daily', 'AWS_daily',]):
    # expid[i], 'ERA5_daily',
    # idataset = 'AWS_daily'
    print('#-------------------------------- ' + str(irow) + ' ' + idataset)
    
    if (idataset == expid[i]):
        wind_u = exp_org_o[expid[i]]['uv']['u'].sel(
            lev=47, time=slice('2014-12-25T23:52:30', '2015-01-16T23:52:30')).sel(
                lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
                lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
                method='nearest',
            ).values
        wind_v = exp_org_o[expid[i]]['uv']['v'].sel(
            lev=47, time=slice('2014-12-25T23:52:30', '2015-01-16T23:52:30')).sel(
                lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
                lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
                method='nearest',
            ).values
    elif (idataset == 'ERA5_daily'):
        wind_u = ERA5_daily_u10_2013_2022.u10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).sel(time=slice('2014-12-25', '2015-01-16')).values
        wind_v = ERA5_daily_v10_2013_2022.v10.sel(
            latitude=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
            longitude=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
            method='nearest',
        ).sel(time=slice('2014-12-25', '2015-01-16')).values
    elif (idataset == 'AWS_daily'):
        wind_u, wind_v = mpcalc.wind_components(AWS_Dome_C['1h']['Vel'].values * units('m/s'), AWS_Dome_C['1h']['Dir'].values * units.deg)
        AWS_Dome_C['1h']['wind_u'] = wind_u.magnitude
        AWS_Dome_C['1h']['wind_v'] = wind_v.magnitude
        AWS_Dome_C['1d'] = AWS_Dome_C['1h'].resample('1d', on='time').mean().reset_index()
        
        itimestart = np.where(AWS_Dome_C['1d']['time'] == '2014-12-25')[0][0]
        itimeend = np.where(AWS_Dome_C['1d']['time'] == '2015-01-16')[0][0]
        wind_u = AWS_Dome_C['1d'].iloc[itimestart:itimeend+1]['wind_u'].values
        wind_v = AWS_Dome_C['1d'].iloc[itimestart:itimeend+1]['wind_v'].values
    
    plt_quiver = ax.quiver(
        time, irow,
        wind_u, wind_v,
        units='height', scale=100, width=0.005,
        headwidth=3, headlength=5,
        )

ax.quiverkey(plt_quiver, X=0.2, Y=0.94, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E',
             labelsep = 0.1)

ax.set_xticks(time[::4])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date', labelpad=6)
plt.xticks(rotation=30, ha='right')

ax.set_ylim(-1, 3)
ax.set_yticklabels([])

plt.text(time[0].values - np.timedelta64(48, 'h'), 0, 'ECHAM6', ha='right', va='center')
plt.text(time[0].values - np.timedelta64(48, 'h'), 1, 'ERA5', ha='right', va='center')
plt.text(time[0].values - np.timedelta64(48, 'h'), 2, 'AWS', ha='right', va='center')

ax.grid(True, which='both',
        linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
fig.subplots_adjust(left=0.2, right=0.98, bottom=0.3, top=0.98)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check wind in forcing and output

isite = 'EDC'

data1 = exp_org_o[expid[i]]['uv']['u'].sel(
    lev=47, time=slice('2013-01-01T23:52:30', '2013-01-31T23:52:30')).sel(
        lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
        lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
        method='nearest',
        ).values

era5_uv_201301 = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/forcing/echam/era5_uv_201301.nc')

data2 = era5_uv_201301.u.sel(time=slice(20130101., 20130131.75)).sel(lev=47,).sel(
    lat=ten_sites_loc[ten_sites_loc.Site==isite].lat.values[0],
    lon=ten_sites_loc[ten_sites_loc.Site==isite].lon.values[0],
    method='nearest',
    ).values

data1 - (data2[0::4] + data2[1::4] + data2[2::4] + data2[3::4])/4

# endregion
# -----------------------------------------------------------------------------
