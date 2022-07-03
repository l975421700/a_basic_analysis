

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import output

exp_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    'pi_echam6_1y_208_3.60',
    'pi_echam6_1y_203_3.60',
    'pi_echam6_1y_212_3.60',
    # 'pi_echam6_1y_213_3.60',
    ]

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    ## echam
    exp_org_o[expid[i]]['echam'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source lat - scaling tagmap with lat

minlat = {}
maxlat = {}
minlat['pi_echam6_1y_208_3.60'] = -90
maxlat['pi_echam6_1y_208_3.60'] = 90

i = 0
expid[i]

ocean_pre = {}
lat_scaled_pre = {}
pre_weighted_lat = {}
ocean_pre_ann = {}
lat_scaled_pre_ann = {}
pre_weighted_lat_ann = {}
ocean_pre_sea = {}
lat_scaled_pre_sea = {}
pre_weighted_lat_sea = {}
ocean_pre_am = {}
lat_scaled_pre_am = {}
pre_weighted_lat_am = {}


ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] +  exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :]).sum(axis=1)
lat_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4, :, :])


#---------------- monthly values

pre_weighted_lat[expid[i]] = lat_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxlat[expid[i]] - minlat[expid[i]]) + minlat[expid[i]]
pre_weighted_lat[expid[i]].values[np.where(ocean_pre[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat[expid[i]] = pre_weighted_lat[expid[i]].rename('pre_weighted_lat')
pre_weighted_lat[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc'
)


#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lat_scaled_pre_ann[expid[i]] = lat_scaled_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lat_ann[expid[i]] = lat_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxlat[expid[i]] - minlat[expid[i]]) + minlat[expid[i]]
pre_weighted_lat_ann[expid[i]].values[np.where(ocean_pre_ann[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_ann[expid[i]] = pre_weighted_lat_ann[expid[i]].rename('pre_weighted_lat_ann')
pre_weighted_lat_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
lat_scaled_pre_sea[expid[i]] = lat_scaled_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lat_sea[expid[i]] = lat_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxlat[expid[i]] - minlat[expid[i]]) + minlat[expid[i]]
pre_weighted_lat_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_sea[expid[i]] = pre_weighted_lat_sea[expid[i]].rename('pre_weighted_lat_sea')
pre_weighted_lat_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
lat_scaled_pre_am[expid[i]] = lat_scaled_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_lat_am[expid[i]] = lat_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxlat[expid[i]] - minlat[expid[i]]) + minlat[expid[i]]
pre_weighted_lat_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_am[expid[i]] = pre_weighted_lat_am[expid[i]].rename('pre_weighted_lat_am')
pre_weighted_lat_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc'
)



# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source lat - lat bins

i = 0
expid[i]

latbins = np.arange(-90, 90.1, 10)
latbins_mid = np.arange(-85, 85.1, 10)

ocean_pre = {}
lat_binned_pre = {}
pre_weighted_lat = {}
ocean_pre_ann = {}
lat_binned_pre_ann = {}
pre_weighted_lat_ann = {}
ocean_pre_sea = {}
lat_binned_pre_sea = {}
pre_weighted_lat_sea = {}
ocean_pre_am = {}
lat_binned_pre_am = {}
pre_weighted_lat_am = {}

lat_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :])
ocean_pre[expid[i]] = lat_binned_pre[expid[i]].sum(axis=1)


#---------------- monthly values

pre_weighted_lat[expid[i]] = (lat_binned_pre[expid[i]] * latbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre[expid[i]]
pre_weighted_lat[expid[i]].values[np.where(ocean_pre[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat[expid[i]] = pre_weighted_lat[expid[i]].rename('pre_weighted_lat')
pre_weighted_lat[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc'
)


#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lat_binned_pre_ann[expid[i]] = lat_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lat_ann[expid[i]] = (lat_binned_pre_ann[expid[i]] * latbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_lat_ann[expid[i]].values[np.where(ocean_pre_ann[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_ann[expid[i]] = pre_weighted_lat_ann[expid[i]].rename('pre_weighted_lat_ann')
pre_weighted_lat_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
lat_binned_pre_sea[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lat_sea[expid[i]] = (lat_binned_pre_sea[expid[i]] * latbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_lat_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_sea[expid[i]] = pre_weighted_lat_sea[expid[i]].rename('pre_weighted_lat_sea')
pre_weighted_lat_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
lat_binned_pre_am[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)

pre_weighted_lat_am[expid[i]] = (lat_binned_pre_am[expid[i]] * latbins_mid[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_lat_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_am[expid[i]] = pre_weighted_lat_am[expid[i]].rename('pre_weighted_lat_am')
pre_weighted_lat_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc'
)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot am/DJF/JJA source lat

#-------- import data

pre_weighted_lat = {}
pre_weighted_lat_ann = {}
pre_weighted_lat_sea = {}
pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    pre_weighted_lat_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set
i = 0
j = 1
lon = pre_weighted_lat[expid[i]].lon
lat = pre_weighted_lat[expid[i]].lat
print('#-------- ' + expid[i] + ' & '+ expid[j])
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.0_' + expid[i] + '_and_' + expid[j] + '_pre_weighted_lat_compare.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'


pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-5, 5.01, 0.5)
pltticks2 = np.arange(-5, 5.01, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()


nrow = 3
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

#-------- annual mean values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[j]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am - \
        pre_weighted_lat_am[expid[j]].pre_weighted_lat_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA') - pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with latitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Partitioning tag map with latitude', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Differences', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot Am, DJF, JJA scaled tagmap with lat [-90, 90]

#-------- import data

pre_weighted_lat = {}
pre_weighted_lat_ann = {}
pre_weighted_lat_sea = {}
pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    pre_weighted_lat_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set

i = 0

lon = pre_weighted_lat[expid[i]].lon
lat = pre_weighted_lat[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.0_' + expid[i] + '_pre_weighted_lat_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'

pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-20, 20.01, 2.5)
pltticks2 = np.arange(-20, 20.01, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()

nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Am, DJF, JJA values
axs[0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)







'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Antarctic plot Am, DJF, JJA scaled tagmap with lat [-90, 90]

#-------- import data

pre_weighted_lat = {}
pre_weighted_lat_ann = {}
pre_weighted_lat_sea = {}
pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    pre_weighted_lat_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set

i = 0

lon = pre_weighted_lat[expid[i]].lon
lat = pre_weighted_lat[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.0_Antarctica_' + expid[i] + '_pre_weighted_lat_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'

pltlevel = np.arange(-60, -29.9, 2.5)
pltticks = np.arange(-60, -29.9, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-10, 10.01, 1)
pltticks2 = np.arange(-10, 10.01, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()

nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

#-------- Am, DJF, JJA values
axs[0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)






'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot lat

i = 0
minsst = -90
maxsst = 90
# ocean_pre1 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
ocean_pre1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4:, :, :]).sum(axis=0)

tsw1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre1 * (maxsst - minsst) + minsst
# stats.describe(ocean_pre1, axis=None)
# np.where(ocean_pre1 < 1e-15)
# ocean_pre1[43, 186]
tsw1.values[np.where(ocean_pre1 < 1e-9)] = np.nan
stats.describe(tsw1, axis=None, nan_policy='omit') # 271.81658443 - 301.80649666
tsw1.to_netcdf('/work/ollie/qigao001/0_backup/lat1.nc')


pltlevel = np.arange(-90, 90.01, 0.1)
pltticks = np.arange(-90, 90.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw1.lon,
    tsw1.lat,
    tsw1,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source lat [$°$]\n1 year simulation, last month, [0, 400]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '0_backup/trial.png')


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source lat - lat bins

i=0
expid[i]

lat = exp_org_o[expid[i]]['wiso'].lat

ocean_pre = {}
lat_binned_pre = {}
pre_weighted_lat = {}
ocean_pre_ann = {}
lat_binned_pre_ann = {}
pre_weighted_lat_ann = {}
ocean_pre_sea = {}
lat_binned_pre_sea = {}
pre_weighted_lat_sea = {}
ocean_pre_am = {}
lat_binned_pre_am = {}
pre_weighted_lat_am = {}


lat_binned_pre[expid[i]] = xr.concat([
    (exp_org_o['pi_echam6_1y_213_3.60']['wiso'].wisoaprl[:, 4:, :, :] + \
        exp_org_o['pi_echam6_1y_213_3.60']['wiso'].wisoaprc[:, 4:, :, :]),
    (exp_org_o['pi_echam6_1y_212_3.60']['wiso'].wisoaprl[:, 4:, :, :] + \
        exp_org_o['pi_echam6_1y_212_3.60']['wiso'].wisoaprc[:, 4:, :, :]),
    ], dim="wisotype")

ocean_pre[expid[i]] = lat_binned_pre[expid[i]].sum(axis=1)


#---------------- monthly values

pre_weighted_lat[expid[i]] = \
    (lat_binned_pre[expid[i]] * lat.values[None, :, None, None]
     ).sum(axis=1) / ocean_pre[expid[i]]

pre_weighted_lat[expid[i]].values[
    np.where(ocean_pre[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat[expid[i]] = \
    pre_weighted_lat[expid[i]].rename('pre_weighted_lat')

pre_weighted_lat[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')



#---------------- annual values
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lat_binned_pre_ann[expid[i]] = lat_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lat_ann[expid[i]] = (lat_binned_pre_ann[expid[i]] * lat.values[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_lat_ann[expid[i]].values[np.where(ocean_pre_ann[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_ann[expid[i]] = pre_weighted_lat_ann[expid[i]].rename('pre_weighted_lat_ann')

pre_weighted_lat_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
lat_binned_pre_sea[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lat_sea[expid[i]] = (lat_binned_pre_sea[expid[i]] * lat.values[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_lat_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_sea[expid[i]] = pre_weighted_lat_sea[expid[i]].rename('pre_weighted_lat_sea')
pre_weighted_lat_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
lat_binned_pre_am[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)

pre_weighted_lat_am[expid[i]] = (lat_binned_pre_am[expid[i]] * lat.values[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_lat_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_am[expid[i]] = pre_weighted_lat_am[expid[i]].rename('pre_weighted_lat_am')
pre_weighted_lat_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc'
)








'''

# 'pi_echam6_1y_213_3.60'
lat[:48]
lat[48:]

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot Jun/Sep/Dec source lat

#-------- import data

pre_weighted_lat = {}
# pre_weighted_lat_ann = {}
# pre_weighted_lat_sea = {}
# pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    # pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    # pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    # pre_weighted_lat_am[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set

lon = pre_weighted_lat[expid[0]].lon
lat = pre_weighted_lat[expid[1]].lat
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.1_pre_weighted_lat_compare_different_bins.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'

pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-5, 5.01, 0.5)
pltticks2 = np.arange(-5, 5.01, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()

nrow = 3
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)


#-------- scaled values J-S-D
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


#-------- Differences to 10 degree bin values J-S-D
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[5, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[8, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[11, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


#-------- Differences to 10 degree bin values J-S-D
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[5, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[8, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[11, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Jun', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Sep', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Dec', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with latitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Differences with partitioning tag\nmap with $10°$ latitude bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with each latitude', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot Am/DJF/JJA source lat

#-------- import data

pre_weighted_lat = {}
pre_weighted_lat_ann = {}
pre_weighted_lat_sea = {}
pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    pre_weighted_lat_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set

lon = pre_weighted_lat[expid[0]].lon
lat = pre_weighted_lat[expid[0]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.1_pre_weighted_lat_compare_different_bins_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'


pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-2.5, 2.51, 0.5)
pltticks2 = np.arange(-2.5, 2.51, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()

nrow = 3
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

#-------- scaled values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[0]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


#-------- Differences to 10 degree bin values
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[0]].pre_weighted_lat_am - pre_weighted_lat_am[expid[1]].pre_weighted_lat_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[1]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='JJA') - pre_weighted_lat_sea[expid[1]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


#-------- Differences to every latitude
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[0]].pre_weighted_lat_am - pre_weighted_lat_am[expid[2]].pre_weighted_lat_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[2]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[0]].pre_weighted_lat_sea.sel(season='JJA') - pre_weighted_lat_sea[expid[2]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with latitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Differences with partitioning tag\nmap with $10°$ latitude bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with each latitude', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''
'''
# endregion
# =============================================================================

