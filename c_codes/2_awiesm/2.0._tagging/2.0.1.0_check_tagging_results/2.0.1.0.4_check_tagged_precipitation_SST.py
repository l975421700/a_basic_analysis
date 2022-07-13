

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
# region import output pi_echam6_1d*

exp_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    # 'pi_echam6_1y_204_3.60',
    # 'pi_echam6_1y_202_3.60',
    'pi_echam6_1y_221_3.70',
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
# region calculate source SST - scaling tagmap with SST

minsst = {}
maxsst = {}
minsst['pi_echam6_1y_204_3.60'] = 260
maxsst['pi_echam6_1y_204_3.60'] = 310
minsst['pi_echam6_1y_205_3.60'] = 200
maxsst['pi_echam6_1y_205_3.60'] = 400
minsst['pi_echam6_1y_206_3.60'] = 0
maxsst['pi_echam6_1y_206_3.60'] = 400
minsst['pi_echam6_1y_207_3.60'] = 0
maxsst['pi_echam6_1y_207_3.60'] = 600
minsst['pi_echam6_1y_210_3.60'] = 0
maxsst['pi_echam6_1y_210_3.60'] = 1000
minsst['pi_echam6_1y_211_3.60'] = 0
maxsst['pi_echam6_1y_211_3.60'] = 2000
i = 0
expid[i]

ocean_pre = {}
sst_scaled_pre = {}
pre_weighted_tsw = {}
ocean_pre_ann = {}
sst_scaled_pre_ann = {}
pre_weighted_tsw_ann = {}
ocean_pre_sea = {}
sst_scaled_pre_sea = {}
pre_weighted_tsw_sea = {}
ocean_pre_am = {}
sst_scaled_pre_am = {}
pre_weighted_tsw_am = {}

ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] +  exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :]).sum(axis=1)
sst_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4, :, :])

#---------------- monthly values

pre_weighted_tsw[expid[i]] = sst_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw[expid[i]].values[np.where(ocean_pre[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc'
)

#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
sst_scaled_pre_ann[expid[i]] = sst_scaled_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_tsw_ann[expid[i]] = sst_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw_ann[expid[i]].values[np.where(ocean_pre_ann[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)


#---------------- seasonal values

# spin up: one year
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
sst_scaled_pre_sea[expid[i]] = sst_scaled_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = sst_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)


#---------------- annual mean values

# spin up: one year
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
sst_scaled_pre_am[expid[i]] = sst_scaled_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = sst_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)



# endregion
# =============================================================================


# =============================================================================
# region plot source SST - scaling tagmap with SST



run_length = '12'
run_units = 'month'
time_step = 11 # for plot



plt_x = pre_weighted_tsw[expid[i]].lon
plt_y = pre_weighted_tsw[expid[i]].lat
plt_z = pre_weighted_tsw[expid[i]][time_step, :, :] - 273.15


#-------------------------------- global plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$] in ' + expid[i] + '\n' + 'Run length: ' + run_length + ' ' + run_units + '; Time period: ' + str(time_step + 1) + '; Scaling factors: ' + '[' + str(minsst[expid[i]]) + ', ' + str(maxsst[expid[i]]) + ']'
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.0_' + expid[i] + '_pre_weighted_tsw_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_' + str(minsst[expid[i]]) + '_' + str(maxsst[expid[i]]) + '_sum_pre.png'


fig, ax = globe_plot()
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)


#-------------------------------- Antarctic plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$]\n' + expid[i]
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.0_' + expid[i] + '_pre_weighted_tsw_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_' + str(minsst[expid[i]]) + '_' + str(maxsst[expid[i]]) + '_sum_pre_Antarctica.png'

fig, ax = hemisphere_plot(northextent=-45)
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)





'''
# ocean_pre[expid[i]] = (
#     exp_org_o[expid[i]]['echam'].aprl[:, :, :] + \
#         exp_org_o[expid[i]]['echam'].aprc[:, :, :]) - \
#             (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :] + \
#                 exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :])

# stats.describe(ocean_pre[expid[i]], axis=None)
# stats.describe(pre_weighted_tsw[expid[i]], axis=None, nan_policy='omit')
# pre_weighted_tsw[expid[i]].to_netcdf('/work/ollie/qigao001/0_backup/test.nc')

i = 0
minsst = 260
maxsst = 310

ocean_pre1 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
tsw1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre1 * (maxsst - minsst) + minsst
tsw1.values[np.where(ocean_pre1 < 1e-9)] = np.nan
stats.describe(tsw1, axis=None, nan_policy='omit') # 271.81658443 - 301.80649666


i=1
minsst = 0
maxsst = 400
ocean_pre2 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
tsw2 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre2 * (maxsst - minsst) + minsst
tsw2.values[np.where(ocean_pre2 < 1e-9)] = np.nan
stats.describe(tsw2, axis=None, nan_policy='omit') # 229.00815916 - 304.1026493
tsw2.to_netcdf('/work/ollie/qigao001/0_backup/tsw2.nc')
test = tsw1 - tsw2
stats.describe(test, axis=None, nan_policy='omit')
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')

np.where(test>50)
tsw1[94, 79] # 281.60002634
tsw2[94, 79] # 229.00815916
ocean_pre2[94, 79] # 6.69987505e-08
(exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, 94, 79] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, 94, 79]) # 3.83581513e-08

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw1.lon,
    tsw1.lat,
    tsw1 - 273.15,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source SST [$°C$]\n1 year simulation, last month, [260. 310]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '/work/ollie/qigao001/0_backup/trial.png')



pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw2.lon,
    tsw2.lat,
    tsw2 - 273.15,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source SST [$°C$]\n1 year simulation, last month, [0, 400]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '/work/ollie/qigao001/0_backup/trial2.png')
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source SST - 16 SST bins

i = 0
expid[i]

sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))
# sstbins_mid = np.arange(-1, 29.1, 2)
sstbins_mid = np.concatenate((np.array([271.38 - zerok]), np.arange(1, 29.1, 2)))
ocean_pre = {}
sst_binned_pre = {}
pre_weighted_tsw = {}
ocean_pre_ann = {}
sst_binned_pre_ann = {}
pre_weighted_tsw_ann = {}
ocean_pre_sea = {}
sst_binned_pre_sea = {}
pre_weighted_tsw_sea = {}
ocean_pre_am = {}
sst_binned_pre_am = {}
pre_weighted_tsw_am = {}


sst_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :])
ocean_pre[expid[i]] = sst_binned_pre[expid[i]].sum(axis=1)

#---------------- monthly values

pre_weighted_tsw[expid[i]] = ( sst_binned_pre[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre[expid[i]]
pre_weighted_tsw[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc'
)

#---------------- annual values
sst_binned_pre_ann[expid[i]] = sst_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_tsw_ann[expid[i]] = ( sst_binned_pre_ann[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_tsw_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)

#---------------- seasonal values
# spin up: one year

sst_binned_pre_sea[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = ( sst_binned_pre_sea[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)

'''
pre_weighted_tsw_sea[expid[i]][3, 46, 54]
(sst_binned_pre_sea[expid[i]][3, :, 46, 54] * sstbins_mid).sum()
ocean_pre_sea[expid[i]][3, 46, 54]

'''
#---------------- annual mean values

# spin up: one year

sst_binned_pre_am[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = ( sst_binned_pre_am[expid[i]] * sstbins_mid[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)

# endregion
# =============================================================================

# =============================================================================
# =============================================================================
# region plot source SST - SST bins



run_length = '12'
run_units = 'month'
time_step = 11 # for plot


plt_x = pre_weighted_tsw_bin[expid[i]].lon
plt_y = pre_weighted_tsw_bin[expid[i]].lat
plt_z = pre_weighted_tsw_bin[expid[i]][time_step, :, :]


#-------------------------------- global plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$] in ' + expid[i] + '\n' + 'Run length: ' + run_length + ' ' + run_units + '; Time period: ' + str(time_step + 1) + '; Bins: [-2, 30, 2]'
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.1_' + expid[i] + '_pre_weighted_tsw_bin_' + run_length + '_' + run_units + '_' + str(time_step+1) + '.png'


fig, ax = globe_plot()
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)


#-------------------------------- Antarctic plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$]\n' + expid[i]
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.1_' + expid[i] + '_pre_weighted_tsw_bin_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_Antarctica.png'

fig, ax = hemisphere_plot(northextent=-45)
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)





# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot am/DJF/JJA source SST

#-------- import data

pre_weighted_tsw = {}
pre_weighted_tsw_ann = {}
pre_weighted_tsw_sea = {}
pre_weighted_tsw_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')
    pre_weighted_tsw_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc')
    pre_weighted_tsw_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc')
    pre_weighted_tsw_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc')

# stats.describe(pre_weighted_tsw[expid[i]].pre_weighted_tsw, axis=None, nan_policy='omit')
# stats.describe(pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea, axis=None, nan_policy='omit')

#-------- basic set
i = 0
j = 1
lon = pre_weighted_tsw[expid[i]].lon
lat = pre_weighted_tsw[expid[i]].lat
print('#-------- ' + expid[i] + ' & '+ expid[j])
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_' + expid[i] + '_and_' + expid[j] + '_pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()
# pltcmp = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

pltlevel2 = np.arange(-1, 1.01, 0.125)
pltticks2 = np.arange(-1, 1.01, 0.25)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()
# pltcmp2 = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

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
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[j]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am - \
        pre_weighted_tsw_am[expid[j]].pre_weighted_tsw_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA') - pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='JJA'),
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
    -0.05, 0.5, 'Scaling tag map with SST', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Partitioning tag map based on SST', transform=axs[1, 0].transAxes,
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

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot 12th month source SST, multiple scaling factors


#-------- import data

pre_weighted_tsw = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')

# stats.describe(pre_weighted_tsw[expid[i]].pre_weighted_tsw, axis=None, nan_policy='omit')


#-------- basic set

lon = pre_weighted_tsw[expid[0]].lon
lat = pre_weighted_tsw[expid[0]].lat

print('#-------- Control: ' + expid[0])
print('#-------- Test: ' + expid[1] + ' & ' + expid[2] + ' & ' + expid[3] + ' & ' + expid[4])

mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
# time_step = 11
# output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_multiple_scaling_factors__pre_weighted_tsw_compare.png'
time_step = 5
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2.7_multiple_scaling_factors__pre_weighted_tsw_compare_6th_month.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-2, 2.01, 0.25)
pltticks2 = np.arange(-2, 2.01, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()


nrow = 3
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow!=0) | (jcol ==0)):
            axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)
        else:
            axs[irow, jcol].axis('off')


#-------- 12th month values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[1]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[2]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[3]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[4]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[time_step, :, :] - pre_weighted_tsw[expid[1]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[time_step, :, :] - pre_weighted_tsw[expid[2]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[time_step, :, :] - pre_weighted_tsw[expid[3]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[time_step, :, :] - pre_weighted_tsw[expid[4]].pre_weighted_tsw[time_step, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


plt.text(
    0.5, 1.05, 'Partitioning tag map based on SST',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [260, 310]',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [200, 400]',
    transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 400]',
    transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 600]',
    transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot 36th month source SST, two scaling factors


#-------- import data

pre_weighted_tsw = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')


#-------- basic set

lon = pre_weighted_tsw[expid[0]].lon
lat = pre_weighted_tsw[expid[0]].lat

print('#-------- Control: ' + expid[0] + ' & ' + expid[1])
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration

output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_' + expid[0] + '_and_' + expid[1] + '_pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-2, 2.01, 0.01)
pltticks2 = np.arange(-2, 2.01, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()

nrow = 1
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                                 add_grid_labels=False)


#-------- 12th month values
axs[0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[35, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[1]].pre_weighted_tsw[35, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[35, :, :] - pre_weighted_tsw[expid[1]].pre_weighted_tsw[35, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [260, 310]',
    transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 400]',
    transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Differences',
    transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.6),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot Am, DJF, JJA scaled tagmap with [260, 310]


#-------- import data

pre_weighted_tsw = {}
pre_weighted_tsw_ann = {}
pre_weighted_tsw_sea = {}
pre_weighted_tsw_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')
    # pre_weighted_tsw_ann[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc')
    # pre_weighted_tsw_sea[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc')
    # pre_weighted_tsw_am[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc')

#-------- basic set

i = 0

lon = pre_weighted_tsw[expid[i]].lon
lat = pre_weighted_tsw[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2.5_' + expid[i] + '_pre_weighted_tsw_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-10, 10.01, 1)
pltticks2 = np.arange(-10, 10.01, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()

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
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
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
test = pre_weighted_tsw['pi_echam6_1y_204_3.60'] - pre_weighted_tsw['pi_echam6_1y_202_3.60']
test.to_netcdf('scratch/test/test.nc')

test1 = pre_weighted_tsw['pi_echam6_1y_206_3.60'] - pre_weighted_tsw['pi_echam6_1y_202_3.60']
test1.to_netcdf('scratch/test/test1.nc')

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Antarctic plot Am, DJF, JJA scaled tagmap with [260, 310]

#-------- import data

pre_weighted_tsw = {}
pre_weighted_tsw_ann = {}
pre_weighted_tsw_sea = {}
pre_weighted_tsw_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')
    pre_weighted_tsw_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc')
    pre_weighted_tsw_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc')
    pre_weighted_tsw_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc')


#-------- basic set

i = 0

lon = pre_weighted_tsw[expid[i]].lon
lat = pre_weighted_tsw[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2.5_Antarctica_' + expid[i] + '_pre_weighted_tsw_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(2, 20.01, 1)
pltticks = np.arange(2, 20.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-8, 8.01, 1)
pltticks2 = np.arange(-8, 8.01, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()

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
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
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


fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.94)
fig.savefig(output_png)



stats.describe(pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'), axis=None, nan_policy='omit')

'''
fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot 12th month source SST, 2nd set of multiple scaling factors


#-------- import data

pre_weighted_tsw = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')

# stats.describe(pre_weighted_tsw[expid[i]].pre_weighted_tsw, axis=None, nan_policy='omit')


#-------- basic set

lon = pre_weighted_tsw[expid[0]].lon
lat = pre_weighted_tsw[expid[0]].lat

print('#-------- Control: ' + expid[0])
print('#-------- Test: ' + expid[1] + ' & ' + expid[2] + ' & ' + expid[3] + ' & ' + expid[4])

mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration

output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2.6_2nd_multiple_scaling_factors__pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-2, 2.01, 0.25)
pltticks2 = np.arange(-2, 2.01, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()


nrow = 3
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow!=0) | (jcol ==0)):
            axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)
        else:
            axs[irow, jcol].axis('off')


#-------- 12th month values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[11, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[1]].pre_weighted_tsw[11, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[2]].pre_weighted_tsw[11, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[3]].pre_weighted_tsw[11, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[4]].pre_weighted_tsw[11, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[11, :, :] - pre_weighted_tsw[expid[1]].pre_weighted_tsw[11, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[11, :, :] - pre_weighted_tsw[expid[2]].pre_weighted_tsw[11, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[11, :, :] - pre_weighted_tsw[expid[3]].pre_weighted_tsw[11, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[11, :, :] - pre_weighted_tsw[expid[4]].pre_weighted_tsw[11, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


plt.text(
    0.5, 1.05, 'Partitioning tag map based on SST',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [260, 310]',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 400]',
    transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 1000]',
    transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 2000]',
    transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source SST - 58 SST bins

i = 0
expid[i]

sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 0.5), np.array([100])))
# sstbins_mid = np.arange(-0.25, 28.251, 0.5)
sstbins_mid = np.concatenate((np.array([271.38 - zerok]), np.arange(0.25, 28.251, 0.5)))

ocean_pre = {}
sst_binned_pre = {}
pre_weighted_tsw = {}
ocean_pre_ann = {}
sst_binned_pre_ann = {}
pre_weighted_tsw_ann = {}
ocean_pre_sea = {}
sst_binned_pre_sea = {}
pre_weighted_tsw_sea = {}
ocean_pre_am = {}
sst_binned_pre_am = {}
pre_weighted_tsw_am = {}


sst_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :])
ocean_pre[expid[i]] = sst_binned_pre[expid[i]].sum(axis=1)

#---------------- monthly values

pre_weighted_tsw[expid[i]] = ( sst_binned_pre[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre[expid[i]]
pre_weighted_tsw[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc'
)

#---------------- annual values
sst_binned_pre_ann[expid[i]] = sst_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_tsw_ann[expid[i]] = ( sst_binned_pre_ann[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_tsw_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)

#---------------- seasonal values
# spin up: one year

sst_binned_pre_sea[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = ( sst_binned_pre_sea[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)

#---------------- annual mean values

# spin up: one year

sst_binned_pre_am[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = ( sst_binned_pre_am[expid[i]] * sstbins_mid[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region plot am/DJF/JJA source SST - 58 SST bins

#-------- import data

pre_weighted_tsw = {}
pre_weighted_tsw_ann = {}
pre_weighted_tsw_sea = {}
pre_weighted_tsw_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')
    pre_weighted_tsw_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc')
    pre_weighted_tsw_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc')
    pre_weighted_tsw_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc')

# stats.describe(pre_weighted_tsw[expid[i]].pre_weighted_tsw, axis=None, nan_policy='omit')
# stats.describe(pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea, axis=None, nan_policy='omit')

#-------- basic set
i = 0
lon = pre_weighted_tsw[expid[i]].lon
lat = pre_weighted_tsw[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2.8_pre_weighted_tsw_compare_different_bins_am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()
# pltcmp = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

pltlevel2 = np.arange(-1, 1.01, 0.125)
pltticks2 = np.arange(-1, 1.01, 0.25)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()
# pltcmp2 = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

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
    lon, lat, pre_weighted_tsw_am[expid[0]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[0]].pre_weighted_tsw_am - \
        pre_weighted_tsw_am[expid[1]].pre_weighted_tsw_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[0]].pre_weighted_tsw_am - \
        pre_weighted_tsw_am[expid[2]].pre_weighted_tsw_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[1]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[2]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='JJA') - pre_weighted_tsw_sea[expid[1]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[0]].pre_weighted_tsw_sea.sel(season='JJA') - pre_weighted_tsw_sea[expid[2]].pre_weighted_tsw_sea.sel(season='JJA'),
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
    -0.05, 0.5, 'Scaling tag map with SST', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Differences with partitioning tag\nmap with $2°C$ SST bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with $0.5°C$ SST bins', transform=axs[2, 0].transAxes,
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

# endregion
# =============================================================================

