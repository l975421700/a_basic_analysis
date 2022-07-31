

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

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

from a_basic_analysis.b_module.source_properties import (
    source_properties,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_402_4.7',
    # 'pi_m_404_4.7',
    # 'pi_m_408_4.8',
    ]

# region import output

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset( exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

itag = 8
ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
# itag = 2
# ntags = [0, 0, 13, 0, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 31, 0, 0,   0, 0, 0, 0, 0,   7]

# region set indices for specific set of tracers

kwiso2 = 3

if (itag == 0):
    kstart = kwiso2 + 0
    kend   = kwiso2 + ntags[0]
else:
    kstart = kwiso2 + sum(ntags[:itag])
    kend   = kwiso2 + sum(ntags[:(itag+1)])

print(kstart); print(kend)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate source rh2m - scaling tagmap with rh2m

minrh2m = {}
maxrh2m = {}
minrh2m['pi_m_402_4.7'] = 0
maxrh2m['pi_m_402_4.7'] = 1.6

ocean_pre = {}
rh2m_scaled_pre = {}
pre_weighted_rh2m = {}
ocean_pre_ann = {}
rh2m_scaled_pre_ann = {}
pre_weighted_rh2m_ann = {}
ocean_pre_sea = {}
rh2m_scaled_pre_sea = {}
pre_weighted_rh2m_sea = {}
ocean_pre_am = {}
rh2m_scaled_pre_am = {}
pre_weighted_rh2m_am = {}

i = 0
expid[i]


#---------------- calculate precipitation

#---- monthly pre
ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype').compute()[120:]
rh2m_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=kstart+2) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=kstart+2)).compute()[120:]

rh2m_scaled_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8] = 0
ocean_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8] = 0

#---- annual pre
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True).compute()
rh2m_scaled_pre_ann[expid[i]] = rh2m_scaled_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True).compute()

#---- seasonal mean pre
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]].groupby('time.season').mean(dim="time", skipna=True).compute()
rh2m_scaled_pre_sea[expid[i]] = rh2m_scaled_pre[expid[i]].groupby('time.season').mean(dim="time", skipna=True).compute()

#---- annual mean pre
ocean_pre_am[expid[i]] = ocean_pre[expid[i]].mean(dim="time", skipna=True).compute()
rh2m_scaled_pre_am[expid[i]] = rh2m_scaled_pre[expid[i]].mean(dim="time", skipna=True).compute()

#---------------- monthly values

pre_weighted_rh2m[expid[i]] = (rh2m_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxrh2m[expid[i]] - minrh2m[expid[i]]) + minrh2m[expid[i]]).compute()
pre_weighted_rh2m[expid[i]].values[ocean_pre[expid[i]].values < 2e-8] = np.nan
pre_weighted_rh2m[expid[i]] = pre_weighted_rh2m[expid[i]].rename('pre_weighted_rh2m')
pre_weighted_rh2m[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.nc')

#---------------- annual values

pre_weighted_rh2m_ann[expid[i]] = (rh2m_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxrh2m[expid[i]] - minrh2m[expid[i]]) + minrh2m[expid[i]]).compute()
pre_weighted_rh2m_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 2e-8] = np.nan
pre_weighted_rh2m_ann[expid[i]] = pre_weighted_rh2m_ann[expid[i]].rename('pre_weighted_rh2m_ann')
pre_weighted_rh2m_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_ann.nc'
)

#---------------- seasonal values

pre_weighted_rh2m_sea[expid[i]] = (rh2m_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxrh2m[expid[i]] - minrh2m[expid[i]]) + minrh2m[expid[i]]).compute()
pre_weighted_rh2m_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 2e-8] = np.nan
pre_weighted_rh2m_sea[expid[i]] = pre_weighted_rh2m_sea[expid[i]].rename('pre_weighted_rh2m_sea')
pre_weighted_rh2m_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_sea.nc'
)

#---------------- annual mean values

pre_weighted_rh2m_am[expid[i]] = (rh2m_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxrh2m[expid[i]] - minrh2m[expid[i]]) + minrh2m[expid[i]]).compute()
pre_weighted_rh2m_am[expid[i]].values[ocean_pre_am[expid[i]].values < 2e-8] = np.nan
pre_weighted_rh2m_am[expid[i]] = pre_weighted_rh2m_am[expid[i]].rename('pre_weighted_rh2m_am')
pre_weighted_rh2m_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_am.nc'
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source rh2m - binning tagmap with rh2m

i = 0
expid[i]

# rh2mbins = np.concatenate((np.array([0]), np.arange(0.55, 1.05 + 1e-4, 0.05), np.array([2])))
# rh2mbins_mid = np.arange(0.525, 1.075 + 1e-4, 0.05)
rh2mbins = np.concatenate((np.array([0]), np.arange(0.52, 1.08 + 1e-4, 0.02), np.array([2])))
rh2mbins_mid = np.arange(0.51, 1.09 + 1e-4, 0.02)


ocean_pre = {}
rh2m_binned_pre = {}
pre_weighted_rh2m = {}
ocean_pre_ann = {}
rh2m_binned_pre_ann = {}
pre_weighted_rh2m_ann = {}
ocean_pre_sea = {}
rh2m_binned_pre_sea = {}
pre_weighted_rh2m_sea = {}
ocean_pre_am = {}
rh2m_binned_pre_am = {}
pre_weighted_rh2m_am = {}


rh2m_binned_pre[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kend))
ocean_pre[expid[i]] = rh2m_binned_pre[expid[i]].sum(dim='wisotype')

#---------------- monthly values

pre_weighted_rh2m[expid[i]] = ( rh2m_binned_pre[expid[i]] * rh2mbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre[expid[i]]
pre_weighted_rh2m[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_rh2m[expid[i]] = pre_weighted_rh2m[expid[i]].rename('pre_weighted_rh2m')
pre_weighted_rh2m[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.nc'
)

#---------------- annual values
rh2m_binned_pre_ann[expid[i]] = rh2m_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_rh2m_ann[expid[i]] = ( rh2m_binned_pre_ann[expid[i]] * rh2mbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_ann[expid[i]]
pre_weighted_rh2m_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_rh2m_ann[expid[i]] = pre_weighted_rh2m_ann[expid[i]].rename('pre_weighted_rh2m_ann')
pre_weighted_rh2m_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_ann.nc'
)

#---------------- seasonal values
# spin up: one year

rh2m_binned_pre_sea[expid[i]] = rh2m_binned_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_rh2m_sea[expid[i]] = ( rh2m_binned_pre_sea[expid[i]] * rh2mbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_sea[expid[i]]
pre_weighted_rh2m_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_rh2m_sea[expid[i]] = pre_weighted_rh2m_sea[expid[i]].rename('pre_weighted_rh2m_sea')
pre_weighted_rh2m_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_sea.nc'
)

#---------------- annual mean values
# spin up: one year

rh2m_binned_pre_am[expid[i]] = rh2m_binned_pre[expid[i]][12:].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_rh2m_am[expid[i]] = ( rh2m_binned_pre_am[expid[i]] * rh2mbins_mid[:, None, None]).sum(dim='wisotype') / ocean_pre_am[expid[i]]
pre_weighted_rh2m_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_rh2m_am[expid[i]] = pre_weighted_rh2m_am[expid[i]].rename('pre_weighted_rh2m_am')
pre_weighted_rh2m_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_am.nc'
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region cross check am/DJF/JJA source rh2m between scaling and binning tagmap

#-------- import data


pre_weighted_rh2m = {}
pre_weighted_rh2m_ann = {}
pre_weighted_rh2m_sea = {}
pre_weighted_rh2m_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_rh2m[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.nc')
    pre_weighted_rh2m_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_ann.nc')
    pre_weighted_rh2m_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_sea.nc')
    pre_weighted_rh2m_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_am.nc')

#-------- basic set
i = 0
lon = pre_weighted_rh2m[expid[i]].lon
lat = pre_weighted_rh2m[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.2_rh2m/' + '6.1.0.2.0.0_cross check am_DJF_JJA pre_weighted_rh2m from scaling and binning tagmap.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source rh2m [$\%$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source rh2m [$\%$]'

pltlevel = np.arange(60, 90 + 1e-4, 1.5)
pltticks = np.arange(60, 90 + 1e-4, 3)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-2.5, -0.25 + 1e-4, 0.25), np.arange(0.25, 2.5 + 1e-4, 0.25)))
pltticks2 = np.arange(-2.5, 2.5 + 1e-4, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=False)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1).reversed()

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
    lon, lat, pre_weighted_rh2m_am[expid[0]].pre_weighted_rh2m_am * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, (pre_weighted_rh2m_am[expid[0]].pre_weighted_rh2m_am - \
        pre_weighted_rh2m_am[expid[1]].pre_weighted_rh2m_am) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, (pre_weighted_rh2m_am[expid[0]].pre_weighted_rh2m_am - \
        pre_weighted_rh2m_am[expid[2]].pre_weighted_rh2m_am) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='DJF') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='DJF') - pre_weighted_rh2m_sea[expid[1]].pre_weighted_rh2m_sea.sel(season='DJF')) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='DJF') - pre_weighted_rh2m_sea[expid[2]].pre_weighted_rh2m_sea.sel(season='DJF')) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='JJA') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='JJA') - pre_weighted_rh2m_sea[expid[1]].pre_weighted_rh2m_sea.sel(season='JJA')) * 100,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[0]].pre_weighted_rh2m_sea.sel(season='JJA') - pre_weighted_rh2m_sea[expid[2]].pre_weighted_rh2m_sea.sel(season='JJA')) * 100,
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
    -0.05, 0.5, 'Scaling tag map with rh2m', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with $5\%$ rh2m bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with $2\%$ rh2m bins', transform=axs[2, 0].transAxes,
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA scaled tagmap with rh2m


#-------- import data

pre_weighted_rh2m = {}
pre_weighted_rh2m_ann = {}
pre_weighted_rh2m_sea = {}
pre_weighted_rh2m_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_rh2m[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.nc')
    pre_weighted_rh2m_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_ann.nc')
    pre_weighted_rh2m_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_sea.nc')
    pre_weighted_rh2m_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_am.nc')

#-------- basic set

i = 0

lon = pre_weighted_rh2m[expid[i]].lon
lat = pre_weighted_rh2m[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.2_rh2m/' + '6.1.0.2.0.1_am_DJF_JJA pre_weighted_rh2m ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source rh2m [$\%$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source rh2m [$\%$]'


pltlevel = np.arange(60, 90 + 1e-4, 1.5)
pltticks = np.arange(60, 90 + 1e-4, 3)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-20, 20 + 1e-4, 2)
pltticks2 = np.arange(-20, 20 + 1e-4, 4)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1).reversed()

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
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_rh2m_am[expid[i]].pre_weighted_rh2m_am * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='DJF') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='JJA') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='DJF') - pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='JJA')) * 100,
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
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region Antarctic plot am/DJF/JJA/DJF-JJA scaled tagmap with rh2m

#-------- import data

pre_weighted_rh2m = {}
pre_weighted_rh2m_ann = {}
pre_weighted_rh2m_sea = {}
pre_weighted_rh2m_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_rh2m[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.nc')
    pre_weighted_rh2m_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_ann.nc')
    pre_weighted_rh2m_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_sea.nc')
    pre_weighted_rh2m_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m_am.nc')


#-------- basic set

i = 0

lon = pre_weighted_rh2m[expid[i]].lon
lat = pre_weighted_rh2m[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.2_rh2m/' + '6.1.0.2.0.2_Antarctica am_DJF_JJA pre_weighted_rh2m ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source rh2m [$\%$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source rh2m [$\%$]'

pltlevel = np.arange(73, 85 + 1e-4, 1)
pltticks = np.arange(73, 85 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 1)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('bwr', len(pltlevel2)-1).reversed()


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
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_rh2m_am[expid[i]].pre_weighted_rh2m_am * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='DJF') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='JJA') * 100,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, (pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='DJF') - pre_weighted_rh2m_sea[expid[i]].pre_weighted_rh2m_sea.sel(season='JJA')) * 100,
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
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)


fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.94)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------





