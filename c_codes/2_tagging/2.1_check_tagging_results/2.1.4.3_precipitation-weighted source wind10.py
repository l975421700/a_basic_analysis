

# =============================================================================
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

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_402_4.7',
    # 'pi_m_405_4.7',
    # 'pi_m_409_4.8',
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
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# =============================================================================


# =============================================================================

itag = 9
ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
# itag = 3
# ntags = [0, 0, 0, 18, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 0, 35, 0,   0, 0, 0, 0, 0,   7]


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
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source wind10 - scaling tagmap with wind10

minwind10 = {}
maxwind10 = {}
minwind10['pi_m_402_4.7'] = 0
maxwind10['pi_m_402_4.7'] = 28
i = 0
expid[i]

ocean_pre = {}
wind10_scaled_pre = {}
pre_weighted_wind10 = {}
ocean_pre_ann = {}
wind10_scaled_pre_ann = {}
pre_weighted_wind10_ann = {}
ocean_pre_sea = {}
wind10_scaled_pre_sea = {}
pre_weighted_wind10_sea = {}
ocean_pre_am = {}
wind10_scaled_pre_am = {}
pre_weighted_wind10_am = {}

ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype')
wind10_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=kstart+2) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=kstart+2))

#---------------- monthly values

pre_weighted_wind10[expid[i]] = wind10_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxwind10[expid[i]] - minwind10[expid[i]]) + minwind10[expid[i]]
pre_weighted_wind10[expid[i]].values[ocean_pre[expid[i]] < 1e-9] = np.nan
pre_weighted_wind10[expid[i]] = pre_weighted_wind10[expid[i]].rename('pre_weighted_wind10')
pre_weighted_wind10[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.nc')

#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
wind10_scaled_pre_ann[expid[i]] = wind10_scaled_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_wind10_ann[expid[i]] = wind10_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxwind10[expid[i]] - minwind10[expid[i]]) + minwind10[expid[i]]
pre_weighted_wind10_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_ann[expid[i]] = pre_weighted_wind10_ann[expid[i]].rename('pre_weighted_wind10_ann')
pre_weighted_wind10_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_ann.nc'
)


#---------------- seasonal values

# spin up: one year
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)
wind10_scaled_pre_sea[expid[i]] = wind10_scaled_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_wind10_sea[expid[i]] = wind10_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxwind10[expid[i]] - minwind10[expid[i]]) + minwind10[expid[i]]
pre_weighted_wind10_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_sea[expid[i]] = pre_weighted_wind10_sea[expid[i]].rename('pre_weighted_wind10_sea')
pre_weighted_wind10_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_sea.nc'
)

#---------------- annual mean values

# spin up: one year
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)
wind10_scaled_pre_am[expid[i]] = wind10_scaled_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_wind10_am[expid[i]] = wind10_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxwind10[expid[i]] - minwind10[expid[i]]) + minwind10[expid[i]]
pre_weighted_wind10_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_am[expid[i]] = pre_weighted_wind10_am[expid[i]].rename('pre_weighted_wind10_am')
pre_weighted_wind10_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_am.nc'
)



# endregion
# =============================================================================


# =============================================================================
# region calculate source wind10 - binning tagmap with wind10

i = 0
expid[i]

# wind10bins = np.concatenate((np.array([0]), np.arange(1, 16 + 1e-4, 1), np.array([100])))
# wind10bins_mid = np.concatenate((np.arange(0.5, 16 + 1e-4, 1), np.array([19])))
# wind10bins_mid = np.arange(0.5, 16.5 + 1e-4, 1)
wind10bins = np.concatenate((np.array([0]), np.arange(0.5, 16.5 + 1e-4, 0.5), np.array([100])))
wind10bins_mid = np.concatenate((np.arange(0.25, 16.25 + 1e-4, 0.5), np.array([19])))
# wind10bins_mid = np.arange(0.25, 16.75 + 1e-4, 0.5)


ocean_pre = {}
wind10_binned_pre = {}
pre_weighted_wind10 = {}
ocean_pre_ann = {}
wind10_binned_pre_ann = {}
pre_weighted_wind10_ann = {}
ocean_pre_sea = {}
wind10_binned_pre_sea = {}
pre_weighted_wind10_sea = {}
ocean_pre_am = {}
wind10_binned_pre_am = {}
pre_weighted_wind10_am = {}


wind10_binned_pre[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kend))
ocean_pre[expid[i]] = wind10_binned_pre[expid[i]].sum(dim='wisotype')

#---------------- monthly values

pre_weighted_wind10[expid[i]] = ( wind10_binned_pre[expid[i]] * wind10bins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre[expid[i]]
pre_weighted_wind10[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10[expid[i]] = pre_weighted_wind10[expid[i]].rename('pre_weighted_wind10')
pre_weighted_wind10[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.nc'
)

#---------------- annual values
wind10_binned_pre_ann[expid[i]] = wind10_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_wind10_ann[expid[i]] = ( wind10_binned_pre_ann[expid[i]] * wind10bins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_ann[expid[i]]
pre_weighted_wind10_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_ann[expid[i]] = pre_weighted_wind10_ann[expid[i]].rename('pre_weighted_wind10_ann')
pre_weighted_wind10_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_ann.nc'
)

#---------------- seasonal values
# spin up: one year

wind10_binned_pre_sea[expid[i]] = wind10_binned_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_wind10_sea[expid[i]] = ( wind10_binned_pre_sea[expid[i]] * wind10bins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_sea[expid[i]]
pre_weighted_wind10_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_sea[expid[i]] = pre_weighted_wind10_sea[expid[i]].rename('pre_weighted_wind10_sea')
pre_weighted_wind10_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_sea.nc'
)

#---------------- annual mean values
# spin up: one year

wind10_binned_pre_am[expid[i]] = wind10_binned_pre[expid[i]][12:].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_wind10_am[expid[i]] = ( wind10_binned_pre_am[expid[i]] * wind10bins_mid[:, None, None]).sum(dim='wisotype') / ocean_pre_am[expid[i]]
pre_weighted_wind10_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_wind10_am[expid[i]] = pre_weighted_wind10_am[expid[i]].rename('pre_weighted_wind10_am')
pre_weighted_wind10_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_am.nc'
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region cross check am/DJF/JJA source wind10 between scaling and binning tagmap

#-------- import data


pre_weighted_wind10 = {}
pre_weighted_wind10_ann = {}
pre_weighted_wind10_sea = {}
pre_weighted_wind10_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_wind10[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.nc')
    pre_weighted_wind10_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_ann.nc')
    pre_weighted_wind10_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_sea.nc')
    pre_weighted_wind10_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_am.nc')

#-------- basic set
i = 0
lon = pre_weighted_wind10[expid[i]].lon
lat = pre_weighted_wind10[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.3_wind10/' + '6.1.0.3.0.0_cross check am_DJF_JJA pre_weighted_wind10 from scaling and binning tagmap.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'

pltlevel = np.arange(4, 14 + 1e-4, 1)
pltticks = np.arange(4, 14 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-0.5, -0.05 + 1e-4, 0.05), np.arange(0.05, 0.5 + 1e-4, 0.05)))
pltticks2 = np.arange(-0.5, 0.5 + 1e-4, 0.1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

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
    lon, lat, pre_weighted_wind10_am[expid[0]].pre_weighted_wind10_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_wind10_am[expid[0]].pre_weighted_wind10_am - \
        pre_weighted_wind10_am[expid[1]].pre_weighted_wind10_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_wind10_am[expid[0]].pre_weighted_wind10_am - \
        pre_weighted_wind10_am[expid[2]].pre_weighted_wind10_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='DJF') - pre_weighted_wind10_sea[expid[1]].pre_weighted_wind10_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='DJF') - pre_weighted_wind10_sea[expid[2]].pre_weighted_wind10_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='JJA') - pre_weighted_wind10_sea[expid[1]].pre_weighted_wind10_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[0]].pre_weighted_wind10_sea.sel(season='JJA') - pre_weighted_wind10_sea[expid[2]].pre_weighted_wind10_sea.sel(season='JJA'),
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
    -0.05, 0.5, 'Scaling tag map with wind10', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $1 \; m \; s^{-1}$ wind10 bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $0.5 \; m \; s^{-1}$ wind10 bins', transform=axs[2, 0].transAxes,
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


# =============================================================================
# =============================================================================
# region plot am/DJF/JJA/DJF-JJA scaled tagmap with wind10


#-------- import data

pre_weighted_wind10 = {}
pre_weighted_wind10_ann = {}
pre_weighted_wind10_sea = {}
pre_weighted_wind10_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_wind10[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.nc')
    pre_weighted_wind10_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_ann.nc')
    pre_weighted_wind10_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_sea.nc')
    pre_weighted_wind10_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_am.nc')

#-------- basic set

i = 0

lon = pre_weighted_wind10[expid[i]].lon
lat = pre_weighted_wind10[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.3_wind10/' + '6.1.0.3.0.1_am_DJF_JJA pre_weighted_wind10 ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'

pltlevel = np.arange(4, 14 + 1e-4, 1)
pltticks = np.arange(4, 14 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-6, -1 + 1e-4, 1), np.arange(1, 6 + 1e-4, 1)))
pltticks2 = np.arange(-6, 6 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


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
    lon, lat, pre_weighted_wind10_am[expid[i]].pre_weighted_wind10_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='DJF') - pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='JJA'),
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


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region Antarctic plot Am, DJF, JJA scaled tagmap with [260, 310]

#-------- import data

pre_weighted_wind10 = {}
pre_weighted_wind10_ann = {}
pre_weighted_wind10_sea = {}
pre_weighted_wind10_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_wind10[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.nc')
    pre_weighted_wind10_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_ann.nc')
    pre_weighted_wind10_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_sea.nc')
    pre_weighted_wind10_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10_am.nc')


#-------- basic set

i = 0

lon = pre_weighted_wind10[expid[i]].lon
lat = pre_weighted_wind10[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.3_wind10/' + '6.1.0.3.0.2_Antarctica am_DJF_JJA pre_weighted_wind10 ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source wind10 [$m \; s^{-1}$]'

pltlevel = np.arange(8, 13 + 1e-4, 0.5)
pltticks = np.arange(8, 13 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-4, -0.5 + 1e-4, 0.5), np.arange(0.5, 4 + 1e-4, 0.5)))
pltticks2 = np.arange(-4, 4 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()

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
    lon, lat, pre_weighted_wind10_am[expid[i]].pre_weighted_wind10_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='DJF') - pre_weighted_wind10_sea[expid[i]].pre_weighted_wind10_sea.sel(season='JJA'),
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

# endregion
# =============================================================================





