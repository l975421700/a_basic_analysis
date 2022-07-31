

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
)


from a_basic_analysis.b_module.namelist import (
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
    # 'pi_m_403_4.7',
    # 'pi_m_407_4.8',
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
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

itag = 7
ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
# itag = 1
# ntags = [0, 17, 0, 0, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 35, 0, 0, 0,   0, 0, 0, 0, 0,   7]

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
# region calculate source SST - scaling tagmap with SST


#---------------- basic settings

minsst = {}
maxsst = {}
minsst['pi_m_402_4.7'] = 268.15
maxsst['pi_m_402_4.7'] = 318.15

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

i = 0
expid[i]


#---------------- calculate precipitation

#---- monthly pre
ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype').compute()[120:]
sst_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=kstart+2) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=kstart+2)).compute()[120:]

sst_scaled_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8] = 0
ocean_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8] = 0

#---- annual pre
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True).compute()
sst_scaled_pre_ann[expid[i]] = sst_scaled_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True).compute()

#---- seasonal mean pre
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]].groupby('time.season').mean(dim="time", skipna=True).compute()
sst_scaled_pre_sea[expid[i]] = sst_scaled_pre[expid[i]].groupby('time.season').mean(dim="time", skipna=True).compute()

#---- annual mean pre
ocean_pre_am[expid[i]] = ocean_pre[expid[i]].mean(dim="time", skipna=True).compute()
sst_scaled_pre_am[expid[i]] = sst_scaled_pre[expid[i]].mean(dim="time", skipna=True).compute()

#---------------- monthly values

pre_weighted_tsw[expid[i]] = (sst_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok).compute()
pre_weighted_tsw[expid[i]].values[ocean_pre[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')

#---------------- annual values

pre_weighted_tsw_ann[expid[i]] = (sst_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok).compute()
pre_weighted_tsw_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)


#---------------- seasonal values

pre_weighted_tsw_sea[expid[i]] = (sst_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok).compute()
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)


#---------------- annual mean values


pre_weighted_tsw_am[expid[i]] = (sst_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok).compute()
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)


'''
np.max(sst_scaled_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8])
np.max(ocean_pre[expid[i]].values[ocean_pre[expid[i]] < 2e-8])
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source SST - binning tagmap with SST

i = 0
expid[i]

# sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))
# sstbins_mid = np.arange(-1, 29.1, 2)
sstbins = np.concatenate((np.array([-100]), np.arange(-1, 31.1, 1), np.array([100])))
sstbins_mid = np.arange(-1.5, 31.51, 1)


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


sst_binned_pre[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kend))
ocean_pre[expid[i]] = sst_binned_pre[expid[i]].sum(dim='wisotype')

#---------------- monthly values

pre_weighted_tsw[expid[i]] = ( sst_binned_pre[expid[i]] * sstbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre[expid[i]]
pre_weighted_tsw[expid[i]].values[ocean_pre[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc'
)

#---------------- annual values
sst_binned_pre_ann[expid[i]] = sst_binned_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').mean(dim="time", skipna=True)

pre_weighted_tsw_ann[expid[i]] = ( sst_binned_pre_ann[expid[i]] * sstbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_ann[expid[i]]
pre_weighted_tsw_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)

#---------------- seasonal values
# spin up: one year

sst_binned_pre_sea[expid[i]] = sst_binned_pre[expid[i]][12:].groupby('time.season').mean(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').mean(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = ( sst_binned_pre_sea[expid[i]] * sstbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_sea[expid[i]]
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)

#---------------- annual mean values
# spin up: one year

sst_binned_pre_am[expid[i]] = sst_binned_pre[expid[i]][12:].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = ( sst_binned_pre_am[expid[i]] * sstbins_mid[:, None, None]).sum(dim='wisotype') / ocean_pre_am[expid[i]]
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 2e-8] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region cross check am/DJF/JJA source SST between scaling and binning tagmap

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
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.3.0_cross check am_DJF_JJA pre_weighted_tsw from scaling and binning tagmap.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'

pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-0.5, -0.1 + 1e-4, 0.1), np.arange(0.1, 0.5 + 1e-4, 0.1)))
pltticks2 = np.arange(-0.5, 0.5 + 1e-4, 0.1, dtype=np.float64)
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
    -0.08, 0.5, 'Differences with partitioning\ntag map with $2°C$ SST bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with $1°C$ SST bins', transform=axs[2, 0].transAxes,
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
# region plot am/DJF/JJA/DJF-JJA scaled tagmap with SST [-5, 45]


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
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.3.1_am_DJF_JJA pre_weighted_tsw ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'


pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 1)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
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
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
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
# region Antarctic plot am/DJF/JJA/DJF-JJA scaled tagmap with SST [-5, 45]

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
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.3.2_Antarctica am_DJF_JJA pre_weighted_tsw ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'

pltlevel = np.arange(8, 20 + 1e-4, 1)
pltticks = np.arange(8, 20 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
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
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
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


# -----------------------------------------------------------------------------
# region check correlation coefficient between source SST and lat

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


pre_weighted_lat_am[expid[i]].pre_weighted_lat_am.sel(lat=slice(-60, -90))
pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am.sel(lat=slice(-60, -90))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------





