

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
    'pi_m_412_4.9',
    'pi_m_413_4.10',
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

# itag = 6
# ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
itag = 13
# ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   0, 0, 19]
ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   0, 0, 37]

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
# region calculate source lon - scaling tagmap with lon

minlon = {}
maxlon = {}
minlon['pi_m_402_4.7'] = 0
maxlon['pi_m_402_4.7'] = 360
i = 0
expid[i]

ocean_pre = {}
lon_scaled_pre = {}
pre_weighted_lon = {}
ocean_pre_ann = {}
lon_scaled_pre_ann = {}
pre_weighted_lon_ann = {}
ocean_pre_sea = {}
lon_scaled_pre_sea = {}
pre_weighted_lon_sea = {}
ocean_pre_am = {}
lon_scaled_pre_am = {}
pre_weighted_lon_am = {}

ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype')
lon_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=kstart+2) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=kstart+2))

#---------------- monthly values

pre_weighted_lon[expid[i]] = lon_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (maxlon[expid[i]] - minlon[expid[i]]) + minlon[expid[i]]
pre_weighted_lon[expid[i]].values[ocean_pre[expid[i]] < 1e-9] = np.nan
pre_weighted_lon[expid[i]] = pre_weighted_lon[expid[i]].rename('pre_weighted_lon')
pre_weighted_lon[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')

#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lon_scaled_pre_ann[expid[i]] = lon_scaled_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lon_ann[expid[i]] = lon_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (maxlon[expid[i]] - minlon[expid[i]]) + minlon[expid[i]]
pre_weighted_lon_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_ann[expid[i]] = pre_weighted_lon_ann[expid[i]].rename('pre_weighted_lon_ann')
pre_weighted_lon_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc'
)


#---------------- seasonal values

# spin up: one year
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:36].groupby('time.season').sum(dim="time", skipna=True)
lon_scaled_pre_sea[expid[i]] = lon_scaled_pre[expid[i]][12:36].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lon_sea[expid[i]] = lon_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxlon[expid[i]] - minlon[expid[i]]) + minlon[expid[i]]
pre_weighted_lon_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].rename('pre_weighted_lon_sea')
pre_weighted_lon_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc'
)


#---------------- annual mean values

# spin up: one year
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:36].mean(dim="time", skipna=True)
lon_scaled_pre_am[expid[i]] = lon_scaled_pre[expid[i]][12:36].mean(dim="time", skipna=True)

pre_weighted_lon_am[expid[i]] = lon_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxlon[expid[i]] - minlon[expid[i]]) + minlon[expid[i]]
pre_weighted_lon_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_am[expid[i]] = pre_weighted_lon_am[expid[i]].rename('pre_weighted_lon_am')
pre_weighted_lon_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc'
)



# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate source lon - binning tagmap with lon

i = 0
expid[i]

# lonbins = np.concatenate((np.array([-1]), np.arange(20, 340+1e-4, 20), np.array([361])))
# lonbins_mid = np.arange(10, 350.1, 20)
lonbins = np.concatenate((np.array([-1]), np.arange(10, 350+1e-4, 10), np.array([361])))
lonbins_mid = np.arange(5, 355+1e-4, 10)

ocean_pre = {}
lon_binned_pre = {}
pre_weighted_lon = {}
ocean_pre_ann = {}
lon_binned_pre_ann = {}
pre_weighted_lon_ann = {}
ocean_pre_sea = {}
lon_binned_pre_sea = {}
pre_weighted_lon_sea = {}
ocean_pre_am = {}
lon_binned_pre_am = {}
pre_weighted_lon_am = {}

lon_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kend)))
ocean_pre[expid[i]] = lon_binned_pre[expid[i]].sum(dim='wisotype')


#---------------- monthly values

pre_weighted_lon[expid[i]] = (lon_binned_pre[expid[i]] * lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre[expid[i]]
pre_weighted_lon[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon[expid[i]] = pre_weighted_lon[expid[i]].rename('pre_weighted_lon')
pre_weighted_lon[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc'
)


#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lon_binned_pre_ann[expid[i]] = lon_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lon_ann[expid[i]] = (lon_binned_pre_ann[expid[i]] * lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_ann[expid[i]]
pre_weighted_lon_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_ann[expid[i]] = pre_weighted_lon_ann[expid[i]].rename('pre_weighted_lon_ann')
pre_weighted_lon_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)
lon_binned_pre_sea[expid[i]] = lon_binned_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lon_sea[expid[i]] = (lon_binned_pre_sea[expid[i]] * lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_sea[expid[i]]
pre_weighted_lon_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].rename('pre_weighted_lon_sea')
pre_weighted_lon_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)
lon_binned_pre_am[expid[i]] = lon_binned_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_lon_am[expid[i]] = (lon_binned_pre_am[expid[i]] * lonbins_mid[:, None, None]).sum(dim='wisotype') / ocean_pre_am[expid[i]]
pre_weighted_lon_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_am[expid[i]] = pre_weighted_lon_am[expid[i]].rename('pre_weighted_lon_am')
pre_weighted_lon_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc'
)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region cross check am/DJF/JJA source lon between scaling and binning tagmap

#-------- import data

pre_weighted_lon = {}
pre_weighted_lon_ann = {}
pre_weighted_lon_sea = {}
pre_weighted_lon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')
    pre_weighted_lon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc')
    pre_weighted_lon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc')
    pre_weighted_lon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')


#-------- basic set

lon = pre_weighted_lon[expid[0]].lon
lat = pre_weighted_lon[expid[0]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.4_lon/' + '6.1.0.4.0.0_cross check am_DJF_JJA pre_weighted_lon from scaling and binning tagmap.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'


pltlevel = np.arange(0, 360 + 1e-4, 5)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-5, -1 + 1e-4, 1), np.arange(1, 5 + 1e-4, 1)))
pltticks2 = np.arange(-5, 5 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

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
    lon, lat, pre_weighted_lon_am[expid[0]].pre_weighted_lon_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lon_am[expid[0]].pre_weighted_lon_am - pre_weighted_lon_am[expid[1]].pre_weighted_lon_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lon_am[expid[0]].pre_weighted_lon_am - pre_weighted_lon_am[expid[2]].pre_weighted_lon_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='DJF') - pre_weighted_lon_sea[expid[1]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='DJF') - pre_weighted_lon_sea[expid[2]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='JJA') - pre_weighted_lon_sea[expid[1]].pre_weighted_lon_sea.sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='JJA') - pre_weighted_lon_sea[expid[2]].pre_weighted_lon_sea.sel(season='JJA'),
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
    -0.05, 0.5, 'Scaling tag map with longitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $20°$ longitude bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $10°$ longitude bins', transform=axs[2, 0].transAxes,
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






