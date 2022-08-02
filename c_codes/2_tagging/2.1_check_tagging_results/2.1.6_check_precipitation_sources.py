

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_402_4.7',
    ]

# region import output

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

itag = 10
ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]

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
# region calculate precipitation source fractions: pi_geo_tagmap

# Antarctica, SH sea ice, other Land, other Ocean

i = 0
expid[i]

pre_alltagged_am  = {}
pre_Antarctica_am = {}
pre_SHseaice_am   = {}
pre_oland_am      = {}
pre_oocean_am     = {}

pre_Antarctica_am_frc = {}
pre_SHseaice_am_frc   = {}
pre_oland_am_frc      = {}
pre_oocean_am_frc     = {}

pre_alltagged_am[expid[i]]  = ((exp_org_o[expid[i]]['wiso'].wisoaprl[120:].sel(wisotype=slice(kstart+1, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc[120:].sel(wisotype=slice(kstart+1, kend))).sum(dim='wisotype')).mean(dim='time')

pre_Antarctica_am[expid[i]] = ((exp_org_o[expid[i]]['wiso'].wisoaprl[120:].sel(wisotype=kstart+3) + exp_org_o[expid[i]]['wiso'].wisoaprc[120:].sel(wisotype=kstart+3))).mean(dim='time')

pre_SHseaice_am[expid[i]]   = ((exp_org_o[expid[i]]['wiso'].wisoaprl[120:].sel(wisotype=kend) + exp_org_o[expid[i]]['wiso'].wisoaprc[120:].sel(wisotype=kend))).mean(dim='time')

pre_oland_am[expid[i]]      = ((exp_org_o[expid[i]]['wiso'].wisoaprl[120:].sel(wisotype=slice(kstart+1, kstart+2)) + exp_org_o[expid[i]]['wiso'].wisoaprc[120:].sel(wisotype=slice(kstart+1, kstart+2))).sum(dim='wisotype')).mean(dim='time')

pre_oocean_am[expid[i]]     = ((exp_org_o[expid[i]]['wiso'].wisoaprl[120:].sel(wisotype=slice(kstart+4, kstart+6)) + exp_org_o[expid[i]]['wiso'].wisoaprc[120:].sel(wisotype=slice(kstart+4, kstart+6))).sum(dim='wisotype')).mean(dim='time')


# np.max(abs(pre_alltagged_am[expid[i]].values - (pre_Antarctica_am[expid[i]].values + pre_SHseaice_am[expid[i]].values + pre_oland_am[expid[i]].values + pre_oocean_am[expid[i]].values)))
# np.min(pre_alltagged_am[expid[i]].values)


pre_Antarctica_am_frc[expid[i]] = pre_Antarctica_am[expid[i]] / pre_alltagged_am[expid[i]] * 100
pre_Antarctica_am_frc[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_Antarctica_am_frc.nc')

pre_SHseaice_am_frc[expid[i]] = pre_SHseaice_am[expid[i]] / pre_alltagged_am[expid[i]] * 100
pre_SHseaice_am_frc[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_SHseaice_am_frc.nc')

pre_oland_am_frc[expid[i]] = pre_oland_am[expid[i]] / pre_alltagged_am[expid[i]] * 100
pre_oland_am_frc[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_oland_am_frc.nc')

pre_oocean_am_frc[expid[i]] = pre_oocean_am[expid[i]] / pre_alltagged_am[expid[i]] * 100
pre_oocean_am_frc[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_oocean_am_frc.nc')




'''
test = pre_alltagged_am[expid[i]].values - (pre_Antarctica_am[expid[i]].values + pre_SHseaice_am[expid[i]].values + pre_oland_am[expid[i]].values + pre_oocean_am[expid[i]].values)
wheremax = np.where(abs(test) == np.max(abs(test)))
np.max(abs(test))
test[wheremax]
pre_alltagged_am[expid[i]].values[wheremax]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot precipitation source fractions pi_geo_tagmap

lon = exp_org_o[expid[i]]['wiso'].lon
lat = exp_org_o[expid[i]]['wiso'].lat

#-------- precipitation from Antarctica

pltlevel = np.arange(0, 12.01, 1)
pltticks = np.arange(0, 12.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, pre_Antarctica_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nAntarctica [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.1.0_Antarctica_' + expid[i] + '_precipitation_from_Antarctica.png')


#-------- SH sea ice

pltlevel = np.arange(0, 40.01, 5)
pltticks = np.arange(0, 40.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, pre_SHseaice_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nSH sea ice covered area [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.1.1_Antarctica_' + expid[i] + '_precipitation_from_SHseaice.png')


#-------- other Land

pltlevel = np.arange(0, 12.01, 1)
pltticks = np.arange(0, 12.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, pre_oland_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nland excl. Antarctica [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.1.2_Antarctica_' + expid[i] + '_precipitation_from_oland.png')


#-------- other Ocean

pltlevel = np.arange(50, 100.01, 5)
pltticks = np.arange(50, 100.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Greens', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, pre_oocean_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nocean excl. SH sea ice covered area [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.1.3_Antarctica_' + expid[i] + '_precipitation_from_oocean.png')



'''
np.max(abs((pre_Antarctica_am_frc[expid[i]] + pre_SHseaice_am_frc[expid[i]] + pre_oland_am_frc[expid[i]] + pre_oocean_am_frc[expid[i]]).values - 100))
'''

# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate precipitation source fractions: tagmap_echam6_t63_1_47.nc

i = 0
expid[i]

tagged_pre = {}
land_pre = {}
Antarctic_pre = {}
tagged_pre_am = {}
land_pre_am = {}
Antarctic_pre_am = {}

land_pre_am_frc = {}
Antarctic_pre_am_frc = {}

tagged_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[12:, 3:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[12:, 3:, :, :]).sum(axis=1)
land_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[12:, 41:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[12:, 41:, :, :]).sum(axis=1)
Antarctic_pre[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl[12:, 41, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[12:, 41, :, :]

tagged_pre_am[expid[i]] = tagged_pre[expid[i]].mean(dim='time')
land_pre_am[expid[i]] = land_pre[expid[i]].mean(dim='time')
Antarctic_pre_am[expid[i]] = Antarctic_pre[expid[i]].mean(dim='time')

land_pre_am_frc[expid[i]] = land_pre_am[expid[i]] / tagged_pre_am[expid[i]] * 100
Antarctic_pre_am_frc[expid[i]] = Antarctic_pre_am[expid[i]] / tagged_pre_am[expid[i]] * 100

'''
where_max = np.where(Antarctic_pre_am_frc[expid[i]] == np.max(Antarctic_pre_am_frc[expid[i]]))
Antarctic_pre_am_frc[expid[i]][where_max]
stats.describe(Antarctic_pre[expid[i]], axis=None)
stats.describe(land_pre_am_frc[expid[i]], axis=None)
stats.describe(Antarctic_pre_am_frc[expid[i]], axis=None)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot precipitation source fractions

lon = exp_org_o[expid[i]]['wiso'].lon
lat = exp_org_o[expid[i]]['wiso'].lat

pltlevel = np.arange(0, 80.01, 5)
pltticks = np.arange(0, 80.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

fig, ax = globe_plot()

plt_cmp = ax.pcolormesh(
    lon, lat, land_pre_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('Fraction of annual mean precipitation from land [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.0_' + expid[i] + '_precipitation_from_land.png')


pltlevel = np.arange(0, 12.01, 1)
pltticks = np.arange(0, 12.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, land_pre_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation\nfrom land [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.0.2_Antarctica_' + expid[i] + '_precipitation_from_land.png')


pltlevel = np.arange(0, 12.01, 1)
pltticks = np.arange(0, 12.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, Antarctic_pre_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation\nfrom Antarctica [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.0.3_Antarctica_' + expid[i] + '_precipitation_from_Antarctica.png')


pltlevel = np.arange(0, 12.01, 1)
pltticks = np.arange(0, 12.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

fig, ax = hemisphere_plot(northextent=-60)

plt_cmp = ax.pcolormesh(
    lon, lat, land_pre_am_frc[expid[i]] - Antarctic_pre_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation\nfrom land, excl. Antarctica [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.0.4_Antarctica_' + expid[i] + '_precipitation_from_land_excl_Antarctica.png')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate precipitation source fractions: tagmap_ls_0.nc


i = 0
expid[i]

tagged_pre = {}
land_pre = {}
tagged_pre_am = {}
land_pre_am = {}

land_pre_am_frc = {}

tagged_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[12:, 3:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[12:, 3:, :, :]).sum(axis=1)
land_pre[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl[12:, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[12:, 3, :, :]

tagged_pre_am[expid[i]] = tagged_pre[expid[i]].mean(dim='time')
land_pre_am[expid[i]] = land_pre[expid[i]].mean(dim='time')

land_pre_am_frc[expid[i]] = land_pre_am[expid[i]] / tagged_pre_am[expid[i]] * 100



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot precipitation source fractions

lon = exp_org_o[expid[i]]['wiso'].lon
lat = exp_org_o[expid[i]]['wiso'].lat

pltlevel = np.arange(0, 80.01, 5)
pltticks = np.arange(0, 80.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)).reversed()

fig, ax = globe_plot()

plt_cmp = ax.pcolormesh(
    lon, lat, land_pre_am_frc[expid[i]],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('Fraction of annual mean precipitation from land [%]', linespacing=2)
fig.savefig('figures/6_awi/6.1_echam6/6.1.2_precipitation_sources/6.1.2.0.0.1_' + expid[i] + '_precipitation_from_land.png')


# endregion
# -----------------------------------------------------------------------------




