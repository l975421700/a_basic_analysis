

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

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'

expid = [
    'pi_echam6_1y_209_3.60',
    # 'pi_echam6_1y_204_3.60'
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
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
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
# =============================================================================
