

# =============================================================================
# region import packages


# basic library
import numpy as np
import xarray as xr
import datetime
import glob
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# plot
import matplotlib.path as mpath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
mpl.rcParams['figure.dpi'] = 600
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'data_source/bg_cartopy'
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from geopy import distance
from scipy import linalg
from scipy import stats
from sklearn import mixture
import metpy.calc as mpcalc
from metpy.units import units

# self defined function and namelist
from a00_basic_analysis.b_module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot1,
    hemisphere_plot,
)

from a00_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

import warnings
warnings.filterwarnings('ignore')

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate monthly 2m temperature in era5


era5_mon_sl_79_21_2mtem = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_2mtem.nc')

t2m = xr.concat((
    era5_mon_sl_79_21_2mtem.t2m[:-1, 0, :, :],
    era5_mon_sl_79_21_2mtem.t2m[-1:, 1, :, :]), dim='time')
# np.isnan(t2m).sum()

t2m_mon_average = mon_sea_ann_average(t2m, 'time.month')
# stats.describe(t2m, axis=None)

pltlevel = np.arange(210, 310.01, 0.5)
pltticks = np.arange(210, 310.01, 10)

pltlevel_nh = np.arange(210, 300.01, 0.5)
pltticks_nh = np.arange(210, 300.01, 10)

pltlevel_sh = np.arange(210, 300.01, 0.5)
pltticks_sh = np.arange(210, 300.01, 10)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot global monthly temperature in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values,
    t2m_mon_average.sel(month=1),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 (1979-2021)')
# ax.set_title(month[0], pad=5, size=10)
ax.text(
    0.5, 1.05, month[0], backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)
fig.savefig('figures/00_test/trial.png', dpi=1200)

# endregion
# =============================================================================


# =============================================================================
# region animate global monthly temperature in era5

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
ims = []
for i in range(12):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values,
        t2m_mon_average.sel(month=(i+1)),
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),
        )
    textinfo = ax.text(
        0.5, 1.05, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in the ERA5 reanalysis (1979-2021)')
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.00 monthly temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot NH monthly temperature in era5
fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values[0:241],
    t2m_mon_average.sel(month=1)[0:241, ],
    norm=BoundaryNorm(pltlevel_nh, ncolors=len(pltlevel_nh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_nh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_nh, extend='max')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 (1979-2021)')
ax.text(
    -0.1, 1, month[0], backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.savefig('figures/00_test/trial.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate NH monthly temperature in era5

fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)
ims = []
for i in range(12):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values[0:241],
        t2m_mon_average.sel(month=(i+1))[0:241, ],
        norm=BoundaryNorm(pltlevel_nh, ncolors=len(pltlevel_nh), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel_nh)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    textinfo = ax.text(
        -0.1, 1, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_nh, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 (1979-2021)')

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.01 NH monthly temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot SH monthly temperature in era5
fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values[480:],
    t2m_mon_average.sel(month=1)[480:, ],
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 (1979-2021)')
ax.text(
    -0.1, 1, month[0], backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.savefig('figures/00_test/trial.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate SH monthly temperature in era5

fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)
ims = []
for i in range(12):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values[480:],
        t2m_mon_average.sel(month=(i+1))[480:, ],
        norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    textinfo = ax.text(
        -0.1, 1, month[i], backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(12))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Monthly 2m temperature [K] in ERA5 (1979-2021)')

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.02 SH monthly temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region calculate annual temperature in era5

era5_mon_sl_79_21_2mtem = xr.open_dataset(
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_2mtem.nc')

t2m = xr.concat((
    era5_mon_sl_79_21_2mtem.t2m[:-1, 0, :, :],
    era5_mon_sl_79_21_2mtem.t2m[-1:, 1, :, :]), dim='time')

t2m_ann_average = mon_sea_ann_average(t2m, 'time.year')
# stats.describe(t2m_ann_average, axis =None)
pltlevel = np.arange(220, 300.01, 0.5)
pltticks = np.arange(220, 300.01, 10)

pltlevel_nh = np.arange(230, 290.01, 0.5)
pltticks_nh = np.arange(230, 290.01, 10)

pltlevel_sh = np.arange(220, 290.01, 0.5)
pltticks_sh = np.arange(220, 290.01, 10)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot global annual temperature in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values,
    t2m_ann_average.sel(year=1979),
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')
ax.text(
    0.5, 1.05, '1979', backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)
fig.savefig('figures/00_test/trial.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate global annual temperature in era5
fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
ims = []

for i in range(len(t2m_ann_average.year) - 1):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values,
        t2m_ann_average.sel(year=(i+1979)),
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    
    textinfo = ax.text(
        0.5, 1.05, t2m_ann_average.year[i].values, backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(t2m_ann_average.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.06,
    fraction=0.09, shrink=0.6, aspect=25, anchor=(0.5, -0.6),
    ticks=pltticks, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.995)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.03 Annual temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot NH annual temperature in era5
fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values[0:241],
    t2m_ann_average.sel(year=1979)[0:241, ],
    norm=BoundaryNorm(pltlevel_nh, ncolors=len(pltlevel_nh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_nh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_nh, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')
ax.text(
    -0.1, 1, '1979', backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.savefig('figures/00_test/trial.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate NH annual temperature in era5
fig, ax = hemisphere_plot(southextent=45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(len(t2m_ann_average.year) - 1):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values[0:241],
        t2m_ann_average.sel(year=(i+1979))[0:241, ],
        norm=BoundaryNorm(pltlevel_nh, ncolors=len(pltlevel_nh), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel_nh)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    
    textinfo = ax.text(
        -0.1, 1, t2m_ann_average.year[i].values, backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(t2m_ann_average.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_nh, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.04 NH Annual temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region plot SH annual temperature in era5
fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    era5_mon_sl_79_21_2mtem.longitude.values,
    era5_mon_sl_79_21_2mtem.latitude.values[480:],
    t2m_ann_average.sel(year=1979)[480:, ],
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')
ax.text(
    -0.1, 1, '1979', backgroundcolor='white',
    transform=ax.transAxes, fontweight='bold', ha='center', va='center',
)
fig.savefig('figures/00_test/trial.png', dpi=1200)

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region animate SH annual temperature in era5
fig, ax = hemisphere_plot(northextent=-45, sb_length=2000, sb_barheight=200,)
ims = []

for i in range(len(t2m_ann_average.year) - 1):  # range(2):  #
    # i=0
    plt_cmp = ax.pcolormesh(
        era5_mon_sl_79_21_2mtem.longitude.values,
        era5_mon_sl_79_21_2mtem.latitude.values[480:],
        t2m_ann_average.sel(year=(i+1979))[480:, ],
        norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
        cmap=cm.get_cmap('viridis', len(pltlevel_sh)), rasterized=True,
        transform=ccrs.PlateCarree(),
    )
    
    textinfo = ax.text(
        -0.1, 1, t2m_ann_average.year[i].values, backgroundcolor='white',
        transform=ax.transAxes, fontweight='bold', ha='center', va='center',
    )
    
    ims.append([plt_cmp, textinfo])
    print(str(i) + '/' + str(len(t2m_ann_average.year)))

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08,
    fraction=0.07, shrink=1, aspect=25, anchor=(0.5, 1), panchor=(0.5, 0),
    ticks=pltticks_sh, extend='both')
cbar.ax.set_xlabel(
    'Annual 2m temperature [K] in the ERA5 reanalysis')

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/02_era5/02_03_era5_tem/02.03.05 SH Annual temperature in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/02_era5/02_01_era5_pre/02.01.04 NH Annual precipitation in ERA5.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
# endregion
# =============================================================================

