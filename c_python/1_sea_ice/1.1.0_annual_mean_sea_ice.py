

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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})


# self defined
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
)

from a00_basic_analysis.b_module.b0_cmip6.b0_0_data_download import (
    esgf_search,
)


# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc, HadGEM3-GC31-LL, piControl, r1i1p1f1

siconc_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))

siconc_hg3_ll_pi = xr.open_mfdataset(
    siconc_hg3_ll_pi_fl[0:5], data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_hg3_ll_pi = siconc_hg3_ll_pi.siconc.mean(axis=0).values
# np.isnan(am_siconc_hg3_ll_pi).sum()

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

plt_cmp = ax.pcolormesh(
    siconc_hg3_ll_pi.longitude,
    siconc_hg3_ll_pi.latitude,
    am_siconc_hg3_ll_pi,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "500 years annual mean sea ice area fraction [%]\nHadGEM3-GC31-LL, piControl, r1i1p1f1",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.0 annual mean siconc in HadGEM3-GC31-LL piControl r1i1p1f1.png')

'''
# check


# files from vittoria
dir_vittoria = '/gws/nopw/j04/pmip4_vol1/users/vittoria/'
fl_si = np.array(sorted(
    glob.glob(dir_vittoria + 'seaice_annual_uba937_' + '*.nc')
))
fs_si = xr.open_mfdataset(
    fl_si, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')
seaicearea = fs_si.aice[0, :, :].copy().values
seaicearea[np.where(seaicearea < 0)] = np.nan
seaicearea[np.where(seaicearea > 1)] = np.nan

fig, ax = hemisphere_plot(southextent=30, sb_length=2000, sb_barheight=200,)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

plt_cmp = ax.pcolormesh(
    fs_si.aice.lon, fs_si.aice.lat, seaicearea,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Sea ice area")

fig.savefig('figures/00_test/trial')

# ax.add_feature(cfeature.LAND, zorder=3)
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc, HadGEM3-GC31-LL, historical, r1i1p1f3

siconc_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc',
)))


siconc_hg3_ll_hi_r1 = xr.open_mfdataset(
    siconc_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_hg3_ll_hi_r1 = siconc_hg3_ll_hi_r1.siconc.mean(axis=0).values
# np.isnan(am_siconc_hg3_ll_hi_r1).sum()

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

plt_cmp = ax.pcolormesh(
    siconc_hg3_ll_hi_r1.longitude,
    siconc_hg3_ll_hi_r1.latitude,
    am_siconc_hg3_ll_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "165 years (1850-2014) annual mean sea ice area fraction [%]\nHadGEM3-GC31-LL, historical, r1i1p1f3",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.1 annual mean siconc in HadGEM3-GC31-LL historical r1i1p1f3.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc difference, HadGEM3-GC31-LL, (historical, r1i1p1f3) - (piControl, r1i1p1f1)

siconc_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))
siconc_hg3_ll_pi = xr.open_mfdataset(
    siconc_hg3_ll_pi_fl[0:5], data_vars='minimal',
    coords='minimal', compat='override',
)
am_siconc_hg3_ll_pi = siconc_hg3_ll_pi.siconc.mean(axis=0).values

siconc_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc',
)))
siconc_hg3_ll_hi_r1 = xr.open_mfdataset(
    siconc_hg3_ll_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
am_siconc_hg3_ll_hi_r1 = siconc_hg3_ll_hi_r1.siconc.mean(axis=0).values

dif_am_siconc_hg3_ll_hi_r1_pi = am_siconc_hg3_ll_hi_r1 - am_siconc_hg3_ll_pi
# stats.describe(am_siconc_hg3_ll_pi, axis=None, nan_policy='omit')
# stats.describe(am_siconc_hg3_ll_hi_r1, axis=None, nan_policy='omit')
# stats.describe(dif_am_siconc_hg3_ll_hi_r1_pi, axis=None, nan_policy='omit')

pltlevel = np.arange(-15, 15.01, 0.2)
pltticks = np.arange(-15, 15.01, 5)

from matplotlib.colors import ListedColormap
cmp_top = cm.get_cmap('Blues_r', int(np.floor(len(pltlevel) / 2)))
cmp_bottom = cm.get_cmap('Reds', int(np.floor(len(pltlevel) / 2)))
cmp_colors = np.vstack(
    (cmp_top(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2)))),
     [1, 1, 1, 1],
     cmp_bottom(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2))))))
cmp_cmap = ListedColormap(cmp_colors, name='RedsBlues_r')


fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

plt_cmp = ax.pcolormesh(
    siconc_hg3_ll_pi.longitude,
    siconc_hg3_ll_pi.latitude,
    dif_am_siconc_hg3_ll_hi_r1_pi,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Annual mean sea ice area fraction [%], HadGEM3-GC31-LL\n(historical, r1i1p1f3, 165 y) - (piControl, r1i1p1f1, 500 y)",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.2 annual mean siconc difference in HadGEM3-GC31-LL historical r1i1p1f3_piControl r1i1p1f1.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc, HadGEM3-GC31-MM, piControl, r1i1p1f1


siconc_hg3_mm_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_*.nc',
)))

siconc_hg3_mm_pi = xr.open_mfdataset(
    siconc_hg3_mm_pi_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_hg3_mm_pi = siconc_hg3_mm_pi.siconc.mean(axis=0).values
# np.isnan(am_siconc_hg3_mm_pi).sum()

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

plt_cmp = ax.pcolormesh(
    siconc_hg3_mm_pi.longitude,
    siconc_hg3_mm_pi.latitude,
    am_siconc_hg3_mm_pi,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "500 years annual mean sea ice area fraction [%]\nHadGEM3-GC31-MM, piControl, r1i1p1f1",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.3 annual mean siconc in HadGEM3-GC31-MM piControl r1i1p1f1.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc, HadGEM3-GC31-MM, historical, r1i1p1f3

siconc_hg3_mm_hi_r1_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r1i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-MM_historical_r1i1p1f3_gn_*.nc',
)))
siconc_hg3_mm_hi_r1 = xr.open_mfdataset(
    siconc_hg3_mm_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_hg3_mm_hi_r1 = siconc_hg3_mm_hi_r1.siconc.mean(axis=0).values
# np.isnan(am_siconc_hg3_mm_hi_r1).sum()

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

plt_cmp = ax.pcolormesh(
    siconc_hg3_mm_hi_r1.longitude,
    siconc_hg3_mm_hi_r1.latitude,
    am_siconc_hg3_mm_hi_r1,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "165 years (1850-2014) annual mean sea ice area fraction [%]\nHadGEM3-GC31-MM, historical, r1i1p1f3",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.4 annual mean siconc in HadGEM3-GC31-MM historical r1i1p1f3.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc difference, HadGEM3-GC31-MM, (historical, r1i1p1f3) - (piControl, r1i1p1f1)

siconc_hg3_mm_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_*.nc',
)))
siconc_hg3_mm_pi = xr.open_mfdataset(
    siconc_hg3_mm_pi_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
am_siconc_hg3_mm_pi = siconc_hg3_mm_pi.siconc.mean(axis=0).values


siconc_hg3_mm_hi_r1_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r1i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-MM_historical_r1i1p1f3_gn_*.nc',
)))
siconc_hg3_mm_hi_r1 = xr.open_mfdataset(
    siconc_hg3_mm_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)
am_siconc_hg3_mm_hi_r1 = siconc_hg3_mm_hi_r1.siconc.mean(axis=0).values


dif_am_siconc_hg3_mm_hi_r1_pi = am_siconc_hg3_mm_hi_r1 - am_siconc_hg3_mm_pi
# stats.describe(am_siconc_hg3_mm_pi, axis=None, nan_policy='omit')
# stats.describe(am_siconc_hg3_mm_hi_r1, axis=None, nan_policy='omit')
# stats.describe(dif_am_siconc_hg3_mm_hi_r1_pi, axis=None, nan_policy='omit')

pltlevel = np.arange(-15, 15.01, 0.2)
pltticks = np.arange(-15, 15.01, 5)

from matplotlib.colors import ListedColormap
cmp_top = cm.get_cmap('Blues_r', int(np.floor(len(pltlevel) / 2)))
cmp_bottom = cm.get_cmap('Reds', int(np.floor(len(pltlevel) / 2)))
cmp_colors = np.vstack(
    (cmp_top(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2)))),
     [1, 1, 1, 1],
     cmp_bottom(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2))))))
cmp_cmap = ListedColormap(cmp_colors, name='RedsBlues_r')


fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

plt_cmp = ax.pcolormesh(
    siconc_hg3_mm_pi.longitude,
    siconc_hg3_mm_pi.latitude,
    dif_am_siconc_hg3_mm_hi_r1_pi,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cmp_cmap.reversed(), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "Annual mean sea ice area fraction [%], HadGEM3-GC31-MM\n(historical, r1i1p1f3, 165 y) - (piControl, r1i1p1f1, 500 y)",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.5 annual mean siconc difference in HadGEM3-GC31-MM historical r1i1p1f3_piControl r1i1p1f1.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region annual mean siconc, UKESM1-0-LL, piControl, r1i1p1f2

siconc_ue_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/SImon/siconc/gn/latest/siconc_SImon_UKESM1-0-LL_piControl_r1i1p1f2_gn_*.nc',
)))

siconc_hg3_ll_pi = xr.open_mfdataset(
    siconc_hg3_ll_pi_fl[0:5], data_vars='minimal',
    coords='minimal', compat='override',
)






am_siconc_hg3_ll_pi = siconc_hg3_ll_pi.siconc.mean(axis=0).values
# np.isnan(am_siconc_hg3_ll_pi).sum()

fig, ax = hemisphere_plot(
    northextent=-45, sb_length=2000, sb_barheight=200,
    figsize=np.array([8.8, 9.8]) / 2.54, fm_bottom=0.12, fm_top=0.98, )

pltlevel = np.arange(0, 100.01, 0.5)
pltticks = np.arange(0, 100.01, 20)

plt_cmp = ax.pcolormesh(
    siconc_hg3_ll_pi.longitude,
    siconc_hg3_ll_pi.latitude,
    am_siconc_hg3_ll_pi,
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "500 years annual mean sea ice area fraction [%]\nHadGEM3-GC31-LL, piControl, r1i1p1f1",
    linespacing=1.4,)

fig.savefig(
    'figures/3_sea_ice/3.0_climatology_of_sea_ice/3.0.0.0 annual mean siconc in HadGEM3-GC31-LL piControl r1i1p1f1.png')

'''
# check
'''
# endregion
# =============================================================================


# =============================================================================
# region check the annual mean siconc with function esgf_search


esgf_siconc_hg3_ll_pi_fl = esgf_search(
    activity_id='CMIP', table_id='SImon', variable_id='siconc',
    experiment_id='piControl', institution_id="MOHC",
    source_id="HadGEM3-GC31-LL", member_id="r1i1p1f1")

esgf_siconc_hg3_ll_pi = xr.open_mfdataset(
    esgf_siconc_hg3_ll_pi_fl[43:48], combine='by_coords',
    chunks={'time': 600})

am_esgf_siconc_hg3_ll_pi = esgf_siconc_hg3_ll_pi.siconc.mean(axis=0)

am_esgf_siconc_hg3_ll_pi.load()


# endregion
# =============================================================================

# ds = xr.open_dataset(
#     '/badc/cmip6/data/CMIP6/PMIP/NERC/HadGEM3-GC31-LL/midHolocene/r1i1p1f1/Amon/pr/gn/v20210111/pr_Amon_HadGEM3-GC31-LL_midHolocene_r1i1p1f1_gn_225001-234912.nc')

# ds1 = xr.open_dataset(
#     '/badc/cmip6/data/CMIP6/PMIP/NERC/HadGEM3-GC31-LL/lig127k/r1i1p1f1/Amon/pr/gn/v20210114/pr_Amon_HadGEM3-GC31-LL_lig127k_r1i1p1f1_gn_185001-204912.nc')



