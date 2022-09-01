

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
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
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]

# region import output

exp_org_o = {}

i = 0
exp_org_o[expid[i]] = {}

filenames_sf_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
exp_org_o[expid[i]]['sf_wiso'] = xr.open_mfdataset(filenames_sf_wiso[120:], data_vars='minimal', coords='minimal', parallel=True)


'''
#-------- check evap
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:], data_vars='minimal', coords='minimal', parallel=True)

np.max(abs(exp_org_o[expid[i]]['echam'].evap.values - exp_org_o[expid[i]]['sf_wiso'].wisoevap[:, 0].values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann evap

i = 0
expid[i]

wisoevap = {}


wisoevap[expid[i]] = (exp_org_o[expid[i]]['sf_wiso'].wisoevap[:, :3]).copy().compute()

wisoevap[expid[i]] = wisoevap[expid[i]].rename('wisoevap')

wisoevap_alltime = {}
wisoevap_alltime[expid[i]] = mon_sea_ann(var_monthly = wisoevap[expid[i]])


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl',
          'wb') as f:
    pickle.dump(wisoevap_alltime[expid[i]], f)


'''
#-------- check
i = 0
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:], data_vars='minimal', coords='minimal', parallel=True)

wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].evap.values - wisoevap_alltime[expid[i]]['mon'][:, 0].values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot aprt and evap


i = 0
expid[i]
wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

#-------- basic settings

lon = wisoevap_alltime[expid[i]]['am'].lon
lat = wisoevap_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt and evap am_sm.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Evaporation [$mm \; day^{-1}$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

pltlevel2 = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks2 = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)


nrow = 3
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol == 0) | (jcol == 1)):
            axs[irow, jcol] = globe_plot(
                ax_org = axs[irow, jcol], add_grid_labels=False)
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt_mesh2 = axs[0, 1].pcolormesh(
    lon, lat, wisoevap_alltime[expid[i]]['am'][0] * 3600 * 24 * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- sm pre
axs[1, 0].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- sm evap
axs[2, 0].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * 3600 * 24 * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * 3600 * 24 * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * 3600 * 24 * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * 3600 * 24 * (-1),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean precipitation', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean evaporation', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF precipitation', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'MAM precipitation', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA precipitation', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'SON precipitation', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF evaporation', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'MAM evaporation', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA evaporation', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'SON evaporation', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-4.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.75, top = 0.96)
fig.savefig(output_png)





'''
# wisoaprt_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')
# wisoevap_alltime[expid[i]]['am'].to_netcdf('scratch/test/test2.nc')

#---- std
# stats.describe(wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24, axis=None)
# (wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24 == (wisoaprt_alltime[expid[i]]['ann'][:, 0] * 3600 * 24).std(dim='time')).all()
# np.max(abs(wisoaprt_alltime[expid[i]]['ann'][:, 0].std(dim='time') * 3600 * 24 - (wisoaprt_alltime[expid[i]]['ann'][:, 0] * 3600 * 24).std(dim='time')))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot aprt and evap Antarctica


i = 0
expid[i]
wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)


#-------- basic set

lon = wisoevap_alltime[expid[i]]['am'].lon
lat = wisoevap_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt and evap am_sm Antarctica.png'
cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Evaporation/Precipitation [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-15, 15 + 1e-4, 3)
pltticks2 = np.arange(-15, 15 + 1e-4, 6)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        if ((irow != 0) | (jcol == 0) | (jcol == 1)):
            axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])
        else:
            axs[irow, jcol].axis('off')

#-------- Am
plt_mesh1 = axs[0, 0].pcolormesh(
    lon, lat, wisoaprt_alltime[expid[i]]['am'][0] * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    lon, lat, wisoevap_alltime[expid[i]]['am'][0] * (-100) / wisoaprt_alltime[expid[i]]['am'][0],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- sm pre
axs[1, 0].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 3].pcolormesh(
    lon, lat,
    wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * 3600 * 24,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- sm evap/pre
plt_mesh2 = axs[2, 0].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='DJF') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='MAM') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='MAM'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='JJA') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 3].pcolormesh(
    lon, lat,
    wisoevap_alltime[expid[i]]['sm'][:, 0].sel(season='SON') * (-100) / wisoaprt_alltime[expid[i]]['sm'][:, 0].sel(season='SON'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean precipitation', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean evaporation/precipitation', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF precipitation', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM precipitation', transform=axs[1, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA precipitation', transform=axs[1, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON precipitation', transform=axs[1, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF evaporation/precipitation', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM evaporation/precipitation', transform=axs[2, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA evaporation/precipitation', transform=axs[2, 2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON evaporation/precipitation', transform=axs[2, 3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------




