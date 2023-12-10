

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
i = 0


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
from statsmodels.stats import multitest
import pycircstat as circ
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

epe_st_weighted_wind10 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_wind10.pkl', 'rb') as f:
    epe_st_weighted_wind10[expid[i]] = pickle.load(f)

dc_st_weighted_wind10 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_st_weighted_wind10.pkl', 'rb') as f:
    dc_st_weighted_wind10[expid[i]] = pickle.load(f)

lon = epe_st_weighted_wind10[expid[i]]['90%']['am'].lon
lat = epe_st_weighted_wind10[expid[i]]['90%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_st_weighted_wind10 - dc_st_weighted_wind10) am Antarctica

iqtl = '90%'
plt_data = (epe_st_weighted_wind10[expid[i]][iqtl]['am'] - \
    dc_st_weighted_wind10[expid[i]][iqtl]['am']).compute()
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.4_source_wind10/6.1.7.0.4 ' + expid[i] + ' epe_st_weighted_wind10 - dc_st_weighted_wind10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.8, cm_max=0.8, cm_interval1=0.2, cm_interval2=0.2, cmap='PuOr',
    reversed=True, asymmetric=True,)
pltticks[-5] = 0

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

# plt1 = ax.pcolormesh(
#     lon,
#     lat,
#     plt_data,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
plt1 = plot_t63_contourf(
    lon, lat, plt_data, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
ax.add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

ttest_fdr_res = ttest_fdr_control(
    epe_st_weighted_wind10[expid[i]][iqtl]['ann'],
    dc_st_weighted_wind10[expid[i]][iqtl]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    y=lat_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    s=1.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source vel10 anomalies [$m \; s^{-1}$]', linespacing=2)
fig.savefig(output_png)



'''
(epe_st_weighted_wind10[expid[i]][iqtl]['am'] - pre_weighted_wind10[expid[i]]['am']).to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_st_weighted_wind10 '90%'-dc_st_weighted_wind10 '10%') am Antarctica

# iqtl = '90%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.4_source_wind10/6.1.7.0.4 ' + expid[i] + ' epe_st_weighted_wind10_90 - dc_st_weighted_wind10_10 am Antarctica.png'

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=-1.2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='PuOr',
#     reversed=True, asymmetric=True,)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=3, cm_interval1=0.5, cm_interval2=0.5, cmap='PuOr',
    reversed=False, asymmetric=True,)

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    epe_st_weighted_wind10[expid[i]]['90%']['am'] - \
        dc_st_weighted_wind10[expid[i]]['10%']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    epe_st_weighted_wind10[expid[i]]['90%']['ann'],
    dc_st_weighted_wind10[expid[i]]['10%']['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('EPE-LP source vel10 [$m \; s^{-1}$]', linespacing=2)
fig.savefig(output_png)



'''
(epe_st_weighted_wind10[expid[i]][iqtl]['am'] - pre_weighted_wind10[expid[i]]['am']).to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region estimate statistics


echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


iqtl = '90%'
wind10_diff = epe_st_weighted_wind10[expid[i]][iqtl]['am'] - \
    dc_st_weighted_wind10[expid[i]][iqtl]['am']


for imask in ['AIS', 'EAIS', 'WAIS', 'AP']:
    # imask = 'AIS'
    print('#-------- ' + imask)
    
    mask = echam6_t63_ais_mask['mask'][imask]
    
    ave_wind10_diff = np.average(
        wind10_diff.values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],
    )
    
    print(str(np.round(ave_wind10_diff, 2)))


echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)
mask_low = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values < 2250)

for mask in[mask_high, mask_low]:
    
    ave_wind10_diff = np.average(
        wind10_diff.values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],
    )
    
    print(str(np.round(ave_wind10_diff, 1)))

echam6_t63_ais_mask['mask'][imask].sum()
mask_high.sum() + mask_low.sum()




with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

for isite in ['EDC', 'Halley']:
    # isite = 'EDC'
    print(isite)
    
    res = wind10_diff.values[
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']]
    
    print(np.round(res, 1))





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (dc_st_weighted_wind10_10 - epe_st_weighted_wind10_10) am Antarctica

iqtl = '10%'
plt_data = dc_st_weighted_wind10[expid[i]][iqtl]['am'] - \
    epe_st_weighted_wind10[expid[i]][iqtl]['am']

plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.4_source_wind10/6.1.7.0.4 ' + expid[i] + ' dc_st_weighted_wind10_10 - epe_st_weighted_wind10_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-1, cm_max=1, cm_interval1=0.25, cm_interval2=0.25, cmap='PuOr',
    reversed=True, asymmetric=True,)
# pltticks[-5] = 0

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    plt_data,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    dc_st_weighted_wind10[expid[i]][iqtl]['ann'],
    epe_st_weighted_wind10[expid[i]][iqtl]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    y=lat_2d[ttest_fdr_res & echam6_t63_ais_mask['mask']['AIS']],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('LP source wind10 anomalies [$m \; s^{-1}$]', linespacing=2)
fig.savefig(output_png)



'''
(epe_st_weighted_wind10[expid[i]][iqtl]['am'] - pre_weighted_wind10[expid[i]]['am']).to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


