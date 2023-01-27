

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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

transport_distance_epe_st = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance_epe_st.pkl', 'rb') as f:
    transport_distance_epe_st[expid[i]] = pickle.load(f)

transport_distance_dc_st = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance_dc_st.pkl', 'rb') as f:
    transport_distance_dc_st[expid[i]] = pickle.load(f)

lon = transport_distance_epe_st[expid[i]]['90%']['am'].lon
lat = transport_distance_epe_st[expid[i]]['90%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (transport_distance_epe_st - transport_distance_dc_st) am Antarctica

iqtl = '90%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.5_travel_distance/6.1.7.0.5 ' + expid[i] + ' transport_distance_epe_st - transport_distance_dc_st am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=18, cm_interval1=2, cm_interval2=2, cmap='BrBG',
    reversed=False)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    (transport_distance_epe_st[expid[i]][iqtl]['am'] - \
        transport_distance_dc_st[expid[i]][iqtl]['am']) / 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    transport_distance_epe_st[expid[i]][iqtl]['ann'],
    transport_distance_dc_st[expid[i]][iqtl]['ann'],
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
cbar.ax.set_xlabel('EPE source-sink distance anomalies [$10^{2} \; km$]',
                   linespacing=2, fontsize=8)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (transport_distance_epe_st - transport_distance_dc_st) am Antarctica

# iqtl = '90%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.5_travel_distance/6.1.7.0.5 ' + expid[i] + ' transport_distance_epe_st_90 - transport_distance_dc_st_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-25, cm_max=45, cm_interval1=5, cm_interval2=5, cmap='BrBG',
    reversed=False, asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    (transport_distance_epe_st[expid[i]]['90%']['am'] - \
        transport_distance_dc_st[expid[i]]['10%']['am']) / 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    transport_distance_epe_st[expid[i]]['90%']['ann'],
    transport_distance_dc_st[expid[i]]['10%']['ann'],
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
cbar.ax.set_xlabel('EPE-DD source-sink distance [$10^{2} \; km$]',
                   linespacing=2, fontsize=8)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region estimate statistics


echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)


iqtl = '90%'

distance_diff = (transport_distance_epe_st[expid[i]][iqtl]['am'] - \
    transport_distance_dc_st[expid[i]][iqtl]['am'])



for imask in ['AIS', 'EAIS', 'WAIS', 'AP']:
    # imask = 'AIS'
    print('#-------- ' + imask)
    
    mask = echam6_t63_ais_mask['mask'][imask]
    
    ave_distance_diff = np.average(
        distance_diff.values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],
    )
    
    print(str(np.round(ave_distance_diff, 0)))


echam6_t63_geosp = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/input/echam/unit.24')
echam6_t63_surface_height = geopotential_to_height(
    echam6_t63_geosp.GEOSP * (units.m / units.s)**2)

imask = 'AIS'
mask_high = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values >= 2250)
mask_low = echam6_t63_ais_mask['mask'][imask] & \
    (echam6_t63_surface_height.values < 2250)

for mask in[mask_high, mask_low]:
    
    ave_distance_diff = np.average(
        distance_diff.values[mask],
        weights = echam6_t63_cellarea.cell_area.values[mask],
    )
    
    print(str(np.round(ave_distance_diff, 1)))

echam6_t63_ais_mask['mask'][imask].sum()
mask_high.sum() + mask_low.sum()




with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

for isite in ['EDC', 'Halley']:
    # isite = 'EDC'
    print(isite)
    
    res = distance_diff.values[
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']]
    
    print(np.round(res, 1))





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (transport_distance_dc_st_10 - transport_distance_epe_st_10) am Antarctica

iqtl = '10%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.5_travel_distance/6.1.7.0.5 ' + expid[i] + ' transport_distance_dc_st_10 - transport_distance_epe_st_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-35, cm_max=25, cm_interval1=5, cm_interval2=5, cmap='BrBG',
    reversed=False, asymmetric=True,)

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    (transport_distance_dc_st[expid[i]][iqtl]['am'] - \
        transport_distance_epe_st[expid[i]][iqtl]['am']) / 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    transport_distance_dc_st[expid[i]][iqtl]['ann'],
    transport_distance_epe_st[expid[i]][iqtl]['ann'],
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
cbar.ax.set_xlabel('LP source-sink distance anomalies [$10^{2} \; km$]',
                   linespacing=2, fontsize=8)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


