

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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

# plot
import proplot as pplt
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

epe_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_weighted_lat.pkl', 'rb') as f:
    epe_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

quantiles = {'90%': 0.9, '95%': 0.95, '99%': 0.99}


'''
wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)


# wrong way to get epe_source_lat
epe_source_lat = {}

for iqtl in quantiles.keys():
    # iqtl = '90%'
    print(iqtl)
    epe_source_lat[iqtl] = {}
    
    # am
    epe_source_lat[iqtl]['am'] = pre_weighted_lat[expid[i]]['daily'].weighted(
        wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl]).mean(dim='time')
    # daily

#-------- check
ilat = 40
ilon = 90

iqtl = '90%'
# am
res001 = epe_source_lat[iqtl]['am'][ilat, ilon].values
data = pre_weighted_lat[expid[i]]['daily'][:, ilat, ilon].values
weights = wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl][:, ilat, ilon]
res002 = np.average(data[np.isfinite(data)], weights=weights[np.isfinite(data)])
print(res001 == res002)
print((res001 - res002) / res001)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_source_lat - pre_weighted_lat) am Antarctica

iqtl = '95%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.0_source_lat/6.1.7.0.0 ' + expid[i] + ' epe_source_lat - pre_weighted_lat am Antarctica.png'

pltlevel = np.arange(0, 10 + 1e-4, 1)
pltticks = np.arange(0, 10 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()
# pltcmp = pplt.Colormap('DryWet', samples=len(pltlevel)-1)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    epe_weighted_lat[expid[i]][iqtl]['ann'],
    pre_weighted_lat[expid[i]]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('EPE source latitude anomalies [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)



'''
(epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am']).to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (epe_source_lat - pre_weighted_lat) am

iqtl = '95%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.0_source_lat/6.1.7.0.0 ' + expid[i] + ' epe_source_lat - pre_weighted_lat am.png'

pltlevel = np.arange(-10, 10 + 1e-4, 1)
pltticks = np.arange(-10, 10 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()
# pltcmp = pplt.Colormap('DryWet', samples=len(pltlevel)-1)

fig, ax = globe_plot()
# fig, ax = hemisphere_plot(
#     northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    epe_weighted_lat[expid[i]][iqtl]['ann'],
    pre_weighted_lat[expid[i]]['ann'],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(length=2, width=0.4, labelsize=8)
cbar.ax.set_xlabel('EPE source latitude anomalies [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)


'''
(epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am']).to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------

