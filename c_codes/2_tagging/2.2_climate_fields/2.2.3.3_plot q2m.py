

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
    plot_maxmin_points,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec_num,
    month_dec,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
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

q2m_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q2m_alltime.pkl', 'rb') as f:
    q2m_alltime[expid[i]] = pickle.load(f)


lon = q2m_alltime[expid[i]]['am'].lon
lat = q2m_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am q2m Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.7_q2m/6.1.2.7 ' + expid[i] + ' q2m am Antarctica.png'

pltlevel = np.array([0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltticks = np.array([0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PRGn', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    q2m_alltime[expid[i]]['am'] * 1000,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('2-metre specific humidity (q2m) [$g \; kg^{-1}$]',
                   linespacing=1.5,)
fig.savefig(output_png)



'''
stats.describe(q2m_alltime[expid[i]]['am'].sel(lat=slice(-20, -90)),
               axis=None, nan_policy='omit')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot DJF-JJA q2m Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.2_climatology/6.1.2.7_q2m/6.1.2.7 ' + expid[i] + ' q2m DJF-JJA Antarctica.png'

pltlevel = np.array([0, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8])
pltticks = np.array([0, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('broc', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt_mesh1 = ax.pcolormesh(
    lon, lat,
    (q2m_alltime[expid[i]]['sm'].sel(season='DJF') - \
        q2m_alltime[expid[i]]['sm'].sel(season='JJA')) * 1000,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
ttest_fdr_res = ttest_fdr_control(
    q2m_alltime[expid[i]]['sea'][3::4],
    q2m_alltime[expid[i]]['sea'][1::4],
    )
ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.95, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.set_xlabel('DJF - JJA q2m [$g \; kg^{-1}$]',
                   linespacing=1.5,)
fig.savefig(output_png)



'''
    # 100 * (q2m_alltime[expid[i]]['sm'].sel(season='DJF') / \
    #     q2m_alltime[expid[i]]['sm'].sel(season='JJA') - 1),

stats.describe(q2m_alltime[expid[i]]['am'].sel(lat=slice(-20, -90)),
               axis=None, nan_policy='omit')

'''
# endregion
# -----------------------------------------------------------------------------

