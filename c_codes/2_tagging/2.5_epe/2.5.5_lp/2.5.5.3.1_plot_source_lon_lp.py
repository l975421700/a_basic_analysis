

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
from scipy.stats import pearsonr

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

lp_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lp_weighted_lon.pkl', 'rb') as f:
    lp_weighted_lon[expid[i]] = pickle.load(f)

lpr_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lpr_weighted_lon.pkl', 'rb') as f:
    lpr_weighted_lon[expid[i]] = pickle.load(f)

lon = lpr_weighted_lon[expid[i]]['10%']['am'].lon
lat = lpr_weighted_lon[expid[i]]['10%']['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot (lp_weighted_lon '10%'-lpr_weighted_lon '10%') am Antarctica

iqtl = '10%'
plt_data = calc_lon_diff(
    lp_weighted_lon[expid[i]][iqtl]['am'],
    lpr_weighted_lon[expid[i]][iqtl]['am'],
    )
# plt_data.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan


output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.0_pre_source/6.1.7.0.1_source_lon/6.1.7.0.1 ' + expid[i] + ' lp_weighted_lon_10 - lpr_weighted_lon_10 am Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn',
    reversed=True, asymmetric=True)

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

wwtest_res = circ.watson_williams(
    lp_weighted_lon[expid[i]][iqtl]['ann'] * np.pi / 180,
    lpr_weighted_lon[expid[i]][iqtl]['ann'] * np.pi / 180,
    axis=0,
    )[0] < 0.05
ax.scatter(
    x=lon_2d[wwtest_res & echam6_t63_ais_mask['mask']['AIS']],
    y=lat_2d[wwtest_res & echam6_t63_ais_mask['mask']['AIS']],
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
cbar.ax.set_xlabel('LP source longitude anomalies [$°$]', linespacing=2)
fig.savefig(output_png, dpi=600)


'''
'''
# endregion
# -----------------------------------------------------------------------------





