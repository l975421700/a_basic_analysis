

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
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

pre_weighted_RHsst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_RHsst.pkl', 'rb') as f:
    pre_weighted_RHsst[expid[i]] = pickle.load(f)

lon = pre_weighted_RHsst[expid[i]]['am'].lon
lat = pre_weighted_RHsst[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


'''
with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=60, cm_max=80, cm_interval1=1, cm_interval2=2, cmap='viridis',)

fig, ax = globe_plot()

plt_mesh1 = plot_t63_contourf(
    lon, lat, pre_weighted_RHsst[expid[i]]['ann'][0] * 100, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('Source RHsst [$\%$]', linespacing=2)
fig.savefig('figures/test/test.png')






# endregion
# -----------------------------------------------------------------------------


