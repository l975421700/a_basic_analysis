

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
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
sys.path.append('/albedo/work/user/qigao001')

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
    find_ilat_ilon,
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

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_aprt.pkl', 'rb') as f:
    contributions2site_aprt = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

# with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2AIS_aprt.pkl', 'rb') as f:
#     contributions2AIS_aprt = pickle.load(f)

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source contributions to aprt at sites


pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues',
    reversed=False)

for isite in ['EDC']:
    # isite = 'EDC'
    # isite = 'Rothera'
    # ten_sites_loc.Site
    print('#-------- ' + isite)
    
    # output_png = 'figures/6_awi/6.1_echam6/6.1.10_pre_sources/6.1.10.0 ' + expid[i] + ' source contributions to aprt at ' + isite + '.png'
    output_png = 'figures/1_study_area/1.2.0 ' + expid[i] + ' source contributions to aprt at ' + isite + '.png'
    
    fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7]) / 2.54,)
    
    isitelat = ten_sites_loc.lat[ten_sites_loc.Site == isite].values[0]
    isitelon = ten_sites_loc.lon[ten_sites_loc.Site == isite].values[0]
    cplot_ice_cores(isitelon, isitelat, ax, edgecolors = 'red')
    
    plt_data = (contributions2site_aprt[isite] * 1000).compute()
    plt_data.values[plt_data.values == 0] = np.nan
    
    plt1 = ax.pcolormesh(
        contributions2site_aprt[isite].lon,
        contributions2site_aprt[isite].lat,
        plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    cbar = fig.colorbar(
        plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
        pad=0.02, fraction=0.2,
        )
    cbar.ax.set_xlabel(
        'Contributions to precipitation\nat ' + isite + ' [‰]',
        linespacing=1.5)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(output_png)




'''
isite = 'EDC'
contributions2site_aprt[isite].sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source contributions to aprt over AIS

grid_contributions = contributions2AIS_aprt.sum(dim=['lat_t63', 'lon_t63']).compute()

grid_contributions = (grid_contributions / grid_contributions.sum().compute()).compute()

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.1, cmap='Blues',
    reversed=False)

output_png = 'figures/6_awi/6.1_echam6/6.1.10_pre_sources/6.1.10.0 ' + expid[i] + ' source contributions to AIS aprt.png'

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7]) / 2.54,)

plt1 = ax.pcolormesh(
    contributions2AIS_aprt.lon_1deg,
    contributions2AIS_aprt.lat_1deg,
    grid_contributions * 1000,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Contributions to precipitation\nover AIS [‰]', linespacing=1.5)
cbar.ax.tick_params(labelsize=8)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check total amount


data1 = contributions2AIS_aprt.sum(dim=['lat_1deg', 'lon_1deg'])

data2 = ocean_aprt_alltime[expid[i]]['am'].sel(var_names='lat') * 18262

np.max(abs((data1.values[echam6_t63_ais_mask['mask']['AIS']] - data2.values[echam6_t63_ais_mask['mask']['AIS']]) / data2.values[echam6_t63_ais_mask['mask']['AIS']]))

ocean_aprt_alltime[expid[i]]['daily']

# endregion
# -----------------------------------------------------------------------------

