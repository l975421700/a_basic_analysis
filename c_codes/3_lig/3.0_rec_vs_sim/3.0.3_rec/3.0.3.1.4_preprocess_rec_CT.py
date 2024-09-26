

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
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp

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
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
    find_ilat_ilon_general,
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

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    mon_sea_ann,
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

lig_recs = {}
lig_recs['CT'] = {}

lig_recs['CT']['original'] = pd.read_csv(
    'data_sources/LIG/CT2023/Global_SST_early_LIG_annual.tab',
    sep='\t', header=0, skiprows=310,
)

lig_recs['CT']['org_SO'] = lig_recs['CT']['original'].loc[lig_recs['CT']['original']['Latitude'] <= -40]

# np.mean(lig_recs['CT']['original'].loc[lig_recs['CT']['original']['Latitude'] <= -40]['T anomaly [°C] (Annual LIG temperature anomal...).1'])

'''
lig_recs['CT']['org_SO'].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot data

output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec am sst lig-pi CT20.png'

cbar_label = 'Annual SST anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='RdBu',)

max_size = 80
scale_size = 16

fig, ax = hemisphere_plot(northextent=-38,)

plt_scatter = ax.scatter(
    x = lig_recs['CT']['org_SO'].Longitude,
    y = lig_recs['CT']['org_SO'].Latitude,
    c = lig_recs['CT']['org_SO']['T anomaly [°C] (Annual LIG temperature anomal...).1'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2, format=remove_trailing_zero_pos,
    )
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
fig.savefig(output_png)

'''
28 rows
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare with simulations

hadcm3_output_regridded_alltime = pd.read_pickle('scratch/share/from_rahul/data_qingang/hadcm3_output_regridded_alltime.pkl')

data = lig_recs['CT']['org_SO']['T anomaly [°C] (Annual LIG temperature anomal...).1']

data1 = np.zeros_like(data)

data2 = find_multi_gridvalue_at_site(
    lig_recs['CT']['org_SO'].Latitude.values,
    lig_recs['CT']['org_SO'].Longitude.values,
    hadcm3_output_regridded_alltime['LIG_PI']['SST']['am'].lat.values,
    hadcm3_output_regridded_alltime['LIG_PI']['SST']['am'].lon.values,
    hadcm3_output_regridded_alltime['LIG_PI']['SST']['am'].squeeze().values
)

data3 = find_multi_gridvalue_at_site(
    lig_recs['CT']['org_SO'].Latitude.values,
    lig_recs['CT']['org_SO'].Longitude.values,
    hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['am'].lat.values,
    hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['am'].lon.values,
    hadcm3_output_regridded_alltime['LIG0.25_PI']['SST']['am'].squeeze().values
)

subset = np.isfinite(data) & np.isfinite(data1)
np.sqrt(np.average(np.square(data[subset] - data1[subset])))

subset = np.isfinite(data) & np.isfinite(data2)
np.sqrt(np.average(np.square(data[subset] - data2[subset])))

subset = np.isfinite(data) & np.isfinite(data3)
np.sqrt(np.average(np.square(data[subset] - data3[subset])))


# endregion
# -----------------------------------------------------------------------------

