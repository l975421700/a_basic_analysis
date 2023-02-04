

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
from scipy.stats import pearsonr

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
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
    remove_trailing_zero_pos_abs,
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
    ten_sites_names,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
    calc_lon_diff_np,
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

# epe_st sources
epe_st_sources_sites_binned = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_sources_sites_binned.pkl', 'rb') as f:
    epe_st_sources_sites_binned[expid[i]] = pickle.load(f)

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)

wind10_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_alltime.pkl', 'rb') as f:
    wind10_alltime[expid[i]] = pickle.load(f)

lon = wind10_alltime[expid[i]]['am'].lon.values
lat = wind10_alltime[expid[i]]['am'].lat.values


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wind10 at sources

wind10_sources = {}

isite = 'EDC'
# isite = 'Halley'
wind10_sources[isite] = \
    epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am.copy()
wind10_sources[isite][:] = 0


for iqtl in range(len(wind10_sources[isite])):
    # iqtl = 0
    
    source_lat = \
        epe_st_sources_sites_binned[expid[i]]['lat'][isite]['am'].am[iqtl]
    
    source_lon = \
        epe_st_sources_sites_binned[expid[i]]['lon'][isite]['am'].am[iqtl]
    
    sources_ind = find_ilat_ilon(source_lat, source_lon, lat, lon)
    
    wind10_sources[isite][iqtl] = \
        wind10_alltime[expid[i]]['am'][sources_ind[0], sources_ind[1]].values

print('#-------- ' + isite)
print(pearsonr(
    wind10_sources[isite],
    epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am,
))

print(np.mean(
    wind10_sources[isite] - \
        epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am,
        ))



'''
np.std(
    wind10_sources[isite] - \
        epe_st_sources_sites_binned[expid[i]]['wind10'][isite]['am'].am,
        )

'''
# endregion
# -----------------------------------------------------------------------------

