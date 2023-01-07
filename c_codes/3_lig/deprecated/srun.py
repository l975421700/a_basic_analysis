

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
sys.path.append('/home/users/qino')

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
# region get ensemble statistics

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

sic_regrid_alltime_ens_stats = {}
sic_regrid_alltime_ens_stats['lig'] = {}
sic_regrid_alltime_ens_stats['pi'] = {}
sic_regrid_alltime_ens_stats['lig_pi'] = {}

for ialltime in lig_sic_regrid_alltime_ens.keys():
    # ialltime = 'mon'
    print(ialltime)
    
    sic_regrid_alltime_ens_stats['lig'][ialltime] = {}
    sic_regrid_alltime_ens_stats['pi'][ialltime] = {}
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime] = {}
    
    sic_regrid_alltime_ens_stats['lig'][ialltime]['mean'] = \
        lig_sic_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sic_regrid_alltime_ens_stats['lig'][ialltime]['std'] = \
        lig_sic_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sic_regrid_alltime_ens_stats['pi'][ialltime]['mean'] = \
        pi_sic_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sic_regrid_alltime_ens_stats['pi'][ialltime]['std'] = \
        pi_sic_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'] = \
        (lig_sic_regrid_alltime_ens[ialltime] - \
            pi_sic_regrid_alltime_ens[ialltime].values).mean(
                dim='ensemble', skipna=True,).compute()
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'] = \
        (lig_sic_regrid_alltime_ens[ialltime] - \
            pi_sic_regrid_alltime_ens[ialltime].values).std(
                dim='ensemble', skipna=True, ddof=1).compute()

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'wb') as f:
    pickle.dump(sic_regrid_alltime_ens_stats, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

ialltime = 'am'

data1 = sic_regrid_alltime_ens_stats['lig'][ialltime]['mean'].values
data2 = lig_sic_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['pi'][ialltime]['mean'].values
data2 = pi_sic_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data1 = sic_regrid_alltime_ens_stats['lig'][ialltime]['std'].values
data2 = lig_sic_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['pi'][ialltime]['std'].values
data2 = pi_sic_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'].values
data2 = (lig_sic_regrid_alltime_ens[ialltime] - \
    pi_sic_regrid_alltime_ens[ialltime].values).mean(
        dim='ensemble', skipna=True,).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'].values
data2 = (lig_sic_regrid_alltime_ens[ialltime] - \
    pi_sic_regrid_alltime_ens[ialltime].values).std(
        dim='ensemble', skipna=True,ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




'''
# endregion
# -----------------------------------------------------------------------------


