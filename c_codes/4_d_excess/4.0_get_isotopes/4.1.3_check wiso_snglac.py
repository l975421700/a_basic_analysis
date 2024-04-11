

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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
# sys.path.append('/work/ollie/qigao001')

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
from statsmodels.stats import multitest
import pycircstat as circ
import math

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
    regrid,
    mean_over_ais,
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
    cplot_ttest,
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

exp_out_wiso = xr.open_mfdataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/test/unknown/test_2000??.01_wiso.nc')

VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)


dO18_alltime = {}
dD_alltime = {}
d_ln_alltime = {}
d_excess_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)


dO18_q_sfc_alltime = {}
dD_q_sfc_alltime = {}
d_ln_q_sfc_alltime = {}
d_excess_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
        dO18_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
        dD_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
        d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check am isotopes

# wisosnglac
wisosnglac_am = exp_out_wiso.wisosnglac.mean(dim='time')

dO18_am = (((wisosnglac_am.sel(wisotype=2) / wisosnglac_am.sel(wisotype=1)) / VSMOW_O18 - 1) * 1000).compute()

dD_am = (((wisosnglac_am.sel(wisotype=3) / wisosnglac_am.sel(wisotype=1)) / VSMOW_D - 1) * 1000).compute()

d_xs_am = dD_am - 8 * dO18_am

ln_dD = 1000 * np.log(1 + dD_am / 1000)
ln_d18O = 1000 * np.log(1 + dO18_am / 1000)
d_ln_am = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

# wisoaprt
wisoaprt_am = (exp_out_wiso.wisoaprl + exp_out_wiso.wisoaprc).mean(dim='time')

pre_dO18_am = (((wisoaprt_am.sel(wisotype=2) / wisoaprt_am.sel(wisotype=1)) / VSMOW_O18 - 1) * 1000).compute()

pre_dD_am = (((wisoaprt_am.sel(wisotype=3) / wisoaprt_am.sel(wisotype=1)) / VSMOW_D - 1) * 1000).compute()

pre_d_xs_am = pre_dD_am - 8 * pre_dO18_am

pre_ln_dD = 1000 * np.log(1 + pre_dD_am / 1000)
pre_ln_d18O = 1000 * np.log(1 + pre_dO18_am / 1000)
pre_d_ln_am = pre_ln_dD - 8.47 * pre_ln_d18O + 0.0285 * (pre_ln_d18O ** 2)




print(dD_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -324.41843
print(dO18_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -41.37838
print(d_xs_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 6.608612
print(d_ln_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 16.646545

print(pre_dD_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -389.4058
print(pre_dO18_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -49.57062
print(pre_d_xs_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 7.1591797
print(pre_d_ln_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 10.972351

print(dD_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -370.18395893
print(dO18_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -46.99330032
print(d_ln_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 11.39127319
print(d_excess_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 5.76244364


print(dD_q_sfc_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -449.14849036
print(dO18_q_sfc_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # -58.66757505
print(d_ln_q_sfc_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 19.97272969
print(d_excess_q_sfc_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest')) # 20.19211003


'''
dO18_daily = (((exp_out_wiso.wisosnglac.sel(wisotype=2) / exp_out_wiso.wisosnglac.sel(wisotype=1)) / VSMOW_O18 - 1) * 1000).compute()

dD_daily = (((exp_out_wiso.wisosnglac.sel(wisotype=3) / exp_out_wiso.wisosnglac.sel(wisotype=1)) / VSMOW_D - 1) * 1000).compute()

stats.describe(dD_daily.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest'))
# -353.72043 to -288.62332
'''
# endregion
# -----------------------------------------------------------------------------


