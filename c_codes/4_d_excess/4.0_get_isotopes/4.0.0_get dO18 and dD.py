

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

# setting
VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266

# threshold to calculate dO18 and dD
wiso_calc_min = 0.05 / 2.628e6

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann dO18

dO18_alltime = {}
dO18_alltime[expid[i]] = {}

for ialltime in wisoaprt_alltime[expid[i]].keys():
    # ialltime = 'am'
    # ialltime = 'daily'
    print(ialltime)
    
    dO18_alltime[expid[i]][ialltime] = \
        (((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=2) / \
            wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)) / \
                VSMOW_O18 - 1) * 1000).compute()
    
    dO18_alltime[expid[i]][ialltime].values[
        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min
        ] = np.nan
    
    dO18_alltime[expid[i]][ialltime] = \
        dO18_alltime[expid[i]][ialltime].rename('dO18')

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'wb') as f:
    pickle.dump(dO18_alltime[expid[i]], f)





'''
#-------------------------------- check 1st
dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

ialltime = 'ann'

ccc = dO18_alltime[expid[i]][ialltime].values
ddd = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=2) / wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) / VSMOW_O18 - 1) * 1000).compute().values

ddd[wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min] = np.nan

(ccc[np.isfinite(ccc)] == ddd[np.isfinite(ddd)]).all()

#-------------------------------- check 2nd

dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

dO18_alltime[expid[i]]['am'].to_netcdf('scratch/test/run/test.nc')

(wisoaprt_alltime[expid[i]][ialltime][0].values < wiso_calc_min).sum()

np.isnan(dO18_alltime[expid[i]][ialltime].values).sum()

(wisoaprt_alltime[expid[i]]['daily'][0].values < wiso_calc_min).sum()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann dD

dD_alltime = {}
dD_alltime[expid[i]] = {}

for ialltime in wisoaprt_alltime[expid[i]].keys():
    # ialltime = 'am'
    # ialltime = 'daily'
    print(ialltime)
    
    dD_alltime[expid[i]][ialltime] = \
        (((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=3) / \
            wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)) / \
                VSMOW_D - 1) * 1000).compute()
    
    dD_alltime[expid[i]][ialltime].values[
        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min
        ] = np.nan
    
    dD_alltime[expid[i]][ialltime] = \
        dD_alltime[expid[i]][ialltime].rename('dD')

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'wb') as f:
    pickle.dump(dD_alltime[expid[i]], f)




'''
#-------------------------------- check 1st
dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

ccc = dD_alltime[expid[i]][ialltime].values
ddd = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=3) / wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) / VSMOW_D - 1) * 1000).compute().values

ddd[wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min] = np.nan

(ccc[np.isfinite(ccc)] == ddd[np.isfinite(ddd)]).all()

#-------------------------------- check 2nd

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

dD_alltime[expid[i]]['ann'].to_netcdf('scratch/test/run/test.nc')

'''
# endregion
# -----------------------------------------------------------------------------

