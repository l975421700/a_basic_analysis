

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_1d_800_5.0',
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
# region check range of RHsst

exp_org_o = {}
exp_org_o[expid[i]] = {}

exp_org_o[expid[i]]['surf'] = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/' + expid[i] + '/outdata/echam/' + expid[i] + '_200001.01_surf.nc')

exp_org_o[expid[i]]['echam'] = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/pi_1d_800_5.0/unknown/pi_1d_800_5.0_200001.01_echam.nc')

RHsst = exp_org_o[expid[i]]['surf'].zqklevw / exp_org_o[expid[i]]['surf'].zqsw

RHsst.to_netcdf('scratch/test/test0.nc')

stats.describe(RHsst.values.flatten(), nan_policy='omit')
stats.describe(RHsst.values[exp_org_o[expid[i]]['echam'].slm.values == 0], nan_policy='omit')




'''
open_ocean = (exp_org_o[expid[i]]['echam'].slm.values == 0) & (exp_org_o[expid[i]]['echam']['seaice'].values == 0)

qklev = exp_org_o[expid[i]]['echam']['q'].sel(lev=47).values[open_ocean]
qklevw = exp_org_o[expid[i]]['surf'].zqklevw.values[open_ocean]

stats.describe(abs(qklevw - qklev) / qklev)

qdiff = (exp_org_o[expid[i]]['surf'].zqklevw - exp_org_o[expid[i]]['echam']['q'].sel(lev=47)) / exp_org_o[expid[i]]['echam']['q'].sel(lev=47)
qdiff.values[open_ocean == False] = np.nan
qdiff.to_netcdf('scratch/test/test0.nc')

np.nanmax(qdiff)
'''
# endregion
# -----------------------------------------------------------------------------



