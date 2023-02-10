

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
    find_cumulative_threshold,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

# quantile_interval  = np.arange(1, 99 + 1e-4, 1, dtype=np.int64)
quantile_interval  = np.arange(10, 50 + 1e-4, 10, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

'''
quantile_interval
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region find a quantile


wisoaprt_cum_qtl = {}
wisoaprt_cum_qtl[expid[i]] = {}
wisoaprt_cum_qtl[expid[i]]['quantiles'] = {}
wisoaprt_cum_qtl[expid[i]]['mask'] = {}

for iqtl in quantiles.keys():
    # iqtl = '20%'
    print(iqtl + ': ' + str(quantiles[iqtl]))
    
    #-------- calculate quantiles
    wisoaprt_cum_qtl[expid[i]]['quantiles'][iqtl] = \
        xr.apply_ufunc(
            find_cumulative_threshold,
            wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1),
            input_core_dims=[["time"]],
            kwargs={'threshold': quantiles[iqtl]},
            dask = 'allowed', vectorize = True).compute()
    
    #-------- get mask
    wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] = \
        (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1) <= \
            wisoaprt_cum_qtl[expid[i]]['quantiles'][iqtl]).compute()


import os, psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 2**30)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_cum_qtl.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_cum_qtl[expid[i]], f)


'''
# 39min to run, mpp120

#-------------------------------- check 2

wisoaprt_cum_qtl = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_cum_qtl.pkl',
    'rb') as f:
    wisoaprt_cum_qtl[expid[i]] = pickle.load(f)

ilat = 70
ilon = 90

iqtl = '30%'
aprt_data = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].values
res1 = find_cumulative_threshold(aprt_data, threshold=quantiles[iqtl])
res2 = wisoaprt_cum_qtl[expid[i]]['quantiles'][iqtl][ilat, ilon].values
print(res1 == res2)

res3 = (aprt_data <= res1)
res4 = wisoaprt_cum_qtl[expid[i]]['mask'][iqtl][:, ilat, ilon].values
print((res3 == res4).all())



#-------------------------------- check 1 with function


res = xr.apply_ufunc(
    find_cumulative_threshold,
    wisoaprt_alltime[expid[i]]['daily'],
    input_core_dims=[["time"]],
    kwargs={'threshold': 0.1},
    dask = 'allowed', vectorize = True)

ilat = 40
ilon = 60

aprt_data = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].values
res1 = find_cumulative_threshold(aprt_data)

print(res1 == res[0, ilat, ilon].values)

'''
# endregion
# -----------------------------------------------------------------------------

