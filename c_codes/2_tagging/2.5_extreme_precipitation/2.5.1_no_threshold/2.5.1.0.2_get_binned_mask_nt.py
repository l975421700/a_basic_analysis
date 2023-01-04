

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
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoaprt_epe_nt = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_nt.pkl',
    'rb') as f:
    wisoaprt_epe_nt[expid[i]] = pickle.load(f)

quantile_interval  = np.arange(1, 99 + 1e-4, 1, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get percentiles 1:99:1

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

wisoaprt_mask_bin_nt = {}

# !1% (< 1%)
wisoaprt_mask_bin_nt['0.5%'] = (wisoaprt_epe_nt[expid[i]]['mask']['1%'] == False)

for iind in range(98):
    # iind = 0
    iqtl1 = list(quantiles_bin.keys())[1:-1][iind]
    iqtl2 = list(quantiles.keys())[iind]
    iqtl3 = list(quantiles.keys())[iind + 1]
    
    print(iqtl1 + ' vs. ' + iqtl2 + ' & ' + iqtl3)
    
    # e.g. 1.5% : 1% (>= 1%) & !2% (< 2%)
    
    wisoaprt_mask_bin_nt[iqtl1] = (wisoaprt_epe_nt[expid[i]]['mask'][iqtl2] & \
        (wisoaprt_epe_nt[expid[i]]['mask'][iqtl3] == False))

# 99% (>= 99%)
wisoaprt_mask_bin_nt['99.5%'] = wisoaprt_epe_nt[expid[i]]['mask']['99%']

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_nt.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_mask_bin_nt, f)






'''
# 2 min to run
#SBATCH --time=00:30:00
#SBATCH --partition=fat

#-------------------------------- check

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_nt.pkl',
    'rb') as f:
    wisoaprt_mask_bin_nt = pickle.load(f)

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

wisoaprt_mask_and = wisoaprt_mask_bin_nt['0.5%'].sum()
wisoaprt_mask_or  = wisoaprt_mask_bin_nt['0.5%'].copy()

for iqtl in list(quantiles_bin.keys())[1:]:
    print(iqtl)
    
    wisoaprt_mask_and = wisoaprt_mask_and + wisoaprt_mask_bin_nt[iqtl].sum()
    wisoaprt_mask_or  = (wisoaprt_mask_or  | wisoaprt_mask_bin_nt[iqtl])

print(wisoaprt_mask_and.values)
print(wisoaprt_mask_or.sum().values)
print(wisoaprt_mask_and.values == wisoaprt_mask_or.sum().values)

res01 = wisoaprt_mask_bin_nt['0.5%'].copy()
res02 = wisoaprt_mask_bin_nt['45.5%'].copy()
res03 = wisoaprt_mask_bin_nt['99.5%'].copy()

del wisoaprt_mask_bin_nt

wisoaprt_epe_nt = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_nt.pkl',
    'rb') as f:
    wisoaprt_epe_nt[expid[i]] = pickle.load(f)

res11 = (wisoaprt_epe_nt[expid[i]]['mask']['1%'] == False)
res12 = (wisoaprt_epe_nt[expid[i]]['mask']['45%'] & \
    (wisoaprt_epe_nt[expid[i]]['mask']['46%'] == False))
res13 = wisoaprt_epe_nt[expid[i]]['mask']['99%']

print((res01 == res11).all().values)
print((res02 == res12).all().values)
print((res03 == res13).all().values)

del wisoaprt_epe_nt


'''
# endregion
# -----------------------------------------------------------------------------



