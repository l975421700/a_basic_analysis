

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
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
import matplotlib.patches as mpatches

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
    change_snsbar_width,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoaprt_mean_over_ais = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl', 'rb') as f:
    wisoaprt_mean_over_ais[expid[i]] = pickle.load(f)

with open('scratch/products/era5/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)




'''
mmaprt1 = aprt_geo7_spave['AIS']['mm'].sum(dim='wisotype') * seconds_per_d * month_days
mmaprt2 = wisoaprt_mean_over_ais[expid[i]]['mm'] * seconds_per_d * month_days
(mmaprt1 - mmaprt2) / mmaprt2
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region histogram: annual circle of aprt over AIS in ECHAM and ERA5

pre_mm_over_ais = pd.DataFrame(columns=(
    'Data', 'Month', 'pre_mon',))

pre_mm_over_ais = pd.concat(
    [pre_mm_over_ais,
     pd.DataFrame(data={
         'Data': 'ECHAM6',
         'Month': np.tile(month, int(len(wisoaprt_mean_over_ais[expid[i]]['mon']) / 12)),
         'pre_mon': wisoaprt_mean_over_ais[expid[i]]['mon'].values * 3600 * 24 * np.tile(month_days, int(len(wisoaprt_mean_over_ais[expid[i]]['mon']) / 12)),
         })],
        ignore_index=True,)

pre_mm_over_ais = pd.concat(
    [pre_mm_over_ais,
     pd.DataFrame(data={
         'Data': 'ERA5',
         'Month': np.tile(month, int(len(tp_era5_mean_over_ais['mon']) / 12)),
         'pre_mon': tp_era5_mean_over_ais['mon'].values * np.tile(month_days, int(len(tp_era5_mean_over_ais['mon']) / 12)),
         })],
        ignore_index=True,)

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.1_aprt_ann_circle/6.1.4.0.1 ' + expid[i] + ' era5 ann circle histogram over AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_bar = sns.barplot(
    data = pre_mm_over_ais,
    x = 'Month',
    y = 'pre_mon',
    hue = 'Data', hue_order = ['ERA5', 'ECHAM6'],
    palette=['tab:blue', 'tab:orange',],
    ci = 'sd', errwidth=0.75, capsize=0.1,
)
lgd_handles = [mpatches.Patch(color=x, label=y) for x, y in \
    zip(['tab:blue', 'tab:orange',], ['ERA5', 'ECHAM6 PI'])]

plt.legend(
    handles=lgd_handles,
    labels=['ERA5', 'ECHAM6 PI'],
    loc='upper right', handlelength=1, framealpha = 0.5, )

ax.set_xlabel('Monthly precipitation over AIS [$mm \; mon^{-1}$]')
ax.set_ylabel(None)
# ax.set_ylim(0, 20)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)



pre_mm_over_ais.loc[pre_mm_over_ais.Data == 'ECHAM6'].pre_mon.mean()
pre_mm_over_ais.loc[pre_mm_over_ais.Data == 'ERA5'].pre_mon.mean()

'''
np.max(abs(
    wisoaprt_mean_over_ais[expid[i]]['mm'] - \
        wisoaprt_mean_over_ais[expid[i]]['mon'].groupby('time.month').mean(skipna=True).compute()))

        #  'Month': month,
        #  'pre_mm': wisoaprt_mean_over_ais[expid[i]]['mm'] * 3600 * 24 * month_days,
        #  'pre_std': wisoaprt_mean_over_ais[expid[i]]['mon_std'] * 3600 * 24 * month_days,

        #  'Month': month,
        #  'pre_mm': tp_era5_mean_over_ais['mm'] * month_days,
        #  'pre_std': tp_era5_mean_over_ais['mon_std'] * month_days,
'''
# endregion
# -----------------------------------------------------------------------------



