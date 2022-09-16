

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'rb') as f:
    aprt_frc_AIS_alltime = pickle.load(f)


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

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

sns.barplot(
    data = pre_mm_over_ais,
    x = 'Month',
    y = 'pre_mon',
    hue = 'Data', hue_order = ['ERA5', 'ECHAM6'],
    palette=['tab:blue', 'tab:orange',],
    ci = 'sd', errwidth=0.75, capsize=0.1,
)
plt.legend(loc='upper right', handlelength=1, framealpha = 0.5, )

ax.set_xlabel('Monthly precipitation over AIS [$mm \; mon^{-1}$]')
ax.set_ylabel(None)
# ax.set_ylim(0, 20)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)





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


# -----------------------------------------------------------------------------
# region histogram: annual circle of aprt frac over AIS

# imask = 'AIS'
imask = 'EAIS'

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.1_aprt_ann_circle/6.1.4.0.1 ' + expid[i] + ' ann circle aprt frc over ' + imask + '.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
lgd_handles = []
colors = ['royalblue', 'darkblue', 'deepskyblue', 'lightblue', ]
regions = ['Antarctica', 'Land excl. Antarctica', 'SH sea ice', 'Open ocean']


for count, iregion in enumerate(regions):
    print(str(count) + ': ' + iregion)
    
    sns.barplot(
        x = month, y = aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS,
        color=colors[count],
    )
    lgd_handles += [mpatches.Patch(color=colors[count], label=iregion)]

change_snsbar_width(ax, .7)

plt.legend(
    handles=lgd_handles,
    loc='lower right', handlelength=1, framealpha = 1, )

ax.set_xlabel('Fraction of precipitation over ' + imask + ' from each region [$\%$]')
ax.set_ylabel(None)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 100+1e-4, 10))

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.98)
fig.savefig(output_png)


#-------------------------------- ann+std

imask = 'AIS'

#---- Antarctica: 1.0% ± 0.07%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['Land excl. Antarctica'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- other land: 4.2% ± 0.28%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['Land excl. Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['SH sea ice'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- SH sea ice: 11.5% ± 0.72%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Open ocean'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['SH sea ice'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['Open ocean'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- SH sea ice: 83.3% ± 0.80%

ann_values = aprt_frc_AIS_alltime[imask]['ann']['Open ocean'].frc_AIS.values
am_values = aprt_frc_AIS_alltime[imask]['am']['Open ocean'].frc_AIS.values
# ann_values.mean()
am_values
ann_values.std()






'''

https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot

min(aprt_frc_AIS['AIS']['Open ocean'].frc_AIS)
'''
# endregion
# -----------------------------------------------------------------------------

