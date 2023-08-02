

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress
from scipy.stats import pearsonr

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
    monthini,
    plot_labels,
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
    xr_par_cor,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import Dome C records and clean it

bs13_dc_records = {}

bs13_dc_records['daily'] = pd.read_excel(
    'data_sources/Dome_C_records/tc-10-2415-2016-supplement.xlsx',
    header=0, skiprows=6,)

# clean it
bs13_dc_records['daily']['date'] = pd.to_datetime(
    bs13_dc_records['daily'][['YEAR', 'MONTH', 'DAY', ]])

bs13_dc_records['daily'] = bs13_dc_records['daily'][
    bs13_dc_records['daily']['YEAR'] > 2007]

bs13_dc_records['daily'] = bs13_dc_records['daily'].rename(columns={
    'T_AWS(C)': 'temp2',
    'P_AWS(hPa)': 'pressure',
    'WS_AWS(m/s)': 'wind_speed',
    'DIR_AWS(°)': 'wind_direction',
    'd18O': 'dO18',
    'deuterium excess': 'd_excess',
    'TOTAL(mm_w.e)': 'wisoaprt',
})

bs13_dc_records['daily'] = bs13_dc_records['daily'][[
    'temp2', 'pressure', 'wind_speed', 'wind_direction',
    'dD', 'dO18', 'd_excess', 'wisoaprt', 'date']]

ln_dD = 1000 * np.log(1 + bs13_dc_records['daily']['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + bs13_dc_records['daily']['dO18'] / 1000)

bs13_dc_records['daily']['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


bs13_dc_records['mon'] = bs13_dc_records['daily'].resample(
    'M', on='date').mean().reset_index()

bs13_dc_records['mm'] = bs13_dc_records['daily'].groupby(
    bs13_dc_records['daily']['date'].dt.month).mean()

bs13_dc_records['am'] = bs13_dc_records['daily'].mean()


'''
bs13_dc_records['daily'].columns
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot data

marker='o'
linewidth=1

for ialltime in ['daily', 'mon', 'mm', ]:
    # ialltime = 'daily'
    print(ialltime)
    
    for ivar in ['temp2', 'pressure', 'wind_speed', 'dD', 'dO18', 'd_excess', 'wisoaprt', 'd_ln']:
        # ivar = 'temp2'
        # ['temp2', 'pressure', 'wind_speed', 'dD', 'dO18', 'd_excess', 'wisoaprt', 'd_ln']
        print(ivar)
        
        if (ialltime == 'daily'):
            rotation = 45
            xticks = pd.date_range(start='2008-01', end='2011-06', freq='6M',)
            xlabels = (str(date)[:7] for date in xticks)
            markersize=0
        elif (ialltime == 'mon'):
            rotation = 45
            xticks = pd.date_range(start='2008-01', end='2011-06', freq='6M',)
            xlabels = (str(date)[:7] for date in xticks)
            markersize=3
        elif (ialltime == 'mm'):
            rotation = 0
            xticks = bs13_dc_records['mm']['date'].values
            xlabels = monthini
            markersize=3
        
        output_png = 'figures/8_d-excess/8.0_records/8.0.4_dome_c/8.0.4.0.0 Dome C ' + ialltime + ' record of ' + ivar + '.png'
        
        fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
        
        ax.plot(
            bs13_dc_records[ialltime]['date'].values,
            bs13_dc_records[ialltime][ivar].values,
            marker=marker, linewidth=linewidth, markersize=markersize,
        )
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        plt.xticks(rotation=rotation)
        
        ax.set_ylabel('Observed Dome C ' + plot_labels[ivar], labelpad=6)
        
        ax.grid(True, which='both',
                linewidth=0.4, color='gray', alpha=0.75, linestyle=':')
        
        fig.subplots_adjust(left=0.2, right=0.92, bottom=0.18, top=0.95)
        fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check some statistics

#--------
# 1096
No_days = len(bs13_dc_records['daily']['wisoaprt'])
# 600
No_days_pre = (bs13_dc_records['daily']['wisoaprt'] > 0).sum()
# 494
No_days_wiso = np.isfinite(bs13_dc_records['daily']['dD']).sum()
# 357
No_days_pre_wiso = ((bs13_dc_records['daily']['wisoaprt'] > 0) & np.isfinite(bs13_dc_records['daily']['dD'])).sum()




#-------- annual mean and pre-weighted temp2 are similar (diff: 0.4)

subset = np.isfinite(bs13_dc_records['daily']['temp2']) & np.isfinite(bs13_dc_records['daily']['wisoaprt'])
print(np.average(
    bs13_dc_records['daily']['temp2'][subset],
    weights = bs13_dc_records['daily']['wisoaprt'][subset],
    ))
print(bs13_dc_records['am']['temp2'])




#-------- annual mean and pre-weighted dD (diff: 33)

subset = np.isfinite(bs13_dc_records['daily']['dD']) & np.isfinite(bs13_dc_records['daily']['wisoaprt'])
print(np.average(
    bs13_dc_records['daily']['dD'][subset],
    weights = bs13_dc_records['daily']['wisoaprt'][subset],
    ))
print(bs13_dc_records['am']['dD'])




#-------- annual mean and pre-weighted dO18 (diff: 5)

subset = np.isfinite(bs13_dc_records['daily']['dO18']) & np.isfinite(bs13_dc_records['daily']['wisoaprt'])
print(np.average(
    bs13_dc_records['daily']['dO18'][subset],
    weights = bs13_dc_records['daily']['wisoaprt'][subset],
    ))
print(bs13_dc_records['am']['dO18'])



subset = np.isfinite(bs13_dc_records['daily']['dD']) & np.isfinite(bs13_dc_records['daily']['temp2'])

np.round(pearsonr(
    bs13_dc_records['daily']['dD'][subset],
    bs13_dc_records['daily']['temp2'][subset],
).statistic ** 2, 3)



fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)

ax.scatter(
    bs13_dc_records['daily']['temp2'][subset].values,
    bs13_dc_records['daily']['dD'][subset].values,
)

fig.subplots_adjust(left=0.2, right=0.92, bottom=0.18, top=0.95)
fig.savefig('figures/test/test.png')


'''
(np.isfinite(bs13_dc_records['daily']['dD']) != np.isfinite(bs13_dc_records['daily']['dO18'])).sum()
'''
# endregion
# -----------------------------------------------------------------------------


