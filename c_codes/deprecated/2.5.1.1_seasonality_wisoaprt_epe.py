

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
)

from a_basic_analysis.b_module.namelist import (
    month,
    monthini,
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

wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate and plot Monthly EPE days at each site

epe_days_mon = wisoaprt_epe[expid[i]]['mask']['90%'].resample(
    {'time': '1M'}).sum().compute()
epe_days_alltime = mon_sea_ann(var_monthly=epe_days_mon)


for isite in stations_sites.Site:
    # isite = 'EDC'
    print(isite)
    
    # plot
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1.0_seasonality/6.1.7.1.0 ' + expid[i] + ' monthly epe_days at ' + isite + '.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    ax.bar(
        x = np.arange(0, 12, 1),
        height = epe_days_alltime['mm'][
            :,
            t63_sites_indices[isite]['ilat'],
            t63_sites_indices[isite]['ilon']],
        color = 'lightgray',
        )
    
    plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='k',)
    
    ax.set_xlabel('Monthly EPE days [$\#$]')
    ax.set_xticks(np.arange(0, 12, 1))
    ax.set_xticklabels(monthini)
    ax.set_xlim(-0.5, 11.5)
    
    ax.set_ylabel(None)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    
    ax.grid(True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.25, top=0.98)
    fig.savefig(output_png)









'''
epe_days_alltime['am'][
    t63_sites_indices[isite]['ilat'],
    t63_sites_indices[isite]['ilon']] * 12

ilat = 48
ilon = 90
epe_days_alltime['mm'][:, ilat, ilon]
epe_days_mon[:, ilat, ilon].groupby('time.month').mean().compute()
'''

# endregion
# -----------------------------------------------------------------------------

