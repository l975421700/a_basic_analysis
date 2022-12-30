

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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

quantile_interval  = np.arange(1, 99 + 1e-4, 1, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))


'''
lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

# quantiles = {'90%': 0.9, '95%': 0.95, '99%': 0.99}

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt_epe_nt

wisoaprt_epe_nt = {}
wisoaprt_epe_nt[expid[i]] = {}
wisoaprt_epe_nt[expid[i]]['quantiles'] = {}
wisoaprt_epe_nt[expid[i]]['mask'] = {}
wisoaprt_epe_nt[expid[i]]['masked_data'] = {}

# set a threshold of 0 mm/d
wisoaprt_epe_nt[expid[i]]['masked_data']['original'] = \
    wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy().compute()

for iqtl in quantiles.keys():
    # iqtl = '90%'
    print(iqtl + ': ' + str(quantiles[iqtl]))
    
    #-------- calculate quantiles
    wisoaprt_epe_nt[expid[i]]['quantiles'][iqtl] = \
        wisoaprt_epe_nt[expid[i]]['masked_data']['original'].quantile(
            quantiles[iqtl], dim='time', skipna=True).compute()
    
    #-------- get mask
    wisoaprt_epe_nt[expid[i]]['mask'][iqtl] = \
        (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy() >= \
            wisoaprt_epe_nt[expid[i]]['quantiles'][iqtl]).compute()

import os, psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 2**30)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_nt.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_epe_nt[expid[i]], f)


'''
# 21 min to run
#SBATCH --time=00:30:00
#SBATCH --partition=fat

#-------------------------------- check
wisoaprt_epe_nt = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_nt.pkl',
    'rb') as f:
    wisoaprt_epe_nt[expid[i]] = pickle.load(f)

ilat=48
ilon=90

#-------- check ['masked_data']['original']
res001 = wisoaprt_epe_nt[expid[i]]['masked_data']['original'][:, ilat, ilon]
res002 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].copy().where(
    wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= (0 / seconds_per_d),
    other=np.nan,)
print((res001[np.isfinite(res001)] == res002[np.isfinite(res002)]).all().values)

for iqtl in quantiles.keys():
    print('#-------- ' + iqtl)
    # iqtl = '90%'
    #-------- check ['quantiles'][iqtl]
    res01 = wisoaprt_epe_nt[expid[i]]['quantiles'][iqtl][ilat, ilon].values
    res02 = np.nanquantile(res002, quantiles[iqtl],)
    print(res01 == res02)
    #-------- check ['mask'][iqtl]
    res11 = wisoaprt_epe_nt[expid[i]]['mask'][iqtl][:, ilat, ilon]
    res12 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= res02
    print((res11 == res12).all().values)




#-------------------------------- check size information
wisoaprt_epe_nt = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe_nt.pkl',
    'rb') as f:
    wisoaprt_epe_nt[expid[i]] = pickle.load(f)

from pympler import asizeof
asizeof.asizeof(wisoaprt_epe_nt) / 2**30
asizeof.asizeof(wisoaprt_epe_nt[expid[i]]['quantiles']) / 2**30
asizeof.asizeof(wisoaprt_epe_nt[expid[i]]['mask']) / 2**30
asizeof.asizeof(wisoaprt_epe_nt[expid[i]]['frc_aprt']) / 2**30
asizeof.asizeof(wisoaprt_alltime) / 2**30


'''
# endregion
# -----------------------------------------------------------------------------

