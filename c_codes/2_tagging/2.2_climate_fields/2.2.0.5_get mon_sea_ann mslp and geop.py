

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0
ifile_start = 0
ifile_end   = 600

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
    plot_maxmin_points,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_psl_zh = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_psl_zh.nc'))

exp_org_o[expid[i]]['psl_zh'] = xr.open_mfdataset(
    filenames_psl_zh[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann psl and gh

psl_zh = {}
psl_zh[expid[i]] = {}

psl_zh[expid[i]]['psl'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['psl_zh'].psl)
psl_zh[expid[i]]['zh'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['psl_zh'].zh)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'wb') as f:
    pickle.dump(psl_zh[expid[i]], f)


'''
#-------------------------------- check
psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

# calculate manually
def time_weighted_mean(ds):
    return ds.weighted(ds.time.dt.days_in_month).mean('time', skipna=False)

test = {}
test['mon'] = exp_org_o[expid[i]]['psl_zh'].psl.copy()
test['sea'] = exp_org_o[expid[i]]['psl_zh'].psl.resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = exp_org_o[expid[i]]['psl_zh'].psl.resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()



(psl_zh[expid[i]]['psl']['mon'].values[np.isfinite(psl_zh[expid[i]]['psl']['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(psl_zh[expid[i]]['psl']['sea'].values[np.isfinite(psl_zh[expid[i]]['psl']['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(psl_zh[expid[i]]['psl']['ann'].values[np.isfinite(psl_zh[expid[i]]['psl']['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(psl_zh[expid[i]]['psl']['mm'].values[np.isfinite(psl_zh[expid[i]]['psl']['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(psl_zh[expid[i]]['psl']['sm'].values[np.isfinite(psl_zh[expid[i]]['psl']['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(psl_zh[expid[i]]['psl']['am'].values[np.isfinite(psl_zh[expid[i]]['psl']['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------





