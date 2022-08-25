

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
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec_num,
    month_dec,
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
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]

# region import output

i = 0
expid[i]

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_g3b_1m = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_g3b_1m.nc'))

exp_org_o[expid[i]]['g3b_1m'] = xr.open_mfdataset(filenames_g3b_1m[120:], data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann temp2

i = 0
expid[i]

temp2 = {}
temp2[expid[i]] = mon_sea_ann(var_monthly=(
    exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2.pkl', 'wb') as f:
    pickle.dump(temp2[expid[i]], f)




'''
#-------- check
i = 0
expid[i]
temp2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2.pkl', 'rb') as f:
    temp2[expid[i]] = pickle.load(f)


test = {}
test['mon'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute().copy()
test['sea'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(temp2[expid[i]]['mon'].values[np.isfinite(temp2[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(temp2[expid[i]]['sea'].values[np.isfinite(temp2[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(temp2[expid[i]]['ann'].values[np.isfinite(temp2[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(temp2[expid[i]]['mm'].values[np.isfinite(temp2[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(temp2[expid[i]]['sm'].values[np.isfinite(temp2[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(temp2[expid[i]]['am'].values[np.isfinite(temp2[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am_sm temp2, psl, and 850hPa wind

i = 0
expid[i]

temp2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2.pkl', 'rb') as f:
    temp2[expid[i]] = pickle.load(f)

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

uv_plev = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl', 'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)





'''
stats.describe(temp2[expid[i]]['am'].sel(lat=slice(-45, -90)), axis=None,
               nan_policy='omit') # -60 - 10
'''
# endregion
# -----------------------------------------------------------------------------


