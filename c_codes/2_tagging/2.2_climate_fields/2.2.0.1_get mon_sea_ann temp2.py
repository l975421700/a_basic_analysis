

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0
ifile_start = 120
ifile_end   = 720

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
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_g3b_1m = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_g3b_1m.nc'))

exp_org_o[expid[i]]['g3b_1m'] = xr.open_mfdataset(
    filenames_g3b_1m[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann temp2

temp2_alltime = {}
temp2_alltime[expid[i]] = mon_sea_ann(var_monthly=(
    exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'wb') as f:
    pickle.dump(temp2_alltime[expid[i]], f)


'''
#-------------------------------- check
temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)

test = {}
test['mon'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute().copy()
test['sea'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(temp2_alltime[expid[i]]['mon'].values[np.isfinite(temp2_alltime[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(temp2_alltime[expid[i]]['sea'].values[np.isfinite(temp2_alltime[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(temp2_alltime[expid[i]]['ann'].values[np.isfinite(temp2_alltime[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(temp2_alltime[expid[i]]['mm'].values[np.isfinite(temp2_alltime[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(temp2_alltime[expid[i]]['sm'].values[np.isfinite(temp2_alltime[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(temp2_alltime[expid[i]]['am'].values[np.isfinite(temp2_alltime[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------



