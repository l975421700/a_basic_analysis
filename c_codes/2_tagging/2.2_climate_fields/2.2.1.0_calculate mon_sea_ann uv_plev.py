

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0
ifile_start = 120
ifile_end =   1080


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
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
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

filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))

exp_org_o[expid[i]]['uvq_plev'] = xr.open_mfdataset(
    filenames_uvq_plev[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann uv_plev


uv_plev = {}
uv_plev[expid[i]] = {}

uv_plev[expid[i]]['u'] = mon_sea_ann(
    var_monthly=exp_org_o[expid[i]]['uvq_plev'].u)
uv_plev[expid[i]]['v'] = mon_sea_ann(
    var_monthly=exp_org_o[expid[i]]['uvq_plev'].v)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl',
    'wb') as f:
    pickle.dump(uv_plev[expid[i]], f)


'''
#---------------- check mon

uv_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl',
    'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)

filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))

ifile = -1
ncfile = xr.open_dataset(filenames_uvq_plev[ifile_start:ifile_end][ifile])

np.max(abs(uv_plev[expid[i]]['u']['mon'][ifile] - ncfile.u[0])).values
np.max(abs(uv_plev[expid[i]]['v']['mon'][ifile] - ncfile.v[0])).values


#---------------- check others
test = {}
test['mon'] = uv_plev[expid[i]]['u']['mon']
test['sea'] = test['mon'].resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = test['mon'].resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(uv_plev[expid[i]]['u']['mon'].values[np.isfinite(uv_plev[expid[i]]['u']['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(uv_plev[expid[i]]['u']['sea'].values[np.isfinite(uv_plev[expid[i]]['u']['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(uv_plev[expid[i]]['u']['ann'].values[np.isfinite(uv_plev[expid[i]]['u']['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(uv_plev[expid[i]]['u']['mm'].values[np.isfinite(uv_plev[expid[i]]['u']['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(uv_plev[expid[i]]['u']['sm'].values[np.isfinite(uv_plev[expid[i]]['u']['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(uv_plev[expid[i]]['u']['am'].values[np.isfinite(uv_plev[expid[i]]['u']['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()

'''
# endregion
# -----------------------------------------------------------------------------




