

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i = 0
ifile_start = 12
ifile_end   = 516

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

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    zerok,
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
    # data_vars='minimal', coords='minimal', parallel=True,
    )

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann tsw

tsw_alltime = {}
tsw_alltime[expid[i]] = mon_sea_ann(var_monthly=(
    exp_org_o[expid[i]]['g3b_1m'].tsw - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'wb') as f:
    pickle.dump(tsw_alltime[expid[i]], f)


'''
#-------------------------------- check
tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

test = {}
test['mon'] = (exp_org_o[expid[i]]['g3b_1m'].tsw - zerok).compute().copy()
test['sea'] = (exp_org_o[expid[i]]['g3b_1m'].tsw - zerok).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (exp_org_o[expid[i]]['g3b_1m'].tsw - zerok).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

print((tsw_alltime[expid[i]]['mon'].values[np.isfinite(tsw_alltime[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all())
print((tsw_alltime[expid[i]]['sea'].values[np.isfinite(tsw_alltime[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all())
print((tsw_alltime[expid[i]]['ann'].values[np.isfinite(tsw_alltime[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all())
print((tsw_alltime[expid[i]]['mm'].values[np.isfinite(tsw_alltime[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all())
print((tsw_alltime[expid[i]]['sm'].values[np.isfinite(tsw_alltime[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all())
print((tsw_alltime[expid[i]]['am'].values[np.isfinite(tsw_alltime[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann seaice

seaice_alltime = {}
seaice_alltime[expid[i]] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['g3b_1m'].seaice)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.seaice_alltime.pkl', 'wb') as f:
    pickle.dump(seaice_alltime[expid[i]], f)


'''
#-------------------------------- check
seaice_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.seaice_alltime.pkl', 'rb') as f:
    seaice_alltime[expid[i]] = pickle.load(f)

test = {}
test['mon'] = (exp_org_o[expid[i]]['g3b_1m'].seaice).compute().copy()
test['sea'] = (exp_org_o[expid[i]]['g3b_1m'].seaice).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (exp_org_o[expid[i]]['g3b_1m'].seaice).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(seaice_alltime[expid[i]]['mon'].values[np.isfinite(seaice_alltime[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(seaice_alltime[expid[i]]['sea'].values[np.isfinite(seaice_alltime[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(seaice_alltime[expid[i]]['ann'].values[np.isfinite(seaice_alltime[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(seaice_alltime[expid[i]]['mm'].values[np.isfinite(seaice_alltime[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(seaice_alltime[expid[i]]['sm'].values[np.isfinite(seaice_alltime[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(seaice_alltime[expid[i]]['am'].values[np.isfinite(seaice_alltime[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------

