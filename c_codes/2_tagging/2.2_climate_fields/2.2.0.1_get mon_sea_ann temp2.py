

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    'nudged_701_5.0',
    ]
i=0

ifile_start = 12 #0 #120
ifile_end   = 516 #1740 #840

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


# monthly
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
# region calculate mon_sea_ann temp2

temp2_alltime2 = {}
temp2_alltime2[expid[i]] = mon_sea_ann(var_monthly=(
    exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime2.pkl', 'wb') as f:
    pickle.dump(temp2_alltime2[expid[i]], f)


'''
#-------------------------------- check
temp2_alltime2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime2.pkl', 'rb') as f:
    temp2_alltime2[expid[i]] = pickle.load(f)

test = {}
test['mon'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).compute().copy()
test['sea'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (exp_org_o[expid[i]]['g3b_1m'].temp2 - zerok).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(temp2_alltime2[expid[i]]['mon'].values[np.isfinite(temp2_alltime2[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(temp2_alltime2[expid[i]]['sea'].values[np.isfinite(temp2_alltime2[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(temp2_alltime2[expid[i]]['ann'].values[np.isfinite(temp2_alltime2[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(temp2_alltime2[expid[i]]['mm'].values[np.isfinite(temp2_alltime2[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(temp2_alltime2[expid[i]]['sm'].values[np.isfinite(temp2_alltime2[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(temp2_alltime2[expid[i]]['am'].values[np.isfinite(temp2_alltime2[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------


# daily
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_g3b_1d = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_g3b_1d.nc'))

exp_org_o[expid[i]]['g3b_1d'] = xr.open_mfdataset(
    filenames_g3b_1d[ifile_start:ifile_end],
    )

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann temp2

temp2_alltime = {}
temp2_alltime[expid[i]] = mon_sea_ann((
    exp_org_o[expid[i]]['g3b_1d'].temp2 - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'wb') as f:
    pickle.dump(temp2_alltime[expid[i]], f)


'''
#-------------------------------- check
temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)

test = {}
test['daily'] = (exp_org_o[expid[i]]['g3b_1d'].temp2 - zerok).compute().copy()
test['mon'] = (exp_org_o[expid[i]]['g3b_1d'].temp2 - zerok).resample({'time': '1M'}).mean(skipna=False).compute()
test['sea'] = (test['mon']).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (test['mon']).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(temp2_alltime[expid[i]]['daily'].values[np.isfinite(temp2_alltime[expid[i]]['daily'].values)] == test['daily'].values[np.isfinite(test['daily'].values)]).all()
(temp2_alltime[expid[i]]['mon'].values[np.isfinite(temp2_alltime[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(temp2_alltime[expid[i]]['sea'].values[np.isfinite(temp2_alltime[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(temp2_alltime[expid[i]]['ann'].values[np.isfinite(temp2_alltime[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(temp2_alltime[expid[i]]['mm'].values[np.isfinite(temp2_alltime[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(temp2_alltime[expid[i]]['sm'].values[np.isfinite(temp2_alltime[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(temp2_alltime[expid[i]]['am'].values[np.isfinite(temp2_alltime[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check two calculations

temp2_alltime2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime2.pkl', 'rb') as f:
    temp2_alltime2[expid[i]] = pickle.load(f)

temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)

for ialltime in temp2_alltime2[expid[i]].keys():
    print(ialltime)
    
    print(np.max(abs(temp2_alltime2[expid[i]][ialltime].values - temp2_alltime[expid[i]][ialltime].values)))

# endregion
# -----------------------------------------------------------------------------

