

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'hist_700_5.0',
    # 'nudged_701_5.0',
    'nudged_703_6.0_k52',
    ]
i=0


ifile_start = 0 #0 #120
ifile_end   = 528 #1740 #840


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
)

from a_basic_analysis.b_module.namelist import (
    zerok,
)


# endregion
# -----------------------------------------------------------------------------


# Monthly
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(
    filenames_echam[ifile_start:ifile_end],
    )


'''
https://stackoverflow.com/questions/56590075/xarray-open-mfdataset-for-a-small-subset-of-variables
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region

tsurf_alltime = {}
tsurf_alltime[expid[i]] = mon_sea_ann(var_monthly=(
    exp_org_o[expid[i]]['echam'].tsurf - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'wb') as f:
    pickle.dump(tsurf_alltime[expid[i]], f)


'''
tsurf_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'rb') as f:
    tsurf_alltime[expid[i]] = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# Daily
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_g3b_6h = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_g3b_6h.nc'))

exp_org_o[expid[i]]['g3b_6h'] = xr.open_mfdataset(
    filenames_g3b_6h[ifile_start:ifile_end],
    )

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann tsurf

tsurf_alltime = {}
tsurf_alltime[expid[i]] = mon_sea_ann(var_6hourly=(
    exp_org_o[expid[i]]['g3b_6h'].tsurf - zerok).compute())

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'wb') as f:
    pickle.dump(tsurf_alltime[expid[i]], f)


'''
#-------------------------------- check
tsurf_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsurf_alltime.pkl', 'rb') as f:
    tsurf_alltime[expid[i]] = pickle.load(f)

test = {}
test['daily'] = (exp_org_o[expid[i]]['g3b_1d'].tsurf - zerok).compute().copy()
test['mon'] = (exp_org_o[expid[i]]['g3b_1d'].tsurf - zerok).resample({'time': '1M'}).mean(skipna=False).compute()
test['sea'] = (test['mon']).resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = (test['mon']).resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(tsurf_alltime[expid[i]]['daily'].values[np.isfinite(tsurf_alltime[expid[i]]['daily'].values)] == test['daily'].values[np.isfinite(test['daily'].values)]).all()
(tsurf_alltime[expid[i]]['mon'].values[np.isfinite(tsurf_alltime[expid[i]]['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(tsurf_alltime[expid[i]]['sea'].values[np.isfinite(tsurf_alltime[expid[i]]['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(tsurf_alltime[expid[i]]['ann'].values[np.isfinite(tsurf_alltime[expid[i]]['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(tsurf_alltime[expid[i]]['mm'].values[np.isfinite(tsurf_alltime[expid[i]]['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(tsurf_alltime[expid[i]]['sm'].values[np.isfinite(tsurf_alltime[expid[i]]['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(tsurf_alltime[expid[i]]['am'].values[np.isfinite(tsurf_alltime[expid[i]]['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()


'''
# endregion
# -----------------------------------------------------------------------------


