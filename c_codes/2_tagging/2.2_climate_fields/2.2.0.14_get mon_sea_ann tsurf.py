

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
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


