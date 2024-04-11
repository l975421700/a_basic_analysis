

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --partition=fat --nodes=1 --mem=256GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
    ]
i = 0

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
import psutil

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import pandas as pd

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_daily_uv = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.daily_uv.nc'))

exp_org_o[expid[i]]['daily_uv'] = xr.open_mfdataset(
    filenames_daily_uv[ifile_start:ifile_end],
)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann daily uv

daily_uv_ml_k_alltime = {}
daily_uv_ml_k_alltime[expid[i]] = {}

daily_uv_ml_k_alltime[expid[i]]['u'] = mon_sea_ann(exp_org_o[expid[i]]['daily_uv'].u.sel(lev=47))

daily_uv_ml_k_alltime[expid[i]]['v'] = mon_sea_ann(exp_org_o[expid[i]]['daily_uv'].v.sel(lev=47))

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.daily_uv_ml_k_alltime.pkl', 'wb') as f:
    pickle.dump(daily_uv_ml_k_alltime[expid[i]], f)



'''
# check
daily_uv_ml_k_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.daily_uv_ml_k_alltime.pkl', 'rb') as f:
    daily_uv_ml_k_alltime[expid[i]] = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------
