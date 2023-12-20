

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
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

filenames_wiso_q_daily = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_daily.nc'))
exp_org_o[expid[i]]['wiso_q_daily'] = xr.open_mfdataset(
    filenames_wiso_q_daily[ifile_start:ifile_end],
    chunks={'time': 10}
    )


'''
exp_org_o[expid[i]]['wiso_q_daily']
'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --mem=240GB
# -----------------------------------------------------------------------------
# region get mon_sea_ann q16o, q18o, and qhdo

wiso_q_daily = {}
wiso_q_daily[expid[i]] = {}

wiso_q_daily[expid[i]]['q16o'] = (exp_org_o[expid[i]]['wiso_q_daily']['q16o'] + exp_org_o[expid[i]]['wiso_q_daily']['xl16o'] + exp_org_o[expid[i]]['wiso_q_daily']['xi16o']).compute()

wiso_q_daily[expid[i]]['q18o'] = (exp_org_o[expid[i]]['wiso_q_daily']['q18o'] + exp_org_o[expid[i]]['wiso_q_daily']['xl18o'] + exp_org_o[expid[i]]['wiso_q_daily']['xi18o']).compute()

wiso_q_daily[expid[i]]['qhdo'] = (exp_org_o[expid[i]]['wiso_q_daily']['qhdo'] + exp_org_o[expid[i]]['wiso_q_daily']['xlhdo'] + exp_org_o[expid[i]]['wiso_q_daily']['xihdo']).compute()

print(psutil.Process().memory_info().rss / (2 ** 30))

del exp_org_o

print(psutil.Process().memory_info().rss / (2 ** 30))

wiso_q_daily_alltime = {}
wiso_q_daily_alltime[expid[i]] = {}

wiso_q_daily_alltime[expid[i]]['q16o'] = mon_sea_ann(
    wiso_q_daily[expid[i]]['q16o'], lcopy=False)

wiso_q_daily_alltime[expid[i]]['q18o'] = mon_sea_ann(
    wiso_q_daily[expid[i]]['q18o'], lcopy=False)

wiso_q_daily_alltime[expid[i]]['qhdo'] = mon_sea_ann(
    wiso_q_daily[expid[i]]['qhdo'], lcopy=False)

print(psutil.Process().memory_info().rss / (2 ** 30))

del wiso_q_daily

print(psutil.Process().memory_info().rss / (2 ** 30))

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_daily_alltime.pkl', 'wb') as f:
    pickle.dump(wiso_q_daily_alltime[expid[i]], f)



'''
wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------
