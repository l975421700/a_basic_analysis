

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    # 'nudged_701_5.0',
    
    'nudged_703_6.0_k52',
    ]
i = 0


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

# endregion
# -----------------------------------------------------------------------------


# Daily
# -----------------------------------------------------------------------------
# region import data

wiso_q_daily_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_daily_alltime.pkl', 'rb') as f:
    wiso_q_daily_alltime[expid[i]] = pickle.load(f)

# setting
VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann dO18

dO18_q_alltime = {}
dO18_q_alltime[expid[i]] = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'am'
    print(ialltime)
    
    dO18_q_alltime[expid[i]][ialltime] = \
        (((wiso_q_daily_alltime[expid[i]]['q18o'][ialltime] / \
            wiso_q_daily_alltime[expid[i]]['q16o'][ialltime]) / \
                VSMOW_O18 - 1) * 1000).compute()
    
    dO18_q_alltime[expid[i]][ialltime] = \
        dO18_q_alltime[expid[i]][ialltime].rename('dO18')

#-------- monthly without monthly mean
dO18_q_alltime[expid[i]]['mon no mm'] = (dO18_q_alltime[expid[i]]['mon'].groupby('time.month') - dO18_q_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
dO18_q_alltime[expid[i]]['ann no am'] = (dO18_q_alltime[expid[i]]['ann'] - dO18_q_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(dO18_q_alltime[expid[i]], f)


'''
dO18_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl', 'rb') as f:
    dO18_q_alltime[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------




