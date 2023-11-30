

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    # 'nudged_712_6.0_k52_2yr',
    # 'nudged_713_6.0_2yr',
    # 'nudged_714_6.0_k52_88_2yr',
    # 'nudged_715_6.0_k43_2yr',
    # 'nudged_716_6.0_I01_2yr',
    # 'nudged_717_6.0_I03_2yr',
    # 'nudged_718_6.0_S3_2yr',
    # 'nudged_719_6.0_S6_2yr',
    
    'nudged_705_6.0',
    # 'nudged_706_6.0_k52_88',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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


# -----------------------------------------------------------------------------
# region import data

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)


# setting
VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann dO18

dO18_q_sfc_alltime = {}
dO18_q_sfc_alltime[expid[i]] = {}

for ialltime in ['6h', 'daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = '6h'
    print(ialltime)
    
    dO18_q_sfc_alltime[expid[i]][ialltime] = \
        (((wiso_q_6h_sfc_alltime[expid[i]]['q18o'][ialltime].sel(lev=47) / \
            wiso_q_6h_sfc_alltime[expid[i]]['q16o'][ialltime].sel(lev=47)) / \
                VSMOW_O18 - 1) * 1000).compute()
    
    dO18_q_sfc_alltime[expid[i]][ialltime] = \
        dO18_q_sfc_alltime[expid[i]][ialltime].rename('dO18')

#-------- monthly without monthly mean
dO18_q_sfc_alltime[expid[i]]['mon no mm'] = (dO18_q_sfc_alltime[expid[i]]['mon'].groupby('time.month') - dO18_q_sfc_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
dO18_q_sfc_alltime[expid[i]]['ann no am'] = (dO18_q_sfc_alltime[expid[i]]['ann'] - dO18_q_sfc_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(dO18_q_sfc_alltime[expid[i]], f)


'''
dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann dD

dD_q_sfc_alltime = {}
dD_q_sfc_alltime[expid[i]] = {}

for ialltime in ['6h', 'daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'am'
    print(ialltime)
    
    dD_q_sfc_alltime[expid[i]][ialltime] = \
        (((wiso_q_6h_sfc_alltime[expid[i]]['qhdo'][ialltime].sel(lev=47) / \
            wiso_q_6h_sfc_alltime[expid[i]]['q16o'][ialltime].sel(lev=47)) / \
                VSMOW_D - 1) * 1000).compute()
    
    dD_q_sfc_alltime[expid[i]][ialltime] = \
        dD_q_sfc_alltime[expid[i]][ialltime].rename('dD')

#-------- monthly without monthly mean
dD_q_sfc_alltime[expid[i]]['mon no mm'] = (dD_q_sfc_alltime[expid[i]]['mon'].groupby('time.month') - dD_q_sfc_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
dD_q_sfc_alltime[expid[i]]['ann no am'] = (dD_q_sfc_alltime[expid[i]]['ann'] - dD_q_sfc_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(dD_q_sfc_alltime[expid[i]], f)




'''
dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------

