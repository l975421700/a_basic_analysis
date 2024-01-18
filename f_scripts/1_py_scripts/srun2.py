

exp_odir = 'albedo_scratch/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    # 'nudged_701_5.0',
    
    # 'nudged_703_6.0_k52',
    
    # 'nudged_705_6.0',
    # 'nudged_707_6.0_k43',
    'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
    ]
i = 0

ifile_start = 0 #1380 #12 #0 #120
ifile_end   = 528 #1740 #516 #1740 #840

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
sys.path.append('/home/users/qino')
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
import xesmf as xe
import pandas as pd


from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    mean_over_ais,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
    filenames_wiso[ifile_start:ifile_end],
    )

'''
#-------- check pre
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
nc1 = xr.open_dataset(filenames_wiso[ifile])
nc2 = xr.open_dataset(filenames_echam[ifile])

np.max(abs(nc1.wisoaprl[:, 0].mean(dim='time').values - nc2.aprl[0].values))


#-------- input previous files

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso_daily'] = xr.open_mfdataset(filenames_wiso_daily, data_vars='minimal', coords='minimal', parallel=True)
        
        filenames_echam_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_echam.nc'))
        exp_org_o[expid[i]]['echam_daily'] = xr.open_mfdataset(filenames_echam_daily[120:], data_vars='minimal', coords='minimal', parallel=True)

'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region get mon/sea/ann wisoaprt

wisoaprt = {}
wisoaprt[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl[:, :3] + \
        exp_org_o[expid[i]]['wiso'].wisoaprc[:, :3].values).compute()

wisoaprt[expid[i]] = wisoaprt[expid[i]].rename('wisoaprt')

wisoaprt_alltime = {}
wisoaprt_alltime[expid[i]] = mon_sea_ann(var_6hourly=wisoaprt[expid[i]])


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'wb') as f:
    pickle.dump(wisoaprt_alltime[expid[i]], f)


'''
#-------- check calculation
wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[120:1080][ifile])

(wisoaprt_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values)).all()

(wisoaprt_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values).mean(dim='time')).all()


#-------- check simulation of aprt and wisoaprt
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:180], data_vars='minimal', coords='minimal', parallel=True)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl.values + \
    exp_org_o[expid[i]]['echam'].aprc.values - \
        wisoaprt_alltime[expid[i]]['mon'][:60, 0].values))

'''
# endregion
# -----------------------------------------------------------------------------

