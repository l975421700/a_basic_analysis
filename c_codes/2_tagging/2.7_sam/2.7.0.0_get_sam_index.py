

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    'hist_700_5.0',
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

from a_basic_analysis.b_module.basic_calculations import (
    get_mon_sam,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon sam

lat = psl_zh[expid[i]]['psl']['mon'].lat
mslp = psl_zh[expid[i]]['psl']['mon']

sam_index = get_mon_sam(lat, mslp)

sam_mon = xr.Dataset(
    {'sam': (('time'), sam_index),},
    coords={'time': temp2_alltime[expid[i]]['mon'].time,},
    )

sam_mon.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')

'''
# Write output file
d = {}
d['time'] = psl_zh[expid[i]]['psl']['mon'].time
d['sam'] = (['time'], sam_index)
sam_mon = xr.Dataset(d)
sam_mon.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')


sam_mon = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')
sam_mon.sam.values
'''
# endregion
# -----------------------------------------------------------------------------


