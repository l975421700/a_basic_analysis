

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i=0

# -----------------------------------------------------------------------------
# region import packages

# management
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle

from a_basic_analysis.b_module.source_properties import (
    sincoslon_2_lon,
    calc_lon_diff,
)

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region estimate lon

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_sinlon.pkl', 'rb') as f:
    q_sfc_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_coslon.pkl', 'rb') as f:
    q_sfc_weighted_coslon = pickle.load(f)

q_sfc_weighted_lon = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    q_sfc_weighted_lon[ialltime] = sincoslon_2_lon(
        q_sfc_weighted_sinlon[ialltime], q_sfc_weighted_coslon[ialltime],
        var_name='q_sfc_weighted_lon',
    )

#-------- monthly without monthly mean
q_sfc_weighted_lon['mon no mm'] = calc_lon_diff(
    q_sfc_weighted_lon['mon'].groupby('time.month'),
    q_sfc_weighted_lon['mon'].groupby('time.month').mean(skipna=True),)

#-------- annual without annual mean
q_sfc_weighted_lon['ann no am'] = calc_lon_diff(
    q_sfc_weighted_lon['ann'],
    q_sfc_weighted_lon['ann'].mean(dim='time', skipna=True),)

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lon.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_sfc_weighted_lon, f)




'''
#-------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lon.pkl', 'rb') as f:
    q_sfc_weighted_lon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_sinlon.pkl', 'rb') as f:
    q_sfc_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_coslon.pkl', 'rb') as f:
    q_sfc_weighted_coslon = pickle.load(f)

ialltime = 'sea'
itime = 20
ilat = 50
ilon = 90

sinlon = q_sfc_weighted_sinlon[ialltime][itime, ilat, ilon].values
coslon = q_sfc_weighted_coslon[ialltime][itime, ilat, ilon].values
lon = q_sfc_weighted_lon[ialltime][itime, ilat, ilon].values

lon_new = (np.arctan2(sinlon, coslon) * 180 / np.pi)
if lon_new < 0:
    lon_new += 360

print(lon)
print(lon_new)
'''
# endregion
# -----------------------------------------------------------------------------



