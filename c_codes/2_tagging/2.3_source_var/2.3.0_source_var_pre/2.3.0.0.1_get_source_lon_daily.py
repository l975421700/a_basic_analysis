

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    'pi_601_5.1',
    ]
i=0

output_dir = exp_odir + expid[i] + '/analysis/echam/'

# -----------------------------------------------------------------------------
# region import packages

# management
import warnings
warnings.filterwarnings('ignore')
import os

# data analysis
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
# region import data

with open(output_dir + expid[i] + '.pre_weighted_sinlon.pkl', 'rb') as f:
    pre_weighted_sinlon = pickle.load(f)

with open(output_dir + expid[i] + '.pre_weighted_coslon.pkl', 'rb') as f:
    pre_weighted_coslon = pickle.load(f)

pre_weighted_lon = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    pre_weighted_lon[ialltime] = sincoslon_2_lon(
        pre_weighted_sinlon[ialltime], pre_weighted_coslon[ialltime]
    )

#-------- monthly without monthly mean
pre_weighted_lon['mon no mm'] = calc_lon_diff(
    pre_weighted_lon['mon'].groupby('time.month'),
    pre_weighted_lon['mon'].groupby('time.month').mean(skipna=True),)

#-------- annual without annual mean
pre_weighted_lon['ann no am'] = calc_lon_diff(
    pre_weighted_lon['ann'],
    pre_weighted_lon['ann'].mean(dim='time', skipna=True),)

output_file = output_dir + expid[i] + '.pre_weighted_lon.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(pre_weighted_lon, f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region copy output

import shutil

# src_exp = 'pi_600_5.0'
src_exp = 'pi_601_5.1'

expid = [
    'pi_602_5.2',
    'pi_605_5.5',
    'pi_606_5.6',
    'pi_609_5.7',
    # 'pi_610_5.8',
    ]

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    
    input_file = exp_odir + src_exp + '/analysis/echam/' + src_exp + '.pre_weighted_lon.pkl'
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    shutil.copy2(input_file, output_file)

# endregion
# -----------------------------------------------------------------------------
