

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    sincoslon_2_lon,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

i=0
print('#-------- ' + expid[i])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon.pkl', 'rb') as f:
    pre_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon.pkl', 'rb') as f:
    pre_weighted_coslon = pickle.load(f)

pre_weighted_lon = {}

for ialltime in pre_weighted_sinlon.keys():
    print(ialltime)
    
    pre_weighted_lon[ialltime] = sincoslon_2_lon(
        pre_weighted_sinlon[ialltime], pre_weighted_coslon[ialltime]
    )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl',
          'wb') as f:
    pickle.dump(pre_weighted_lon, f)

# endregion
# -----------------------------------------------------------------------------

