

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i=0

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
# region import data can estimate lon

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_sinlon.pkl', 'rb') as f:
    q_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_coslon.pkl', 'rb') as f:
    q_weighted_coslon = pickle.load(f)

q_weighted_lon = {}

for ialltime in q_weighted_sinlon.keys():
    print(ialltime)
    
    q_weighted_lon[ialltime] = sincoslon_2_lon(
        q_weighted_sinlon[ialltime], q_weighted_coslon[ialltime],
        var_name='q_weighted_lon',
    )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lon.pkl',
          'wb') as f:
    pickle.dump(q_weighted_lon, f)



'''
#-------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_lon.pkl', 'rb') as f:
    q_weighted_lon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_sinlon.pkl', 'rb') as f:
    q_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_coslon.pkl', 'rb') as f:
    q_weighted_coslon = pickle.load(f)

ialltime = 'sea'
itime = 20
plev = 0
ilat = 40
ilon = 90

sinlon = q_weighted_sinlon[ialltime][itime, plev, ilat, ilon].values
coslon = q_weighted_coslon[ialltime][itime, plev, ilat, ilon].values
lon = q_weighted_lon[ialltime][itime, plev, ilat, ilon].values

lon_new = (np.arctan2(sinlon, coslon) * 180 / np.pi)
if lon_new < 0:
    lon_new += 360

print(lon)
print(lon_new)
'''
# endregion
# -----------------------------------------------------------------------------



