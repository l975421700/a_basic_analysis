

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
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
# region import data

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_sinlon.pkl', 'rb') as f:
    dc_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_coslon.pkl', 'rb') as f:
    dc_weighted_coslon = pickle.load(f)

dc_weighted_lon = {}

for iqtl in dc_weighted_sinlon.keys():
    dc_weighted_lon[iqtl] = {}
    
    for ialltime in dc_weighted_sinlon[iqtl].keys():
        print(iqtl + ' - ' + ialltime)
        
        dc_weighted_lon[iqtl][ialltime] = sincoslon_2_lon(
            dc_weighted_sinlon[iqtl][ialltime],
            dc_weighted_coslon[iqtl][ialltime],
        )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_lon.pkl',
          'wb') as f:
    pickle.dump(dc_weighted_lon, f)


'''
#-------------------------------- check

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_lon.pkl', 'rb') as f:
    dc_weighted_lon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_sinlon.pkl', 'rb') as f:
    dc_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dc_weighted_coslon.pkl', 'rb') as f:
    dc_weighted_coslon = pickle.load(f)

itime = -1
ilat = 40
ilon = 90
for iqtl in dc_weighted_sinlon.keys():
    # iqtl = '90%'
    for ialltime in dc_weighted_sinlon[iqtl].keys():
        # ialltime = 'am'
        print(iqtl + ' - ' + ialltime)
        
        if (ialltime != 'am'):
            # ialltime = 'mon'
            sinlon = dc_weighted_sinlon[iqtl][ialltime][itime, ilat, ilon]
            coslon = dc_weighted_coslon[iqtl][ialltime][itime, ilat, ilon]
            lon1 = dc_weighted_lon[iqtl][ialltime][itime, ilat, ilon]
            lon2 = np.arctan2(sinlon, coslon) * 180 / np.pi
            if (lon2 < 0): lon2 += 360
            print(((lon1 - lon2) / lon1).values)
        else:
            sinlon = dc_weighted_sinlon[iqtl][ialltime][ilat, ilon]
            coslon = dc_weighted_coslon[iqtl][ialltime][ilat, ilon]
            lon1 = dc_weighted_lon[iqtl][ialltime][ilat, ilon]
            lon2 = np.arctan2(sinlon, coslon) * 180 / np.pi
            if (lon2 < 0): lon2 += 360
            print(((lon1 - lon2) / lon1).values)


pre_weighted_lon = (np.arctan2(sinlon, coslon) * 180 / np.pi).compute()
pre_weighted_lon.values[pre_weighted_lon.values < 0] += 360
'''
# endregion
# -----------------------------------------------------------------------------

