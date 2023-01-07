

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

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_sinlon_binned.pkl', 'rb') as f:
    epe_st_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_coslon_binned.pkl', 'rb') as f:
    epe_st_weighted_coslon = pickle.load(f)

epe_st_weighted_lon = {}

for iqtl in epe_st_weighted_sinlon.keys():
    epe_st_weighted_lon[iqtl] = {}
    
    for ialltime in epe_st_weighted_sinlon[iqtl].keys():
        print(iqtl + ' - ' + ialltime)
        
        epe_st_weighted_lon[iqtl][ialltime] = sincoslon_2_lon(
            epe_st_weighted_sinlon[iqtl][ialltime],
            epe_st_weighted_coslon[iqtl][ialltime],
        )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_lon_binned.pkl',
          'wb') as f:
    pickle.dump(epe_st_weighted_lon, f)


'''
#-------------------------------- check
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_sinlon_binned.pkl', 'rb') as f:
    epe_st_weighted_sinlon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_coslon_binned.pkl', 'rb') as f:
    epe_st_weighted_coslon = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_weighted_lon_binned.pkl', 'rb') as f:
    epe_st_weighted_lon = pickle.load(f)

itime = -1
ilat = 48
ilon = 90
for iqtl in epe_st_weighted_sinlon.keys():
    # iqtl = '93.5%'
    for ialltime in epe_st_weighted_sinlon[iqtl].keys():
        # ialltime = 'sea'
        print(iqtl + ' - ' + ialltime)
        
        if (ialltime != 'am'):
            # ialltime = 'mon'
            sinlon = epe_st_weighted_sinlon[iqtl][ialltime][itime, ilat, ilon]
            coslon = epe_st_weighted_coslon[iqtl][ialltime][itime, ilat, ilon]
            lon1 = epe_st_weighted_lon[iqtl][ialltime][itime, ilat, ilon]
            lon2 = np.arctan2(sinlon, coslon) * 180 / np.pi
            if (lon2 < 0): lon2 += 360
            print(((lon1 - lon2) / lon1).values)
        else:
            sinlon = epe_st_weighted_sinlon[iqtl][ialltime][ilat, ilon]
            coslon = epe_st_weighted_coslon[iqtl][ialltime][ilat, ilon]
            lon1 = epe_st_weighted_lon[iqtl][ialltime][ilat, ilon]
            lon2 = np.arctan2(sinlon, coslon) * 180 / np.pi
            if (lon2 < 0): lon2 += 360
            print(((lon1 - lon2) / lon1).values)

'''
# endregion
# -----------------------------------------------------------------------------

