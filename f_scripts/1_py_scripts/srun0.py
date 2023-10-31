

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i = 0

ifile_start = 12
ifile_end   = 516

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
import pandas as pd

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
exp_org_o[expid[i]]['wiso_q_6h_sfc'] = xr.open_mfdataset(
    filenames_wiso_q_6h_sfc[ifile_start:ifile_end],
    )


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann q_sfc from 7 geo regions

time = exp_org_o[expid[i]]['wiso_q_6h_sfc'].time
lon  = exp_org_o[expid[i]]['wiso_q_6h_sfc'].lon
lat  = exp_org_o[expid[i]]['wiso_q_6h_sfc'].lat

geo_regions = [
    'AIS', 'Land excl. AIS', 'Atlantic Ocean',
    'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean',
    'Open Ocean', 'Sum']
q_count = {'AIS': 13, 'Land excl. AIS': 14,
             'Atlantic Ocean': 15, 'Indian Ocean': 16, 'Pacific Ocean': 17,
             'SH seaice': 18, 'Southern Ocean': 19}

q_geo7_sfc = {}
q_geo7_sfc[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(geo_regions), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'geo_regions':  geo_regions,
        'lat':          lat,
        'lon':          lon,
    }
)

for iregion in geo_regions[:-2]:
    # iregion = 'AIS'
    print('#-------------------------------- ' + iregion)
    print(q_count[iregion])
    
    q_geo7_sfc[expid[i]].sel(geo_regions=iregion)[:] = \
        (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str(q_count[iregion])] + \
            exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str(q_count[iregion])] + \
                exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str(q_count[iregion])]).sel(lev=47).compute()

q_geo7_sfc[expid[i]].sel(geo_regions='Open Ocean')[:] = \
    q_geo7_sfc[expid[i]].sel(geo_regions=[
        'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'Southern Ocean',
        ]).sum(dim='geo_regions', skipna=False).compute()

q_geo7_sfc[expid[i]].sel(geo_regions='Sum')[:] = \
    q_geo7_sfc[expid[i]].sel(geo_regions=[
        'AIS', 'Land excl. AIS', 'Atlantic Ocean',
        'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean',
        ]).sum(dim='geo_regions', skipna=False).compute()

q_geo7_sfc_alltiime = {}
q_geo7_sfc_alltiime[expid[i]] = mon_sea_ann(var_6hourly=q_geo7_sfc[expid[i]])

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_geo7_sfc_alltiime[expid[i]], f)


# endregion
# -----------------------------------------------------------------------------
