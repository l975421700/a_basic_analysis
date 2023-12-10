

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


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
    
    'nudged_703_6.0_k52',
    # 'nudged_705_6.0',
    # 'nudged_706_6.0_k52_88',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
    ]
i = 0

ifile_start = 0 #12
ifile_end   = 528 #516

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


#SBATCH --time=12:00:00
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




'''
#-------------------------------- check

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
wiso_6h_sfc = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

data1 = q_geo7_sfc_alltiime[expid[i]]['6h'].sel(geo_regions='Pacific Ocean')[-124:].values
data2 = (wiso_6h_sfc['q_17'] + wiso_6h_sfc['xl_17'] + wiso_6h_sfc['xi_17']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = q_geo7_sfc_alltiime[expid[i]]['6h'].sel(geo_regions='Open Ocean')[-124:].values
data2 = (
    wiso_6h_sfc['q_15'] + wiso_6h_sfc['xl_15'] + wiso_6h_sfc['xi_15'] + \
        wiso_6h_sfc['q_16'] + wiso_6h_sfc['xl_16'] + wiso_6h_sfc['xi_16'] + \
            wiso_6h_sfc['q_17'] + wiso_6h_sfc['xl_17'] + wiso_6h_sfc['xi_17'] + \
                wiso_6h_sfc['q_19'] + wiso_6h_sfc['xl_19'] + wiso_6h_sfc['xi_19']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
print(np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data1[np.isfinite(data1)]))


data1 = q_geo7_sfc_alltiime[expid[i]]['6h'].sel(geo_regions='Sum')[-124:].values
data2 = (
    wiso_6h_sfc['q_13'] + wiso_6h_sfc['xl_13'] + wiso_6h_sfc['xi_13'] + \
    wiso_6h_sfc['q_14'] + wiso_6h_sfc['xl_14'] + wiso_6h_sfc['xi_14'] + \
    wiso_6h_sfc['q_15'] + wiso_6h_sfc['xl_15'] + wiso_6h_sfc['xi_15'] + \
    wiso_6h_sfc['q_16'] + wiso_6h_sfc['xl_16'] + wiso_6h_sfc['xi_16'] + \
    wiso_6h_sfc['q_17'] + wiso_6h_sfc['xl_17'] + wiso_6h_sfc['xi_17'] + \
    wiso_6h_sfc['q_18'] + wiso_6h_sfc['xl_18'] + wiso_6h_sfc['xi_18'] + \
    wiso_6h_sfc['q_19'] + wiso_6h_sfc['xl_19'] + wiso_6h_sfc['xi_19']
    ).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
print(np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data1[np.isfinite(data1)]))

#-------------------------------- check 2

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

data1 = q_geo7_sfc_alltiime[expid[i]]['am'][:7].sum(dim='geo_regions').values
data2 = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['am'].squeeze().values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))

data1 = q_geo7_sfc_alltiime[expid[i]]['am'].sel(geo_regions='Sum').values
data2 = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['am'].squeeze().values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))


#-------------------------------- check 3

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

# Open ocean + Arctic sea ice
# data1 = q_geo7_sfc_alltiime[expid[i]]['am'].sel(geo_regions=[
#     'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'Southern Ocean',
#     ], lat=slice(0, -90)).sum(dim='geo_regions', skipna=False).compute().values
data1 = q_geo7_sfc_alltiime[expid[i]]['am'].sel(geo_regions=['Open Ocean'], lat=slice(0, -90)).squeeze().values
# Open ocean
data2 = ocean_q_sfc_alltime[expid[i]]['am'].sel(var_names='lat', lat=slice(0, -90)).values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))
print(np.max(abs(data1[subset] - data2[subset])))

data1 = ocean_q_sfc_alltime[expid[i]]['am'].sel(var_names='lat').values
data2 = ocean_q_sfc_alltime[expid[i]]['am'].sel(var_names='coslon').values
subset = np.isfinite(data1) & np.isfinite(data2)
np.max(abs(data1[subset] - data2[subset]) / data2[subset])

'''
# endregion
# -----------------------------------------------------------------------------
