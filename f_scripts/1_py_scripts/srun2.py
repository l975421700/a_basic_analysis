

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
    ]
i = 0

ifile_start = 0 #0 #120
ifile_end   = 528 #1740 #840

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

filenames_wiso_q_daily = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_daily.nc'))
exp_org_o[expid[i]]['wiso_q_daily'] = xr.open_mfdataset(
    filenames_wiso_q_daily[ifile_start:ifile_end],
    chunks={'time': 10}
    )


'''
exp_org_o[expid[i]]['wiso_q_daily']
'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --mem=240GB
# -----------------------------------------------------------------------------
# region get mon_sea_ann q from 7 geo regions

time = exp_org_o[expid[i]]['wiso_q_daily'].time
lon  = exp_org_o[expid[i]]['wiso_q_daily'].lon
lat  = exp_org_o[expid[i]]['wiso_q_daily'].lat
lev  = exp_org_o[expid[i]]['wiso_q_daily'].lev

geo_regions = [
    'AIS', 'Land excl. AIS', 'Atlantic Ocean',
    'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean',
    'Open Ocean', 'Sum']
q_count = {'AIS': 13, 'Land excl. AIS': 14,
             'Atlantic Ocean': 15, 'Indian Ocean': 16, 'Pacific Ocean': 17,
             'SH seaice': 18, 'Southern Ocean': 19}

q_geo7 = {}
q_geo7[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(geo_regions), len(lev), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'geo_regions':  geo_regions,
        'lev':          lev,
        'lat':          lat,
        'lon':          lon,
    }
)

for iregion in geo_regions[:-2]:
    # iregion = 'AIS'
    print('#-------------------------------- ' + iregion)
    print(q_count[iregion])
    
    q_geo7[expid[i]].sel(geo_regions=iregion)[:] = \
        (exp_org_o[expid[i]]['wiso_q_daily']['q_' + str(q_count[iregion])] + \
            exp_org_o[expid[i]]['wiso_q_daily']['xl_' + str(q_count[iregion])] + \
                exp_org_o[expid[i]]['wiso_q_daily']['xi_' + str(q_count[iregion])]).compute()

print(psutil.Process().memory_info().rss / (2 ** 30))

del exp_org_o

print(psutil.Process().memory_info().rss / (2 ** 30))

q_geo7[expid[i]].sel(geo_regions='Open Ocean')[:] = \
    q_geo7[expid[i]].sel(geo_regions=[
        'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'Southern Ocean',
        ]).sum(dim='geo_regions', skipna=False).compute()

q_geo7[expid[i]].sel(geo_regions='Sum')[:] = \
    q_geo7[expid[i]].sel(geo_regions=[
        'AIS', 'Land excl. AIS', 'Atlantic Ocean',
        'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean',
        ]).sum(dim='geo_regions', skipna=False).compute()

q_geo7_alltiime = {}
q_geo7_alltiime[expid[i]] = mon_sea_ann(q_geo7[expid[i]], lcopy=False)

print(psutil.Process().memory_info().rss / (2 ** 30))

del q_geo7

print(psutil.Process().memory_info().rss / (2 ** 30))

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_geo7_alltiime[expid[i]], f)


'''
#-------------------------------- check

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))

ifile = -10
wiso_q_plev = xr.open_dataset(filenames_wiso_q_plev[ifile_start:ifile_end][ifile])

data1 = q_geo7_alltiime[expid[i]]['mon'].sel(geo_regions='Pacific Ocean')[ifile].values
data2 = (wiso_q_plev['q_17'] + wiso_q_plev['xl_17'] + wiso_q_plev['xi_17']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = q_geo7_alltiime[expid[i]]['mon'].sel(geo_regions='Open Ocean')[ifile].values
data2 = (
    wiso_q_plev['q_15'] + wiso_q_plev['xl_15'] + wiso_q_plev['xi_15'] + \
        wiso_q_plev['q_16'] + wiso_q_plev['xl_16'] + wiso_q_plev['xi_16'] + \
            wiso_q_plev['q_17'] + wiso_q_plev['xl_17'] + wiso_q_plev['xi_17'] + \
                wiso_q_plev['q_19'] + wiso_q_plev['xl_19'] + wiso_q_plev['xi_19']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
print(np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data1[np.isfinite(data1)]))


data1 = q_geo7_alltiime[expid[i]]['mon'].sel(geo_regions='Sum')[ifile].values
data2 = (
    wiso_q_plev['q_13'] + wiso_q_plev['xl_13'] + wiso_q_plev['xi_13'] + \
    wiso_q_plev['q_14'] + wiso_q_plev['xl_14'] + wiso_q_plev['xi_14'] + \
    wiso_q_plev['q_15'] + wiso_q_plev['xl_15'] + wiso_q_plev['xi_15'] + \
    wiso_q_plev['q_16'] + wiso_q_plev['xl_16'] + wiso_q_plev['xi_16'] + \
    wiso_q_plev['q_17'] + wiso_q_plev['xl_17'] + wiso_q_plev['xi_17'] + \
    wiso_q_plev['q_18'] + wiso_q_plev['xl_18'] + wiso_q_plev['xi_18'] + \
    wiso_q_plev['q_19'] + wiso_q_plev['xl_19'] + wiso_q_plev['xi_19']
    ).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
print(np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data1[np.isfinite(data1)]))

#-------------------------------- check 2

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)

data1 = q_geo7_alltiime[expid[i]]['am'][:7].sum(dim='geo_regions').values
data2 = wiso_q_plev_alltime[expid[i]]['q16o']['am'].values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))

data1 = q_geo7_alltiime[expid[i]]['am'].sel(geo_regions='Sum').values
data2 = wiso_q_plev_alltime[expid[i]]['q16o']['am'].values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))


#-------------------------------- check 3

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

# Open ocean + Arctic sea ice
data1 = q_geo7_alltiime[expid[i]]['am'].sel(geo_regions=[
    'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'Southern Ocean',
    ], lat=slice(0, -90), plev=1e+5).sum(dim='geo_regions', skipna=False).compute().values
# data1 = q_geo7_alltiime[expid[i]]['am'].sel(geo_regions=['Open Ocean'], lat=slice(0, -90), plev=1e+5).squeeze().values
# Open ocean
data2 = ocean_q_alltime[expid[i]]['am'].sel(var_names='lat', lat=slice(0, -90), plev=1e+5).values
subset = np.isfinite(data1) & np.isfinite(data2)
print(np.max(abs(data1[subset] - data2[subset]) / data2[subset]))

data1 = ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').values
data2 = ocean_q_alltime[expid[i]]['am'].sel(var_names='coslon').values
subset = np.isfinite(data1) & np.isfinite(data2)
np.max(abs(data1[subset] - data2[subset]) / data2[subset])

'''
# endregion
# -----------------------------------------------------------------------------
