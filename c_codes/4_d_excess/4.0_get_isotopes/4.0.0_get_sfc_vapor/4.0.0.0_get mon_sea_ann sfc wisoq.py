

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


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_q_sfc

time = exp_org_o[expid[i]]['wiso_q_6h_sfc'].time
lon  = exp_org_o[expid[i]]['wiso_q_6h_sfc'].lon
lat  = exp_org_o[expid[i]]['wiso_q_6h_sfc'].lat

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0,  3, 0]
kwiso2 = 0
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'RHsst',]
itags = [5, 7, 8, 9, 11, 12, 14]

ocean_q_sfc = {}
ocean_q_sfc[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'lat':          lat,
        'lon':          lon,
    }
)

for count,var_name in enumerate(var_names):
    # count = 0; var_name = 'lat'
    
    kstart = kwiso2 + sum(ntags[:itags[count]])
    
    str_ind1 = str(kstart + 2)
    str_ind2 = str(kstart + 3)
    
    if (len(str_ind1) == 1): str_ind1 = '0' + str_ind1
    if (len(str_ind2) == 1): str_ind2 = '0' + str_ind2
    
    print(str(count) + ' : ' + var_name + ' : ' + str(itags[count]) + \
        ' : ' + str_ind1 + ' : ' + str_ind2)
    
    ocean_q_sfc[expid[i]].sel(var_names=var_name)[:] = \
        (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str_ind1] + \
            exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str_ind2] + \
                exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str_ind1] + \
                    exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str_ind2] + \
                        exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str_ind1] + \
                            exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str_ind2]
                            ).sel(lev=47)

ocean_q_sfc_alltime = {}
ocean_q_sfc_alltime[expid[i]] = mon_sea_ann(
    var_6hourly=ocean_q_sfc[expid[i]],)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_q_sfc[expid[i]]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'wb') as f:
    pickle.dump(ocean_q_sfc_alltime[expid[i]], f)



'''
#-------------------------------- check

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)
print(psutil.Process().memory_info().rss / (2 ** 30))


#---------------- check correspondance between varnames
data1 = ocean_q_sfc_alltime[expid[i]]['ann'].sel(var_names='lat').values
data2 = ocean_q_sfc_alltime[expid[i]]['ann'].sel(var_names='wind10').values
print(np.max(abs((data1[np.isfinite(data1)] - data2[np.isfinite(data2)])/data1[np.isfinite(data1)])))

#---------------- check 6h data
exp_org_o = {}
exp_org_o[expid[i]] = {}
filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

data1 = (ncfile.q_02[:, 0] + ncfile.q_03[:, 0] + \
    ncfile.xl_02[:, 0] + ncfile.xl_03[:, 0] + \
        ncfile.xi_02[:, 0] + ncfile.xi_03[:, 0]).values
data2 = (ocean_q_sfc_alltime[expid[i]]['6h'][-124:].sel(var_names='lat')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

data1 = (ncfile.q_24[:, 0] + ncfile.q_25[:, 0] + \
    ncfile.xl_24[:, 0] + ncfile.xl_25[:, 0] + \
        ncfile.xi_24[:, 0] + ncfile.xi_25[:, 0]).values
data2 = (ocean_q_sfc_alltime[expid[i]]['6h'][-124:].sel(var_names='coslon')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

#---------------- from 6h to daily
exp_org_o = {}
exp_org_o[expid[i]] = {}
filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

data1 = (ncfile.q_02[:, 0] + ncfile.q_03[:, 0] + \
    ncfile.xl_02[:, 0] + ncfile.xl_03[:, 0] + \
        ncfile.xi_02[:, 0] + ncfile.xi_03[:, 0]).resample({'time': '1d'}).mean(skipna=False).values
data2 = (ocean_q_sfc_alltime[expid[i]]['daily'][-31:].sel(var_names='lat')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

data1 = (ncfile.q_24[:, 0] + ncfile.q_25[:, 0] + \
    ncfile.xl_24[:, 0] + ncfile.xl_25[:, 0] + \
        ncfile.xi_24[:, 0] + ncfile.xi_25[:, 0]).resample({'time': '1d'}).mean(skipna=False).values
data2 = (ocean_q_sfc_alltime[expid[i]]['daily'][-31:].sel(var_names='coslon')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

#---------------- from 6h to monthly
exp_org_o = {}
exp_org_o[expid[i]] = {}
filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

data1 = (ncfile.q_02[:, 0] + ncfile.q_03[:, 0] + \
    ncfile.xl_02[:, 0] + ncfile.xl_03[:, 0] + \
        ncfile.xi_02[:, 0] + ncfile.xi_03[:, 0]).resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False).values
data2 = (ocean_q_sfc_alltime[expid[i]]['mon'][-1].sel(var_names='lat')).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data1 = (ncfile.q_24[:, 0] + ncfile.q_25[:, 0] + \
    ncfile.xl_24[:, 0] + ncfile.xl_25[:, 0] + \
        ncfile.xi_24[:, 0] + ncfile.xi_25[:, 0]).resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False).values
data2 = (ocean_q_sfc_alltime[expid[i]]['mon'][-1].sel(var_names='coslon')).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann q16o, q18o, and qhdo

wiso_q_6h_sfc = {}
wiso_q_6h_sfc[expid[i]] = {}

wiso_q_6h_sfc[expid[i]]['q16o'] = (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q16o'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl16o'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi16o']).compute()

wiso_q_6h_sfc[expid[i]]['q18o'] = (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q18o'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl18o'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi18o']).compute()

wiso_q_6h_sfc[expid[i]]['qhdo'] = (exp_org_o[expid[i]]['wiso_q_6h_sfc']['qhdo'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xlhdo'] + exp_org_o[expid[i]]['wiso_q_6h_sfc']['xihdo']).compute()


wiso_q_6h_sfc_alltime = {}
wiso_q_6h_sfc_alltime[expid[i]] = {}

wiso_q_6h_sfc_alltime[expid[i]]['q16o'] = mon_sea_ann(
    var_6hourly=wiso_q_6h_sfc[expid[i]]['q16o'])

wiso_q_6h_sfc_alltime[expid[i]]['q18o'] = mon_sea_ann(
    var_6hourly=wiso_q_6h_sfc[expid[i]]['q18o'])

wiso_q_6h_sfc_alltime[expid[i]]['qhdo'] = mon_sea_ann(
    var_6hourly=wiso_q_6h_sfc[expid[i]]['qhdo'])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'wb') as f:
    pickle.dump(wiso_q_6h_sfc_alltime[expid[i]], f)




'''
#-------------------------------- check

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

exp_org_o = {}
exp_org_o[expid[i]] = {}
filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

#---------------- check 6h

print((wiso_q_6h_sfc_alltime[expid[i]]['q16o']['6h'][-124:].sel(lev=47) == (ncfile.q16o + ncfile.xl16o + ncfile.xi16o).sel(lev=47)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['q18o']['6h'][-124:].sel(lev=47) == (ncfile.q18o + ncfile.xl18o + ncfile.xi18o).sel(lev=47)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['qhdo']['6h'][-124:].sel(lev=47) == (ncfile.qhdo + ncfile.xlhdo + ncfile.xihdo).sel(lev=47)).all().values)

#---------------- check 1d

print((wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'][-31:].sel(lev=47) == (ncfile.q16o + ncfile.xl16o + ncfile.xi16o).sel(lev=47).resample({'time': '1d'}).mean(skipna=False)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['q18o']['daily'][-31:].sel(lev=47) == (ncfile.q18o + ncfile.xl18o + ncfile.xi18o).sel(lev=47).resample({'time': '1d'}).mean(skipna=False)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['qhdo']['daily'][-31:].sel(lev=47) == (ncfile.qhdo + ncfile.xlhdo + ncfile.xihdo).sel(lev=47).resample({'time': '1d'}).mean(skipna=False)).all().values)

#---------------- check 1d

print((wiso_q_6h_sfc_alltime[expid[i]]['q16o']['mon'][-1].sel(lev=47) == (ncfile.q16o + ncfile.xl16o + ncfile.xi16o).sel(lev=47).resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['q18o']['mon'][-1].sel(lev=47) == (ncfile.q18o + ncfile.xl18o + ncfile.xi18o).sel(lev=47).resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False)).all().values)
print((wiso_q_6h_sfc_alltime[expid[i]]['qhdo']['mon'][-1].sel(lev=47) == (ncfile.qhdo + ncfile.xlhdo + ncfile.xihdo).sel(lev=47).resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False)).all().values)

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


# -----------------------------------------------------------------------------
# region get mon_sea_ann q_sfc_frc from 7 geo regions

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

q_geo7_sfc_frc_alltime = {}
q_geo7_sfc_frc_alltime[expid[i]] = {}

for ialltime in q_geo7_sfc_alltiime[expid[i]].keys():
    # ialltime = 'mm'
    print('#-------------------------------- ' + ialltime)
    
    q_geo7_sfc_frc_alltime[expid[i]][ialltime] = \
        (q_geo7_sfc_alltiime[expid[i]][ialltime].sel(geo_regions=[
            'AIS', 'Land excl. AIS', 'Atlantic Ocean', 'Indian Ocean',
            'Pacific Ocean', 'SH seaice', 'Southern Ocean', 'Open Ocean',
        ]) / q_geo7_sfc_alltiime[expid[i]][ialltime].sel(geo_regions='Sum') * 100
         ).compute()

del q_geo7_sfc_alltiime

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_geo7_sfc_frc_alltime[expid[i]], f)




'''
#-------------------------------- check

q_geo7_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_frc_alltime.pkl', 'rb') as f:
    q_geo7_sfc_frc_alltime[expid[i]] = pickle.load(f)

q_geo7_sfc_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_sfc_alltiime.pkl', 'rb') as f:
    q_geo7_sfc_alltiime[expid[i]] = pickle.load(f)

# check 1
ialltime = 'mon'
iregion = 'Atlantic Ocean'

data1 = (q_geo7_sfc_alltiime[expid[i]][ialltime].sel(geo_regions=iregion) / q_geo7_sfc_alltiime[expid[i]][ialltime].sel(geo_regions='Sum') * 100).values
data2 = q_geo7_sfc_frc_alltime[expid[i]][ialltime].sel(geo_regions=iregion).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

ialltime = 'mon'
data = q_geo7_sfc_frc_alltime[expid[i]][ialltime].sel(geo_regions=['AIS', 'Land excl. AIS', 'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean']).sum(dim='geo_regions', skipna=False)
print(np.nanmin(data))
print(np.nanmax(data))


q_geo7_sfc_frc_alltime[expid[i]]['am'].to_netcdf('scratch/test/test0.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_q_frc

ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

ocean_q_sfc_frc_alltime = {}
ocean_q_sfc_frc_alltime[expid[i]] = {}

for ialltime in ocean_q_sfc_alltime[expid[i]].keys():
    # ialltime = 'mon'
    print('#-------------------------------- ' + ialltime)
    
    ocean_q_sfc_frc_alltime[expid[i]][ialltime] = (
        ocean_q_sfc_alltime[expid[i]][ialltime] / wiso_q_6h_sfc_alltime[expid[i]]['q16o'][ialltime] * 100).compute().squeeze()


output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_frc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ocean_q_sfc_frc_alltime[expid[i]], f)




'''
#-------------------------------- check
ocean_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_alltime[expid[i]] = pickle.load(f)

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

ocean_q_sfc_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_sfc_frc_alltime.pkl', 'rb') as f:
    ocean_q_sfc_frc_alltime[expid[i]] = pickle.load(f)


ialltime = 'daily'
data1 = (ocean_q_sfc_alltime[expid[i]][ialltime] / wiso_q_6h_sfc_alltime[expid[i]]['q16o'][ialltime] * 100).compute().squeeze().values
data2 = ocean_q_sfc_frc_alltime[expid[i]][ialltime].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

'''
# endregion
# -----------------------------------------------------------------------------
