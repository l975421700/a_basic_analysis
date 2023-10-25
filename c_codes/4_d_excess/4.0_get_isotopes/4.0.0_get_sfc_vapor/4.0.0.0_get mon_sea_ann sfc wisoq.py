

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i = 0

ifile_start = 12
ifile_end   = 516 # 516

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

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 0
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon',]
itags = [5, 7, 8, 9, 11, 12]

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

