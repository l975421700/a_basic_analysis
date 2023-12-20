

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
# region get mon_sea_ann ocean_q

time = exp_org_o[expid[i]]['wiso_q_daily'].time
lon  = exp_org_o[expid[i]]['wiso_q_daily'].lon
lat  = exp_org_o[expid[i]]['wiso_q_daily'].lat
lev  = exp_org_o[expid[i]]['wiso_q_daily'].lev

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0,  3, 0]
kwiso2 = 0
var_names = ['lat',]
itags = [5,]
# var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'RHsst']
# itags = [5, 7, 8, 9, 11, 12, 14]

ocean_q = {}
ocean_q[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(lev), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'lev':          lev,
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
    
    ocean_q[expid[i]].sel(var_names=var_name)[:] = \
        (exp_org_o[expid[i]]['wiso_q_daily']['q_' + str_ind1] + \
            exp_org_o[expid[i]]['wiso_q_daily']['q_' + str_ind2] + \
                exp_org_o[expid[i]]['wiso_q_daily']['xl_' + str_ind1] + \
                    exp_org_o[expid[i]]['wiso_q_daily']['xl_' + str_ind2] + \
                        exp_org_o[expid[i]]['wiso_q_daily']['xi_' + str_ind1] + \
                            exp_org_o[expid[i]]['wiso_q_daily']['xi_' + str_ind2]
                            ).compute()

print(psutil.Process().memory_info().rss / (2 ** 30))

del exp_org_o

print(psutil.Process().memory_info().rss / (2 ** 30))

ocean_q_alltime = {}
ocean_q_alltime[expid[i]] = mon_sea_ann(ocean_q[expid[i]], lcopy=False)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_q

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'wb') as f:
    pickle.dump(ocean_q_alltime[expid[i]], f)



'''
#-------------------------------- check

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)
print(psutil.Process().memory_info().rss / (2 ** 30))

data1 = ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').values
data2 = ocean_q_alltime[expid[i]]['am'].sel(var_names='sinlon').values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data2[np.isfinite(data2)])

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_plev[ifile_start:ifile_end][ifile])

data1 = (ncfile.q_02[0] + ncfile.q_03[0] + ncfile.xl_02[0] + ncfile.xl_03[0] + \
    ncfile.xi_02[0] + ncfile.xi_03[0]).values
data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='lat')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

data1 = (ncfile.q_24[0] + ncfile.q_25[0] + ncfile.xl_24[0] + ncfile.xl_25[0] + \
    ncfile.xi_24[0] + ncfile.xi_25[0]).values
data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='coslon')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


# data1 = (ncfile.q_16[0] + ncfile.q_18[0] + ncfile.xl_16[0] + ncfile.xl_18[0] + \
#     ncfile.xi_16[0] + ncfile.xi_18[0]).values
# data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='geo7')).values
# (data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

#-------- include that of geo7
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'geo7']
ocean_q[expid[i]].sel(var_names='geo7')[:] = \
    (exp_org_o[expid[i]]['wiso_q_plev']['q_16'] + \
        exp_org_o[expid[i]]['wiso_q_plev']['q_18'] + \
            exp_org_o[expid[i]]['wiso_q_plev']['xl_16'] + \
                exp_org_o[expid[i]]['wiso_q_plev']['xl_18'] + \
                    exp_org_o[expid[i]]['wiso_q_plev']['xi_16'] + \
                        exp_org_o[expid[i]]['wiso_q_plev']['xi_18']
                        )
'''
# endregion
# -----------------------------------------------------------------------------


