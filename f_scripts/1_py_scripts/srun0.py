

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
<<<<<<< Updated upstream
    'nudged_703_6.0_k52',
    ]
i = 0

ifile_start = 0 #0 #120
ifile_end   = 528 #1740 #840
=======
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    'hist_700_5.0',
    # 'nudged_701_5.0',
    
    # 'nudged_703_6.0_k52',
    ]
i = 0

ifile_start = 1380 #12 #0 #120
ifile_end   = 1740 #516 #1740 #840
>>>>>>> Stashed changes

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
sys.path.append('/home/users/qino')
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
import xesmf as xe
import pandas as pd


from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    mean_over_ais,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

<<<<<<< Updated upstream
filenames_wiso_q_daily = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_daily.nc'))
exp_org_o[expid[i]]['wiso_q_daily'] = xr.open_mfdataset(
    filenames_wiso_q_daily[ifile_start:ifile_end],
    chunks={'time': 10}
=======
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
    filenames_wiso[ifile_start:ifile_end],
>>>>>>> Stashed changes
    )

'''
<<<<<<< Updated upstream
exp_org_o[expid[i]]['wiso_q_daily']
=======
#-------- check pre
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
nc1 = xr.open_dataset(filenames_wiso[ifile])
nc2 = xr.open_dataset(filenames_echam[ifile])

np.max(abs(nc1.wisoaprl[:, 0].mean(dim='time').values - nc2.aprl[0].values))


#-------- input previous files

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso_daily'] = xr.open_mfdataset(filenames_wiso_daily, data_vars='minimal', coords='minimal', parallel=True)
        
        filenames_echam_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_echam.nc'))
        exp_org_o[expid[i]]['echam_daily'] = xr.open_mfdataset(filenames_echam_daily[120:], data_vars='minimal', coords='minimal', parallel=True)

>>>>>>> Stashed changes
'''
# endregion
# -----------------------------------------------------------------------------


<<<<<<< Updated upstream
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
=======
#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region get mon/sea/ann wisoaprt

wisoaprt = {}
wisoaprt[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl[:, :3] + \
        exp_org_o[expid[i]]['wiso'].wisoaprc[:, :3].values).compute()

wisoaprt[expid[i]] = wisoaprt[expid[i]].rename('wisoaprt')

wisoaprt_alltime = {}
wisoaprt_alltime[expid[i]] = mon_sea_ann(wisoaprt[expid[i]])

>>>>>>> Stashed changes

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'wb') as f:
    pickle.dump(wisoaprt_alltime[expid[i]], f)


'''
#-------- check calculation
wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

<<<<<<< Updated upstream
ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)
print(psutil.Process().memory_info().rss / (2 ** 30))
=======
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[120:1080][ifile])

(wisoaprt_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values)).all()

(wisoaprt_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values).mean(dim='time')).all()
>>>>>>> Stashed changes

data1 = ocean_q_alltime[expid[i]]['am'].sel(var_names='lat').values
data2 = ocean_q_alltime[expid[i]]['am'].sel(var_names='sinlon').values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data2[np.isfinite(data2)])

<<<<<<< Updated upstream
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
=======
#-------- check simulation of aprt and wisoaprt
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:180], data_vars='minimal', coords='minimal', parallel=True)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl.values + \
    exp_org_o[expid[i]]['echam'].aprc.values - \
        wisoaprt_alltime[expid[i]]['mon'][:60, 0].values))
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
