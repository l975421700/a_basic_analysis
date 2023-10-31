

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    'nudged_701_5.0',
    ]
i = 0

ifile_start = 12 #0 #120
ifile_end   = 516 #1740 #840

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

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))
exp_org_o[expid[i]]['wiso_q_plev'] = xr.open_mfdataset(
    filenames_wiso_q_plev[ifile_start:ifile_end],
    )


'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_q

time = exp_org_o[expid[i]]['wiso_q_plev'].time
lon  = exp_org_o[expid[i]]['wiso_q_plev'].lon
lat  = exp_org_o[expid[i]]['wiso_q_plev'].lat
plev = exp_org_o[expid[i]]['wiso_q_plev'].plev

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 0
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon',]
itags = [5, 7, 8, 9, 11, 12]

ocean_q = {}
ocean_q[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(plev), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'plev':         plev,
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
        (exp_org_o[expid[i]]['wiso_q_plev']['q_' + str_ind1] + \
            exp_org_o[expid[i]]['wiso_q_plev']['q_' + str_ind2] + \
                exp_org_o[expid[i]]['wiso_q_plev']['xl_' + str_ind1] + \
                    exp_org_o[expid[i]]['wiso_q_plev']['xl_' + str_ind2] + \
                        exp_org_o[expid[i]]['wiso_q_plev']['xi_' + str_ind1] + \
                            exp_org_o[expid[i]]['wiso_q_plev']['xi_' + str_ind2]
                            ).compute()

ocean_q_alltime = {}
ocean_q_alltime[expid[i]] = mon_sea_ann(var_monthly=ocean_q[expid[i]],)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_q[expid[i]]

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


# -----------------------------------------------------------------------------
# region get mon_sea_ann q16o, q18o, and qhdo

wiso_q_plev = {}
wiso_q_plev[expid[i]] = {}

wiso_q_plev[expid[i]]['q16o'] = (exp_org_o[expid[i]]['wiso_q_plev']['q16o'] + exp_org_o[expid[i]]['wiso_q_plev']['xl16o'] + exp_org_o[expid[i]]['wiso_q_plev']['xi16o']).compute()

wiso_q_plev[expid[i]]['q18o'] = (exp_org_o[expid[i]]['wiso_q_plev']['q18o'] + exp_org_o[expid[i]]['wiso_q_plev']['xl18o'] + exp_org_o[expid[i]]['wiso_q_plev']['xi18o']).compute()

wiso_q_plev[expid[i]]['qhdo'] = (exp_org_o[expid[i]]['wiso_q_plev']['qhdo'] + exp_org_o[expid[i]]['wiso_q_plev']['xlhdo'] + exp_org_o[expid[i]]['wiso_q_plev']['xihdo']).compute()


wiso_q_plev_alltime = {}
wiso_q_plev_alltime[expid[i]] = {}

wiso_q_plev_alltime[expid[i]]['q16o'] = mon_sea_ann(
    var_monthly=wiso_q_plev[expid[i]]['q16o'])

wiso_q_plev_alltime[expid[i]]['q18o'] = mon_sea_ann(
    var_monthly=wiso_q_plev[expid[i]]['q18o'])

wiso_q_plev_alltime[expid[i]]['qhdo'] = mon_sea_ann(
    var_monthly=wiso_q_plev[expid[i]]['qhdo'])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'wb') as f:
    pickle.dump(wiso_q_plev_alltime[expid[i]], f)



'''
wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region get mon_sea_ann q from 7 geo regions

time = exp_org_o[expid[i]]['wiso_q_plev'].time
lon  = exp_org_o[expid[i]]['wiso_q_plev'].lon
lat  = exp_org_o[expid[i]]['wiso_q_plev'].lat
plev = exp_org_o[expid[i]]['wiso_q_plev'].plev

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
        (len(time), len(geo_regions), len(plev), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'geo_regions':  geo_regions,
        'plev':         plev,
        'lat':          lat,
        'lon':          lon,
    }
)

for iregion in geo_regions[:-2]:
    # iregion = 'AIS'
    print('#-------------------------------- ' + iregion)
    print(q_count[iregion])
    
    q_geo7[expid[i]].sel(geo_regions=iregion)[:] = \
        (exp_org_o[expid[i]]['wiso_q_plev']['q_' + str(q_count[iregion])] + \
            exp_org_o[expid[i]]['wiso_q_plev']['xl_' + str(q_count[iregion])] + \
                exp_org_o[expid[i]]['wiso_q_plev']['xi_' + str(q_count[iregion])]).compute()

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
q_geo7_alltiime[expid[i]] = mon_sea_ann(var_monthly=q_geo7[expid[i]])

print(psutil.Process().memory_info().rss / (2 ** 30))

del q_geo7[expid[i]]

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


# -----------------------------------------------------------------------------
# region get mon_sea_ann q_frc from 7 geo regions

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

q_geo7_frc_alltime = {}
q_geo7_frc_alltime[expid[i]] = {}

for ialltime in q_geo7_alltiime[expid[i]].keys():
    # ialltime = 'mm'
    print('#-------------------------------- ' + ialltime)
    
    q_geo7_frc_alltime[expid[i]][ialltime] = \
        (q_geo7_alltiime[expid[i]][ialltime].sel(geo_regions=[
            'AIS', 'Land excl. AIS', 'Atlantic Ocean', 'Indian Ocean',
            'Pacific Ocean', 'SH seaice', 'Southern Ocean', 'Open Ocean',
        ]) / q_geo7_alltiime[expid[i]][ialltime].sel(geo_regions='Sum') * 100
         ).compute()


output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_frc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_geo7_frc_alltime[expid[i]], f)




'''
#-------------------------------- check

q_geo7_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_frc_alltime.pkl', 'rb') as f:
    q_geo7_frc_alltime[expid[i]] = pickle.load(f)

q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

# check 1
ialltime = 'mon'
iregion = 'Atlantic Ocean'

data1 = (q_geo7_alltiime[expid[i]][ialltime].sel(geo_regions=iregion) / q_geo7_alltiime[expid[i]][ialltime].sel(geo_regions='Sum') * 100).values
data2 = q_geo7_frc_alltime[expid[i]][ialltime].sel(geo_regions=iregion).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

ialltime = 'mon'
data = q_geo7_frc_alltime[expid[i]][ialltime].sel(geo_regions=['AIS', 'Land excl. AIS', 'Atlantic Ocean', 'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean']).sum(dim='geo_regions', skipna=False)
print(np.nanmin(data))
print(np.nanmax(data))


q_geo7_frc_alltime[expid[i]]['am'].to_netcdf('scratch/test/test0.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_q_frc

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)

ocean_q_frc_alltime = {}
ocean_q_frc_alltime[expid[i]] = {}

for ialltime in ocean_q_alltime[expid[i]].keys():
    # ialltime = 'mon'
    print('#-------------------------------- ' + ialltime)
    
    ocean_q_frc_alltime[expid[i]][ialltime] = (
        ocean_q_alltime[expid[i]][ialltime] / wiso_q_plev_alltime[expid[i]]['q16o'][ialltime] * 100).compute()


output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_frc_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ocean_q_frc_alltime[expid[i]], f)




'''
#-------------------------------- check
ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)

wiso_q_plev_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_plev_alltime.pkl', 'rb') as f:
    wiso_q_plev_alltime[expid[i]] = pickle.load(f)

ocean_q_frc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_frc_alltime.pkl', 'rb') as f:
    ocean_q_frc_alltime[expid[i]] = pickle.load(f)


ialltime = 'am'
data1 = (ocean_q_alltime[expid[i]][ialltime] / wiso_q_plev_alltime[expid[i]]['q16o'][ialltime] * 100).compute().values
data2 = ocean_q_frc_alltime[expid[i]][ialltime].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

'''
# endregion
# -----------------------------------------------------------------------------

