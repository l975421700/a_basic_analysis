# #SBATCH --time=00:30:00


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
<<<<<<< Updated upstream
    'nudged_703_6.0_k52',
=======
    # 'pi_600_5.0',
    'hist_700_5.0',
    # 'nudged_701_5.0',
    # 'pi_1d_803_6.0',
    # 'nudged_703_6.0_k52',
>>>>>>> Stashed changes
    ]
i=0

<<<<<<< Updated upstream
ifile_start = 0 #0 #120
ifile_end   = 528 #1740 #840
=======
output_dir = exp_odir + expid[i] + '/analysis/echam/'

ifile_start = 1380 #0 #120
ifile_end   = 1740 # 528 #1740 #840

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0,  3, 0]

# var_name  = 'sst'
# itag      = 7
# min_sf    = 268.15
# max_sf    = 318.15

# var_name  = 'lat'
# itag      = 5
# min_sf    = -90
# max_sf    = 90

var_name  = 'rh2m'
itag      = 8
min_sf    = 0
max_sf    = 1.6

# var_name  = 'wind10'
# itag      = 9
# min_sf    = 0
# max_sf    = 28

# var_name  = 'sinlon'
# itag      = 11
# min_sf    = -1
# max_sf    = 1

# var_name  = 'coslon'
# itag      = 12
# min_sf    = -1
# max_sf    = 1

# var_name  = 'RHsst'
# itag      = 14
# min_sf    = 0
# max_sf    = 1.4
>>>>>>> Stashed changes

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
sys.path.append('/home/users/qino')
import os

# data analysis
# import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle

from a_basic_analysis.b_module.source_properties import (
    source_properties,
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

fl_wiso_daily = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'
        ))

<<<<<<< Updated upstream
filenames_wiso_q_daily = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_daily.nc'))
exp_org_o[expid[i]]['wiso_q_daily'] = xr.open_mfdataset(
    filenames_wiso_q_daily[ifile_start:ifile_end],
    chunks={'time': 10}
    )


'''
exp_org_o[expid[i]]['wiso_q_daily']
'''
=======
exp_out_wiso_daily = xr.open_mfdataset(
    fl_wiso_daily[ifile_start:ifile_end],
    )

>>>>>>> Stashed changes
# endregion
# -----------------------------------------------------------------------------


<<<<<<< Updated upstream
#SBATCH --mem=240GB
# -----------------------------------------------------------------------------
# region get mon_sea_ann q from 7 geo regions

time = exp_org_o[expid[i]]['wiso_q_daily'].time
lon  = exp_org_o[expid[i]]['wiso_q_daily'].lon
lat  = exp_org_o[expid[i]]['wiso_q_daily'].lat
lev  = exp_org_o[expid[i]]['wiso_q_daily'].lev
=======
# -----------------------------------------------------------------------------
# region set indices

kwiso2 = 3
>>>>>>> Stashed changes

kstart = kwiso2 + sum(ntags[:itag])
kend   = kwiso2 + sum(ntags[:(itag+1)])

<<<<<<< Updated upstream
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
=======
print(kstart); print(kend)
>>>>>>> Stashed changes

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source var

#-------------------------------- precipitation

ocean_pre = (
    exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) + \
        exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))
        ).sum(dim='wisotype').compute()
var_scaled_pre = (
    exp_out_wiso_daily.wisoaprl.sel(wisotype=kstart+2) + \
        exp_out_wiso_daily.wisoaprc.sel(wisotype=kstart+2)).compute()

var_scaled_pre.values[ocean_pre.values < 2e-8] = 0
ocean_pre.values[ocean_pre.values < 2e-8] = 0


#-------- monthly/seasonal/annual (mean) values

ocean_pre_alltime      = mon_sea_ann(ocean_pre)
var_scaled_pre_alltime = mon_sea_ann(var_scaled_pre)

#-------------------------------- pre-weighted var

pre_weighted_var = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
<<<<<<< Updated upstream
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
=======
    pre_weighted_var[ialltime] = source_properties(
        var_scaled_pre_alltime[ialltime],
        ocean_pre_alltime[ialltime],
        min_sf, max_sf,
        var_name,
    )

#-------- monthly without monthly mean
pre_weighted_var['mon no mm'] = (pre_weighted_var['mon'].groupby('time.month') - pre_weighted_var['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
pre_weighted_var['ann no am'] = (pre_weighted_var['ann'] - pre_weighted_var['ann'].mean(dim='time', skipna=True)).compute()

output_file = output_dir + expid[i] + '.pre_weighted_' + var_name + '.pkl'
>>>>>>> Stashed changes

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
<<<<<<< Updated upstream
    pickle.dump(q_geo7_alltiime[expid[i]], f)
=======
    pickle.dump(pre_weighted_var, f)
>>>>>>> Stashed changes


'''
#-------- import data
pre_weighted_lat = {}

<<<<<<< Updated upstream
q_geo7_alltiime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_geo7_alltiime.pkl', 'rb') as f:
    q_geo7_alltiime[expid[i]] = pickle.load(f)

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))

ifile = -10
wiso_q_plev = xr.open_dataset(filenames_wiso_q_plev[ifile_start:ifile_end][ifile])

data1 = q_geo7_alltiime[expid[i]]['mon'].sel(geo_regions='Pacific Ocean')[ifile].values
data2 = (wiso_q_plev['q_17'] + wiso_q_plev['xl_17'] + wiso_q_plev['xi_17']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
=======
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

#-------- check precipitation sources are bit identical

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]
>>>>>>> Stashed changes

source_var = ['latitude', 'longitude', 'SST', 'rh2m', 'wind10', 'distance']
pre_weighted_var = {}

<<<<<<< Updated upstream
data1 = q_geo7_alltiime[expid[i]]['mon'].sel(geo_regions='Open Ocean')[ifile].values
data2 = (
    wiso_q_plev['q_15'] + wiso_q_plev['xl_15'] + wiso_q_plev['xi_15'] + \
        wiso_q_plev['q_16'] + wiso_q_plev['xl_16'] + wiso_q_plev['xi_16'] + \
            wiso_q_plev['q_17'] + wiso_q_plev['xl_17'] + wiso_q_plev['xi_17'] + \
                wiso_q_plev['q_19'] + wiso_q_plev['xl_19'] + wiso_q_plev['xi_19']).compute().squeeze().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
print(np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]) / data1[np.isfinite(data1)]))
=======
for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_lon.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
        prefix + '.transport_distance.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)
>>>>>>> Stashed changes

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

<<<<<<< Updated upstream
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
=======
for ivar in source_var:
    # ivar = 'SST'
    print('#---------------- ' + ivar)
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
        # ialltime = 'am'
        print('#-------- ' + ialltime)
        
        for i in [1, 2, 3]:
            # i = 1
            print('#---- expid 0 vs. ' + str(i))
            
            data1 = pre_weighted_var[expid[0]][ivar][ialltime].values
            data2 = pre_weighted_var[expid[i]][ivar][ialltime].values
            
            data1 = data1[np.isfinite(data1)]
            data2 = data2[np.isfinite(data2)]
            
            print((data1 == data2).all())

>>>>>>> Stashed changes

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region copy output

# import shutil

# src_exp = 'pi_600_5.0'
# # src_exp = 'pi_601_5.1'

# expid = [
#     # 'pi_602_5.2',
#     # 'pi_605_5.5',
#     # 'pi_606_5.6',
#     # 'pi_609_5.7',
#     'pi_610_5.8',
#     ]

# for var_name in ['sst', 'lat', 'rh2m', 'wind10', 'sinlon', 'coslon']:
#     print('#---------------- ' + var_name)
    
#     for i in range(len(expid)):
#         print('#-------- ' + expid[i])
        
#         input_file = exp_odir + src_exp + '/analysis/echam/' + src_exp + '.pre_weighted_' + var_name + '.pkl'
        
#         output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + var_name + '.pkl'
        
#         if (os.path.isfile(output_file)):
#             os.remove(output_file)
        
#         shutil.copy2(input_file, output_file)

# endregion
# -----------------------------------------------------------------------------

