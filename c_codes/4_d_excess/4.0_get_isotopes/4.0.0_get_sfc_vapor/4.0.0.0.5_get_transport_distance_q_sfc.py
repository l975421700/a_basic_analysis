#SBATCH --time=00:30:00


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    
    'nudged_703_6.0_k52',
    ]
i = 0

# -----------------------------------------------------------------------------
# region import packages

# management
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
import datetime

# data analysis
import numpy as np
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from haversine import haversine_vector


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

q_sfc_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lon.pkl', 'rb') as f:
    q_sfc_weighted_lon[expid[i]] = pickle.load(f)

q_sfc_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_lat.pkl', 'rb') as f:
    q_sfc_weighted_lat[expid[i]] = pickle.load(f)

lon = q_sfc_weighted_lat[expid[i]]['am'].lon
lat = q_sfc_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


'''
import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get transport distance

q_sfc_transport_distance = {}
q_sfc_transport_distance[expid[i]] = {}

begin_time = datetime.datetime.now()
print(begin_time)

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'daily'
    # ialltime = 'ann'
    
    q_sfc_transport_distance[expid[i]][ialltime] = q_sfc_weighted_lat[expid[i]][ialltime].copy().rename('q_sfc_transport_distance')
    q_sfc_transport_distance[expid[i]][ialltime][:] = 0
    
    if (ialltime in ['daily', 'mon', 'sea', 'ann']):
        print(ialltime)
        
        years = np.unique(q_sfc_transport_distance[expid[i]][ialltime].time.dt.year)
        for iyear in years:
            # iyear = 2010
            print(str(iyear) + ' / ' + str(years[-1]))
            
            time_indices = np.where(
                q_sfc_transport_distance[expid[i]][ialltime].time.dt.year == iyear)
            
            b_lon_2d = np.broadcast_to(
                lon_2d,
                q_sfc_transport_distance[expid[i]][ialltime][time_indices].shape,
                )
            b_lat_2d = np.broadcast_to(
                lat_2d,
                q_sfc_transport_distance[expid[i]][ialltime][time_indices].shape,
                )
            b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
            b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
            local_pairs = [[x, y] for x, y in zip(b_lat_2d_flatten, b_lon_2d_flatten)]
            
            lon_src_flatten = q_sfc_weighted_lon[expid[i]][
                ialltime][time_indices].values.reshape(-1, 1).copy()
            lat_src_flatten = q_sfc_weighted_lat[expid[i]][
                ialltime][time_indices].values.reshape(-1, 1).copy()
            source_pairs = [[x, y] for x, y in zip(
                lat_src_flatten, lon_src_flatten)]
            
            q_sfc_transport_distance[expid[i]][ialltime][time_indices] = \
                haversine_vector(
                local_pairs, source_pairs, normalize=True).reshape(
                    q_sfc_transport_distance[expid[i]][ialltime][time_indices].shape)
            
            print(datetime.datetime.now() - begin_time)
            
    elif (ialltime in ['mm', 'sm', 'am']):
        print(ialltime)
        b_lon_2d = np.broadcast_to(
            lon_2d, q_sfc_weighted_lat[expid[i]][ialltime].shape, )
        b_lat_2d = np.broadcast_to(
            lat_2d, q_sfc_weighted_lat[expid[i]][ialltime].shape, )
        b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
        b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
        local_pairs = [[x, y] for x, y in zip(b_lat_2d_flatten, b_lon_2d_flatten)]

        lon_src_flatten = q_sfc_weighted_lon[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        lat_src_flatten = q_sfc_weighted_lat[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        source_pairs = [[x, y] for x, y in zip(lat_src_flatten, lon_src_flatten)]

        q_sfc_transport_distance[expid[i]][ialltime][:] = haversine_vector(
                    local_pairs, source_pairs, normalize=True).reshape(
                        q_sfc_weighted_lat[expid[i]][ialltime].shape)

#-------- monthly without monthly mean
q_sfc_transport_distance[expid[i]]['mon no mm'] = (q_sfc_transport_distance[expid[i]]['mon'].groupby('time.month') - q_sfc_transport_distance[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
q_sfc_transport_distance[expid[i]]['ann no am'] = (q_sfc_transport_distance[expid[i]]['ann'] - q_sfc_transport_distance[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_transport_distance.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_sfc_transport_distance[expid[i]], f)




'''
#-------------------------------- check

from haversine import haversine

q_sfc_transport_distance = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_transport_distance.pkl', 'rb') as f:
    q_sfc_transport_distance[expid[i]] = pickle.load(f)

ilat = 50
ilon = 90

for ialltime in ['daily', 'mon', 'ann', 'mm', 'sm']:
    # ialltime = 'mm'
    itime = -3
    
    local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
    source = [q_sfc_weighted_lat[expid[i]][ialltime][itime, ilat, ilon].values,
              q_sfc_weighted_lon[expid[i]][ialltime][itime, ilat, ilon].values,]

    print(haversine(local, source, normalize=True))
    print(q_sfc_transport_distance[expid[i]][ialltime][itime, ilat, ilon].values)

ialltime = 'am'

local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
source = [q_sfc_weighted_lat[expid[i]][ialltime][ilat, ilon].values,
          q_sfc_weighted_lon[expid[i]][ialltime][ilat, ilon].values,]

print(haversine(local, source, normalize=True))
print(q_sfc_transport_distance[expid[i]][ialltime][ilat, ilon].values)




'''
# endregion
# -----------------------------------------------------------------------------


