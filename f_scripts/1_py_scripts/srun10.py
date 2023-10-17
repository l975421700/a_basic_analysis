#SBATCH --time=00:30:00


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'hist_700_5.0',
    'nudged_701_5.0',
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

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


'''
print(psutil.Process().memory_info().rss / (2 ** 30))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get transport distance

transport_distance = {}
transport_distance[expid[i]] = {}

begin_time = datetime.datetime.now()
print(begin_time)

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'daily'
    # ialltime = 'ann'
    
    transport_distance[expid[i]][ialltime] = pre_weighted_lat[expid[i]][ialltime].copy().rename('transport_distance')
    transport_distance[expid[i]][ialltime][:] = 0
    
    if (ialltime in ['daily', 'mon', 'sea', 'ann']):
        print(ialltime)
        
        years = np.unique(transport_distance[expid[i]][ialltime].time.dt.year)
        for iyear in years:
            # iyear = 2010
            print(str(iyear) + ' / ' + str(years[-1]))
            
            time_indices = np.where(
                transport_distance[expid[i]][ialltime].time.dt.year == iyear)
            
            b_lon_2d = np.broadcast_to(
                lon_2d,
                transport_distance[expid[i]][ialltime][time_indices].shape,
                )
            b_lat_2d = np.broadcast_to(
                lat_2d,
                transport_distance[expid[i]][ialltime][time_indices].shape,
                )
            b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
            b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
            local_pairs = [[x, y] for x, y in zip(b_lat_2d_flatten, b_lon_2d_flatten)]
            
            lon_src_flatten = pre_weighted_lon[expid[i]][
                ialltime][time_indices].values.reshape(-1, 1).copy()
            lat_src_flatten = pre_weighted_lat[expid[i]][
                ialltime][time_indices].values.reshape(-1, 1).copy()
            source_pairs = [[x, y] for x, y in zip(
                lat_src_flatten, lon_src_flatten)]
            
            transport_distance[expid[i]][ialltime][time_indices] = \
                haversine_vector(
                local_pairs, source_pairs, normalize=True).reshape(
                    transport_distance[expid[i]][ialltime][time_indices].shape)
            
            print(datetime.datetime.now() - begin_time)
            
    elif (ialltime in ['mm', 'sm', 'am']):
        print(ialltime)
        b_lon_2d = np.broadcast_to(
            lon_2d, pre_weighted_lat[expid[i]][ialltime].shape, )
        b_lat_2d = np.broadcast_to(
            lat_2d, pre_weighted_lat[expid[i]][ialltime].shape, )
        b_lon_2d_flatten = b_lon_2d.reshape(-1, 1)
        b_lat_2d_flatten = b_lat_2d.reshape(-1, 1)
        local_pairs = [[x, y] for x, y in zip(b_lat_2d_flatten, b_lon_2d_flatten)]

        lon_src_flatten = pre_weighted_lon[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        lat_src_flatten = pre_weighted_lat[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        source_pairs = [[x, y] for x, y in zip(lat_src_flatten, lon_src_flatten)]

        transport_distance[expid[i]][ialltime][:] = haversine_vector(
                    local_pairs, source_pairs, normalize=True).reshape(
                        pre_weighted_lat[expid[i]][ialltime].shape)

#-------- monthly without monthly mean
transport_distance[expid[i]]['mon no mm'] = (transport_distance[expid[i]]['mon'].groupby('time.month') - transport_distance[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
transport_distance[expid[i]]['ann no am'] = (transport_distance[expid[i]]['ann'] - transport_distance[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(transport_distance[expid[i]], f)


'''
#-------------------------------- check

from geopy.distance import geodesic, great_circle
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from haversine import haversine, Unit, haversine_vector

transport_distance = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.transport_distance.pkl', 'rb') as f:
    transport_distance[expid[i]] = pickle.load(f)

transport_distance[expid[i]]['am'].to_netcdf('scratch/test/test.nc')

ilat = 40
ilon = 90

for ialltime in ['daily', 'mon', 'ann', 'mm', 'sm']:
    # ialltime = 'mm'
    itime = -4
    
    local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
    source = [pre_weighted_lat[expid[i]][ialltime][itime, ilat, ilon].values,
              pre_weighted_lon[expid[i]][ialltime][itime, ilat, ilon].values,]

    # print(geodesic(local, source).km)
    # print(great_circle(local, source).km)

    # local_in_radians = [radians(_) for _ in local]
    # source_in_radians = [radians(_) for _ in source]
    # result = haversine_distances([local_in_radians, source_in_radians])
    # print((result * 6371000/1000)[0, 1])

    print(haversine(local, source, normalize=True))

    print(transport_distance[expid[i]][ialltime][itime, ilat, ilon].values)

ialltime = 'am'

local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
source = [pre_weighted_lat[expid[i]][ialltime][ilat, ilon].values,
          pre_weighted_lon[expid[i]][ialltime][ilat, ilon].values,]

print(haversine(local, source, normalize=True))
print(transport_distance[expid[i]][ialltime][ilat, ilon].values)




#-------- Function to normalize longitude

def lon_180(lon):
    lon_copy = lon.copy()
    
    if (type(lon_copy) != np.float64):
        lon_copy[lon_copy>180] -= 360
    elif (lon_copy > 180):
        lon_copy -= - 360
    
    return(lon_copy)




#-------- previous trial by calculating in order

            # for ilat in range(len(lat)):
            #     for ilon in range(len(lon)):
                    
            #         # itime = 0; ilat = 0; ilon = 0
                    
            #         local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
            #         source = [
            #             pre_weighted_lat[expid[i]][ialltime][
            #                 itime, ilat, ilon].values,
            #             pre_weighted_lon[expid[i]][ialltime][
            #                 itime, ilat, ilon].values,]
                    
            #         if (np.isnan(source).sum() > 0):
            #             transport_distance[expid[i]][ialltime][
            #                 itime, ilat, ilon] = np.nan
            #         else:
            #             transport_distance[expid[i]][ialltime][
            #                 itime, ilat, ilon] = geodesic(local, source).km

        
        # for ilat in range(len(lat)):
        #     for ilon in range(len(lon)):
                
        #         # ilat = 48; ilon = 96
                
        #         local = [lat_2d[ilat, ilon], lon_2d[ilat, ilon]]
        #         source = [
        #             pre_weighted_lat[expid[i]][ialltime][ilat, ilon].values,
        #             pre_weighted_lon[expid[i]][ialltime][ilat, ilon].values,]
                
        #         if (np.isnan(source).sum() > 0):
        #             transport_distance[expid[i]][ialltime][
        #                 ilat, ilon] = np.nan
        #         else:
        #             transport_distance[expid[i]][ialltime][
        #                 ilat, ilon] = geodesic(local, source)


#-------- previous trial calculate for each timestep

    if (ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']):
        
        transport_distance[expid[i]][ialltime] = pre_weighted_lat[expid[i]][ialltime].rename('transport_distance')
        transport_distance[expid[i]][ialltime][:] = 0
        
        b_lon_2d = np.broadcast_to(
                lon_2d, pre_weighted_lat[expid[i]][ialltime].shape, )
        b_lat_2d = np.broadcast_to(
            lat_2d, pre_weighted_lat[expid[i]][ialltime].shape, )
        b_lon_2d_flatten = b_lon_2d.reshape(-1, 1).copy()
        b_lat_2d_flatten = b_lat_2d.reshape(-1, 1).copy()
        
        for itime in range(pre_weighted_lat[expid[i]][ialltime].shape[0]):
            # itime = 0
            
            
            
            
            
            local_pairs = [[x, y] for x, y in zip(lat_2d_flatten, lon_2d_flatten)]
            
            
            lon_src_flatten = pre_weighted_lon[expid[i]][ialltime][
                itime].values.reshape(-1, 1).copy()
            lat_src_flatten = pre_weighted_lat[expid[i]][ialltime][
                itime].values.reshape(-1, 1).copy()
            source_pairs = [[x, y] for x, y in zip(lat_src_flatten, lon_src_flatten)]
            
            transport_distance[expid[i]][ialltime][itime, ] = haversine_vector(
                local_pairs, source_pairs, normalize=True).reshape(lon_2d.shape)
            
            if (itime % 100 == 0):
                print(str(itime) + ': ' + str(datetime.datetime.now() - begin_time))
            
    elif (ialltime in ['am']):
        # ialltime = 'am'
        print(ialltime)
        transport_distance[expid[i]][ialltime] = pre_weighted_lat[expid[i]][ialltime].rename('transport_distance')
        transport_distance[expid[i]][ialltime][:] = 0
        
        lon_src_flatten = pre_weighted_lon[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        lat_src_flatten = pre_weighted_lat[expid[i]][
            ialltime].values.reshape(-1, 1).copy()
        source_pairs = [[x, y] for x, y in zip(lat_src_flatten, lon_src_flatten)]
        
        transport_distance[expid[i]][ialltime][:] = haversine_vector(
            local_pairs, source_pairs, normalize=True).reshape(lon_2d.shape)

'''
# endregion
# -----------------------------------------------------------------------------


