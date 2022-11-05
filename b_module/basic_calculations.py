

# -----------------------------------------------------------------------------
# region Function to calculate time weighted mean values

def time_weighted_mean(ds):
    '''
    #---- Input
    ds: xarray.DataArray
    '''
    
    return ds.weighted(ds.time.dt.days_in_month).mean(dim='time', skipna=False)

# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region Function to calculate monthly/seasonal/annual (mean) values

def mon_sea_ann(
    var_daily = None, var_monthly = None,
    lcopy = True,
    ):
    '''
    #---- Input
    var_daily:   xarray.DataArray, daily variables, must have time dimension
    var_monthly: xarray.DataArray, monthly variables, must have time dimension
    lcopy:       whether to use copy of original var
    
    #---- Output
    var_alltime
    
    '''
    
    var_alltime = {}
    
    if not var_daily is None:
        if lcopy:
            var_alltime['daily'] = var_daily.copy()
        else:
            var_alltime['daily'] = var_daily
        
        #-------- monthly
        var_alltime['mon'] = var_daily.resample({'time': '1M'}).mean(skipna=False).compute()
        
        #-------- seasonal
        # var_alltime['sea'] = var_daily.resample({'time': 'Q-FEB'}).mean(skipna=skipna)[1:-1].compute()
        
        #-------- annual
        # var_alltime['ann'] = var_daily.resample({'time': '1Y'}).mean(skipna=skipna).compute()
        
    elif not var_monthly is None:
        #-------- monthly
        var_alltime['mon'] = var_monthly.copy()
    
    #-------- seasonal
    var_alltime['sea'] = var_alltime['mon'].resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
    
    #-------- annual
    var_alltime['ann'] = var_alltime['mon'].resample({'time': '1Y'}).map(time_weighted_mean).compute()
    
    #-------- monthly mean
    var_alltime['mm'] = var_alltime['mon'].groupby('time.month').mean(skipna=True).compute()
    
    #-------- seasonal mean
    var_alltime['sm'] = var_alltime['sea'].groupby('time.season').mean(skipna=True).compute()
    
    #-------- annual mean
    var_alltime['am'] = var_alltime['ann'].mean(dim='time', skipna=True).compute()
    return(var_alltime)




'''
#---- skipna
mon:    False
sea:    False
ann:    False
# otherwise, the results will be biased

mm:     True
sm:     True
am:     True
# it's okay to miss several month/season/year


#-------------------------------- check
import xarray as xr
import pandas as pd
import numpy as np

x = np.arange(0, 360, 1)
y = np.arange(-90, 90, 1)

#-------- check daily data

time = pd.date_range( "2001-01-01-00", "2009-12-31-00", freq="1D")

ds = xr.DataArray(
    data = np.random.rand(len(time),len(x), len(y)),
    coords={
            "time": time,
            "x": x,
            "y": y,
        }
)

ds_alltime = mon_sea_ann(ds)

# calculation in function and manually
(ds_alltime['mon'] == ds.resample({'time': '1M'}).mean()).all().values
(ds_alltime['sea'] == ds_alltime['mon'].resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()).all().values
(ds_alltime['ann'] == ds_alltime['mon'].resample({'time': '1Y'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['mm'] == ds_alltime['mon'].groupby('time.month').mean()).all().values
(ds_alltime['sm'] == ds_alltime['sea'].groupby('time.season').mean()).all().values
(ds_alltime['am'] == ds_alltime['ann'].mean(dim='time')).all().values

# mon
ilat=30
ilon=20
ds[-31:, ilat, ilon].values.mean()
ds_alltime['mon'][-1, ilat, ilon].values

# sea
ilat=30
ilon=40
ds[59:151, ilat, ilon].values.mean()
np.average(
    ds_alltime['mon'][2:5, ilat, ilon],
    weights=ds_alltime['mon'][2:5, ilat, ilon].time.dt.days_in_month,
)
ds_alltime['sea'][0, ilat, ilon].values

# ann
ilat=30
ilon=40
ds[:365, ilat, ilon].mean().values
np.average(
    ds_alltime['mon'][0:12, ilat, ilon],
    weights=ds_alltime['mon'][0:12, ilat, ilon].time.dt.days_in_month,
)
ds_alltime['ann'][0, ilat, ilon].values

# mm
ilat=30
ilon=60
ds_alltime['mon'].sel(time=(ds_alltime['mon'].time.dt.month==6).values)[:, ilat, ilon].mean().values
ds_alltime['mm'][5, ilat, ilon]

# sm
ds_alltime['sea'].sel(time=(ds_alltime['sea'].time.dt.month==8).values)[:, ilat, ilon].mean().values
ds_alltime['sm'].sel(season='JJA')[ilat, ilon].values

# am
ds_alltime['ann'][:, ilat, ilon].mean().values
ds_alltime['am'][ilat, ilon].values


#-------- check monthly data

time = pd.date_range( "2001-01-01-00", "2009-12-31-00", freq="1M")

ds = xr.DataArray(
    data = np.random.rand(len(time),len(x), len(y)),
    coords={
            "time": time,
            "x": x,
            "y": y,
        }
)

ds_alltime = mon_sea_ann(var_monthly = ds)

(ds_alltime['mon'] == ds).all().values
(ds_alltime['sea'] == ds.resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()).all().values
(ds_alltime['ann'] == ds.resample({'time': '1Y'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['mm'] == ds_alltime['mon'].groupby('time.month').mean()).all().values
(ds_alltime['sm'] == ds_alltime['sea'].groupby('time.season').mean()).all().values
(ds_alltime['am'] == ds_alltime['ann'].mean(dim='time')).all().values

# seasonal
i1 = 30
i3 = 40
i4 = 60
np.average(
    ds[(i1*3 + 2):(i1*3 + 5), i3, i4],
    weights = ds[(i1*3 + 2):(i1*3 + 5), i3, i4].time.dt.days_in_month)
ds_alltime['sea'][i1, i3, i4].values

# annual
i1 = 6
i3 = 30
i4 = 60
np.average(
    ds[(i1*12 + 0):(i1*12 + 12), i3, i4],
    weights = ds[(i1*12 + 0):(i1*12 + 12), i3, i4].time.dt.days_in_month)
ds_alltime['ann'][i1, i3, i4].values


test1 = ds.weighted(ds.time.dt.days_in_month).mean(dim='time', skipna=True).compute()
test2 = test1.values[np.isfinite(test1.values)] - ds_alltime['am'].values[np.isfinite(ds_alltime['am'].values)]
wheremax = np.where(abs(test2) == np.max(abs(test2)))
test2[wheremax]
np.max(abs(test2))
test1.values[np.isfinite(test1.values)][wheremax]
ds_alltime['am'].values[np.isfinite(ds_alltime['am'].values)][wheremax]

# (np.isfinite(test1.values) == np.isfinite(ds_alltime['am'].values)).all()


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region functions to calculate monthly/seasonal/annual weighted average

def mon_sea_ann_average(ds, average, skipna = True):
    '''
    ds: xarray.DataArray, monthly mean values
    average: 'time.month', 'time.season', 'time.year'
    '''
    month_length = ds.time.dt.days_in_month
    
    weights = (
        month_length.groupby(average) /
        month_length.groupby(average).sum()
    )
    
    ds_weighted = (
        ds * weights).groupby(average).sum(dim="time", skipna=skipna)
    
    # Calculate the weighted average
    return ds_weighted

'''
# check the monthly average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month

weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

pre_mon_average0 = (pre * weights).groupby('time.month').sum(dim="time")
pre_mon_average1 = mon_sea_ann_average(pre, 'time.month')
(pre_mon_average0 == pre_mon_average1).all()


# check the seasonal average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month
weights = (
    month_length.groupby("time.season") /
    month_length.groupby("time.season").sum()
    )
pre_sea_average0 = (pre * weights).groupby("time.season").sum(dim="time")
pre_sea_average1 = mon_sea_ann_average(pre, 'time.season')
(pre_sea_average0 == pre_sea_average1).all()

# check the annual average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
    )
pre_ann_average0 = (pre * weights).groupby("time.year").sum(dim="time")
pre_ann_average1 = mon_sea_ann_average(pre, 'time.year')
(pre_ann_average0 == pre_ann_average1).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region functions to regrid a dataset to another grid

def regrid(
    ds_in, ds_out=None, grid_spacing=1, method='bilinear',
    periodic=True, ignore_degenerate=True, unmapped_to_nan=True,
    extrap_method='nearest_s2d'):
    '''
    ds_in: original xarray.DataArray
    ds_out: xarray.DataArray with target grid, default None
    grid_spacing: 1
    periodic: True, When dealing with global grids, we need to set periodic=True, otherwise data along the meridian line will be missing.
    ignore_degenerate: Ignore degenerate cells when checking the input Grids or Meshes for errors. If this is set to True, then the regridding proceeds, but degenerate cells will be skipped. If set to False, a degenerate cell produces an error. This currently only applies to CONSERVE, other regrid methods currently always skip degenerate cells. If None, defaults to False.
    '''
    
    import xesmf as xe
    
    ds_in_copy = ds_in.copy()
    
    if (ds_out is None):
        ds_out = xe.util.grid_global(grid_spacing, grid_spacing)
    
    regridder = xe.Regridder(
        ds_in_copy, ds_out, method, periodic=periodic,
        ignore_degenerate=ignore_degenerate, unmapped_to_nan=unmapped_to_nan,
        extrap_method=extrap_method,)
    return regridder(ds_in_copy)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region function to check if a point lies within a Path
# Attention!!!: It does not work with South Pole

def points_in_polygon(lon, lat, polygon,):
    '''
    ---- Input
    lon: a numpy array of lon of points, 1d or 2d
    lat: a numpy array of lat of points, 1d or 2d
    polygon: a list of [(lon, lat), ...] pair or a matplotlib.path.Path object
    
    ---- Output
    
    '''
    import numpy as np
    from matplotlib.path import Path
    
    coors = np.hstack((lon.reshape(-1, 1), lat.reshape(-1, 1)))
    
    if (type(polygon) == list):
        polygon = Path(polygon)
    
    mask = polygon.contains_points(coors)
    
    if (lon.shape[-1] > 1):
        mask = mask.reshape(lon.shape[0], lon.shape[1])
    
    mask01 = np.zeros(lon.shape)
    mask01[mask] = 1
    
    return (mask, mask01)

'''
EAIS_mask, EAIS_mask01 = points_in_polygon(
    lon, lat, Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords),
                   closed=True))

# EAIS_path = Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords))
# WAIS_path = Path(list(ais_imbie2.to_crs(4326).geometry[1].exterior.coords))
# ap_path = Path(list(ais_imbie2.to_crs(4326).geometry[3].exterior.coords))
# np.unique(Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords),
# closed=True).codes, return_counts=True)

ax.contour(lon, lat, EAIS_mask01, colors='blue', levels=np.array([0.5]),
transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region function to generate mask for three AIS

def create_ais_mask(lon, lat, ais_shpfile, cellarea):
    '''
    #---- Input
    lon:         numpy.ndarray, 1D longitude
    lat:         numpy.ndarray, 1D latitude
    ais_shpfile: geopandas.geodataframe.GeoDataFrame, returned by gpd.read_file('*.shp')
    cellarea:    numpy.ndarray, 2D cell areas
    
    #---- Output
    ais_mask: dictionary, {'mask', 'mask01', 'cellarea'}
    '''
    
    import numpy as np
    import geopandas as gpd
    from geopandas.tools import sjoin
    
    lon_2d, lat_2d = np.meshgrid(lon, lat,)
    
    lon_lat = gpd.GeoDataFrame(
        crs="EPSG:4326", geometry=gpd.points_from_xy(
            lon_2d.reshape(-1, 1), lat_2d.reshape(-1, 1))).to_crs(3031)
    pointInPolys = sjoin(lon_lat, ais_shpfile, how='left')
    regions = pointInPolys.Regions.to_numpy().reshape(
        lon_2d.shape[0], lon_2d.shape[1])
    
    ais_mask = {}
    ais_mask['mask'] = {}
    ais_mask['mask']['EAIS'] = (regions == 'East')
    ais_mask['mask']['WAIS'] = (regions == 'West')
    ais_mask['mask']['AP'] = (regions == 'Peninsula')
    ais_mask['mask']['AIS'] = (ais_mask['mask']['EAIS'] | ais_mask['mask']['WAIS'] | ais_mask['mask']['AP'])
    
    ais_mask['mask01'] = {}
    ais_mask['mask01']['EAIS'] = np.zeros(regions.shape)
    ais_mask['mask01']['EAIS'][ais_mask['mask']['EAIS']] = 1
    
    ais_mask['mask01']['WAIS'] = np.zeros(regions.shape)
    ais_mask['mask01']['WAIS'][ais_mask['mask']['WAIS']] = 1
    
    ais_mask['mask01']['AP'] = np.zeros(regions.shape)
    ais_mask['mask01']['AP'][ais_mask['mask']['AP']] = 1
    
    ais_mask['mask01']['AIS'] = np.zeros(regions.shape)
    ais_mask['mask01']['AIS'][ais_mask['mask']['AIS']] = 1
    
    ais_mask['cellarea'] = {}
    ais_mask['cellarea']['EAIS'] = cellarea[ais_mask['mask']['EAIS']].sum() / 10**6
    ais_mask['cellarea']['WAIS'] = cellarea[ais_mask['mask']['WAIS']].sum() / 10**6
    ais_mask['cellarea']['AP'] = cellarea[ais_mask['mask']['AP']].sum() / 10**6
    ais_mask['cellarea']['AIS'] = cellarea[ais_mask['mask']['AIS']].sum() / 10**6
    
    return(ais_mask)


'''
#-------------------------------- check
from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,)
from matplotlib.colors import BoundaryNorm
from matplotlib import cm, pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
mpl.rcParams['axes.linewidth'] = 0.2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import xarray as xr
import numpy as np

echam6_t63_slm = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_slm.nc')
with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)

#-------- cell area is on the right order
echam6_t63_ais_mask['cellarea']
# 9759740.151912227 + 2144474.2285643886 + 265318.13031548966 = 12169532.510792103


#-------- mask01 looks right

fig, ax = hemisphere_plot(northextent=-60)

ax.contour(
    echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
    echam6_t63_ais_mask['mask01']['EAIS'],
    colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
    echam6_t63_ais_mask['mask01']['WAIS'],
    colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
    echam6_t63_ais_mask['mask01']['AP'],
    colors='yellow', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
# ax.contour(
#     echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
#     echam6_t63_ais_mask['mask01']['AIS'],
#     colors='m', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/test.png')

((echam6_t63_ais_mask['mask01']['EAIS'] == 1) == echam6_t63_ais_mask['mask']['EAIS']).all()
((echam6_t63_ais_mask['mask01']['WAIS'] == 1) == echam6_t63_ais_mask['mask']['WAIS']).all()
((echam6_t63_ais_mask['mask01']['AP'] == 1) == echam6_t63_ais_mask['mask']['AP']).all()
((echam6_t63_ais_mask['mask01']['AIS'] == 1) == echam6_t63_ais_mask['mask']['AIS']).all()



#-------------------------------- derivation

one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')
lon, lat = np.meshgrid(
    one_degree_grids_cdo_area.lon, one_degree_grids_cdo_area.lat, )

ais_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp'
)

#1 geopandas.tools.sjoin point and polygon geometry
lon_lat = gpd.GeoDataFrame(
    crs="EPSG:4326", geometry=gpd.points_from_xy(
        lon.reshape(-1, 1), lat.reshape(-1, 1))).to_crs(3031)

from geopandas.tools import sjoin
pointInPolys = sjoin(lon_lat, ais_imbie2, how='left')
# pointInPolys = pointInPolys.groupby([pointInPolys.index], as_index=False).nth(0)
regions = pointInPolys.Regions.to_numpy().reshape(lon.shape[0], lon.shape[1])
# pointInPolys.Regions.value_counts(dropna=False)

EAIS_mask = (regions == 'East')
EAIS_mask01 = np.zeros(EAIS_mask.shape)
EAIS_mask01[EAIS_mask] = 1

WAIS_mask = (regions == 'West')
WAIS_mask01 = np.zeros(WAIS_mask.shape)
WAIS_mask01[WAIS_mask] = 1

ap_mask = (regions == 'Peninsula')
ap_mask01 = np.zeros(ap_mask.shape)
ap_mask01[ap_mask] = 1

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to calculate area-weighted average over AIS

def mean_over_ais(
    ds,
    ais_mask,
    cell_area,
    ):
    '''
    #---- Input
    ds: xarray.DataArray, three dimension (time, lat, lon)
    ais_mask: np.ndarray, two dimension (lat, lon)
    cell_area: np.ndarray, two dimension (lat, lon)
    
    #---- Output
    ds_mean_over_ais: xarray.DataArray, one dimension (time)
    
    '''
    
    import numpy as np
    
    ds_mean_over_ais = ds.mean(axis=(1, 2)).compute()
    ds_mean_over_ais.values[:] = 0
    
    for itime in range(len(ds_mean_over_ais)):
        # itime = 0
        ds_mean_over_ais.values[itime] = np.average(
            ds[itime].values[ais_mask],
            weights = cell_area[ais_mask],
        )
    
    return(ds_mean_over_ais)


'''
#-------------------------------- check
import xarray as xr
import pickle
import numpy as np
import pandas as pd
with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

x = np.linspace(0, 360, echam6_t63_ais_mask['mask']['AIS'].shape[1])
y = np.linspace(-90, 90, echam6_t63_ais_mask['mask']['AIS'].shape[0])
time = pd.date_range( "2001-01-01-00", "2009-12-31-00", freq="1M")

ds = xr.DataArray(
    data = np.random.rand(len(time),len(y), len(x)),
    coords={
            "time": time,
            "y": y,
            "x": x,
        }
)

ds_mean_over_ais = mean_over_ais(
    ds,
    echam6_t63_ais_mask['mask']['AIS'],
    echam6_t63_cellarea.cell_area.values,
    )

itime = 40
np.average(
    ds[itime,].values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)

ds_mean_over_ais[itime].values


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region find ilat/ilon for site lat/lon

def find_ilat_ilon(slat, slon, lat, lon):
    '''
    #-------- Input
    slat: latitude in the range of [-90, 90], scalar
    slon: longitude in the range of [0, 360], scalar
    
    lat:  latitude in the range of [-90, 90], 1d array
    lon:  longitude in the range of [0, 360], 1d array
    '''
    
    import numpy as np
    
    # scale longitude to be between [0, 360]
    if (slon < 0):
        slon += 360
    
    lon[lon < 0] += 360
    
    ilon = np.where(abs(slon - lon) == np.min(abs(slon - lon)))[0][0]
    ilat = np.where(abs(slat - lat) == np.min(abs(slat - lat)))[0][0]
    
    return([ilat, ilon])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region find ilat/ilon for site lat/lon: general approach

def find_ilat_ilon_general(slat, slon, lat, lon):
    '''
    #-------- Input
    slat: site latitude, scalar
    slon: site longitude, scalar
    
    lat:  latitude, 1d or 2d array
    lon:  longitude, 1d or 2d array
    '''
    
    import numpy as np
    from haversine import haversine_vector
    
    if (lon.ndim == 2):
        lon1d = lon.reshape(-1, 1)
        lat1d = lat.reshape(-1, 1)
    elif (lon.ndim == 1):
        lon1d = lon
        lat1d = lat
    
    slocation_pair = [slat, slon]
    location_pairs = [[x, y] for x, y in zip(lat1d, lon1d)]
    
    distances1d = haversine_vector(
        slocation_pair, location_pairs, comb=True, normalize=True,
        )
    
    if (lon.ndim == 2):
        distances = distances1d.reshape(lon.shape)
    elif (lon.ndim == 1):
        distances = distances1d
    
    wheremin = np.where(distances == np.min(distances))
    
    iind0 = wheremin[0][0]
    iind1 = wheremin[-1][0]
    
    return([iind0, iind1])


'''
#-------- check

from haversine import haversine

model = list(lig_sst.keys())[-1]
# model = 'AWI-ESM-1-1-LR'

slat = ec_sst_rec['original'].Latitude[2]
slon = ec_sst_rec['original'].Longitude[2]
lon = pi_sst[model].lon.values
lat = pi_sst[model].lat.values

iind0, iind1 = find_ilat_ilon_general(slat, slon, lat, lon)

if (lon.ndim == 2):
    print(lon[iind0, iind1])
    print(lat[iind0, iind1])
elif (lon.ndim == 1):
    print(lon[iind0])
    print(lat[iind0])

print(slon)
print(slat)

if (lon.ndim == 2):
    print(haversine([slat, slon], [lat[iind0, iind1], lon[iind0, iind1]]))
elif (lon.ndim == 1):
    print(haversine([slat, slon], [lat[iind0], lon[iind0]]))


'''
# endregion
# -----------------------------------------------------------------------------


