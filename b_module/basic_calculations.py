

# -----------------------------------------------------------------------------
# region Function to calculate monthly/seasonal/annual (mean) values

def mon_sea_ann(
    var_daily = None, var_monthly = None, skipna = True,
    ):
    '''
    #---- Input
    var_daily:   xarray.DataArray, daily variables, must have time dimension
    var_monthly: xarray.DataArray, monthly variables, must have time dimension
    
    #---- Output
    var_alltime
    
    '''
    
    var_alltime = {}
    
    if var_monthly is None:
        var_alltime['daily'] = var_daily.copy()
        var_alltime['daily'].values[var_alltime['daily'].values < 2e-8] = 0
        
        #-------- monthly
        var_alltime['mon'] = var_daily.resample({'time': '1M'}).mean(skipna=skipna).compute()
        
        #-------- seasonal
        var_alltime['sea'] = var_daily.resample({'time': 'Q-FEB'}).mean(skipna=skipna)[1:-1].compute()
        
        #-------- annual
        var_alltime['ann'] = var_daily.resample({'time': '1Y'}).mean(skipna=skipna).compute()
        
        #-------- monthly mean
        var_alltime['mm'] = var_alltime['mon'].groupby('time.month').mean(skipna=skipna).compute()
        
        #-------- seasonal mean
        var_alltime['sm'] = var_alltime['sea'].groupby('time.season').mean(skipna=skipna).compute()
        
        #-------- annual mean
        var_alltime['am'] = var_alltime['ann'].mean(dim='time', skipna=skipna).compute()
        
    # else:
        #-------- monthly
        
        
        #-------- seasonal
        
        
        #-------- annual
        
        
        #-------- monthly mean
        
        
        #-------- seasonal mean
        
        
        #-------- annual mean
        
    
    return(var_alltime)


'''
#-------------------------------- check with daily values
#-------- monthly pre

ocean_pre_mon = ocean_pre.resample({'time': '1M'}).mean()
var_scaled_pre_mon = var_scaled_pre.resample({'time': '1M'}).mean()

#-------- seasonal pre

ocean_pre_sea = ocean_pre.resample({'time': 'Q-FEB'}).mean()[1:-1]
var_scaled_pre_sea = var_scaled_pre.resample({'time': 'Q-FEB'}).mean()[1:-1]

#-------- annual pre
ocean_pre_ann = ocean_pre.resample({'time': '1Y'}).mean()
var_scaled_pre_ann = var_scaled_pre.resample({'time': '1Y'}).mean()

#-------- monthly mean pre
ocean_pre_mm = ocean_pre_mon.groupby('time.month').mean()
var_scaled_pre_mm = var_scaled_pre_mon.groupby('time.month').mean()


#-------- seasonal mean pre
ocean_pre_sm = ocean_pre_sea.groupby('time.season').mean()
var_scaled_pre_sm = var_scaled_pre_sea.groupby('time.season').mean()

#-------- annual mean pre
ocean_pre_am = ocean_pre_ann.mean(dim='time')
var_scaled_pre_am = var_scaled_pre_ann.mean(dim='time')

(ocean_pre_alltime['mon'] == ocean_pre_mon).all()
(ocean_pre_alltime['sea'] == ocean_pre_sea).all()
(ocean_pre_alltime['ann'] == ocean_pre_ann).all()
(ocean_pre_alltime['mm'] == ocean_pre_mm).all()
(ocean_pre_alltime['sm'] == ocean_pre_sm).all()
(ocean_pre_alltime['am'] == ocean_pre_am).all()
(var_scaled_pre_alltime['mon'] == var_scaled_pre_mon).all()
(var_scaled_pre_alltime['sea'] == var_scaled_pre_sea).all()
(var_scaled_pre_alltime['ann'] == var_scaled_pre_ann).all()
(var_scaled_pre_alltime['mm'] == var_scaled_pre_mm).all()
(var_scaled_pre_alltime['sm'] == var_scaled_pre_sm).all()
(var_scaled_pre_alltime['am'] == var_scaled_pre_am).all()


#-------------------------------- minor checks during calculation

#---- monthly
ocean_pre[-31:, 30, 30].values.mean()
ocean_pre.resample({'time': '1M'}).mean()[-1, 30, 30].values

var_scaled_pre[-31:, 30, 30].values.mean()
var_scaled_pre.resample({'time': '1M'}).mean()[-1, 30, 30].values

#---- seasonally
ocean_pre[:60, 30, 30].values.mean()
ocean_pre.resample({'time': 'Q-FEB'}).mean()[0, 30, 30].values

ocean_pre[-31:, 30, 30].values.mean()
ocean_pre.resample({'time': 'Q-FEB'}).mean()[-1, 30, 30].values
# ocean_pre.resample({'time': 'QS-DEC'}).mean()

var_scaled_pre[:60, 30, 30].values.mean()
var_scaled_pre.resample({'time': 'Q-FEB'}).mean()[0, 30, 30].values

var_scaled_pre[-31:, 30, 30].values.mean()
var_scaled_pre.resample({'time': 'Q-FEB'}).mean()[-1, 30, 30].values

#---- annual
ocean_pre.resample({'time': '1Y'}).mean()[0, 40, 50].values
ocean_pre[:366, 40, 50].mean().values
var_scaled_pre.resample({'time': '1Y'}).mean()[0, 40, 50].values
var_scaled_pre[:366, 40, 50].mean().values

#---- monthly mean
ocean_pre_mon.sel(time=(ocean_pre_mon.time.dt.month==6).values)[:, 30, 30].mean().values
ocean_pre_mon.groupby('time.month').mean()[5, 30, 30].values
var_scaled_pre_mon.sel(time=(var_scaled_pre_mon.time.dt.month==6).values)[:, 30, 30].mean().values
var_scaled_pre_mon.groupby('time.month').mean()[5, 30, 30].values

#---- seasonal mean
ocean_pre_sea.sel(time=(ocean_pre_sea.time.dt.month==8).values)[:, 30, 30].mean().values
ocean_pre_sm.sel(season='JJA')[30,30].values
var_scaled_pre_sea.sel(time=(var_scaled_pre_sea.time.dt.month==8).values)[:, 30, 30].mean().values
var_scaled_pre_sm.sel(season='JJA')[30,30].values

#---- annual mean
ocean_pre_ann.mean(dim='time')[30, 30].values
ocean_pre_ann[:, 30, 30].mean().values
var_scaled_pre_ann.mean(dim='time')[30, 30].values
var_scaled_pre_ann[:, 30, 30].mean().values

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

def regrid(ds_in, ds_out=None, grid_spacing=1, method='bilinear',
           periodic=True, ignore_degenerate=False):
    '''
    ds_in: original xarray.DataArray
    ds_out: xarray.DataArray with target grid, default None
    grid_spacing: 0.25
    '''
    
    import xesmf as xe
    
    ds_in_copy = ds_in.copy()
    
    if (ds_out is None):
        ds_out = xe.util.grid_global(grid_spacing, grid_spacing)
    
    regridder = xe.Regridder(
        ds_in_copy, ds_out, method, periodic=periodic,
        ignore_degenerate=ignore_degenerate,)
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
eais_mask, eais_mask01 = points_in_polygon(
    lon, lat, Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords),
                   closed=True))

# eais_path = Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords))
# wais_path = Path(list(ais_imbie2.to_crs(4326).geometry[1].exterior.coords))
# ap_path = Path(list(ais_imbie2.to_crs(4326).geometry[3].exterior.coords))
# np.unique(Path(list(ais_imbie2.to_crs(4326).geometry[2].exterior.coords),
# closed=True).codes, return_counts=True)

ax.contour(lon, lat, eais_mask01, colors='blue', levels=np.array([0.5]),
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
    ais_mask['mask']['eais'] = (regions == 'East')
    ais_mask['mask']['wais'] = (regions == 'West')
    ais_mask['mask']['ap'] = (regions == 'Peninsula')
    ais_mask['mask']['ais'] = (ais_mask['mask']['eais'] | ais_mask['mask']['wais'] | ais_mask['mask']['ap'])
    
    ais_mask['mask01'] = {}
    ais_mask['mask01']['eais'] = np.zeros(regions.shape)
    ais_mask['mask01']['eais'][ais_mask['mask']['eais']] = 1
    
    ais_mask['mask01']['wais'] = np.zeros(regions.shape)
    ais_mask['mask01']['wais'][ais_mask['mask']['wais']] = 1
    
    ais_mask['mask01']['ap'] = np.zeros(regions.shape)
    ais_mask['mask01']['ap'][ais_mask['mask']['ap']] = 1
    
    ais_mask['mask01']['ais'] = np.zeros(regions.shape)
    ais_mask['mask01']['ais'][ais_mask['mask']['ais']] = 1
    
    ais_mask['cellarea'] = {}
    ais_mask['cellarea']['eais'] = cellarea[ais_mask['mask']['eais']].sum() / 10**6
    ais_mask['cellarea']['wais'] = cellarea[ais_mask['mask']['wais']].sum() / 10**6
    ais_mask['cellarea']['ap'] = cellarea[ais_mask['mask']['ap']].sum() / 10**6
    ais_mask['cellarea']['ais'] = cellarea[ais_mask['mask']['ais']].sum() / 10**6
    
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

# ax.contour(
#     echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
#     echam6_t63_ais_mask['mask01']['eais'],
#     colors='blue', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
# ax.contour(
#     echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
#     echam6_t63_ais_mask['mask01']['wais'],
#     colors='red', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
# ax.contour(
#     echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
#     echam6_t63_ais_mask['mask01']['ap'],
#     colors='yellow', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
    echam6_t63_ais_mask['mask01']['ais'],
    colors='m', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/test.png')

((echam6_t63_ais_mask['mask01']['eais'] == 1) == echam6_t63_ais_mask['mask']['eais']).all()
((echam6_t63_ais_mask['mask01']['wais'] == 1) == echam6_t63_ais_mask['mask']['wais']).all()
((echam6_t63_ais_mask['mask01']['ap'] == 1) == echam6_t63_ais_mask['mask']['ap']).all()
((echam6_t63_ais_mask['mask01']['ais'] == 1) == echam6_t63_ais_mask['mask']['ais']).all()



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

eais_mask = (regions == 'East')
eais_mask01 = np.zeros(eais_mask.shape)
eais_mask01[eais_mask] = 1

wais_mask = (regions == 'West')
wais_mask01 = np.zeros(wais_mask.shape)
wais_mask01[wais_mask] = 1

ap_mask = (regions == 'Peninsula')
ap_mask01 = np.zeros(ap_mask.shape)
ap_mask01[ap_mask] = 1

'''
# endregion
# -----------------------------------------------------------------------------
