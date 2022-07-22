
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


def create_ais_mask(lon_lat_file = None, ais_file = None):
    '''
    ---- Input
    lon_lat_file: a file contains the desired lon/lat
    ais_file: a shapefile contains the AIS.
    
    ---- Output
    
    '''
    
    import numpy as np
    import xarray as xr
    import geopandas as gpd
    from geopandas.tools import sjoin
    
    if (lon_lat_file is None):
        lon_lat_file = 'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc'
    
    ncfile = xr.open_dataset(lon_lat_file)
    lon, lat = np.meshgrid(ncfile.lon, ncfile.lat,)
    
    if (ais_file is None):
        ais_file = 'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp'
    
    shpfile = gpd.read_file(ais_file)
    
    lon_lat = gpd.GeoDataFrame(
        crs="EPSG:4326", geometry=gpd.points_from_xy(
            lon.reshape(-1, 1), lat.reshape(-1, 1))).to_crs(3031)
    
    pointInPolys = sjoin(lon_lat, shpfile, how='left')
    regions = pointInPolys.Regions.to_numpy().reshape(
        lon.shape[0], lon.shape[1])
    
    eais_mask = (regions == 'East')
    eais_mask01 = np.zeros(eais_mask.shape)
    eais_mask01[eais_mask] = 1
    
    wais_mask = (regions == 'West')
    wais_mask01 = np.zeros(wais_mask.shape)
    wais_mask01[wais_mask] = 1
    
    ap_mask = (regions == 'Peninsula')
    ap_mask01 = np.zeros(ap_mask.shape)
    ap_mask01[ap_mask] = 1
    
    ais_mask = (eais_mask | wais_mask | ap_mask)
    ais_mask01 = np.zeros(ais_mask.shape)
    ais_mask01[ais_mask] = 1
    
    eais_area = ncfile.cell_area.values[eais_mask].sum()
    wais_area = ncfile.cell_area.values[wais_mask].sum()
    ap_area = ncfile.cell_area.values[ap_mask].sum()
    
    return (lon, lat, eais_mask, eais_mask01, wais_mask, wais_mask01, ap_mask,
            ap_mask01, ais_mask, ais_mask01, eais_area, wais_area, ap_area)


'''
# Production run to create area and masks files
import pickle
(lon, lat, eais_mask, eais_mask01, wais_mask, wais_mask01, ap_mask,
 ap_mask01, ais_mask, ais_mask01, eais_area, wais_area, ap_area
 ) = create_ais_mask()

ais_masks = {'lon': lon, 'lat': lat, 'eais_mask': eais_mask,
             'eais_mask01': eais_mask01, 'wais_mask': wais_mask,
             'wais_mask01': wais_mask01, 'ap_mask': ap_mask,
             'ap_mask01': ap_mask01, 'ais_mask': ais_mask,
             'ais_mask01': ais_mask01,}
ais_area = {'eais': eais_area, 'wais': wais_area, 'ap': ap_area,
            'ais': eais_area + wais_area + ap_area,}

with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'wb') as handle:
    pickle.dump(ais_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bas_palaeoclim_qino/others/ais_area.pickle', 'wb') as handle:
    pickle.dump(ais_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''


'''
# check
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,hemisphere_plot,)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pickle

with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)
with open('bas_palaeoclim_qino/others/ais_area.pickle', 'rb') as handle:
    ais_area = pickle.load(handle)
ais_area
# 9761642.045593847, 9620521.647740476, 2110804.571811703, 2038875.6430063567, 232127.50065176177, 232678.32105463636

fig, ax = hemisphere_plot(
    northextent=-60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98, add_atlas=False,)

# fig, ax = framework_plot1("global")
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['eais_mask01'],
    colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['wais_mask01'],
    colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['ap_mask01'],
    colors='yellow', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/0_test/trial.png')
'''


'''
# derivation
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
