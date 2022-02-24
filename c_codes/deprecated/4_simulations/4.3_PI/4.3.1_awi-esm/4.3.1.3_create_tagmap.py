
import numpy as np
import xarray as xr

tagmap = xr.open_dataset('/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap.nc')
lon = tagmap.lon.values
lat = tagmap.lat.values


# Create an identical tagmap3 to tagmap

ntag = 3

tagmap3 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)))),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap3.tagmap.sel(level = 5, lat=slice(60, 30), lon=slice(300, 360))[:, :] = 1
tagmap3.tagmap.sel(level = 6, lat=slice(-40, -70), lon=slice(180, 270))[:, :] = 1

# (tagmap3.tagmap == tagmap.tagmap).all()
# np.sum(tagmap.tagmap, axis =None)

tagmap3.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap3.nc', mode='w')


# Add one more tag area including land+ocean to tagmap

ntag = 4

tagmap4 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)))),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap4.tagmap.sel(level = 5, lat=slice(60, 30), lon=slice(300, 360))[:, :] = 1
tagmap4.tagmap.sel(
    level = 6, lat=slice(-40, -70), lon=slice(180, 270))[:, :] = 1
tagmap4.tagmap.sel(level = 7, lat=slice(60, 30), lon=slice(240, 300))[:, :] = 1

# (tagmap4.tagmap[] == tagmap.tagmap).all()
# np.sum(tagmap.tagmap, axis =None)

tagmap4.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap4.nc', mode='w')

