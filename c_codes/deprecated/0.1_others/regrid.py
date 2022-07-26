# source /home/users/qino/miniconda3/envs/deepice/bin/activate
# conda activate /home/users/qino/miniconda3/envs/deepice

import xarray as xr
import numpy as np
import xesmf as xe

from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,
)


obs_data = xr.open_dataset('/gws/nopw/j04/pmip4_vol1/users/rachel/OBSERVATIONAL_DATA/CLARA_A2/ORD46800/SALmm20160101000000221AVPOSE1NP.nc')
hadgem_data = xr.open_dataset('/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/v20190624/tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-194912.nc')


regridder = xe.Regridder(
    obs_data, xe.util.grid_global(1, 1), 'bilinear')
obs_data_rg_hg3 = regridder(obs_data)
obs_data_rg_hg3.to_netcdf('/home/users/qino/share/test.nc')

import cartopy.crs as ccrs
fig, ax = hemisphere_plot(southextent=60,)
ax.pcolormesh(obs_data.lon.values[np.isfinite(obs_data.lon)],
              obs_data.lat.values[np.isfinite(obs_data.lon)],
              obs_data.SZA[0, :, :].values[np.isfinite(obs_data.lon)],
               transform=ccrs.PlateCarree())
fig.savefig('/home/users/qino/share/test1.png', dpi = 600)



np.nanmean(obs_data_rg_hg3.SZA[0, :, :].values)
np.nanmean(obs_data.SZA[0, :, :].values)

