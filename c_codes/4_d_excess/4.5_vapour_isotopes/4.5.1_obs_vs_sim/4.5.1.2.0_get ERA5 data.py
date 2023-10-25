

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

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily data from hourly data

ERA5_hourly_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_hourly_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_daily_temp2_2013_2022 = ERA5_hourly_temp2_2013_2022.t2m.resample(time='1d').mean().compute()

ERA5_daily_temp2_2013_2022.to_netcdf('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc')





'''
#-------------------------------- check
ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_hourly_temp2_2013_2022.nc', chunks={'time': 720})

(ERA5_hourly_temp2_2013_2022.t2m[-24:].mean(dim='time').values == ERA5_daily_temp2_2013_2022.t2m[-1].values).all()

# ERA5_hourly_temp2_2013_2022.t2m[0:48].resample(time='1d').mean().compute()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann data

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_temp2_2013_2022_alltime = mon_sea_ann(
    var_daily=ERA5_daily_temp2_2013_2022.t2m, lcopy=False)


output_file = 'scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.nc'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_temp2_2013_2022_alltime, f)



'''
with open('scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.nc', 'rb') as f:
    ERA5_temp2_2013_2022_alltime = pickle.load(f)



'''
# endregion
# -----------------------------------------------------------------------------




