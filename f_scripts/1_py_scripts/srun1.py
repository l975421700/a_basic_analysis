

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
# region get daily SIC from hourly SIC

ERA5_hourly_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_hourly_SIC_2013_2022.nc', chunks={'time': 720})

ERA5_daily_SIC_2013_2022 = ERA5_hourly_SIC_2013_2022.siconc.resample(time='1d').mean().compute()

ERA5_daily_SIC_2013_2022.to_netcdf('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_hourly_SIC_2013_2022.nc', chunks={'time': 720})

(ERA5_hourly_SIC_2013_2022.siconc[-24:].mean(dim='time').values == ERA5_daily_SIC_2013_2022.siconc[-1].values).all()
'''
# endregion
# -----------------------------------------------------------------------------



