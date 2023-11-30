

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
from scipy.stats import pearsonr

from a_basic_analysis.b_module.basic_calculations import (
    find_ilat_ilon,
    find_ilat_ilon_general,
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

#-------- import obs

with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

#-------- import ERA5 data

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

MC16_Dome_C_1d_era5 = MC16_Dome_C['1d'].copy()

MC16_Dome_C_1d_era5['t_3m_era5'] = find_multi_gridvalue_at_site_time(
    MC16_Dome_C_1d_era5['time'],
    MC16_Dome_C_1d_era5['lat'],
    MC16_Dome_C_1d_era5['lon'],
    ERA5_daily_temp2_2013_2022.time.sel(time=slice('2014-12-25', '2015-01-16')).values,
    ERA5_daily_temp2_2013_2022.latitude.values,
    ERA5_daily_temp2_2013_2022.longitude.values,
    ERA5_daily_temp2_2013_2022.t2m.sel(time=slice('2014-12-25', '2015-01-16')).values
    )

output_file = 'scratch/ERA5/temp2/MC16_Dome_C_1d_era5.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(MC16_Dome_C_1d_era5, f)




'''
#-------------------------------- check
with open('scratch/ERA5/temp2/MC16_Dome_C_1d_era5.pkl', 'rb') as f:
    MC16_Dome_C_1d_era5 = pickle.load(f)

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ires = 10
stime = MC16_Dome_C_1d_era5['time'][ires]
slat = MC16_Dome_C_1d_era5['lat'][ires]
slon = MC16_Dome_C_1d_era5['lon'][ires]

itime = np.argmin(abs(stime.asm8 - ERA5_daily_temp2_2013_2022.time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    ERA5_daily_temp2_2013_2022.latitude.values,
    ERA5_daily_temp2_2013_2022.longitude.values)

MC16_Dome_C_1d_era5['t_3m_era5'][ires]
ERA5_daily_temp2_2013_2022.t2m[itime, ilat, ilon]



print(pearsonr(MC16_Dome_C_1d_era5['t_3m'], MC16_Dome_C_1d_era5['t_3m_era5']).statistic ** 2)



'''
# endregion
# -----------------------------------------------------------------------------



