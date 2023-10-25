
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

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

with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)

NK16_Australia_Syowa['1d']['q'] = NK16_Australia_Syowa['1d']['humidity'] * 18.01528 / (28.9645 * 1e6)

#-------- import ERA5 data

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

NK16_Australia_Syowa_1d_era5 = NK16_Australia_Syowa['1d'].copy()

NK16_Australia_Syowa_1d_era5['t2m_era5'] = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa_1d_era5['time'],
    NK16_Australia_Syowa_1d_era5['lat'],
    NK16_Australia_Syowa_1d_era5['lon'],
    ERA5_daily_temp2_2013_2022.time.values,
    ERA5_daily_temp2_2013_2022.latitude.values,
    ERA5_daily_temp2_2013_2022.longitude.values,
    ERA5_daily_temp2_2013_2022.t2m.values
    )

output_file = 'scratch/ERA5/temp2/NK16_Australia_Syowa_1d_era5.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(NK16_Australia_Syowa_1d_era5, f)




'''
#-------------------------------- check
with open('scratch/ERA5/temp2/NK16_Australia_Syowa_1d_era5.pkl', 'rb') as f:
    NK16_Australia_Syowa_1d_era5 = pickle.load(f)

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ires = 120
stime = NK16_Australia_Syowa_1d_era5['time'][ires]
slat = NK16_Australia_Syowa_1d_era5['lat'][ires]
slon = NK16_Australia_Syowa_1d_era5['lon'][ires]

itime = np.argmin(abs(stime.asm8 - ERA5_daily_temp2_2013_2022.time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    ERA5_daily_temp2_2013_2022.latitude.values,
    ERA5_daily_temp2_2013_2022.longitude.values)

NK16_Australia_Syowa_1d_era5['t2m_era5'][ires]
ERA5_daily_temp2_2013_2022.t2m[itime, ilat, ilon]



pearsonr(NK16_Australia_Syowa_1d_era5['t_air'], NK16_Australia_Syowa_1d_era5['t2m_era5']).statistic ** 2



'''
# endregion
# -----------------------------------------------------------------------------


