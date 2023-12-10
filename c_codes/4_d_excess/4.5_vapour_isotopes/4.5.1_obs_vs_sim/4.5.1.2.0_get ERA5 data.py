

#SBATCH --time=00:30:00
#SBATCH --mem=240GB


# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=240GB
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
from metpy.calc import specific_humidity_from_dewpoint
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    calc_specific_humidity,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily temp2 from hourly temp2

ERA5_hourly_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_hourly_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_daily_temp2_2013_2022 = ERA5_hourly_temp2_2013_2022.t2m.resample(time='1d').mean().compute()

ERA5_daily_temp2_2013_2022.to_netcdf('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_hourly_temp2_2013_2022.nc', chunks={'time': 720})

(ERA5_hourly_temp2_2013_2022.t2m[-24:].mean(dim='time').values == ERA5_daily_temp2_2013_2022.t2m[-1].values).all()
'''
# endregion
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region get mon_sea_ann temp2

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_temp2_2013_2022_alltime = mon_sea_ann(
    var_daily=ERA5_daily_temp2_2013_2022.t2m, lcopy=False)


output_file = 'scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_temp2_2013_2022_alltime, f)



'''
#-------------------------------- check

with open('scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_temp2_2013_2022_alltime = pickle.load(f)

ERA5_temp2_2013_2022_alltime['am'].to_netcdf('scratch/ERA5/temp2/ERA5_am_temp2_2013_2022.nc')
ERA5_temp2_2013_2022_alltime['sm'].to_netcdf('scratch/ERA5/temp2/ERA5_sm_temp2_2013_2022.nc')

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

itime = 200
data1 = ERA5_temp2_2013_2022_alltime['daily'][itime].values
data2 = ERA5_daily_temp2_2013_2022.t2m[itime].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily SST from hourly SST

ERA5_hourly_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_hourly_SST_2013_2022.nc', chunks={'time': 720})

ERA5_daily_SST_2013_2022 = ERA5_hourly_SST_2013_2022.sst.resample(time='1d').mean().compute()

ERA5_daily_SST_2013_2022.to_netcdf('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_hourly_SST_2013_2022.nc', chunks={'time': 720})

data1 = ERA5_hourly_SST_2013_2022.sst[-24:].mean(dim='time').values
data2 = ERA5_daily_SST_2013_2022.sst[-1].values

(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region get mon_sea_ann SST

ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc', chunks={'time': 720})

ERA5_SST_2013_2022_alltime = mon_sea_ann(
    var_daily=ERA5_daily_SST_2013_2022.sst, lcopy=False)


output_file = 'scratch/ERA5/SST/ERA5_SST_2013_2022_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_SST_2013_2022_alltime, f)



'''
#-------------------------------- check

with open('scratch/ERA5/SST/ERA5_SST_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_SST_2013_2022_alltime = pickle.load(f)

ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc', chunks={'time': 720})

itime = 200
data1 = ERA5_SST_2013_2022_alltime['daily'][itime].values
data2 = ERA5_daily_SST_2013_2022.sst[itime].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily SIC from hourly SIC

ERA5_hourly_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_hourly_SIC_2013_2022.nc', chunks={'time': 720})

ERA5_daily_SIC_2013_2022 = ERA5_hourly_SIC_2013_2022.siconc.resample(time='1d').mean().compute()

ERA5_daily_SIC_2013_2022.to_netcdf('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_hourly_SIC_2013_2022.nc', chunks={'time': 720})

data1 = ERA5_hourly_SIC_2013_2022.siconc[-24:].mean(dim='time').values
data2 = ERA5_daily_SIC_2013_2022.siconc[-1].values

(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region get mon_sea_ann SIC

ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

ERA5_SIC_2013_2022_alltime = mon_sea_ann(
    var_daily=ERA5_daily_SIC_2013_2022.siconc, lcopy=False)


output_file = 'scratch/ERA5/SIC/ERA5_SIC_2013_2022_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_SIC_2013_2022_alltime, f)



'''
#-------------------------------- check

with open('scratch/ERA5/SIC/ERA5_SIC_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_SIC_2013_2022_alltime = pickle.load(f)

ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

itime = 200
data1 = ERA5_SIC_2013_2022_alltime['daily'][itime].values
data2 = ERA5_daily_SIC_2013_2022.siconc[itime].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily prs_sfc from hourly prs_sfc

ERA5_hourly_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_hourly_prs_sfc_2013_2022.nc', chunks={'time': 720})

ERA5_daily_prs_sfc_2013_2022 = ERA5_hourly_prs_sfc_2013_2022.sp.resample(time='1d').mean().compute()

ERA5_daily_prs_sfc_2013_2022.to_netcdf('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_hourly_prs_sfc_2013_2022.nc', chunks={'time': 720})

data1 = ERA5_hourly_prs_sfc_2013_2022.sp[-24:].mean(dim='time').values
data2 = ERA5_daily_prs_sfc_2013_2022.sp[-1].values

(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily t2m_dew from hourly t2m_dew

ERA5_hourly_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_hourly_t2m_dew_2013_2022.nc', chunks={'time': 720})

ERA5_daily_t2m_dew_2013_2022 = ERA5_hourly_t2m_dew_2013_2022.d2m.resample(time='1d').mean().compute()

ERA5_daily_t2m_dew_2013_2022.to_netcdf('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_hourly_t2m_dew_2013_2022.nc', chunks={'time': 720})

data1 = ERA5_hourly_t2m_dew_2013_2022.d2m[-24:].mean(dim='time').values
data2 = ERA5_daily_t2m_dew_2013_2022.d2m[-1].values

(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily specific humidity from daily prs_sfc and t2m_dew

ERA5_daily_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc', chunks={'time': 720})

ERA5_daily_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc', chunks={'time': 720})

ERA5_daily_q_2013_2022 = specific_humidity_from_dewpoint(
    (ERA5_daily_prs_sfc_2013_2022.sp / 100) * units.hPa,
    ERA5_daily_t2m_dew_2013_2022.d2m * units.K,
    ).compute()
ERA5_daily_q_2013_2022 = ERA5_daily_q_2013_2022.rename('q')
ERA5_daily_q_2013_2022.values[:] = ERA5_daily_q_2013_2022.values[:] * 1000
ERA5_daily_q_2013_2022 = ERA5_daily_q_2013_2022.assign_attrs(units='g/kg')

ERA5_daily_q_2013_2022.to_netcdf('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc')


'''
#-------------------------------- check
ERA5_daily_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc', chunks={'time': 720})
ERA5_daily_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc', chunks={'time': 720})
ERA5_daily_q_2013_2022 = xr.open_dataset('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc', chunks={'time': 720})

itime = 100
ilat = 100
ilon = 100

print(specific_humidity_from_dewpoint(
    (ERA5_daily_prs_sfc_2013_2022.sp[itime, ilat, ilon].values / 100) * units.hPa,
    ERA5_daily_t2m_dew_2013_2022.d2m[itime, ilat, ilon].values * units.K,
    ) * 1000)
print(ERA5_daily_q_2013_2022.q[itime, ilat, ilon].values)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily relative humidity from daily prs_sfc and t2m_dew

ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})

ERA5_daily_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc', chunks={'time': 720})

ERA5_daily_rh2m_2013_2022 = relative_humidity_from_dewpoint(
    ERA5_daily_temp2_2013_2022.t2m * units.K,
    ERA5_daily_t2m_dew_2013_2022.d2m * units.K,
    ).compute()
ERA5_daily_rh2m_2013_2022 = ERA5_daily_rh2m_2013_2022.rename('rh2m')
ERA5_daily_rh2m_2013_2022.values[:] = ERA5_daily_rh2m_2013_2022.values[:] * 100
ERA5_daily_rh2m_2013_2022 = ERA5_daily_rh2m_2013_2022.assign_attrs(units='%')
ERA5_daily_rh2m_2013_2022.to_netcdf('scratch/ERA5/rh2m/ERA5_daily_rh2m_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_t2m_dew_2013_2022 = xr.open_dataset('scratch/ERA5/t2m_dew/ERA5_daily_t2m_dew_2013_2022.nc', chunks={'time': 720})
ERA5_daily_temp2_2013_2022 = xr.open_dataset('scratch/ERA5/temp2/ERA5_daily_temp2_2013_2022.nc', chunks={'time': 720})
ERA5_daily_rh2m_2013_2022 = xr.open_dataset('scratch/ERA5/rh2m/ERA5_daily_rh2m_2013_2022.nc', chunks={'time': 720})

itime = 100
ilat = 100
ilon = 100

print(relative_humidity_from_dewpoint(
    ERA5_daily_temp2_2013_2022.t2m[itime, ilat, ilon].values * units.K,
    ERA5_daily_t2m_dew_2013_2022.d2m[itime, ilat, ilon].values * units.K,
) * 100)
print(ERA5_daily_rh2m_2013_2022.rh2m[itime, ilat, ilon].values)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily evaporation from hourly evaporation

ERA5_hourly_evap_2013_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_hourly_evap_2013_2022.nc', chunks={'time': 720})

ERA5_daily_evap_2013_2022 = ERA5_hourly_evap_2013_2022.e.resample(time='1d').mean().compute()

ERA5_daily_evap_2013_2022.to_netcdf('scratch/ERA5/evap/ERA5_daily_evap_2013_2022.nc')




'''
#-------------------------------- check
ERA5_daily_evap_2013_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_daily_evap_2013_2022.nc', chunks={'time': 720})

ERA5_hourly_evap_2013_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_hourly_evap_2013_2022.nc', chunks={'time': 720})

data1 = ERA5_hourly_evap_2013_2022.e[-24:].mean(dim='time').values
data2 = ERA5_daily_evap_2013_2022.e[-1].values

(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region get mon_sea_ann evap

ERA5_daily_evap_2013_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_daily_evap_2013_2022.nc', chunks={'time': 720})

ERA5_evap_2013_2022_alltime = mon_sea_ann(
    var_daily=ERA5_daily_evap_2013_2022.e, lcopy=False)

output_file = 'scratch/ERA5/evap/ERA5_evap_2013_2022_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_evap_2013_2022_alltime, f)



'''
#-------------------------------- check

with open('scratch/ERA5/evap/ERA5_evap_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_evap_2013_2022_alltime = pickle.load(f)

ERA5_daily_evap_2013_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_daily_evap_2013_2022.nc', chunks={'time': 720})

itime = 200
data1 = ERA5_evap_2013_2022_alltime['daily'][itime].values
data2 = ERA5_daily_evap_2013_2022.e[itime].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann evap 1979 to 2022

ERA5_monthly_evap_1940_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_monthly_evap_1940_2022.nc')

ERA5_evap_1979_2022_alltime = mon_sea_ann(
    var_monthly=ERA5_monthly_evap_1940_2022.e.sel(time=slice('1979-01-01', '2022-12-01')))

output_file = 'scratch/ERA5/evap/ERA5_evap_1979_2022_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(ERA5_evap_1979_2022_alltime, f)




'''
#-------------------------------- check
ERA5_monthly_evap_1940_2022 = xr.open_dataset('scratch/ERA5/evap/ERA5_monthly_evap_1940_2022.nc')
with open('scratch/ERA5/evap/ERA5_evap_1979_2022_alltime.pkl', 'rb') as f:
    ERA5_evap_1979_2022_alltime = pickle.load(f)

itime = 20
data1 = ERA5_evap_1979_2022_alltime['mon'][itime].values
data2 = ERA5_monthly_evap_1940_2022.e.sel(time=slice('1979-01-01', '2022-12-01'))[itime].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get RHsst_2m in ERA5

ERA5_daily_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc', chunks={'time': 720})

ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc', chunks={'time': 720})

ERA5_daily_q_2013_2022 = xr.open_dataset('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc', chunks={'time': 720})


q_sst = calc_specific_humidity(
    RH = 1,
    T = ERA5_daily_SST_2013_2022.sst,
    p = ERA5_daily_prs_sfc_2013_2022.sp,
)

ERA5_daily_RHsst_2013_2022 = (ERA5_daily_q_2013_2022.q/1000 / q_sst * 100).compute()

ERA5_daily_RHsst_2013_2022.to_netcdf('scratch/ERA5/RHsst/ERA5_daily_RHsst_2013_2022.nc')



'''
ERA5_daily_RHsst_2013_2022 = xr.open_dataset('scratch/ERA5/RHsst/ERA5_daily_RHsst_2013_2022.nc')
ERA5_daily_prs_sfc_2013_2022 = xr.open_dataset('scratch/ERA5/prs_sfc/ERA5_daily_prs_sfc_2013_2022.nc', chunks={'time': 720})
ERA5_daily_SST_2013_2022 = xr.open_dataset('scratch/ERA5/SST/ERA5_daily_SST_2013_2022.nc', chunks={'time': 720})
ERA5_daily_q_2013_2022 = xr.open_dataset('scratch/ERA5/q/ERA5_daily_q_2013_2022.nc', chunks={'time': 720})

itime = 100
ilat = 20
ilon = 20
data1 = calc_specific_humidity(
    RH = 1,
    T = ERA5_daily_SST_2013_2022.sst[itime, ilat, ilon].values,
    p = ERA5_daily_prs_sfc_2013_2022.sp[itime, ilat, ilon].values,
)
data1 = ERA5_daily_q_2013_2022.q[itime, ilat, ilon].values/1000 / data1 * 100
data2 = ERA5_daily_RHsst_2013_2022['__xarray_dataarray_variable__'][itime, ilat, ilon].values

print(data1)
print(data2)

'''
# endregion
# -----------------------------------------------------------------------------

