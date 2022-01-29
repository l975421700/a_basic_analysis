

# =============================================================================
# region Don't run Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'SIday/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20181218/'

siconc_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

siconc_awc_mr_hi_r1 = xr.open_mfdataset(
    siconc_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_awc_mr_hi_r1 = xr.Dataset(
    data_vars={
        'siconc': (('ncells'), np.zeros((len(siconc_awc_mr_hi_r1.ncells)))),
        'lat_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lat_bnds.data),
        'lon_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lon_bnds.data),
    },
    coords={
        'ncells': siconc_awc_mr_hi_r1.ncells.data,
        'vertices': siconc_awc_mr_hi_r1.vertices.data,
        'lat': siconc_awc_mr_hi_r1.lat.data,
        'lon': siconc_awc_mr_hi_r1.lon.data,
    },
    attrs=siconc_awc_mr_hi_r1.attrs
)

am_siconc_awc_mr_hi_r1_80 = xr.Dataset(
    data_vars={
        'siconc': (('ncells'), np.zeros((len(siconc_awc_mr_hi_r1.ncells)))),
        'lat_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lat_bnds.data),
        'lon_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lon_bnds.data),
    },
    coords={
        'ncells': siconc_awc_mr_hi_r1.ncells.data,
        'vertices': siconc_awc_mr_hi_r1.vertices.data,
        'lat': siconc_awc_mr_hi_r1.lat.data,
        'lon': siconc_awc_mr_hi_r1.lon.data,
    },
    attrs=siconc_awc_mr_hi_r1.attrs
)

am_siconc_awc_mr_hi_r1.siconc[:] = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
    '1979-01-01', '2014-12-31')).mean(axis=0).values
am_siconc_awc_mr_hi_r1_80.siconc[:] = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
    '1980-01-01', '2014-12-31')).mean(axis=0).values

# am_siconc_awc_mr_hi_r1 = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
#     '1979-01-01', '2014-12-31')).mean(axis=0).load()
# am_siconc_awc_mr_hi_r1_80 = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
#     '1980-01-01', '2014-12-31')).mean(axis=0).load()


am_siconc_awc_mr_hi_r1.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')
am_siconc_awc_mr_hi_r1_80.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc')


'''
#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_cdo_remap.nc


# global_2: lonlat (180x90) grid
# global_1: lonlat (360x180) grid
# cdo -P 4 -remapcon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc regridding_weights_AWI-CM-1-1-MR_FESOM.nc

#### generate weights does not work
cdo genycon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
cdo remap,global_1, bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc


# cdo genycon, global_2 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
# cdo genycon, global_1 siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-301012.nc regrid2deg_weights_AWI-ESM-1-1-LR

#### slow
from cdo import Cdo
cdo = Cdo()
cdo.remapcon(
    'r360x180',
    input='/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc',
    output='bas_palaeoclim_qino/scratch/cmip6/historical/siconc/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231_cdo_regrid.nc')

# regrid_weights = xr.open_dataset(
#     '/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/AWI_org/regrid2deg_weights_AWI-ESM-1-1-LR.nc'
# )


stats.describe(am_siconc_awc_mr_hi_r1.siconc, axis=None, nan_policy='omit')
stats.describe(am_siconc_awc_mr_hi_r1_80.siconc, axis=None, nan_policy='omit')
'''
# endregion
# =============================================================================


# =============================================================================
# region Don't run Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'SIday/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20181218/'

siconc_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

siconc_awc_mr_hi_r1 = xr.open_mfdataset(
    siconc_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')

am_siconc_awc_mr_hi_r1_reformat = siconc_awc_mr_hi_r1.copy()

am_siconc_awc_mr_hi_r1_reformat['siconc'] = am_siconc_awc_mr_hi_r1_reformat['siconc'][0, :]

am_siconc_awc_mr_hi_r1_reformat['siconc'][:] = \
    am_siconc_awc_mr_hi_r1.siconc[:].values


am_siconc_awc_mr_hi_r1_reformat.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc')
# am_siconc_awc_mr_hi_r1_80.to_netcdf(
#     'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc')


'''
# check
am_siconc_awc_mr_hi_r1_reformat = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc'
)
am_siconc_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')
(am_siconc_awc_mr_hi_r1_reformat.siconc == am_siconc_awc_mr_hi_r1.siconc).all()


#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat_cdo_remap.nc



cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_cdo_remap.nc


# global_2: lonlat (180x90) grid
# global_1: lonlat (360x180) grid
# cdo -P 4 -remapcon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc regridding_weights_AWI-CM-1-1-MR_FESOM.nc

#### generate weights does not work
cdo genycon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
cdo remap,global_1, bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc


# cdo genycon, global_2 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
# cdo genycon, global_1 siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-301012.nc regrid2deg_weights_AWI-ESM-1-1-LR

#### slow
from cdo import Cdo
cdo = Cdo()
cdo.remapcon(
    'r360x180',
    input='/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc',
    output='bas_palaeoclim_qino/scratch/cmip6/historical/siconc/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231_cdo_regrid.nc')

# regrid_weights = xr.open_dataset(
#     '/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/AWI_org/regrid2deg_weights_AWI-ESM-1-1-LR.nc'
# )


stats.describe(am_siconc_awc_mr_hi_r1.siconc, axis=None, nan_policy='omit')
stats.describe(am_siconc_awc_mr_hi_r1_80.siconc, axis=None, nan_policy='omit')
'''
# endregion
# =============================================================================


# =============================================================================
# region Handling CloudSat raw data

# Using deepice_pynio environment
# conda activate deepice_pynio
# python -c "from IPython import start_ipython; start_ipython()" --no-autoindent

# management
from pyhdf.SD import SD, SDC
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})

# import Nio
# Nio.option_defaults['TimePeriodSuffix'] = True
# cloudsat2007001 = xr.open_dataset(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
#     engine='pynio',
# )

'''

import Nio
Nio.option_defaults
opt = Nio.option_defaults
opt['CompressionLevel'] = -2

# file = Nio.open_file(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf', 'r')
# file.dataset()

# from pyhdf.SD import SD, SDC
# hdf = SD(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
#     SDC.READ)
# hdf.datasets()

# import xarray as xr
# cloudsat2007001_nc = xr.open_dataset(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.nc',
# )

# import pandas
# pandas.read_hdf(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf'
# )

import rioxarray as rxr
cloudsat2007001 = rxr.open_rasterio(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
    masked=True,
)

import rasterio
cloudsat2007001 = rasterio.open(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
)
'''
# endregion
# =============================================================================

