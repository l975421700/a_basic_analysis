

# ERA5 data retrieval
# Source: https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset


# -----------------------------------------------------------------------------
# region evaporation related variables

# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
# date: 20230307

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'evaporation', 'sea_surface_temperature',
            'surface_pressure',
        ],
        'year': '2022',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            0, -180, -90,
            180,
        ],
        'format': 'grib',
    },
    'scratch/products/era5/era5_evap_hourly_variables_2022.grib')


'''
! cdo -f nc copy scratch/products/era5/era5_evap_hourly_variables_2022.grib scratch/products/era5/era5_evap_hourly_variables_2022.nc

https://docs.meteoblue.com/en/meteo/data-sources/era5
10m u-component of wind: instanteneous
10m v-component of wind: instanteneous

2m dewpoint temperature: instanteneous
2m temperature: instanteneous
Evaporation: accumulated
Sea surface temperature: instanteneous
Surface pressure:
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region evaporation related variables 2022-01

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'evaporation', 'sea_surface_temperature',
            'surface_pressure',
        ],
        'year': '2022',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            0, -180, -90,
            180,
        ],
        'format': 'netcdf',
    },
    'scratch/products/era5/era5_evap_hourly_variables_202201.nc')

'''
! cdo -f nc copy scratch/products/era5/era5_evap_hourly_variables_202201.grib scratch/products/era5/era5_evap_hourly_variables_202201.nc

'''
# endregion
# -----------------------------------------------------------------------------

