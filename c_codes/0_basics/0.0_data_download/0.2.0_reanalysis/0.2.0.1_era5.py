

# ERA5 data retrieval
# Source: https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset


import cdsapi
c = cdsapi.Client()


# -----------------------------------------------------------------------------
# region 1979-2014 ERA5 monthly averaged data on single levels from 1979 to present

# retrieval time: 2022-02-24
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/pre/ERA5_mon_sl_197901_201412_pre.nc')



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region deprecated 1979-2021 ERA5 monthly averaged data on single levels from 1979 to present
# folder: mon_sl_79_present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

import cdsapi
c = cdsapi.Client()

# file: era5_mon_sl_79_21_pre.nc
# Monthly mean Total precipitation from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')


# file: era5_mon_sl_79_21_slp.nc
# Monthly mean sea level pressure from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'mean_sea_level_pressure',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_slp.nc')


# file: era5_mon_sl_79_21_2mtem.nc
# Monthly mean 2m temperature from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_2mtem.nc')


# file: era5_mon_sl_79_21_sic.nc
# Monthly mean sea ice cover from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_ice_cover',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')


# file: era5_mon_sl_20_gph.nc
# Monthly mean geopotential height in 2020
# retrieval time: 2021-11-08
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'variable': 'geopotential',
        'year': '2020',
        'product_type': 'monthly_averaged_reanalysis',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_20_gph.nc')


# file: era5_mon_sl_79_21_wind.nc
# Monthly mean 10m u, 10m v, 10m wind speed from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed',
        ],
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_10mwind.nc')


# file: era5_mon_sl_79_21_sst.nc
# Monthly mean Sea surface temperature from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_surface_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sst.nc')


# file: era5_mon_sl_79_21_evp.nc
# Monthly mean Evaporation from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'evaporation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_evp.nc')


# file: era5_mon_sl_79_21_lsmask.nc
# Land Sea Mask on 2020-01
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'land_sea_mask',
        'year': '2020',
        'month': '01',
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_lsmask.nc')

# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region deprecated 1979-2021 ERA5 hourly data on single levels from 1979 to present
# folder: hr_sl_79_present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

import cdsapi
c = cdsapi.Client()

# file: era5_hr_sl_201412_10uv_slp_sst.nc
# Hourly 10m u, 10m v, SST, SLP in 2014
# retrieval time: 2021-12-05
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
            'sea_surface_temperature',
        ],
        'year': '2014',
        'month': '12',
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
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/hr_sl_79_present/era5_hr_sl_201412_10uv_slp_sst.nc')


# endregion
# -----------------------------------------------------------------------------




