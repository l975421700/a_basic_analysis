

# -----------------------------------------------------------------------------
# region Function to recover precipitation-weighted open-oceanic source properties

def source_properties(
    var_scaled_pre, ocean_pre, min_sf, max_sf, var_name,
    prefix = 'pre_weighted_', threshold = 2e-8,
    ):
    '''
    #---- Input
    var_scaled_pre : xarray.DataArray, precipitation scaled with var
    ocean_pre      : xarray.DataArray, open-oceanic precipitation
    min_sf         : parameter, minimum scaling factors
    max_sf         : parameter, maximum scaling factors
    var_name       : variable name
    
    #---- Output
    pre_weighted_var : precipitation-weighted open-oceanic source properties
    
    '''
    
    #---- Import packages
    import numpy as np
    
    #---- estimation on original time intervals
    pre_weighted_var = (var_scaled_pre / ocean_pre.values * (max_sf - min_sf) + min_sf).compute()
    pre_weighted_var.values[ocean_pre.values < threshold] = np.nan
    pre_weighted_var = pre_weighted_var.rename(prefix + var_name)
    
    if (var_name == 'sst'):
        pre_weighted_var.values[:] = pre_weighted_var.values[:] - 273.15
    
    if (var_name == 'rh2m'):
        pre_weighted_var.values[:] = pre_weighted_var.values[:] * 100
    
    return(pre_weighted_var)


'''
#-------- check
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to calculate source lon from sincoslon

def sincoslon_2_lon(
    sinlon, coslon,
    var_name='pre_weighted_lon',
    ):
    '''
    #---- Input
    sinlon: xarray.DataArray,
    coslon: xarray.DataArray,
    
    #---- Output
    pre_weighted_lon
    '''
    
    import numpy as np
    
    pre_weighted_lon = (np.arctan2(sinlon, coslon) * 180 / np.pi).compute()
    pre_weighted_lon.values[pre_weighted_lon.values < 0] += 360
    
    pre_weighted_lon = pre_weighted_lon.rename(var_name)
    
    return(pre_weighted_lon)

'''
test = sincoslon_2_lon(
    pre_weighted_sinlon_am[expid[i]].pre_weighted_sinlon_am,
    pre_weighted_coslon_am[expid[i]].pre_weighted_coslon_am,)

(test.values[np.isfinite(test.values)] == pre_weighted_lon_am[expid[i]].values[np.isfinite(pre_weighted_lon_am[expid[i]].values)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate differences in lon

def calc_lon_diff(lon1, lon2):
    '''
    #---- Input
    lon1: xarray.DataArray, longitude [0, 360]
    lon2: xarray.DataArray, longitude [0, 360]
    
    #---- Output
    lon_diff: differences in longitude
    '''
    
    lon_diff = (lon1 - lon2).compute()
    
    lon_diff.values[lon_diff.values < -180] += 360
    lon_diff.values[lon_diff.values > 180]  -= 360
    
    return(lon_diff)

def calc_lon_diff_np(lon1, lon2):
    '''
    #---- Input
    lon1: numpy.array
    lon2: numpy.array
    
    #---- Output
    lon_diff: differences in longitude
    '''
    
    lon_diff = lon1 - lon2
    
    lon_diff[lon_diff < -180] += 360
    lon_diff[lon_diff > 180]  -= 360
    
    return(lon_diff)


'''
#---- check
diff_pre_weighted_lon = (pre_weighted_lon[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lon[expid[i]]['sm'].sel(season='JJA')).compute()
diff_pre_weighted_lon.values[diff_pre_weighted_lon.values < -180] += 360
diff_pre_weighted_lon.values[diff_pre_weighted_lon.values > 180] -= 360


diff_lon = calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
                         pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'))
(diff_pre_weighted_lon.values[np.isfinite(diff_pre_weighted_lon.values)] == diff_lon.values[np.isfinite(diff_lon.values)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


