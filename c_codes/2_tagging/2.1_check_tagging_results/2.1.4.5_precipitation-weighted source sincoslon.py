

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_411_4.9',
    # 'pi_m_412_4.9',
    # 'pi_m_413_4.10',
    ]

# region import output

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)


'''
        # exp_org_o[expid[i]]['echam'] = xr.open_dataset(
        #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')

        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))

        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# itag = 11
# lsincos = 'sin'
itag = 12
lsincos = 'cos'
ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7, 3, 3, 0]
# itag = 13
# ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   0, 0, 19]
# ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   0, 0, 37]

# region set indices for specific set of tracers

kwiso2 = 3

if (itag == 0):
    kstart = kwiso2 + 0
    kend   = kwiso2 + ntags[0]
else:
    kstart = kwiso2 + sum(ntags[:itag])
    kend   = kwiso2 + sum(ntags[:(itag+1)])

print(kstart); print(kend)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate source sincoslon - scaling tagmap with sincoslon

min_sincoslon = -1
max_sincoslon = 1

ocean_pre = {}
sincoslon_scaled_pre = {}
pre_weighted_sincoslon = {}
ocean_pre_ann = {}
sincoslon_scaled_pre_ann = {}
pre_weighted_sincoslon_ann = {}
ocean_pre_sea = {}
sincoslon_scaled_pre_sea = {}
pre_weighted_sincoslon_sea = {}
ocean_pre_am = {}
sincoslon_scaled_pre_am = {}
pre_weighted_sincoslon_am = {}

ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype')
sincoslon_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=kstart+2) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=kstart+2))

#---------------- monthly values

pre_weighted_sincoslon[expid[i]] = sincoslon_scaled_pre[expid[i]] / ocean_pre[expid[i]] * (max_sincoslon - min_sincoslon) + min_sincoslon
pre_weighted_sincoslon[expid[i]].values[ocean_pre[expid[i]] < 1e-9] = np.nan
pre_weighted_sincoslon[expid[i]] = pre_weighted_sincoslon[expid[i]].rename('pre_weighted_' + lsincos + 'lon')
pre_weighted_sincoslon[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon.nc')

#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
sincoslon_scaled_pre_ann[expid[i]] = sincoslon_scaled_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_sincoslon_ann[expid[i]] = sincoslon_scaled_pre_ann[expid[i]] / ocean_pre_ann[expid[i]] * (max_sincoslon - min_sincoslon) + min_sincoslon
pre_weighted_sincoslon_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_sincoslon_ann[expid[i]] = pre_weighted_sincoslon_ann[expid[i]].rename('pre_weighted_' + lsincos + 'lon_ann')
pre_weighted_sincoslon_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_ann.nc'
)


#---------------- seasonal values

# spin up: one year
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][120:].groupby('time.season').sum(dim="time", skipna=True)
sincoslon_scaled_pre_sea[expid[i]] = sincoslon_scaled_pre[expid[i]][120:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_sincoslon_sea[expid[i]] = sincoslon_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (max_sincoslon - min_sincoslon) + min_sincoslon
pre_weighted_sincoslon_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_sincoslon_sea[expid[i]] = pre_weighted_sincoslon_sea[expid[i]].rename('pre_weighted_' + lsincos + 'lon_sea')
pre_weighted_sincoslon_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_sea.nc'
)


#---------------- annual mean values

# spin up: one year
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][120:].mean(dim="time", skipna=True)
sincoslon_scaled_pre_am[expid[i]] = sincoslon_scaled_pre[expid[i]][120:].mean(dim="time", skipna=True)

pre_weighted_sincoslon_am[expid[i]] = sincoslon_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (max_sincoslon - min_sincoslon) + min_sincoslon
pre_weighted_sincoslon_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_sincoslon_am[expid[i]] = pre_weighted_sincoslon_am[expid[i]].rename('pre_weighted_' + lsincos + 'lon_am')
pre_weighted_sincoslon_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_am.nc'
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate source sincoslon - binning tagmap with lon

i = 0
expid[i]

# lonbins = np.concatenate((np.array([-1]), np.arange(20, 340+1e-4, 20), np.array([361])))
# lonbins_mid = np.arange(10, 350.1, 20)
lonbins = np.concatenate((np.array([-1]), np.arange(10, 350+1e-4, 10), np.array([361])))
lonbins_mid = np.arange(5, 355+1e-4, 10)

# lsincos = 'sin'
# sincos_lonbins_mid = np.sin(lonbins_mid * np.pi / 180)
lsincos = 'cos'
sincos_lonbins_mid = np.cos(lonbins_mid * np.pi / 180)

ocean_pre = {}
lon_binned_pre = {}
pre_weighted_lon = {}
ocean_pre_ann = {}
lon_binned_pre_ann = {}
pre_weighted_lon_ann = {}
ocean_pre_sea = {}
lon_binned_pre_sea = {}
pre_weighted_lon_sea = {}
ocean_pre_am = {}
lon_binned_pre_am = {}
pre_weighted_lon_am = {}

lon_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(kstart+2, kend)))
ocean_pre[expid[i]] = lon_binned_pre[expid[i]].sum(dim='wisotype')


#---------------- monthly values

pre_weighted_lon[expid[i]] = (lon_binned_pre[expid[i]] * sincos_lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre[expid[i]]
pre_weighted_lon[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon[expid[i]] = pre_weighted_lon[expid[i]].rename('pre_weighted_' + lsincos + 'lon')
pre_weighted_lon[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon.nc'
)


#---------------- annual values

ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lon_binned_pre_ann[expid[i]] = lon_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lon_ann[expid[i]] = (lon_binned_pre_ann[expid[i]] * sincos_lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_ann[expid[i]]
pre_weighted_lon_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_ann[expid[i]] = pre_weighted_lon_ann[expid[i]].rename('pre_weighted_' + lsincos + 'lon_ann')
pre_weighted_lon_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)
lon_binned_pre_sea[expid[i]] = lon_binned_pre[expid[i]][12:].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lon_sea[expid[i]] = (lon_binned_pre_sea[expid[i]] * sincos_lonbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_sea[expid[i]]
pre_weighted_lon_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].rename('pre_weighted_' + lsincos + 'lon_sea')
pre_weighted_lon_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:].mean(dim="time", skipna=True)
lon_binned_pre_am[expid[i]] = lon_binned_pre[expid[i]][12:].mean(dim="time", skipna=True)

pre_weighted_lon_am[expid[i]] = (lon_binned_pre_am[expid[i]] * sincos_lonbins_mid[:, None, None]).sum(dim='wisotype') / ocean_pre_am[expid[i]]
pre_weighted_lon_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_lon_am[expid[i]] = pre_weighted_lon_am[expid[i]].rename('pre_weighted_' + lsincos + 'lon_am')
pre_weighted_lon_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_' + lsincos + 'lon_am.nc'
)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source lon from sincoslon

pre_weighted_sinlon = {}
pre_weighted_sinlon_ann = {}
pre_weighted_sinlon_sea = {}
pre_weighted_sinlon_am = {}
pre_weighted_coslon = {}
pre_weighted_coslon_ann = {}
pre_weighted_coslon_sea = {}
pre_weighted_coslon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_sinlon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon.nc')
    pre_weighted_sinlon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_ann.nc')
    pre_weighted_sinlon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_sea.nc')
    pre_weighted_sinlon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_am.nc')
    pre_weighted_coslon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon.nc')
    pre_weighted_coslon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_ann.nc')
    pre_weighted_coslon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_sea.nc')
    pre_weighted_coslon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_am.nc')


pre_weighted_lon = {}
pre_weighted_lon_ann = {}
pre_weighted_lon_sea = {}
pre_weighted_lon_am = {}

pre_weighted_lon[expid[i]] = np.arctan2(pre_weighted_sinlon[expid[i]].pre_weighted_sinlon, pre_weighted_coslon[expid[i]].pre_weighted_coslon)  * 180 / np.pi
pre_weighted_lon[expid[i]] = pre_weighted_lon[expid[i]].rename('pre_weighted_lon')
pre_weighted_lon[expid[i]].values[pre_weighted_lon[expid[i]].values < 0] = pre_weighted_lon[expid[i]].values[pre_weighted_lon[expid[i]].values < 0] + 360
pre_weighted_lon[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')

pre_weighted_lon_ann[expid[i]] = np.arctan2(pre_weighted_sinlon_ann[expid[i]].pre_weighted_sinlon_ann, pre_weighted_coslon_ann[expid[i]].pre_weighted_coslon_ann)  * 180 / np.pi
pre_weighted_lon_ann[expid[i]] = pre_weighted_lon_ann[expid[i]].rename('pre_weighted_lon_ann')
pre_weighted_lon_ann[expid[i]].values[pre_weighted_lon_ann[expid[i]].values < 0] = pre_weighted_lon_ann[expid[i]].values[pre_weighted_lon_ann[expid[i]].values < 0] + 360
pre_weighted_lon_ann[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc')

pre_weighted_lon_sea[expid[i]] = np.arctan2(pre_weighted_sinlon_sea[expid[i]].pre_weighted_sinlon_sea, pre_weighted_coslon_sea[expid[i]].pre_weighted_coslon_sea)  * 180 / np.pi
pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].rename('pre_weighted_lon_sea')
pre_weighted_lon_sea[expid[i]].values[pre_weighted_lon_sea[expid[i]].values < 0] = pre_weighted_lon_sea[expid[i]].values[pre_weighted_lon_sea[expid[i]].values < 0] + 360
pre_weighted_lon_sea[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc')

pre_weighted_lon_am[expid[i]] = np.arctan2(pre_weighted_sinlon_am[expid[i]].pre_weighted_sinlon_am, pre_weighted_coslon_am[expid[i]].pre_weighted_coslon_am)  * 180 / np.pi
pre_weighted_lon_am[expid[i]] = pre_weighted_lon_am[expid[i]].rename('pre_weighted_lon_am')
pre_weighted_lon_am[expid[i]].values[pre_weighted_lon_am[expid[i]].values < 0] = pre_weighted_lon_am[expid[i]].values[pre_weighted_lon_am[expid[i]].values < 0] + 360
pre_weighted_lon_am[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')





'''
aaa = pre_weighted_sinlon[expid[i]].pre_weighted_sinlon[0, 40, 40].values
bbb = pre_weighted_coslon[expid[i]].pre_weighted_coslon[0, 40, 40].values
np.arctan2(aaa, bbb) * 180 / np.pi

ccc = np.array([aaa, bbb])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

np.linalg.norm(ccc)

ddd = unit_vector(ccc)
aaa ** 2 + bbb ** 2
ccc[0] ** 2 + ccc[1] ** 2
ddd[0] ** 2 + ddd[1] ** 2
# https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html

np.arctan2(ddd[0], ddd[1]) * 180 / np.pi

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region cross check am/DJF/JJA source lon between scaling and binning tagmap

#-------------------------------- import data

pre_weighted_lon = {}
pre_weighted_lon_ann = {}
pre_weighted_lon_sea = {}
pre_weighted_lon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')
    pre_weighted_lon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc')
    pre_weighted_lon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc')
    pre_weighted_lon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')


#-------------------------------- calculate differences

diff_pre_weighted_lon = {}
diff_pre_weighted_lon_ann = {}
diff_pre_weighted_lon_sea = {}
diff_pre_weighted_lon_am = {}

for i in range(len(expid) - 1):
    # i = 1
    
    diff_pre_weighted_lon[expid[i+1]] = pre_weighted_lon[expid[0]].pre_weighted_lon[:36] - pre_weighted_lon[expid[i+1]].pre_weighted_lon.values
    diff_pre_weighted_lon[expid[i+1]].values[diff_pre_weighted_lon[expid[i+1]].values < -180] += 360
    diff_pre_weighted_lon[expid[i+1]].values[diff_pre_weighted_lon[expid[i+1]].values > 180] -= 360
    # stats.describe(diff_pre_weighted_lon[expid[i+1]], axis=None, nan_policy='omit')
    # wheremax = np.where(abs(diff_pre_weighted_lon[expid[i+1]]) == np.max(abs(diff_pre_weighted_lon[expid[i+1]])))
    # np.max(abs(diff_pre_weighted_lon[expid[i+1]]))
    # diff_pre_weighted_lon[expid[i+1]].values[wheremax]
    # pre_weighted_lon[expid[0]].pre_weighted_lon[:36].values[wheremax]
    # pre_weighted_lon[expid[i+1]].pre_weighted_lon.values[wheremax]
    
    diff_pre_weighted_lon_ann[expid[i+1]] = pre_weighted_lon_ann[expid[0]].pre_weighted_lon_ann[:3] - pre_weighted_lon_ann[expid[i+1]].pre_weighted_lon_ann.values
    diff_pre_weighted_lon_ann[expid[i+1]].values[diff_pre_weighted_lon_ann[expid[i+1]].values < -180] += 360
    diff_pre_weighted_lon_ann[expid[i+1]].values[diff_pre_weighted_lon_ann[expid[i+1]].values > 180] -= 360
    
    diff_pre_weighted_lon_sea[expid[i+1]] = pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea - pre_weighted_lon_sea[expid[i+1]].pre_weighted_lon_sea.values
    diff_pre_weighted_lon_sea[expid[i+1]].values[diff_pre_weighted_lon_sea[expid[i+1]].values < -180] += 360
    diff_pre_weighted_lon_sea[expid[i+1]].values[diff_pre_weighted_lon_sea[expid[i+1]].values > 180] -= 360
    diff_pre_weighted_lon_sea[expid[i+1]].to_netcdf(exp_odir + expid[i+1] + '/analysis/echam/' + expid[i+1] + '.diff_pre_weighted_lon_sea.nc')
    
    diff_pre_weighted_lon_am[expid[i+1]] = pre_weighted_lon_am[expid[0]].pre_weighted_lon_am - pre_weighted_lon_am[expid[i+1]].pre_weighted_lon_am.values
    diff_pre_weighted_lon_am[expid[i+1]].values[diff_pre_weighted_lon_am[expid[i+1]].values < -180] += 360
    diff_pre_weighted_lon_am[expid[i+1]].values[diff_pre_weighted_lon_am[expid[i+1]].values > 180] -= 360
    
    diff_pre_weighted_lon_am[expid[i+1]].to_netcdf(exp_odir + expid[i+1] + '/analysis/echam/' + expid[i+1] + '.diff_pre_weighted_lon_am.nc')

# stats.describe(diff_pre_weighted_lon_sea[expid[1]], axis=None, nan_policy='omit')
# stats.describe(diff_pre_weighted_lon_sea[expid[2]], axis=None, nan_policy='omit')
# stats.describe(diff_pre_weighted_lon_am[expid[1]], axis=None, nan_policy='omit')
# stats.describe(diff_pre_weighted_lon_am[expid[2]], axis=None, nan_policy='omit')


#-------------------------------- plot

#-------- basic set

lon = pre_weighted_lon[expid[0]].lon
lat = pre_weighted_lon[expid[0]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.4_lon/' + '6.1.0.4.0.0_cross check am_DJF_JJA pre_weighted_lon from scaling and binning tagmap.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'


pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

pltlevel2 = np.concatenate((np.arange(-5, -1 + 1e-4, 1), np.arange(1, 5 + 1e-4, 1)))
pltticks2 = np.arange(-5, 5 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

nrow = 3
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

#-------- annual mean values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lon_am[expid[0]].pre_weighted_lon_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, diff_pre_weighted_lon_am[expid[1]],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, diff_pre_weighted_lon_am[expid[2]],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[1]].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[2]].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[0]].pre_weighted_lon_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[1]].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[2]].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with longitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $20°$ longitude bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Diff. with partitioning tag\nmap with $10°$ longitude bins', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)




'''
    i0=0;i1=2;i2=60;i3=60
    pre_weighted_norm_sincoslon_sea[expid[0]].pre_weighted_norm_sincoslon_sea[i0, i1, i2, i3].values
    pre_weighted_norm_sincoslon_sea[expid[i+1]].pre_weighted_norm_sincoslon_sea[i0, i1, i2, i3].values
    pre_weighted_norm_sincoslon_sea[expid[0]].pre_weighted_norm_sincoslon_sea[i0, i1, i2, i3].values * pre_weighted_norm_sincoslon_sea[expid[i+1]].pre_weighted_norm_sincoslon_sea[i0, i1, i2, i3].values
    dot_product[i0, i1, i2, i3].values

# check where 0 dot product occurs
    stats.describe(dot_product, axis=None, nan_policy='omit')
    stats.describe(np.arccos(dot_product) * 180 / np.pi, axis=None, nan_policy='omit')
    i1,i2,i3 = np.where(dot_product == 0)
    dot_product.values[i1[0],i2[0],i3[0]]
    pre_weighted_norm_sincoslon_sea[expid[0]].pre_weighted_norm_sincoslon_sea.values[:, i1[0],i2[0],i3[0]] * pre_weighted_norm_sincoslon_sea[expid[i+1]].pre_weighted_norm_sincoslon_sea.values[:, i1[0],i2[0],i3[0]]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA scaled tagmap with lat [-90, 90]

#-------- import data

pre_weighted_lon = {}
pre_weighted_lon_ann = {}
pre_weighted_lon_sea = {}
pre_weighted_lon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')
    pre_weighted_lon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc')
    pre_weighted_lon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc')
    pre_weighted_lon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')

diff_pre_weighted_lon_sea = {}
diff_pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='DJF') - pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='JJA')

diff_pre_weighted_lon_sea[expid[i]].values[diff_pre_weighted_lon_sea[expid[i]].values < -180] += 360
diff_pre_weighted_lon_sea[expid[i]].values[diff_pre_weighted_lon_sea[expid[i]].values > 180] -= 360
# diff_pre_weighted_lon_sea[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.diff_pre_weighted_lon_sea.nc')

#-------- basic set

i = 0

lon = pre_weighted_lon[expid[i]].lon
lat = pre_weighted_lon[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.4_lon/' + '6.1.0.4.0.1_am_DJF_JJA pre_weighted_lon ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.concatenate((np.arange(-20, -2.5 + 1e-4, 2.5), np.arange(2.5, 20 + 1e-4, 2.5)))
pltticks2 = np.arange(-20, 20 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Am, DJF, JJA values
axs[0].pcolormesh(
    lon, lat, pre_weighted_lon_am[expid[i]].pre_weighted_lon_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[i]],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Antarctic plot am/DJF/JJA/DJF-JJA scaled tagmap with lat [-90, 90]

#-------- import data

pre_weighted_lon = {}
pre_weighted_lon_ann = {}
pre_weighted_lon_sea = {}
pre_weighted_lon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.nc')
    pre_weighted_lon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_ann.nc')
    pre_weighted_lon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_sea.nc')
    pre_weighted_lon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')

diff_pre_weighted_lon_sea = {}
diff_pre_weighted_lon_sea[expid[i]] = pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='DJF') - pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='JJA')

diff_pre_weighted_lon_sea[expid[i]].values[diff_pre_weighted_lon_sea[expid[i]].values < -180] += 360
diff_pre_weighted_lon_sea[expid[i]].values[diff_pre_weighted_lon_sea[expid[i]].values > 180] -= 360
# diff_pre_weighted_lon_sea[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.diff_pre_weighted_lon_sea.nc')


#-------- basic set

i = 0

lon = pre_weighted_lon[expid[i]].lon
lat = pre_weighted_lon[expid[i]].lat
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.4_lon/' + '6.1.0.4.0.2_Antarctica am_DJF_JJA pre_weighted_lon ' + expid[i] + '.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'


pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.concatenate((np.arange(-30, -2.5 + 1e-4, 2.5), np.arange(2.5, 30 + 1e-4, 2.5)))
pltticks2 = np.arange(-30, 30 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


#-------- plot configuration



nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])


#-------- Am, DJF, JJA values
axs[0].pcolormesh(
    lon, lat, pre_weighted_lon_am[expid[i]].pre_weighted_lon_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lon_sea[expid[i]].pre_weighted_lon_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[3].pcolormesh(
    lon, lat, diff_pre_weighted_lon_sea[expid[i]],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


