

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_402_4.7',
    'pi_m_416_4.9',
    ]

# region import output

i = 0
expid[i]

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))

exp_org_o[expid[i]]['uvq_plev'] = xr.open_mfdataset(filenames_uvq_plev[120:], data_vars='minimal', coords='minimal', parallel=True)


'''
filenames_echam_uvq = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_*_monthly_uvq_plev.01_echam.nc'))
exp_org_o[expid[i]]['echam_uvq'] = xr.open_mfdataset(filenames_echam_uvq, data_vars='minimal', coords='minimal', parallel=True)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate moisture flux

i = 0
expid[i]

moisture_flux = {}
moisture_flux[expid[i]] = {}

q_total = (exp_org_o[expid[i]]['uvq_plev'].q + \
    exp_org_o[expid[i]]['uvq_plev'].xl.values + \
        exp_org_o[expid[i]]['uvq_plev'].xi.values).compute()

zonal_moisture_flux = (exp_org_o[expid[i]]['uvq_plev'].u * q_total.values).compute()
zonal_moisture_flux = zonal_moisture_flux.rename('zonal_moisture_flux')
zonal_moisture_flux.attrs['unit'] = 'kg/kg * m/s'

meridional_moisture_flux = (exp_org_o[expid[i]]['uvq_plev'].v * q_total.values).compute()
meridional_moisture_flux = meridional_moisture_flux.rename('meridional_moisture_flux')
meridional_moisture_flux.attrs['unit'] = 'kg/kg * m/s'

moisture_flux[expid[i]]['zonal'] = mon_sea_ann(var_monthly = zonal_moisture_flux)
moisture_flux[expid[i]]['meridional'] = mon_sea_ann(var_monthly = meridional_moisture_flux)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.pkl', 'wb') as f:
    pickle.dump(moisture_flux[expid[i]], f)



'''
#-------- check

# calculation in function and manually
def time_weighted_mean(ds):
    return ds.weighted(ds.time.dt.days_in_month).mean('time', skipna=True)

test = {}
test['mon'] = zonal_moisture_flux.copy()
test['sea'] = zonal_moisture_flux.resample({'time': 'Q-FEB'}).map(time_weighted_mean)[1:-1].compute()
test['ann'] = zonal_moisture_flux.resample({'time': '1Y'}).map(time_weighted_mean).compute()
test['mm'] = test['mon'].groupby('time.month').mean(skipna=True).compute()
test['sm'] = test['sea'].groupby('time.season').mean(skipna=True).compute()
test['am'] = test['ann'].mean(dim='time', skipna=True).compute()

(moisture_flux[expid[i]]['zonal']['mon'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['mon'].values)] == test['mon'].values[np.isfinite(test['mon'].values)]).all()
(moisture_flux[expid[i]]['zonal']['sea'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['sea'].values)] == test['sea'].values[np.isfinite(test['sea'].values)]).all()
(moisture_flux[expid[i]]['zonal']['ann'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['ann'].values)] == test['ann'].values[np.isfinite(test['ann'].values)]).all()
(moisture_flux[expid[i]]['zonal']['mm'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['mm'].values)] == test['mm'].values[np.isfinite(test['mm'].values)]).all()
(moisture_flux[expid[i]]['zonal']['sm'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['sm'].values)] == test['sm'].values[np.isfinite(test['sm'].values)]).all()
(moisture_flux[expid[i]]['zonal']['am'].values[np.isfinite(moisture_flux[expid[i]]['zonal']['am'].values)] == test['am'].values[np.isfinite(test['am'].values)]).all()

zonal_moisture_flux.weighted(zonal_moisture_flux.time.dt.days_in_month).mean('time', skipna=True).compute()
zonal_moisture_flux.resample({'time': 'Q-FEB'}).mean()


#-------- check

(np.isnan(exp_org_o[expid[i]]['uvq_plev'].q.values) == np.isnan(exp_org_o[expid[i]]['uvq_plev'].xl.values)).all()
(np.isnan(exp_org_o[expid[i]]['uvq_plev'].q.values) == np.isnan(exp_org_o[expid[i]]['uvq_plev'].xi.values)).all()

zonal_moisture_flux.to_netcdf('scratch/test/test1.nc')
meridional_moisture_flux.to_netcdf('scratch/test/test2.nc')

#-------- check calculation of pressure

i0 = 46
i1 = 20
i2 = 30
i3 = 40

exp_org_o[expid[i]]['echam'].hyam[i0].values
exp_org_o[expid[i]]['echam'].hybm[i0].values
exp_org_o[expid[i]]['echam'].aps[i1, i2, i3].values
exp_org_o[expid[i]]['echam'].hyam[i0].values + exp_org_o[expid[i]]['echam'].hybm[i0].values * exp_org_o[expid[i]]['echam'].aps[i1, i2, i3].values

pres[i0, i1, i2, i3].values

#-------- calculate pressure value from original data
pres = xr.DataArray(
    data = exp_org_o[expid[i]]['echam'].hyam.values[None, :, None, None] + exp_org_o[expid[i]]['echam'].hybm.values[None, :, None, None] * exp_org_o[expid[i]]['echam'].aps.values[:, None, :, :],
    coords=dict(
        time=exp_org_o[expid[i]]['echam'].time,
        lev=exp_org_o[expid[i]]['echam'].lev,
        lat=exp_org_o[expid[i]]['echam'].lat,
        lon=exp_org_o[expid[i]]['echam'].lon,),
    attrs=dict(
        units="Pa",),
    name='pres'
)
pres.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pres.nc')


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot moisture flux: along 60 degree South

i = 0
expid[i]

moisture_flux = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.pkl', 'rb') as f:
    moisture_flux[expid[i]] = pickle.load(f)

plev = moisture_flux[expid[i]]['meridional']['am'].plev
lon = moisture_flux[expid[i]]['meridional']['am'].lon
lat = moisture_flux[expid[i]]['meridional']['am'].lat


moisture_flux_am = xr.merge(
    [moisture_flux[expid[i]]['meridional']['am'],
    moisture_flux[expid[i]]['zonal']['am']
    ],
    compat='override')

moisture_flux_am.to_netcdf('scratch/test/test1.nc')

'''

#-------- create cross section

moisture_flux_am = {}

moisture_flux_am[expid[i]] = xr.Dataset(
    {'meridional_moisture_flux': (
        ('plev', 'y', 'x'),
        moisture_flux[expid[i]]['meridional']['am'].values,
        ),
     'zonal_moisture_flux': (
         ('plev', 'y', 'x'),
         moisture_flux[expid[i]]['zonal']['am'].values,
     ),
     'lat': (('y', 'x'), lat_2d),
     'lon': (('y', 'x'), lon_2d),
     },
    coords={
            "plev": plev.values,
            "y": lat.values,
            "x": lon.values,
        }
)

lon_2d, lat_2d = np.meshgrid(lon, lat,)

moisture_flux_am[expid[i]] = xr.merge(
    [moisture_flux[expid[i]]['meridional']['am'],
    moisture_flux[expid[i]]['zonal']['am']],
    compat='override')
moisture_flux_am[expid[i]] = moisture_flux_am[expid[i]].metpy.parse_cf()
# moisture_flux_am[expid[i]]['y'] = moisture_flux_am[expid[i]].lat.values[:, 0]
# moisture_flux_am[expid[i]]['x'] = moisture_flux_am[expid[i]].lon.values[0, :]

startpoint = [-60, -180]
endpoint = [-60, 180]

cross = cross_section(
    moisture_flux_am[expid[i]],
    startpoint, endpoint, steps=180+1,
)
# np.nanmax(cross.meridional_moisture_flux[:, 0] - cross.meridional_moisture_flux[:, 20])

dset_cross = moisture_flux_am[expid[i]].metpy.parse_cf()

dset_cross['y'] = dset_cross['lat'].values[:, 0]
dset_cross['x'] = dset_cross['lon'].values[0, :]

'''
# endregion
# -----------------------------------------------------------------------------

