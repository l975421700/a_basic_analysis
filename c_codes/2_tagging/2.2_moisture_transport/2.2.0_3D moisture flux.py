

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
from metpy.interpolate import cross_section

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
    'pi_m_402_4.7',
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
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        # exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
        #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        filenames_echam_uvq = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_*_monthly_uvq_plev.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        exp_org_o[expid[i]]['echam_uvq'] = xr.open_mfdataset(filenames_echam_uvq, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate moisture flux



q_total = exp_org_o[expid[i]]['echam_uvq'].q + exp_org_o[expid[i]]['echam_uvq'].xl.values + exp_org_o[expid[i]]['echam_uvq'].xi.values

moisture_flux_u = exp_org_o[expid[i]]['echam_uvq'].u * q_total.values
moisture_flux_u = moisture_flux_u.rename('moisture_flux_u')
moisture_flux_u.attrs['unit'] = 'kg/kg * m/s'

moisture_flux_v = exp_org_o[expid[i]]['echam_uvq'].v * q_total.values
moisture_flux_v = moisture_flux_v.rename('moisture_flux_v')
moisture_flux_v.attrs['unit'] = 'kg/kg * m/s'


moisture_flux = xr.merge([moisture_flux_u, moisture_flux_v])
moisture_flux.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.nc')



'''
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
moisture_flux_ann = {}
moisture_flux_sea = {}
moisture_flux_am = {}

moisture_flux[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.nc')
moisture_flux_ann[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux_ann.nc')
moisture_flux_sea[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux_sea.nc')
moisture_flux_am[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux_am.nc')


# create cross section
moisture_flux_am[expid[i]] = moisture_flux_am[expid[i]].metpy.parse_cf()

moisture_flux_am[expid[i]]['y'] = moisture_flux_am[expid[i]].lat.values
moisture_flux_am[expid[i]]['x'] = moisture_flux_am[expid[i]].lon.values

startpoint = [-60, -180]
endpoint = [-60, 180]

cross = cross_section(
    moisture_flux_am[expid[i]],
    startpoint, endpoint, steps=180+1,
).set_coords(('y', 'x'))


'''
cross1 = cross_section(
    dset_cross,
    startpoint1,
    endpoint1,
    steps=int(cross_section_distance1/1.1)+1,
).set_coords(('y', 'x'))
x_km1 = np.linspace(0, cross_section_distance1 * 1000, len(cross1.lon))
windlevel = np.arange(0, 15.1, 0.1)
ticks = np.arange(0, 15.1, 3)
'''


# endregion
# -----------------------------------------------------------------------------

