

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
i = 0


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
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import equivalent_potential_temperature
from metpy.units import units

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
    time_weighted_mean,
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
# region import data

# temperature
st_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.st_plev.pkl',
    'rb') as f:
    st_plev[expid[i]] = pickle.load(f)

# specific humidity
ifile_start = 120
ifile_end =   720
exp_org_o = {}
exp_org_o[expid[i]] = {}
filenames_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_q_plev.nc'))
exp_org_o[expid[i]]['q_plev'] = xr.open_mfdataset(
    filenames_q_plev[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True,)

'''
q_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_plev.pkl',
    'rb') as f:
    q_plev[expid[i]] = pickle.load(f)

tpot_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tpot_plev.pkl',
    'rb') as f:
    tpot_plev[expid[i]] = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get dewpoint_from_specific_humidity

b_plev = np.broadcast_to(
    (st_plev[expid[i]]['mon'].plev.values * units.Pa)[None, :, None, None],
    exp_org_o[expid[i]]['q_plev'].q.shape,)

dew_point = dewpoint_from_specific_humidity(
    pressure=b_plev,
    temperature=st_plev[expid[i]]['mon'].values * units.K,
    specific_humidity=exp_org_o[expid[i]]['q_plev'].q * units('kg/kg'),)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dew_point.pkl',
    'wb') as f:
    pickle.dump(dew_point, f)



'''
#-------------------------------- check
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dew_point.pkl',
    'rb') as f:
    dew_point = pickle.load(f)

itime = 100
iplev = 10
ilat = 40
ilon = 30
pressure = st_plev[expid[i]]['mon'].plev.values[iplev] * units.Pa
temperature = st_plev[expid[i]]['mon'][itime, iplev, ilat, ilon].values * units.K
specific_q = q_plev[expid[i]]['mon'][itime, iplev, ilat, ilon].values * 1000 * units('g/kg')

dewpoint_from_specific_humidity(pressure, temperature, specific_q)
dew_point[itime, iplev, ilat, ilon].magnitude


# https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.dewpoint_from_specific_humidity.html
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon equivalent_potential_temperature

b_plev = np.broadcast_to(
    (st_plev[expid[i]]['mon'].plev.values * units.Pa)[None, :, None, None],
    exp_org_o[expid[i]]['q_plev'].q.shape,)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dew_point.pkl',
    'rb') as f:
    dew_point = pickle.load(f)

theta_e = equivalent_potential_temperature(
    pressure = b_plev,
    temperature = st_plev[expid[i]]['mon'].values * units.K,
    dewpoint = dew_point.magnitude*units.degC,)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e.pkl',
    'wb') as f:
    pickle.dump(theta_e, f)


'''
#-------------------------------- check
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e.pkl',
    'rb') as f:
    theta_e = pickle.load(f)

itime = 100
iplev = 10
ilat = 40
ilon = 30
pressure = st_plev[expid[i]]['mon'].plev.values[iplev] * units.Pa
temperature = st_plev[expid[i]]['mon'][itime, iplev, ilat, ilon].values * units.K
dewpoint = (dew_point[itime, iplev, ilat, ilon].magnitude + zerok) * units.K

equivalent_potential_temperature(pressure, temperature, dewpoint,)

theta_e[itime, iplev, ilat, ilon].magnitude

# https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.equivalent_potential_temperature.html
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann equivalent_potential_temperature

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e.pkl',
    'rb') as f:
    theta_e = pickle.load(f)

theta_e_xr = xr.DataArray(
    data=theta_e.magnitude,
    dims=['time', 'plev', 'lat', 'lon'],
    coords=dict(
        time=st_plev[expid[i]]['mon'].time,
        plev=st_plev[expid[i]]['mon'].plev,
        lon=st_plev[expid[i]]['mon'].lon,
        lat=st_plev[expid[i]]['mon'].lat,),
    attrs=dict(units='K',),)

theta_e_alltime = mon_sea_ann(var_monthly=theta_e_xr)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e_alltime.pkl',
    'wb') as f:
    pickle.dump(theta_e_alltime, f)



'''
#-------------------------------- check
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.theta_e_alltime.pkl',
    'rb') as f:
    theta_e_alltime = pickle.load(f)

data1 = theta_e_alltime['mon'].values
data2 = theta_e.magnitude
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


'''
# endregion
# -----------------------------------------------------------------------------
