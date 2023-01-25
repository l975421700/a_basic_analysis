

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0

ifile_start = 120
ifile_end   = 720 # 1080

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
import psutil

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
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
    mean_over_ais,
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
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
    filenames_wiso[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)


'''
#-------- check pre
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
nc1 = xr.open_dataset(filenames_wiso[ifile])
nc2 = xr.open_dataset(filenames_echam[ifile])

np.max(abs(nc1.wisoaprl[:, 0].mean(dim='time').values - nc2.aprl[0].values))


#-------- input previous files

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso_daily'] = xr.open_mfdataset(filenames_wiso_daily, data_vars='minimal', coords='minimal', parallel=True)
        
        filenames_echam_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_echam.nc'))
        exp_org_o[expid[i]]['echam_daily'] = xr.open_mfdataset(filenames_echam_daily[120:], data_vars='minimal', coords='minimal', parallel=True)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_aprt

time = exp_org_o[expid[i]]['wiso'].time
lon  = exp_org_o[expid[i]]['wiso'].lon
lat  = exp_org_o[expid[i]]['wiso'].lat

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 3
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'geo7']
itags = [5, 7, 8, 9, 11, 12]

ocean_aprt = {}
ocean_aprt[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'lat':          lat,
        'lon':          lon,
    }
)

for count,var_name in enumerate(var_names[:-1]):
    # count = 0; var_name = 'lat'
    kstart = kwiso2 + sum(ntags[:itags[count]])
    
    print(str(count) + ' : ' + var_name + ' : ' + str(itags[count]) + \
        ' : ' + str(kstart))
    
    ocean_aprt[expid[i]].sel(var_names=var_name)[:] = \
        (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(
            wisotype=slice(kstart+2, kstart+3)) + \
                exp_org_o[expid[i]]['wiso'].wisoaprc.sel(
                    wisotype=slice(kstart+2, kstart+3))
                ).sum(dim='wisotype')

ocean_aprt[expid[i]].sel(var_names='geo7')[:] = \
    (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(
        wisotype=[19, 21]) + \
            exp_org_o[expid[i]]['wiso'].wisoaprc.sel(
                wisotype=[19, 21])
            ).sum(dim='wisotype')

ocean_aprt_alltime = {}
ocean_aprt_alltime[expid[i]] = mon_sea_ann(
    ocean_aprt[expid[i]], lcopy = False,)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_aprt[expid[i]]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'wb') as f:
    pickle.dump(ocean_aprt_alltime[expid[i]], f)






'''

#-------------------------------- check ocean_aprt

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

ocean_aprt = {}
ocean_aprt[expid[i]] = ocean_aprt_alltime[expid[i]]['daily']

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[ifile_start:ifile_end][ifile])

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 3
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon']
itags = [5, 7, 8, 9, 11, 12]

ilat = 48
ilon = 90

for count in range(6):
    # count = 5
    print(count)
    
    kstart = kwiso2 + sum(ntags[:itags[count]])

    res1 = ocean_aprt[expid[i]][-31:, :, ilat, ilon].sel(
        var_names = var_names[count])

    res2 = ncfile.wisoaprl[:, :, ilat, ilon].sel(
        wisotype=[kstart+2, kstart+3]).sum(dim='wisotype') + \
            ncfile.wisoaprc[:, :, ilat, ilon].sel(
        wisotype=[kstart+2, kstart+3]).sum(dim='wisotype')

    print(np.max(abs(res1 - res2)).values)

# check 'geo7'
res1 = ocean_aprt[expid[i]][-31:, :, ilat, ilon].sel(var_names = 'geo7')
res2 = ncfile.wisoaprl[:, :, ilat, ilon].sel(
    wisotype=[19, 21]).sum(dim='wisotype') + \
        ncfile.wisoaprc[:, :, ilat, ilon].sel(
    wisotype=[19, 21]).sum(dim='wisotype')
print(np.max(abs(res1 - res2)).values)


#-------------------------------- check alltime calculation
ocean_aprt_alltime[expid[i]].keys()
(ocean_aprt_alltime[expid[i]]['daily'] == ocean_aprt[expid[i]]).all().values

(ocean_aprt_alltime[expid[i]]['mon'] == ocean_aprt_alltime[expid[i]]['daily'].resample({'time': '1M'}).mean()).all()

#-------------------------------- check ocean pre consistency
np.max(abs((ocean_aprt_alltime[expid[i]]['mon'][:, 5] - \
    ocean_aprt_alltime[expid[i]]['mon'][:, 0]) / \
        ocean_aprt_alltime[expid[i]]['mon'][:, 5]))
np.mean(abs(ocean_aprt_alltime[expid[i]]['mon'][:, 5] - ocean_aprt_alltime[expid[i]]['mon'][:, 0]))
np.mean(abs(ocean_aprt_alltime[expid[i]]['mon'][:, 5]))

np.max(abs((ocean_aprt_alltime[expid[i]]['am'][5] - \
    ocean_aprt_alltime[expid[i]]['am'][0]) / \
        ocean_aprt_alltime[expid[i]]['am'][5]))

'''
# endregion
# -----------------------------------------------------------------------------

