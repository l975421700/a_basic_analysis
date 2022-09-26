

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0

ifile_start = 120
ifile_end   = 1080 # 1080

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

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))
exp_org_o[expid[i]]['wiso_q_plev'] = xr.open_mfdataset(
    filenames_wiso_q_plev[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)

'''
ifile = -1
filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))
ncfile = xr.open_dataset(filenames_uvq_plev[ifile_start:ifile_end][ifile])

data1 = ncfile.q[0].values
data2 = exp_org_o[expid[i]]['wiso_q_plev'].q16o[ifile].values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))

data1 = ncfile.xl[0].values
data2 = exp_org_o[expid[i]]['wiso_q_plev'].xl16o[ifile].values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))


data1 = ncfile.xi[0].values
data2 = exp_org_o[expid[i]]['wiso_q_plev'].xi16o[ifile].values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_q

time = exp_org_o[expid[i]]['wiso_q_plev'].time
lon  = exp_org_o[expid[i]]['wiso_q_plev'].lon
lat  = exp_org_o[expid[i]]['wiso_q_plev'].lat
plev = exp_org_o[expid[i]]['wiso_q_plev'].plev

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 0
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon',]
itags = [5, 7, 8, 9, 11, 12]

ocean_q = {}
ocean_q[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(plev), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'plev':         plev,
        'lat':          lat,
        'lon':          lon,
    }
)

for count,var_name in enumerate(var_names):
    # count = 0; var_name = 'lat'
    
    kstart = kwiso2 + sum(ntags[:itags[count]])
    
    str_ind1 = str(kstart + 2)
    str_ind2 = str(kstart + 3)
    
    if (len(str_ind1) == 1): str_ind1 = '0' + str_ind1
    if (len(str_ind2) == 1): str_ind2 = '0' + str_ind2
    
    print(str(count) + ' : ' + var_name + ' : ' + str(itags[count]) + \
        ' : ' + str_ind1 + ' : ' + str_ind2)
    
    ocean_q[expid[i]].sel(var_names=var_name)[:] = \
        (exp_org_o[expid[i]]['wiso_q_plev']['q_' + str_ind1] + \
            exp_org_o[expid[i]]['wiso_q_plev']['q_' + str_ind2] + \
                exp_org_o[expid[i]]['wiso_q_plev']['xl_' + str_ind1] + \
                    exp_org_o[expid[i]]['wiso_q_plev']['xl_' + str_ind2] + \
                        exp_org_o[expid[i]]['wiso_q_plev']['xi_' + str_ind1] + \
                            exp_org_o[expid[i]]['wiso_q_plev']['xi_' + str_ind2]
                            )

ocean_q_alltime = {}
ocean_q_alltime[expid[i]] = mon_sea_ann(
    var_monthly=ocean_q[expid[i]],)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_q[expid[i]]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'wb') as f:
    pickle.dump(ocean_q_alltime[expid[i]], f)



'''
#-------------------------------- check

ocean_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_q_alltime.pkl', 'rb') as f:
    ocean_q_alltime[expid[i]] = pickle.load(f)
print(psutil.Process().memory_info().rss / (2 ** 30))

data1 = ocean_q_alltime[expid[i]]['mon'].sel(var_names='lat').values
data2 = ocean_q_alltime[expid[i]]['mon'].sel(var_names='sinlon').values
np.max(abs(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))

filenames_wiso_q_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso_q_plev[ifile_start:ifile_end][ifile])

data1 = (ncfile.q_02[0] + ncfile.q_03[0] + ncfile.xl_02[0] + ncfile.xl_03[0] + \
    ncfile.xi_02[0] + ncfile.xi_03[0]).values
data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='lat')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

data1 = (ncfile.q_24[0] + ncfile.q_25[0] + ncfile.xl_24[0] + ncfile.xl_25[0] + \
    ncfile.xi_24[0] + ncfile.xi_25[0]).values
data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='coslon')).values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()


# data1 = (ncfile.q_16[0] + ncfile.q_18[0] + ncfile.xl_16[0] + ncfile.xl_18[0] + \
#     ncfile.xi_16[0] + ncfile.xi_18[0]).values
# data2 = (ocean_q_alltime[expid[i]]['mon'][ifile].sel(var_names='geo7')).values
# (data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

#-------- include that of geo7
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'geo7']
ocean_q[expid[i]].sel(var_names='geo7')[:] = \
    (exp_org_o[expid[i]]['wiso_q_plev']['q_16'] + \
        exp_org_o[expid[i]]['wiso_q_plev']['q_18'] + \
            exp_org_o[expid[i]]['wiso_q_plev']['xl_16'] + \
                exp_org_o[expid[i]]['wiso_q_plev']['xl_18'] + \
                    exp_org_o[expid[i]]['wiso_q_plev']['xi_16'] + \
                        exp_org_o[expid[i]]['wiso_q_plev']['xi_18']
                        )
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann wiso_q



# endregion
# -----------------------------------------------------------------------------

