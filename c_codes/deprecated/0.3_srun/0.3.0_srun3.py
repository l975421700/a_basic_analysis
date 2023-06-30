

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    'pi_603_5.3',
    ]
i = 0

ifile_start = 120
ifile_end   = 360 # 1080

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
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
    # data_vars='minimal', coords='minimal', parallel=True,
    )

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
# region get mon/sea/ann aprt_geo7 !!! run in sbatch


aprt_geo7 = {}
aprt_geo7[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(16, 22)) + \
        exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(16, 22))
    ).compute()

aprt_geo7[expid[i]] = aprt_geo7[expid[i]].rename('aprt_geo7')

aprt_geo7_alltime = {}
aprt_geo7_alltime[expid[i]] = mon_sea_ann(aprt_geo7[expid[i]], lcopy = False,)

aprt_geo7_alltime[expid[i]]['sum'] = {}
for ialltime in aprt_geo7_alltime[expid[i]].keys():
    # ialltime = 'daily'
    if (ialltime != 'sum'):
        print(ialltime)
        aprt_geo7_alltime[expid[i]]['sum'][ialltime] = \
            aprt_geo7_alltime[expid[i]][ialltime].sum(dim='wisotype')

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'wb') as f:
    pickle.dump(aprt_geo7_alltime[expid[i]], f)



'''
#-------- check calculation
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[ifile_start:ifile_end][ifile])

(aprt_geo7_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, 15:22] + ncfile.wisoaprc[:, 15:22].values)).all().values

(aprt_geo7_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:,15:22] + ncfile.wisoaprc[:,15:22].values).mean(dim='time')).all().values


#-------- check calculation of sum over wisotypes
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

print((aprt_geo7_alltime[expid[i]]['sum']['daily'] == aprt_geo7_alltime[expid[i]]['daily'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mon'] == aprt_geo7_alltime[expid[i]]['mon'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sea'] == aprt_geo7_alltime[expid[i]]['sea'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['ann'] == aprt_geo7_alltime[expid[i]]['ann'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mm'] == aprt_geo7_alltime[expid[i]]['mm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sm'] == aprt_geo7_alltime[expid[i]]['sm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['am'] == aprt_geo7_alltime[expid[i]]['am'].sum(dim='wisotype')).all().values)


#-------- check am values
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

aprt_geo7_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')

'''
# endregion
# -----------------------------------------------------------------------------

