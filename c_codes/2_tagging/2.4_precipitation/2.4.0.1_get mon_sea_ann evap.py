

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

filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
exp_org_o[expid[i]]['sf_wiso'] = xr.open_mfdataset(
    filenames_sf_wiso[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)


'''
#-------- check evap
filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
filenames_echam = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
ncfile1 = xr.open_dataset(filenames_sf_wiso[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(filenames_echam[ifile_start:ifile_end][ifile])

np.max(abs(ncfile1.wisoevap[0, 0] - ncfile2.evap[0])).values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann evap


wisoevap = {}
wisoevap[expid[i]] = (exp_org_o[expid[i]]['sf_wiso'].wisoevap[:, :3]).copy().compute()

wisoevap[expid[i]] = wisoevap[expid[i]].rename('wisoevap')

wisoevap_alltime = {}
wisoevap_alltime[expid[i]] = mon_sea_ann(var_monthly = wisoevap[expid[i]])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl',
          'wb') as f:
    pickle.dump(wisoevap_alltime[expid[i]], f)


'''
#-------- check
wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
filenames_echam = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = -2
ncfile1 = xr.open_dataset(filenames_sf_wiso[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(filenames_echam[ifile_start:ifile_end][ifile])

(wisoevap_alltime[expid[i]]['mon'][ifile, 0] == ncfile2.evap[0]).all().values
(wisoevap_alltime[expid[i]]['mon'][ifile, :] == ncfile1.wisoevap[0, :3]).all().values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann ERA5 evap

era5_mon_evap_1979_2021 = xr.open_dataset(
    'scratch/products/era5/evap/era5_mon_evap_1979_2021.nc')

# change units to mm/d
era5_mon_evap_1979_2021_alltime = mon_sea_ann(
    var_monthly = (era5_mon_evap_1979_2021.e * 1000).compute()
)

with open('scratch/products/era5/evap/era5_mon_evap_1979_2021_alltime.pkl', 'wb') as f:
    pickle.dump(era5_mon_evap_1979_2021_alltime, f)


'''
#-------- check
with open('scratch/products/era5/evap/era5_mon_evap_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_evap_1979_2021_alltime = pickle.load(f)
era5_mon_evap_1979_2021 = xr.open_dataset(
    'scratch/products/era5/evap/era5_mon_evap_1979_2021.nc')

(era5_mon_evap_1979_2021_alltime['mon'] == (era5_mon_evap_1979_2021.e * 1000)).values.all()

'''
# endregion
# -----------------------------------------------------------------------------


