

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_502_5.0',]
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
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec_num,
    month_dec,
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

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(
    filenames_echam[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann evapiac

evapiac_alltime = {}
evapiac_alltime[expid[i]] = mon_sea_ann(
    var_monthly=exp_org_o[expid[i]]['echam'].evapiac
)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.evapiac_alltime.pkl', 'wb') as f:
    pickle.dump(evapiac_alltime[expid[i]], f)



'''
#-------------------------------- check
evapiac_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.evapiac_alltime.pkl', 'rb') as f:
    evapiac_alltime[expid[i]] = pickle.load(f)

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

itime = -10
ncfile = xr.open_dataset(filenames_echam[ifile_start:ifile_end][itime])

(ncfile.evapiac == evapiac_alltime[expid[i]]['mon'][itime]).all().values

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann evapwac

evapwac_alltime = {}
evapwac_alltime[expid[i]] = mon_sea_ann(
    var_monthly=exp_org_o[expid[i]]['echam'].evapwac
)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.evapwac_alltime.pkl', 'wb') as f:
    pickle.dump(evapwac_alltime[expid[i]], f)



'''
#-------------------------------- check
evapwac_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.evapwac_alltime.pkl', 'rb') as f:
    evapwac_alltime[expid[i]] = pickle.load(f)

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

itime = -10
ncfile = xr.open_dataset(filenames_echam[ifile_start:ifile_end][itime])

(ncfile.evapwac == evapwac_alltime[expid[i]]['mon'][itime]).all().values

evapwac_alltime[expid[i]]['am'].to_netcdf('scratch/test/test.nc')

'''
# endregion
# -----------------------------------------------------------------------------
