

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0
ifile_start = 120
ifile_end =   1080


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

filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))

exp_org_o[expid[i]]['uvq_plev'] = xr.open_mfdataset(
    filenames_uvq_plev[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann q_plev

q_plev = {}
q_plev[expid[i]] = mon_sea_ann(
    var_monthly=(
        exp_org_o[expid[i]]['uvq_plev'].q + \
            exp_org_o[expid[i]]['uvq_plev'].xl + \
                exp_org_o[expid[i]]['uvq_plev'].xi ).compute())

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_plev.pkl',
    'wb') as f:
    pickle.dump(q_plev[expid[i]], f)



'''
#-------------------------------- check

q_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_plev.pkl',
    'rb') as f:
    q_plev[expid[i]] = pickle.load(f)

filenames_uvq_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_uvq_plev[ifile_start:ifile_end][ifile])

data1 = q_plev[expid[i]]['mon'][ifile].values
data2 = (ncfile.q + ncfile.xl + ncfile.xi)[0].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

'''
# endregion
# -----------------------------------------------------------------------------

