

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    'nudged_701_5.0',
    ]
i = 0

ifile_start = 12 #0 #120
ifile_end   = 516 #1740 #840


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
# sys.path.append('/work/ollie/qigao001')

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


# st_plev
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_st_plev = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_st_plev.nc'))

exp_org_o[expid[i]]['st_plev'] = xr.open_mfdataset(
    filenames_st_plev[ifile_start:ifile_end])

'''
data_vars='minimal', coords='minimal', parallel=True
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann st_plev


st_plev = {}

st_plev[expid[i]] = mon_sea_ann(
    var_monthly=exp_org_o[expid[i]]['st_plev'].st)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.st_plev.pkl',
    'wb') as f:
    pickle.dump(st_plev[expid[i]], f)


'''

#-------------------------------- check
st_plev = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.st_plev.pkl',
    'rb') as f:
    st_plev[expid[i]] = pickle.load(f)

data1 = st_plev[expid[i]]['mon'].values
data2 = exp_org_o[expid[i]]['st_plev'].st.values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# st_ml
# -----------------------------------------------------------------------------
# region import output

era5_forcing = xr.open_dataset('/albedo/work/projects/paleo_work/paleodyn_from_work_ollie_projects/paleodyn/nudging/ERA5/atmos/T63/era5T63L47_201012.nc')
era5_forcing = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/forcing/echam/ndg201012.nc')

echam6_output = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/unknown/nudged_701_5.0_201012.01_sp_1m.nc')

bias = era5_forcing['t'].sel(lev=47).mean(dim='time').values - echam6_output['st'].sel(lev=47)[0].values

print(np.mean(abs(bias)))
print(np.max(bias))

bias = era5_forcing['t'].sel(lev=47).mean(dim='time') - echam6_output['st'].sel(lev=47)[0]
# endregion
# -----------------------------------------------------------------------------

