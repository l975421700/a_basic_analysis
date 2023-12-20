

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    'nudged_703_6.0_k52',
    ]
i = 0
ifile_start = 0
ifile_end   = 528

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
    plot_maxmin_points,
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


# Monthly data
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_zh_st_ml = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_geop_t.nc'))

exp_org_o[expid[i]]['zh_st_ml'] = xr.open_mfdataset(
    filenames_zh_st_ml[ifile_start:ifile_end],
    # data_vars='minimal', coords='minimal', parallel=True,
    )


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann zh and st

zh_st_ml = {}
zh_st_ml[expid[i]] = {}

zh_st_ml[expid[i]]['zh'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['zh_st_ml'].zh)
zh_st_ml[expid[i]]['st'] = mon_sea_ann(var_monthly=exp_org_o[expid[i]]['zh_st_ml'].st)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'wb') as f:
    pickle.dump(zh_st_ml[expid[i]], f)




'''
#-------- import data
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]

zh_st_ml = {}
for i in range(len(expid)):
    print(str(i) + ' ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'rb') as f:
        zh_st_ml[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# Daily data
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_zh_st_ml = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.daily_geop_t.nc'))

exp_org_o[expid[i]]['zh_st_ml'] = xr.open_mfdataset(
    filenames_zh_st_ml[ifile_start:ifile_end],
    # data_vars='minimal', coords='minimal', parallel=True,
    )


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann zh and st

zh_st_ml = {}
zh_st_ml[expid[i]] = {}

zh_st_ml[expid[i]]['zh'] = mon_sea_ann(exp_org_o[expid[i]]['zh_st_ml'].zh)
zh_st_ml[expid[i]]['st'] = mon_sea_ann(exp_org_o[expid[i]]['zh_st_ml'].st)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'wb') as f:
    pickle.dump(zh_st_ml[expid[i]], f)




'''
#-------- import data
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
    ]

zh_st_ml = {}
for i in range(len(expid)):
    print(str(i) + ' ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.zh_st_ml.pkl', 'rb') as f:
        zh_st_ml[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


