

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    # 'nudged_701_5.0',
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
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import metpy

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

import metpy.calc
from metpy.units import units

geopot = units.Quantity(psl_zh[expid[i]]['zh']['am'].sel(plev=1e+05), 'm^2/s^2')
height = metpy.calc.geopotential_to_height(geopot)


# endregion
# -----------------------------------------------------------------------------

