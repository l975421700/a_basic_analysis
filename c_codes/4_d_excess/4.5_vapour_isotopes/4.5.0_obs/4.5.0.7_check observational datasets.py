

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
from scipy import stats
# import xesmf as xe
import pandas as pd

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)

with open('data_sources/water_isotopes/BJ19/BJ19_polarstern.pkl', 'rb') as f:
    BJ19_polarstern = pickle.load(f)

with open('data_sources/water_isotopes/IT20/IT20_ACE.pkl', 'rb') as f:
    IT20_ACE = pickle.load(f)

with open('data_sources/water_isotopes/FR16/FR16_Kohnen.pkl', 'rb') as f:
    FR16_Kohnen = pickle.load(f)

with open('data_sources/water_isotopes/SD21/SD21_Neumayer.pkl', 'rb') as f:
    SD21_Neumayer = pickle.load(f)

with open('data_sources/water_isotopes/CB19/CB19_DDU.pkl', 'rb') as f:
    CB19_DDU = pickle.load(f)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check data
NK16_Australia_Syowa['1d']
BJ19_polarstern['1d']
IT20_ACE['1d']

MC16_Dome_C['1d']
SD21_Neumayer['1d']
FR16_Kohnen['1d']

CB19_DDU['1d']
# endregion
# -----------------------------------------------------------------------------


