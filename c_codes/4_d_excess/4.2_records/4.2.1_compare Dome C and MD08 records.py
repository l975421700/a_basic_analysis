

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0'
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
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import linregress
from haversine import haversine_vector

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
import cartopy.feature as cfeature
from matplotlib.ticker import AutoMinorLocator

# self defined
from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,
)

from a_basic_analysis.b_module.namelist import (
    plot_labels,
    expid_colours,
    expid_labels,
)

from a_basic_analysis.b_module.component_plot import (
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

Antarctic_snow_isotopes_sim_grouped_all = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.Antarctic_snow_isotopes_sim_grouped_all.pkl', 'rb') as f:
    Antarctic_snow_isotopes_sim_grouped_all[expid[i]] = pickle.load(f)

with open('data_sources/Dome_C_records/BS13_Dome_C.pkl', 'rb') as f:
    BS13_Dome_C = pickle.load(f)


site_pair = [-75.1, 123.35]
snow_pair = [[x, y] for x, y in zip(
    Antarctic_snow_isotopes_sim_grouped_all[expid[i]]['lat'].values,
    Antarctic_snow_isotopes_sim_grouped_all[expid[i]]['lon'].values,
    )]

distance = haversine_vector(site_pair, snow_pair, normalize=True, comb=True)
loc_ind = np.argmin(distance.flatten())

Antarctic_snow_isotopes_sim_grouped_all[expid[i]].iloc[loc_ind]
BS13_Dome_C['am']

'''
'''
# endregion
# -----------------------------------------------------------------------------


