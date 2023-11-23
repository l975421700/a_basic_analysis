

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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import pearsonr
from scipy.stats import linregress
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath

# self defined
from a_basic_analysis.b_module.mapplot import (
    remove_trailing_zero,
    remove_trailing_zero_pos,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_multi_gridvalue_at_site,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

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


# -----------------------------------------------------------------------------
# region check original CB19 data

CB19_DDU_1min = pd.read_csv(
    'data_sources/water_isotopes/CB19/Breant_2019.txt',
    sep = '\s+', header=0, skiprows=4,)

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
ax.plot(CB19_DDU_1min['d-excess'], ls='-', lw=0.2, )

ax.set_xlabel('Minutes from 2016-12-25 to 2017-02-03')
ax.set_ylabel('$d_{xs}$ [$‰$]')

fig.tight_layout()
fig.savefig('figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.6_CB19/8.3.0.6.0_original minute d_xs.png')


fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
ax.plot(CB19_DDU_1min['dD.1'].values - 8 * CB19_DDU_1min['d18O.1'].values, ls='-', lw=0.2, )

ax.set_xlabel('Minutes from 2016-12-25 to 2017-02-03')
ax.set_ylabel('$d_{xs}$ [$‰$]')

fig.tight_layout()
fig.savefig('figures/8_d-excess/8.3_vapour/8.3.0_obs_vs_sim/8.3.0.6_CB19/8.3.0.6.0_original minute d_xs calculated from corrected data.png')


# endregion
# -----------------------------------------------------------------------------

