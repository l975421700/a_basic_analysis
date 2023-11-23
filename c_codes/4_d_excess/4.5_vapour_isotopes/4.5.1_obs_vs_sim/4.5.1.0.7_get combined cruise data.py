

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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

NK16_Australia_Syowa_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.NK16_Australia_Syowa_1d_sim.pkl', 'rb') as f:
    NK16_Australia_Syowa_1d_sim[expid[i]] = pickle.load(f)

IT20_ACE_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.IT20_ACE_1d_sim.pkl', 'rb') as f:
    IT20_ACE_1d_sim[expid[i]] = pickle.load(f)

BJ19_polarstern_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.BJ19_polarstern_1d_sim.pkl', 'rb') as f:
    BJ19_polarstern_1d_sim[expid[i]] = pickle.load(f)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region combine data

SO_vapor_isotopes = {}

columns_subset = ['time', 'lat', 'lon', 'dD', 'd18O', 'd_xs', 'd_ln', 'q', 'dD_sim', 'd18O_sim', 'd_xs_sim', 'd_ln_sim', 'q_sim']

SO_vapor_isotopes[expid[i]] = pd.concat(
    [NK16_Australia_Syowa_1d_sim[expid[i]][columns_subset].assign(
        Reference='Kurita et al. (2016)'),
     IT20_ACE_1d_sim[expid[i]][columns_subset].assign(
         Reference='Thurnherr et al. (2020)'),
     BJ19_polarstern_1d_sim[expid[i]][columns_subset].assign(
         Reference='Bonne et al. (2019)'),],
    ignore_index=True,)


output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(SO_vapor_isotopes[expid[i]], f)




'''
#-------------------------------- check

SO_vapor_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes.pkl', 'rb') as f:
    SO_vapor_isotopes[expid[i]] = pickle.load(f)

columns_subset = ['time', 'lat', 'lon', 'dD', 'd18O', 'd_xs', 'd_ln', 'q', 'dD_sim', 'd18O_sim', 'd_xs_sim', 'd_ln_sim', 'q_sim']

data1 = SO_vapor_isotopes[expid[i]][SO_vapor_isotopes[expid[i]]['Reference'] == 'Kurita et al. (2016)']
data2 = NK16_Australia_Syowa_1d_sim[expid[i]][columns_subset].assign(Reference='Kurita et al. (2016)')
print(data1.values[data1 != data2])
print(data2.values[data1 != data2])

data1 = SO_vapor_isotopes[expid[i]][SO_vapor_isotopes[expid[i]]['Reference'] == 'Thurnherr et al. (2020)']
data2 = IT20_ACE_1d_sim[expid[i]][columns_subset].assign(
         Reference='Thurnherr et al. (2020)')
print(data1.values[data1.values != data2])
print(data2.values[data1.values != data2])

data1 = SO_vapor_isotopes[expid[i]][SO_vapor_isotopes[expid[i]]['Reference'] == 'Bonne et al. (2019)']
data2 = BJ19_polarstern_1d_sim[expid[i]][columns_subset].assign(
         Reference='Bonne et al. (2019)')
print(data1.values[data1.values != data2])
print(data2.values[data1.values != data2])


SO_vapor_isotopes[expid[i]][(SO_vapor_isotopes[expid[i]]['lat'] >= -60) & (SO_vapor_isotopes[expid[i]]['lat'] <= -20)]

'''
# endregion
# -----------------------------------------------------------------------------



