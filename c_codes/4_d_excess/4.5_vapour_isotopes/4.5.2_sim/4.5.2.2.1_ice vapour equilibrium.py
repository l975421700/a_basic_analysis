

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
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
    hemisphere_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)

from a_basic_analysis.b_module.namelist import (
    panel_labels,
    plot_labels,
    expid_colours,
    expid_labels,
    zerok,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

# 2 m temperature
temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)

# site locations
with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

# precipitation isotopes
dO18_alltime = {}
dD_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)

# surface snow isotopes
exp_out_wiso = xr.open_mfdataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/test/unknown/test_2000??.01_wiso.nc')

VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266

wisosnglac_am = exp_out_wiso.wisosnglac.mean(dim='time')

dO18_am = (((wisosnglac_am.sel(wisotype=2) / wisosnglac_am.sel(wisotype=1)) / VSMOW_O18 - 1) * 1000).compute()

dD_am = (((wisosnglac_am.sel(wisotype=3) / wisosnglac_am.sel(wisotype=1)) / VSMOW_D - 1) * 1000).compute()

d_xs_am = dD_am - 8 * dO18_am

ln_dD = 1000 * np.log(1 + dD_am / 1000)
ln_d18O = 1000 * np.log(1 + dO18_am / 1000)
d_ln_am = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get equilibrium isotopic compositions of vapour


temp2_am = temp2_alltime[expid[i]]['am'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest',) + zerok

alpha_D = np.e ** (0.2133 - 203.10 / temp2_am + 48888 / temp2_am**2)
alpha_O18 = np.e ** (0.0831 - 49.192 / temp2_am + 8312.5 / temp2_am**2)


# vapour isotopes in equilibrium with surface snow

dD = ((dD_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest') / 1000 + 1) / alpha_D - 1) * 1000 # -498.71061428
dO18 = ((dO18_am.sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest') / 1000 + 1) / alpha_O18 - 1) * 1000 # -70.68451744

d_xs = dD - 8 * dO18 # 66.76552524

ln_dD = 1000 * np.log(1 + dD / 1000)
ln_d18O = 1000 * np.log(1 + dO18 / 1000)
d_ln = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2) # 83.49522655


# vapour isotopes in equilibrium with am precipitation

dD = ((dD_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest') / 1000 + 1) / alpha_D - 1) * 1000 # -532.66917167
dO18 = ((dO18_alltime[expid[i]]['am'].sel(lon=t63_sites_indices['EDC']['lon'], lat=t63_sites_indices['EDC']['lat'], method='nearest') / 1000 + 1) / alpha_O18 - 1) * 1000 # -76.12778353

d_xs = dD - 8 * dO18 # 76.35309653

ln_dD = 1000 * np.log(1 + dD / 1000)
ln_d18O = 1000 * np.log(1 + dO18 / 1000)
d_ln = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2) # 88.63631781



dD = dD_alltime[expid[i]]['am'].sel(
    lat=t63_sites_indices['EDC']['lat'],
    lon=t63_sites_indices['EDC']['lon'],
    method='nearest',
    ).values
# -370.18395893


dD_vapour = ((dD / 1000 + 1) / alpha_D - 1) * 1000


print('Fake surface vapour isotopes: ' + str(np.round(dD_vapour, 1)))

((-324 / 1000 + 1) / alpha_D - 1) * 1000
# -498

'''
dD_vapour_output = dD_q_sfc_alltime[expid[i]]['am'].sel(
    lat=t63_sites_indices['EDC']['lat'],
    lon=t63_sites_indices['EDC']['lon'],
    method='nearest',
    ).values
print('Simulated surface vapour isotopes: ' + str(np.round(dD_vapour_output, 1)))

'''
# endregion
# -----------------------------------------------------------------------------


