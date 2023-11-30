

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    
    'nudged_705_6.0',
    # 'nudged_706_6.0_k52_88',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    # 'nudged_711_6.0_S6',
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

T63GR15_jan_surf = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/input/echam/unit.24')
ERA5_daily_SIC_2013_2022 = xr.open_dataset('scratch/ERA5/SIC/ERA5_daily_SIC_2013_2022.nc', chunks={'time': 720})

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get SIC and SLM info and combine data


NK16_1d_SLM = find_multi_gridvalue_at_site(
    NK16_Australia_Syowa_1d_sim[expid[i]]['lat'].values,
    NK16_Australia_Syowa_1d_sim[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
    )
NK16_1d_SIC = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa_1d_sim[expid[i]]['time'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lat'],
    NK16_Australia_Syowa_1d_sim[expid[i]]['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc.values * 100
    )

IT20_1d_SLM = find_multi_gridvalue_at_site(
    IT20_ACE_1d_sim[expid[i]]['lat'].values,
    IT20_ACE_1d_sim[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
    )
IT20_1d_SIC = find_multi_gridvalue_at_site_time(
    IT20_ACE_1d_sim[expid[i]]['time'],
    IT20_ACE_1d_sim[expid[i]]['lat'],
    IT20_ACE_1d_sim[expid[i]]['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc.values * 100
    )

BJ19_1d_SLM = find_multi_gridvalue_at_site(
    BJ19_polarstern_1d_sim[expid[i]]['lat'].values,
    BJ19_polarstern_1d_sim[expid[i]]['lon'].values,
    T63GR15_jan_surf.lat.values,
    T63GR15_jan_surf.lon.values,
    T63GR15_jan_surf.SLM.values,
    )
BJ19_1d_SIC = find_multi_gridvalue_at_site_time(
    BJ19_polarstern_1d_sim[expid[i]]['time'],
    BJ19_polarstern_1d_sim[expid[i]]['lat'],
    BJ19_polarstern_1d_sim[expid[i]]['lon'],
    ERA5_daily_SIC_2013_2022.time.values,
    ERA5_daily_SIC_2013_2022.latitude.values,
    ERA5_daily_SIC_2013_2022.longitude.values,
    ERA5_daily_SIC_2013_2022.siconc.values * 100
    )

NK16_Australia_Syowa_1d_sim[expid[i]]['SLM'] = NK16_1d_SLM
NK16_Australia_Syowa_1d_sim[expid[i]]['SIC'] = NK16_1d_SIC

IT20_ACE_1d_sim[expid[i]]['SLM'] = IT20_1d_SLM
IT20_ACE_1d_sim[expid[i]]['SIC'] = IT20_1d_SIC

BJ19_polarstern_1d_sim[expid[i]]['SLM'] = BJ19_1d_SLM
BJ19_polarstern_1d_sim[expid[i]]['SIC'] = BJ19_1d_SIC


SO_vapor_isotopes_SLMSIC = {}

columns_subset = ['time', 'lat', 'lon', 'dD', 'd18O', 'd_xs', 'd_ln', 'q', 'dD_sim', 'd18O_sim', 'd_xs_sim', 'd_ln_sim', 'q_sim', 'SLM', 'SIC']

SO_vapor_isotopes_SLMSIC[expid[i]] = pd.concat(
    [NK16_Australia_Syowa_1d_sim[expid[i]][columns_subset].assign(
        Reference='Kurita et al. (2016)'),
     IT20_ACE_1d_sim[expid[i]][columns_subset].assign(
         Reference='Thurnherr et al. (2020)'),
     BJ19_polarstern_1d_sim[expid[i]][columns_subset].assign(
         Reference='Bonne et al. (2019)'),],
    ignore_index=True,)


output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(SO_vapor_isotopes_SLMSIC[expid[i]], f)




'''
#-------------------------------- check
SO_vapor_isotopes_SLMSIC = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.SO_vapor_isotopes_SLMSIC.pkl', 'rb') as f:
    SO_vapor_isotopes_SLMSIC[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


