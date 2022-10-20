

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd, circmean

# plot
import proplot as pplt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

aprt_frc = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc.pkl', 'rb') as f:
    aprt_frc[expid[i]] = pickle.load(f)

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)


lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region site_latlon = {'lat':-70, 'lon':90}

site_latlon = {'lat':-70, 'lon':90}

site_latlon['ilat'], site_latlon['ilon'] = \
        find_ilat_ilon(site_latlon['lat'], site_latlon['lon'],
                       lat.values, lon.values)

sm_djf_lat = pre_weighted_lat[expid[i]]['sm'].sel(season='DJF')[
    site_latlon['ilat'], site_latlon['ilon'],].values
sm_jja_lat = pre_weighted_lat[expid[i]]['sm'].sel(season='JJA')[
    site_latlon['ilat'], site_latlon['ilon'],].values

sea_djf_lat = pre_weighted_lat[expid[i]]['sea'].sel(time=(
    pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2))[
        :, site_latlon['ilat'], site_latlon['ilon']].values
sea_jja_lat = pre_weighted_lat[expid[i]]['sea'].sel(time=(
    pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8))[
        :, site_latlon['ilat'], site_latlon['ilon']].values

print(np.round(sm_djf_lat, 1))
print(str(np.round(sea_djf_lat.mean(), 1)) + ' ± ' + \
    str(np.round(sea_djf_lat.std(ddof=1), 1)))

print(np.round(sm_jja_lat, 1))
print(str(np.round(sea_jja_lat.mean(), 1)) + ' ± ' + \
    str(np.round(sea_jja_lat.std(ddof=1), 1)))

sm_djf_lat - sm_jja_lat

'''
site_latlon['lat']
lat.values[site_latlon['ilat']]
site_latlon['lon']
lon.values[site_latlon['ilon']]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region site_latlon = {'lat':-70.4016, 'lon':-64.5230}

site_latlon = {'lat':-70.4016, 'lon':-64.5230}

site_latlon['ilat'], site_latlon['ilon'] = \
        find_ilat_ilon(site_latlon['lat'], site_latlon['lon'],
                       lat.values, lon.values)

wisoaprt_am_icores = wisoaprt_alltime[expid[i]]['am'][
    0, site_latlon['ilat'], site_latlon['ilon']].values * seconds_per_d * 365
wisoaprt_annstd_icores = (wisoaprt_alltime[expid[i]]['ann'][
    :, 0, site_latlon['ilat'], site_latlon['ilon']
    ].values * seconds_per_d * 365).std(ddof=1)
print(str(np.round(wisoaprt_am_icores, 1)) + ' ± ' + \
    str(np.round(wisoaprt_annstd_icores, 1)))
# 579.4 ± 73.7 mm /year

aprt_frc_am_icores = aprt_frc[expid[i]]['Otherocean']['am'][
    site_latlon['ilat'], site_latlon['ilon']].values
aprt_frc_annstd_icores = aprt_frc[expid[i]]['Otherocean']['ann'][
    :, site_latlon['ilat'], site_latlon['ilon']].values.std(ddof=1)
print(str(np.round(aprt_frc_am_icores, 1)) + ' ± ' + \
    str(np.round(aprt_frc_annstd_icores, 1)))
# 84.3 ± 1.2


lat_am_icores = pre_weighted_lat[expid[i]]['am'][
    site_latlon['ilat'], site_latlon['ilon']].values
lat_annstd_icores = pre_weighted_lat[expid[i]]['ann'][
    :, site_latlon['ilat'], site_latlon['ilon']].values.std(ddof=1)
print(str(np.round(lat_am_icores, 1)) + ' ± ' + \
    str(np.round(lat_annstd_icores, 1)))
# -41.2 ± 1.0


lon_am_icores = pre_weighted_lon[expid[i]]['am'][
    site_latlon['ilat'], site_latlon['ilon']].values
lon_annstd_icores = circstd(
    pre_weighted_lon[expid[i]]['ann'][
        :, site_latlon['ilat'], site_latlon['ilon']].values,
    high=360, low=0)
print(str(np.round(lon_am_icores, 1)) + ' ± ' + \
    str(np.round(lon_annstd_icores, 1)))
# 241.8 ± 3.0

'''
site_latlon['lat']
lat.values[site_latlon['ilat']]
site_latlon['lon']
lon.values[site_latlon['ilon']]
'''
# endregion
# -----------------------------------------------------------------------------



