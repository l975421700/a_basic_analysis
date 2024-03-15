

# salloc --account=paleodyn.paleodyn --partition=mpp --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
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
import datetime

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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import multiprocessing

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
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.contributions2site_q_sfc.pkl', 'rb') as f:
    contributions2site_q_sfc = pickle.load(f)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Find the maximum point


par_corr_sources_isotopes_q_sfc={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)


#-------------------------------- Daily negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_min = np.min(daily_par_corr)
where_daily_min = np.where(daily_par_corr == daily_min)
# print(daily_min)
# print(daily_par_corr[where_daily_min[0][0], where_daily_min[1][0]])

daily_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lon.values
daily_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_min[0][0], where_daily_min[1][0]].lat.values

daily_min_ilon = np.where(lon == daily_min_lon)[0][0]
daily_min_ilat = np.where(lat == daily_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_min_ilat, daily_min_ilon])


#-------------------------------- Daily positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'daily'

daily_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
daily_max = np.max(daily_par_corr)
where_daily_max = np.where(daily_par_corr == daily_max)
# print(daily_max)
# print(daily_par_corr[where_daily_max[0][0], where_daily_max[1][0]])

daily_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lon.values
daily_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_daily_max[0][0], where_daily_max[1][0]].lat.values

daily_max_ilon = np.where(lon == daily_max_lon)[0][0]
daily_max_ilat = np.where(lat == daily_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[daily_max_ilat, daily_max_ilon])


#-------------------------------- Annual negative corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_min = np.min(annual_par_corr)
where_annual_min = np.where(annual_par_corr == annual_min)
# print(annual_min)
# print(annual_par_corr[where_annual_min[0][0], where_annual_min[1][0]])

annual_min_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lon.values
annual_min_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_min[0][0], where_annual_min[1][0]].lat.values

annual_min_ilon = np.where(lon == annual_min_lon)[0][0]
annual_min_ilat = np.where(lat == annual_min_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_min_ilat, annual_min_ilon])


#-------------------------------- Annual positive corr.
iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

annual_par_corr = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60)).values
annual_max = np.max(annual_par_corr)
where_annual_max = np.where(annual_par_corr == annual_max)
# print(annual_max)
# print(annual_par_corr[where_annual_max[0][0], where_annual_max[1][0]])

annual_max_lon = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lon.values
annual_max_lat = par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].sel(lat=slice(-20, -60))[where_annual_max[0][0], where_annual_max[1][0]].lat.values

annual_max_ilon = np.where(lon == annual_max_lon)[0][0]
annual_max_ilat = np.where(lat == annual_max_lat)[0][0]
# print(par_corr_sources_isotopes_q_sfc[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'].values[annual_max_ilat, annual_max_ilon])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source contributions to q_sfc

pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
    cm_min=0, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='Blues',
    reversed=False)

output_png = 'figures/8_d-excess/8.1_controls/8.1.5_correlation_analysis/8.1.5.7_sources_isotopes_q/8.1.5.7.0_negative correlation/8.1.5.7.0.4 ' + expid[i] + ' source contributions to q_sfc at daily_min_max.png'

fig, ax = hemisphere_plot(northextent=-10, figsize=np.array([5.8, 7]) / 2.54,)

cplot_ice_cores(daily_min_lon, daily_min_lat, ax, s=12)
cplot_ice_cores(daily_max_lon, daily_max_lat, ax, s=12)

#-------- daily_min
plt_data1 = (contributions2site_q_sfc['daily_min'] * 1000).compute()
plt_data1.values[plt_data1.values == 0] = np.nan

plt1 = ax.pcolormesh(
    plt_data1.lon, plt_data1.lat, plt_data1,
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)

#-------- daily_max
plt_data2 = (contributions2site_q_sfc['daily_max'] * 1000).compute()
plt_data2.values[plt_data2.values == 0] = np.nan

plt2 = ax.pcolormesh(
    plt_data2.lon, plt_data2.lat, plt_data2,
    norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks1, extend='max',
    pad=0.02, fraction=0.2,
    )

cbar.ax.set_xlabel('Contributions to surface vapour [‰]')
fig.savefig(output_png)


'''
stats.describe(contributions2site_q_sfc['daily_min'] * 1000, axis=None)

stats.describe(contributions2site_q_sfc['daily_max'] * 1000, axis=None)
'''
# endregion
# -----------------------------------------------------------------------------
