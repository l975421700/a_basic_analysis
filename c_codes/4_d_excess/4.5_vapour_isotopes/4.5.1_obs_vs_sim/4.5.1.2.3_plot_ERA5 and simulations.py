

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i=0


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
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    ticks_labels,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
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
    plot_labels,
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

with open('scratch/ERA5/temp2/ERA5_temp2_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_temp2_2013_2022_alltime = pickle.load(f)

temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)



'''
# temp2_bias = temp2_alltime[expid[i]]['am'] - (ERA5_temp2_2013_2022_alltime['am'] - zerok)

temp2_alltime[expid[i]]['am'].to_netcdf('scratch/test/test0.nc')
(ERA5_temp2_2013_2022_alltime['am'] - zerok).to_netcdf('scratch/test/test1.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am temp2 and its diff

plt_data1 = temp2_alltime[expid[i]]['ann'][-10:].mean(dim='time')
plt_data2 = (ERA5_temp2_2013_2022_alltime['am'] - zerok).compute()
plt_data3 = regrid(plt_data1) - regrid(plt_data2)

# stats.describe(plt_data2.sel(latitude=slice(-20, -90)).values.flatten())


#-------- plot configuration
output_png = 'figures/test/trial.png'
cbar_label1 = '2 m temperature [$°C$]'
cbar_label2 = 'Differences: (a) - (b) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-28, cm_max=28, cm_interval1=2, cm_interval2=4, cmap='RdBu',
    asymmetric=True,)

pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 1
ncol = 3
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-20, ax_org = axs[jcol])
    # cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

plt1 = plot_t63_contourf(
    plt_data1.lon,
    plt_data1.lat,
    plt_data1,
    axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

axs[1].contourf(
    plt_data2.longitude,
    plt_data2.latitude,
    plt_data2,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[2].contourf(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    levels = pltlevel2, extend='both',
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'ERA5', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(a) - (b)', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(-0.2, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(1.1,-2.2),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check nudging: good

era5_forcing_files = sorted(glob.glob('/albedo/work/projects/paleo_work/paleodyn_from_work_ollie_projects/paleodyn/nudging/ERA5/atmos/T63/era5T63L47_*.nc'))
# era5_forcing_files[12:]

echam6_output_files = sorted(glob.glob('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/unknown/nudged_701_5.0_*.01_sp_1m.nc'))
# echam6_output_files[:-1]

for ifile in np.arange(0, len(era5_forcing_files[12:]), 5):
    # ifile = 24
    print('#-------------------------------- ' + str(ifile))
    # print(era5_forcing_files[12:][ifile])
    # print(echam6_output_files[:-1][ifile])
    
    era5_forcing = xr.open_dataset(era5_forcing_files[12:][ifile])
    echam6_output = xr.open_dataset(echam6_output_files[:-1][ifile])
    
    bias = echam6_output.st.sel(lev=47).squeeze() - \
        era5_forcing.t.sel(lev=47).mean(dim='time')
    
    print(np.max(abs(bias)))
    
    if (np.max(abs(bias)) > 1):
        print(np.max(abs(bias)))

# The temperature at the lowest model level is successfully nudged to ERA5.


'''
# t, svo, sd
era5_forcing = xr.open_dataset('/albedo/work/projects/paleo_work/paleodyn_from_work_ollie_projects/paleodyn/nudging/ERA5/atmos/T63/era5T63L47_201003.nc')

# st, svo, sd
echam6_output = xr.open_dataset('albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_701_5.0/unknown/nudged_701_5.0_201003.01_sp_1m.nc')

bias = era5_forcing['t'].sel(lev=47).mean(dim='time').values - echam6_output['st'].sel(lev=47)[0].values

print(np.mean(abs(bias)))
print(np.max(bias))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sm and am temp2 diff

temp2_alltime_13_22 = mon_sea_ann(
    var_monthly=temp2_alltime[expid[i]]['mon'][-120:,])

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am_sm temp2 differences.png'
cbar_label = 'Differences in temp2 between ECHAM6 nudged simulation and ERA5 [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=1, cm_interval2=1, cmap='RdBu',)

nrow = 1
ncol = 5
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    # cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- differences
plt_data = regrid(temp2_alltime_13_22['sm'].sel(season='DJF')) - regrid(ERA5_temp2_2013_2022_alltime['sm'].sel(season='DJF') - zerok)
plt1 = axs[0].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(temp2_alltime_13_22['sm'].sel(season='MAM')) - regrid(ERA5_temp2_2013_2022_alltime['sm'].sel(season='MAM') - zerok)
axs[1].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(temp2_alltime_13_22['sm'].sel(season='JJA')) - regrid(ERA5_temp2_2013_2022_alltime['sm'].sel(season='JJA') - zerok)
axs[2].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(temp2_alltime_13_22['sm'].sel(season='SON')) - regrid(ERA5_temp2_2013_2022_alltime['sm'].sel(season='SON') - zerok)
axs[3].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(temp2_alltime_13_22['am']) - regrid(ERA5_temp2_2013_2022_alltime['am'] - zerok)
axs[4].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'DJF', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[4].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(0.5, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar.ax.set_xlabel(cbar_label, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sm and am SST diff

tsw_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tsw_alltime.pkl', 'rb') as f:
    tsw_alltime[expid[i]] = pickle.load(f)

with open('scratch/ERA5/SST/ERA5_SST_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_SST_2013_2022_alltime = pickle.load(f)

tsw_alltime_13_22 = mon_sea_ann(
    var_monthly=tsw_alltime[expid[i]]['mon'][-120:,])

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am_sm sst differences.png'
cbar_label = 'Differences in SST between ECHAM6 nudged simulation and ERA5 [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-4, cm_max=4, cm_interval1=1, cm_interval2=1, cmap='RdBu',)

nrow = 1
ncol = 5
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- differences
plt_data = regrid(tsw_alltime_13_22['sm'].sel(season='DJF')) - regrid(ERA5_SST_2013_2022_alltime['sm'].sel(season='DJF') - zerok)
plt1 = axs[0].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(tsw_alltime_13_22['sm'].sel(season='MAM')) - regrid(ERA5_SST_2013_2022_alltime['sm'].sel(season='MAM') - zerok)
axs[1].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(tsw_alltime_13_22['sm'].sel(season='JJA')) - regrid(ERA5_SST_2013_2022_alltime['sm'].sel(season='JJA') - zerok)
axs[2].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(tsw_alltime_13_22['sm'].sel(season='SON')) - regrid(ERA5_SST_2013_2022_alltime['sm'].sel(season='SON') - zerok)
axs[3].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(tsw_alltime_13_22['am']) - regrid(ERA5_SST_2013_2022_alltime['am'] - zerok)
axs[4].contourf(
    plt_data.lon, plt_data.lat, plt_data,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'DJF', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[4].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(0.5, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar.ax.set_xlabel(cbar_label, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot sm and am SIC diff

seaice_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.seaice_alltime.pkl', 'rb') as f:
    seaice_alltime[expid[i]] = pickle.load(f)

with open('scratch/ERA5/SIC/ERA5_SIC_2013_2022_alltime.pkl', 'rb') as f:
    ERA5_SIC_2013_2022_alltime = pickle.load(f)

seaice_alltime_13_22 = mon_sea_ann(
    var_monthly=seaice_alltime[expid[i]]['mon'][-120:,])

output_png = 'figures/8_d-excess/8.3_vapour/8.3.1_sim/8.3.1.0_sim_era5/8.3.1.0.0 ECHAM6_ERA5 am_sm siconc differences.png'
cbar_label = 'Differences in SIC between ECHAM6 nudged simulation and ERA5 [$\%$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=4, cmap='RdBu',)

nrow = 1
ncol = 5
fm_bottom = 2.5 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    plt.text(
        0.05, 0.975, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- differences
plt_data = regrid(seaice_alltime_13_22['sm'].sel(season='DJF')) - regrid(ERA5_SIC_2013_2022_alltime['sm'].sel(season='DJF'))
plt1 = axs[0].contourf(
    plt_data.lon, plt_data.lat, plt_data * 100,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(seaice_alltime_13_22['sm'].sel(season='MAM')) - regrid(ERA5_SIC_2013_2022_alltime['sm'].sel(season='MAM'))
axs[1].contourf(
    plt_data.lon, plt_data.lat, plt_data * 100,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(seaice_alltime_13_22['sm'].sel(season='JJA')) - regrid(ERA5_SIC_2013_2022_alltime['sm'].sel(season='JJA'))
axs[2].contourf(
    plt_data.lon, plt_data.lat, plt_data * 100,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(seaice_alltime_13_22['sm'].sel(season='SON')) - regrid(ERA5_SIC_2013_2022_alltime['sm'].sel(season='SON'))
axs[3].contourf(
    plt_data.lon, plt_data.lat, plt_data * 100,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt_data = regrid(seaice_alltime_13_22['am']) - regrid(ERA5_SIC_2013_2022_alltime['am'])
axs[4].contourf(
    plt_data.lon, plt_data.lat, plt_data * 100,
    levels = pltlevel, extend='both',
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'DJF', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'MAM', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'SON', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[4].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,
    anchor=(0.5, 1), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar.ax.set_xlabel(cbar_label, linespacing=2)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------

