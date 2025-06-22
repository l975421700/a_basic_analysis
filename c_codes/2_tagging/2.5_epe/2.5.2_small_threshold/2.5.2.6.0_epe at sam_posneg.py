
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_705_6.0',
    # 'nudged_703_6.0_k52',
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
from scipy.stats import circmean, circstd

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
    calc_lon_diff_np,
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

sam_daily = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_daily.nc')
sam_posneg_ind = {}
sam_posneg_ind['pos'] = sam_daily.sam > sam_daily.sam.std(ddof = 1)
sam_posneg_ind['neg'] = sam_daily.sam < (-1 * sam_daily.sam.std(ddof = 1))

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_sst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sst.pkl', 'rb') as f:
    pre_weighted_sst[expid[i]] = pickle.load(f)

pre_weighted_RHsst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_RHsst.pkl', 'rb') as f:
    pre_weighted_RHsst[expid[i]] = pickle.load(f)

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

d_ln_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    d_ln_alltime[expid[i]] = pickle.load(f)


'''
# pos_case_time = '2022-03-18'
pos_case_time = '2021-10-21'
neg_case_time = '2022-02-10'
case_time = pos_case_time

for case_time in [pos_case_time, neg_case_time]:
    print(case_time)
    print('SAM:             ' + str(sam_daily.sam.sel(time=case_time).values))
    print('Precipitation:   ' + str(wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1, lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], time=case_time, method='nearest').values * seconds_per_d))
    print('Source lat:      ' + str(pre_weighted_lat[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], time=case_time, method='nearest').values))
    print('Source lon:      ' + str(pre_weighted_lon[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], time=case_time, method='nearest').values))
    print('dD:               ' + str(dD_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], time=case_time, method='nearest').values))
    print('dln:               ' + str(d_ln_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], time=case_time, method='nearest').values))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get epe at sam_pos_neg

EDC_daily_pre = (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1, lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')  * seconds_per_d).compute()

# set a threshold of 0.002 mm/d
EDC_daily_pre = EDC_daily_pre.where(EDC_daily_pre>=0.002, other=np.nan).compute()

iepe = EDC_daily_pre >= EDC_daily_pre.quantile(0.9, dim='time', skipna=True).values

posidx = sam_posneg_ind['pos'] & iepe
negidx = sam_posneg_ind['neg'] & iepe
postime = iepe.time[posidx]
negtime = iepe.time[negidx]


'''
sam_daily.sam.sel(time=negtime)

iepe.time[sam_posneg_ind['pos'] & iepe][-20:]
iepe.time[sam_posneg_ind['neg'] & iepe][-20:]

(sam_posneg_ind['pos'] & iepe).sum()
(sam_posneg_ind['neg'] & iepe).sum()

np.isfinite(EDC_daily_pre).sum()
(EDC_daily_pre >= EDC_daily_pre.quantile(0.9, dim='time', skipna=True).values).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check histogram of wisoaprt/src_lat/deltaD at SAM+- epe

plt_data1 = d_ln_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = d_ln_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=5
binrange=(-10, 40)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))

fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms
# ax.hist(plt_data1, bins=30, alpha=0.5, label='Array 1', color='blue', edgecolor='black')
# ax.hist(plt_data2, bins=30, alpha=0.5, label='Array 2', color='orange', edgecolor='black')
sns.histplot(plt_data1, binwidth=binwidth, binrange=binrange, color='blue', alpha=0.5, label='Array 1', ax=ax, kde=False)
sns.histplot(plt_data2, binwidth=binwidth, binrange=binrange, color='orange', alpha=0.5, label='Array 2', ax=ax, kde=False)


# Customize the plot
ax.set_title('Histograms of Two Arrays', fontsize=16)
ax.set_xlabel('Value', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.legend(fontsize=12)

# Show the plot
fig.savefig('figures/trial.png')



'''

plt_data1 = wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1, lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx] * seconds_per_d
plt_data2 = wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1, lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx] * seconds_per_d
binwidth=0.1
binrange=(0.2,3.2)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))
print(str(np.round(np.mean(plt_data2), 2).values) + '±' + str(np.round(np.std(plt_data2, ddof=1), 2).values))
print(str(np.round(np.mean(plt_data1), 2).values) + '±' + str(np.round(np.std(plt_data1, ddof=1), 2).values))
print(str(np.round((np.mean(plt_data2) - np.mean(plt_data1)) / np.mean(plt_data1) * 100, 0).values))

plt_data1 = dD_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = dD_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=10
binrange=(-480, -200)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))
print(str(np.round(np.mean(plt_data2), 0).values) + '±' + str(np.round(np.std(plt_data2, ddof=1), 0).values))
print(str(np.round(np.mean(plt_data1), 0).values) + '±' + str(np.round(np.std(plt_data1, ddof=1), 0).values))


plt_data1 = d_ln_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = d_ln_alltime[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=10
binrange=(-480, -200)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))
print(str(np.round(np.mean(plt_data2), 1).values) + '±' + str(np.round(np.std(plt_data2, ddof=1), 1).values))
print(str(np.round(np.mean(plt_data1), 1).values) + '±' + str(np.round(np.std(plt_data1, ddof=1), 1).values))


plt_data1 = pre_weighted_lat[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = pre_weighted_lat[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=1
binrange=(-51,-25)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))
print(str(np.round(np.mean(plt_data2), 1).values) + '±' + str(np.round(np.std(plt_data2, ddof=1), 1).values))
print(str(np.round(np.mean(plt_data1), 1).values) + '±' + str(np.round(np.std(plt_data1, ddof=1), 1).values))
print(str(np.round((np.mean(plt_data2) - np.mean(plt_data1)) / np.mean(plt_data1) * 100, 0).values))


plt_data1 = pre_weighted_sst[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = pre_weighted_sst[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=1
binrange=(5,23)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))

plt_data1 = pre_weighted_RHsst[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = pre_weighted_RHsst[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=1
binrange=(5,23)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))


plt_data1 = pre_weighted_lon[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = pre_weighted_lon[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=10
binrange=(0,360)
def mean_longitude(longitudes):
    # Convert degrees to radians
    radians = np.radians(longitudes)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    # Compute the circular mean
    mean_angle = np.arctan2(mean_sin, mean_cos)
    return np.degrees(mean_angle)

print(mean_longitude(plt_data1))
print(mean_longitude(plt_data2))
print(str(int(circmean(plt_data2, high=360, low=0))) + '±' + str(int(circstd(plt_data2, high=360, low=0))))
print(str(int(circmean(plt_data1, high=360, low=0))) + '±' + str(int(circstd(plt_data1, high=360, low=0))))



plt_data1 = transport_distance[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[posidx]
plt_data2 = transport_distance[expid[i]]['daily'].sel(lat=t63_sites_indices['EDC']['lat'], lon=t63_sites_indices['EDC']['lon'], method='nearest')[negidx]
binwidth=1
binrange=(-51,-25)
print(stats.describe(plt_data1))
print(stats.describe(plt_data2))


# (270-204)/204 * 100 = 30% more EPE

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot SAM+- EPE property differences

var = 'd_ln'

if var=='pre_freq':
    pos_data = (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1)[posidx.values] >= 2e-8).sum(dim='time') / posidx.sum() * 100
    neg_data = (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1)[negidx.values] >= 2e-8).sum(dim='time') / negidx.sum() * 100
    diff_data= pos_data-neg_data
    cbar_label1 = 'Precipitation frequency [%]'
    cbar_label2 = 'Difference: (a)-(b) [%]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=50, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='viridis',
        reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=4, cmap='BrBG',
        reversed=False)
elif var=='pre':
    pos_data = wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1)[posidx.values].mean(dim='time') * seconds_per_d
    neg_data = wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1)[negidx.values].mean(dim='time') * seconds_per_d
    diff_data= (pos_data-neg_data)/neg_data * 100
    cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
    cbar_label2 = 'Difference: (a)/(b) - 1 [%]'
    pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
    pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
    pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=40, cmap='BrBG',
        reversed=False)
    extend = 'max'
    extend1 = 'neither'
elif var=='lat':
    pos_data = pre_weighted_lat[expid[i]]['daily'][posidx.values].mean(dim='time', skipna=True)
    neg_data = pre_weighted_lat[expid[i]]['daily'][negidx.values].mean(dim='time', skipna=True)
    diff_data= pos_data-neg_data
    cbar_label1 = 'Source latitude [$°\;S$]'
    cbar_label2 = 'Difference: (a) - (b) [$°$]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-46, cm_max=-36, cm_interval1=1, cm_interval2=2, cmap='viridis',
        reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='BrBG',
        reversed=False)
    extend = 'both'
    extend1 = 'both'
elif var=='lon':
    pos_data = calc_lon_diff_np(circmean(pre_weighted_lon[expid[i]]['daily'][posidx.values], high=360, low=0, axis=0, nan_policy='omit'), lon_2d)
    neg_data = calc_lon_diff_np(circmean(pre_weighted_lon[expid[i]]['daily'][negidx.values], high=360, low=0, axis=0, nan_policy='omit'), lon_2d)
    diff_data= calc_lon_diff_np(pos_data, neg_data)
    cbar_label1 = 'Relative source longitude [$°$]'
    cbar_label2 = 'Difference: (a) - (b) [$°$]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-180, cm_max=180, cm_interval1=30, cm_interval2=60,
        cmap='twilight_shifted',)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',
        reversed=False)
elif var=='dD':
    pos_data = dD_alltime[expid[i]]['daily'][posidx.values].mean(dim='time', skipna=True)
    neg_data = dD_alltime[expid[i]]['daily'][negidx.values].mean(dim='time', skipna=True)
    diff_data= pos_data-neg_data
    cbar_label1 = '$\delta D$ [‰]'
    cbar_label2 = 'Difference: (a) - (b) [‰]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-450, cm_max=-100, cm_interval1=25, cm_interval2=50,
        cmap='viridis', reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',
        reversed=False)
elif var=='d_ln':
    pos_data = d_ln_alltime[expid[i]]['daily'][posidx.values].mean(dim='time', skipna=True)
    neg_data = d_ln_alltime[expid[i]]['daily'][negidx.values].mean(dim='time', skipna=True)
    diff_data= pos_data-neg_data
    cbar_label1 = '$d_{ln}$ [‰]'
    cbar_label2 = 'Difference: (a) - (b) [‰]'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-4, cm_max=20, cm_interval1=2, cm_interval2=4,
        cmap='viridis', reversed=False)
    pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
        cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='BrBG',
        reversed=False)


# plot
opng=f'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.2_posnegSAM EPE {var} at EDC pattern over Antarctica.png'

nrow = 1
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.02, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
    cplot_ice_cores(t63_sites_indices['EDC']['lon'], t63_sites_indices['EDC']['lat'], axs[jcol])

plt1 = axs[0].pcolormesh(lon, lat, pos_data, norm=pltnorm, cmap=pltcmp,
                         transform=ccrs.PlateCarree(),)
plt2 = axs[1].pcolormesh(lon, lat, neg_data, norm=pltnorm, cmap=pltcmp,
                         transform=ccrs.PlateCarree(),)
plt3 = axs[2].pcolormesh(lon, lat, diff_data, norm=pltnorm1, cmap=pltcmp1,
                         transform=ccrs.PlateCarree(),)

for jcol in range(ncol):
    axs[jcol].add_feature(
        cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

plt.text(
    0.5, 1.05, '(a) HP during SAM+ at Dome C', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(b) HP during SAM- at Dome C', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, '(c) Difference between (a) and (b)', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40, extend=extend,
    anchor=(-0.2, 0.4), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1,)
if var=='lat':
    cbar1.ax.set_xticklabels(
        [remove_trailing_zero(x) for x in np.negative(pltticks)])

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40, extend=extend1,
    anchor=(1.1,-3.1),ticks=pltticks1)
cbar2.ax.set_xlabel(cbar_label2,)

fig.subplots_adjust(
    left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(opng)





'''
circmean(np.array([0, 90, 0]), high=360, low=0, nan_policy='omit')
circmean(np.array([120, 360]), high=360, low=0, nan_policy='omit')

mean_longitude(np.array([0, 90, 0]))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate precipitation during SAM+- EPE

icase = 'sam+'
if icase=='sam+':
    ind=posidx.values
elif icase=='sam-':
    ind=negidx.values

omp4 = f'figures/8_d-excess/8.1_controls/8.1.0_SAM/8.1.0.3_{icase} EPE at EDC pre over Antarctica.mp4'

pltlevel = np.array([0, 0.1, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.1, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('viridis', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-60)
cplot_ice_cores(t63_sites_indices['EDC']['lon'], t63_sites_indices['EDC']['lat'], ax)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2, format=remove_trailing_zero_pos)
cbar.ax.set_xlabel('Precipitation [$mm \; day^{-1}$]')

plt_objs = []

def update_frames(itime):
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    plt_data = wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1)[ind][itime] * seconds_per_d
    
    plt1 = ax.pcolormesh(lon, lat.sel(lat=slice(-60, -90)),
                         plt_data.sel(lat=slice(-60, -90)),
                         norm=pltnorm, cmap=pltcmp,
                         transform=ccrs.PlateCarree(),)
    plt_txt = plt.text(
        0, 0.95, str(plt_data.time.values)[:10], transform=ax.transAxes,
        ha='left', va='bottom', rotation='horizontal')
    
    plt_objs=[plt1, plt_txt]
    return(plt_objs)

ax.add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

ani = animation.FuncAnimation(
    fig, update_frames, frames=ind.sum(), interval=1000, blit=False)

ani.save(
    omp4,
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)

# endregion
# -----------------------------------------------------------------------------


