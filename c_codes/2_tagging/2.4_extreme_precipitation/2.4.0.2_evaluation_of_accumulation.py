

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
    regrid,
    mean_over_ais,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]

i = 0
expid[i]
wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

acc_recon_ERAI = xr.open_dataset('data_sources/products/Antarctic_Accumulation_Reconstructions/acc_recon_ERAI.nc')

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am accumulation


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.2_accumulation/' + '6.1.4.2 ' + expid[i] + ' accumulation am Antarctica.png'
cbar_label1 = 'Annual mean accumulation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in annual mean accumulation [$\%$]'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

pltlevel2 = np.arange(-100, 100 + 1e-4, 20)
pltticks2 = np.arange(-100, 100 + 1e-4, 20)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

plt_data1 = (wisoaprt_alltime[expid[i]]['am'][0] + wisoevap_alltime[expid[i]]['am'][0]).compute() * 3600 * 24
plt_data1.values[echam6_t63_ais_mask['mask']['AIS'] == False] = np.nan
plt_data2 = acc_recon_ERAI.recon_acc_bc.mean(dim='years') / 365
plt_data3 = (regrid((wisoaprt_alltime[expid[i]]['am'][0] + wisoevap_alltime[expid[i]]['am'][0]) * 3600 * 24) / regrid(acc_recon_ERAI.recon_acc_bc.mean(dim='years') / 365) - 1).compute() * 100

plt_std1 = ((wisoaprt_alltime[expid[i]]['ann'][:, 0] + wisoevap_alltime[expid[i]]['ann'][:, 0]).std(dim = 'time') * 3600 * 24 / plt_data1 * 100).compute()
plt_std2 = (acc_recon_ERAI.recon_acc_bc.std(dim='years') / 365 / plt_data2 * 100).compute()
# stats.describe(plt_std1, axis=None, nan_policy='omit')
# stats.describe(plt_std2, axis=None, nan_policy='omit')

ctr_level = np.arange(0, 40 + 1e-4, 5)

nrow = 1
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

# ECHAM6
plt1 = axs[0].pcolormesh(
    plt_data1.lon,
    plt_data1.lat,
    plt_data1,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    plt_data2.lon,
    plt_data2.lat,
    plt_data2,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# #-------- differences
plt2 = axs[2].pcolormesh(
    plt_data3.lon,
    plt_data3.lat,
    plt_data3,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt_ctr1 = axs[0].contour(
    plt_std1.lon,
    plt_std1.lat,
    plt_std1,
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[0].clabel(
    plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=5,)

plt_ctr2 = axs[1].contour(
    plt_std2.lon,
    plt_std2.lat,
    plt_std2,
    levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
axs[1].clabel(
    plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
    levels=ctr_level, inline_spacing=10, fontsize=5,)

plt.text(
    0.5, 1.05, 'ECHAM6', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Reconstruction_Medley', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'ECHAM6/Reconstruction_Medley - 1', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.05, 0.975, '(a)', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.05, 0.975, '(b)', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.05, 0.975, '(c)', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')


cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, 0.4), ticks=pltticks, format=remove_trailing_zero_pos, )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.3),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann ERA5 pre

tp_era5_79_14 = xr.open_dataset('scratch/cmip6/hist/pre/tp_ERA5_mon_sl_197901_201412.nc')

# change units to mm/d
tp_era5_79_14_alltime = mon_sea_ann(var_monthly = (tp_era5_79_14.tp * 1000).compute())

# tp_era5_79_14_alltime['mm']

with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'wb') as f:
    pickle.dump(tp_era5_79_14_alltime, f)



'''
sh_pre_level = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
sh_pre_ticks = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01,300)))
sh_pre_norm = BoundaryNorm(sh_pre_level, ncolors=len(sh_pre_level))
sh_pre_cmp = cm.get_cmap('RdBu', len(sh_pre_level))


fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([8.8, 9.8]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.04, fm_top=0.99,
    )

plt_cmp = ax.pcolormesh(
    tp_era5_79_14_alltime['am'].longitude,
    tp_era5_79_14_alltime['am'].latitude,
    tp_era5_79_14_alltime['am'] * 365,
    norm=sh_pre_norm, cmap=sh_pre_cmp, transform=ccrs.PlateCarree(),
)
plt_pre_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_pre_norm, cmap=sh_pre_cmp), ax=ax,
    orientation="horizontal",pad=0.02,shrink=0.9,aspect=40,extend='max',
    # anchor=(-0.2, -0.35), fraction=0.12, panchor=(0.5, 0),
    ticks=sh_pre_ticks)
plt_pre_cbar.ax.set_xlabel('Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014', linespacing=1.5)

fig.savefig('figures/trial.png',)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate monthly pre over AIS

with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'rb') as f:
    tp_era5_79_14_alltime = pickle.load(f)

wisoaprt_mean_over_ais = {}
wisoaprt_mean_over_ais[expid[i]] = {}

for ialltime in wisoaprt_alltime[expid[i]].keys():
    # ialltime = 'mm'
    if (ialltime != 'am'):
        wisoaprt_mean_over_ais[expid[i]][ialltime] = mean_over_ais(
            wisoaprt_alltime[expid[i]][ialltime][:, 0],
            echam6_t63_ais_mask['mask']['AIS'],
            echam6_t63_cellarea.cell_area.values,
            )
    else:
        # ialltime = 'am'
        wisoaprt_mean_over_ais[expid[i]][ialltime] = mean_over_ais(
            wisoaprt_alltime[expid[i]][ialltime][0].expand_dims(dim='time'),
            echam6_t63_ais_mask['mask']['AIS'],
            echam6_t63_cellarea.cell_area.values,
            )
    print(ialltime)

wisoaprt_mean_over_ais[expid[i]]['mon_std'] = wisoaprt_mean_over_ais[expid[i]]['mon'].groupby('time.month').std(skipna=True).compute()
wisoaprt_mean_over_ais[expid[i]]['ann_std'] = wisoaprt_mean_over_ais[expid[i]]['ann'].std(skipna=True).compute()

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl',
          'wb') as f:
    pickle.dump(wisoaprt_mean_over_ais[expid[i]], f)

tp_era5_mean_over_ais = {}

for ialltime in tp_era5_79_14_alltime.keys():
    # ialltime = 'mm'
    if (ialltime != 'am'):
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            tp_era5_79_14_alltime[ialltime],
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    else:
        # ialltime = 'am'
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            tp_era5_79_14_alltime[ialltime].expand_dims(dim='time'),
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    print(ialltime)

tp_era5_mean_over_ais['mon_std'] = tp_era5_mean_over_ais['mon'].groupby('time.month').std(skipna=True).compute()
tp_era5_mean_over_ais['ann_std'] = tp_era5_mean_over_ais['ann'].std(skipna=True).compute()

with open('scratch/cmip6/hist/pre/tp_era5_mean_over_ais.pkl', 'wb') as f:
    pickle.dump(tp_era5_mean_over_ais, f)


'''
#-------------------------------- check
wisoaprt_mean_over_ais = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl', 'rb') as f:
    wisoaprt_mean_over_ais[expid[i]] = pickle.load(f)

with open('scratch/cmip6/hist/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)
with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'rb') as f:
    tp_era5_79_14_alltime = pickle.load(f)

itime = 9
np.average(
    wisoaprt_alltime[expid[i]]['mm'][itime, 0].values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['mm'][itime].values
np.average(
    tp_era5_79_14_alltime['mm'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mm'][itime].values

itime = 30
np.average(
    wisoaprt_alltime[expid[i]]['mon'][itime, 0].values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['mon'][itime].values
np.average(
    tp_era5_79_14_alltime['mon'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mon'][itime].values


np.average(
    wisoaprt_alltime[expid[i]]['am'][0].values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['am'].values
np.average(
    tp_era5_79_14_alltime['am'].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['am'].values


(wisoaprt_mean_over_ais[expid[i]]['mm'] == mean_over_ais(
    wisoaprt_alltime[expid[i]]['mm'][:, 0],
    echam6_t63_ais_mask['mask']['AIS'],
    echam6_t63_cellarea.cell_area.values
    )).all().values

(wisoaprt_mean_over_ais[expid[i]]['mon'] == mean_over_ais(
    wisoaprt_alltime[expid[i]]['mon'][:, 0],
    echam6_t63_ais_mask['mask']['AIS'],
    echam6_t63_cellarea.cell_area.values,
    )).all().values


(tp_era5_mean_over_ais['mm'] == mean_over_ais(
    tp_era5_79_14_alltime['mm'],
    era5_ais_mask['mask']['AIS'],
    era5_cellarea.cell_area.values,
    )).all().values
(tp_era5_mean_over_ais['mon'] == mean_over_ais(
    tp_era5_79_14_alltime['mon'],
    era5_ais_mask['mask']['AIS'],
    era5_cellarea.cell_area.values,
    )).all().values


(wisoaprt_mean_over_ais[expid[i]]['mm'] * 3600 * 24 * month_days).sum()
(tp_era5_mean_over_ais['mm'] * month_days).sum()

wisoaprt_mean_over_ais[expid[i]]['am'].values * 3600 * 24 * 365
tp_era5_mean_over_ais['am'].values * 365

wisoaprt_mean_over_ais[expid[i]]['ann_std'] * 3600 * 24 * 365
tp_era5_mean_over_ais['ann_std'] * 365

# ECHAM6:   160.4 ± 6.3 mm/yr,
# ERA5:     173.6 ± 8.7 mm/yr
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot monthly pre over AIS

i = 0
expid[i]
wisoaprt_mean_over_ais = {}
wisoaprt_mean_over_ais[expid[i]] = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl', 'rb') as f:
    wisoaprt_mean_over_ais[expid[i]] = pickle.load(f)

with open('scratch/cmip6/hist/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)


pre_mm_over_ais = pd.DataFrame(columns=(
    'Data', 'Month', 'pre_mon',))

pre_mm_over_ais = pd.concat(
    [pre_mm_over_ais,
     pd.DataFrame(data={
         'Data': 'ECHAM6',
         'Month': np.tile(month, int(len(wisoaprt_mean_over_ais[expid[i]]['mon']) / 12)),
         'pre_mon': wisoaprt_mean_over_ais[expid[i]]['mon'].values * 3600 * 24 * np.tile(month_days, int(len(wisoaprt_mean_over_ais[expid[i]]['mon']) / 12)),
         })],
        ignore_index=True,)

pre_mm_over_ais = pd.concat(
    [pre_mm_over_ais,
     pd.DataFrame(data={
         'Data': 'ERA5',
         'Month': np.tile(month, int(len(tp_era5_mean_over_ais['mon']) / 12)),
         'pre_mon': tp_era5_mean_over_ais['mon'].values * np.tile(month_days, int(len(tp_era5_mean_over_ais['mon']) / 12)),
         })],
        ignore_index=True,)

output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/6.1.4.1 ' + expid[i] + ' mm precipitation histogram over AIS.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

sns.barplot(
    data = pre_mm_over_ais,
    x = 'Month',
    y = 'pre_mon',
    hue = 'Data', hue_order = ['ERA5', 'ECHAM6'],
    palette=['tab:blue', 'tab:orange',],
    ci = 'sd', errwidth=0.75, capsize=0.1,
)
plt.legend(loc='upper right', handlelength=1, framealpha = 0.5, )

ax.set_ylabel('Monthly precipitation over AIS [$mm \; mon^{-1}$]')

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99)

fig.savefig(output_png)





'''
(wisoaprt_mean_over_ais[expid[i]]['mm'] * 3600 * 24 * month_days).sum()
(tp_era5_mean_over_ais['mm'] * month_days).sum()

# np.max(abs(wisoaprt_mean_over_ais[expid[i]]['mm'] - wisoaprt_mean_over_ais[expid[i]]['mon'].groupby('time.month').mean(skipna=True).compute()))

        #  'Month': month,
        #  'pre_mm': wisoaprt_mean_over_ais[expid[i]]['mm'] * 3600 * 24 * month_days,
        #  'pre_std': wisoaprt_mean_over_ais[expid[i]]['mon_std'] * 3600 * 24 * month_days,

        #  'Month': month,
        #  'pre_mm': tp_era5_mean_over_ais['mm'] * month_days,
        #  'pre_std': tp_era5_mean_over_ais['mon_std'] * month_days,

    # y = 'pre_mm',

'''
# endregion
# -----------------------------------------------------------------------------
