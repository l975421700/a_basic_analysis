

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.rcParams['figure.dpi'] = 600

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion
# =============================================================================


# =============================================================================
# region import output pi_final_qg_tag4_1y_0

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_tag4_1y_0', 'pi_final_qg_tag5_1y_0', 'pi_final_qg_tag5_1y_1',
    ]

awi_esm_o = {}

for i in range(len(expid)):
    # i=0
    
    awi_esm_o[expid[i]] = {}
    
    ## echam
    awi_esm_o[expid[i]]['echam'] = {}
    awi_esm_o[expid[i]]['echam']['echam'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.nc')
    awi_esm_o[expid[i]]['echam']['echam_am'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.am.nc')
    awi_esm_o[expid[i]]['echam']['echam_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.ann.nc')
    
    ## wiso
    awi_esm_o[expid[i]]['wiso'] = {}
    awi_esm_o[expid[i]]['wiso']['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.nc')
    awi_esm_o[expid[i]]['wiso']['wiso_am'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.am.nc')
    awi_esm_o[expid[i]]['wiso']['wiso_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.ann.nc')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region time series of precipitation

echam6_t63_slm_area = xr.open_dataset('/work/ollie/qigao001/output/scratch/others/land_sea_masks/ECHAM6_T63_slm_area.nc')

ts_echam_pre = {}
ts_wiso_pre = {}

for i in range(len(expid)):
    ts_echam_pre[expid[i]] = (((awi_esm_o[expid[i]]['echam']['echam'].aprl + awi_esm_o[expid[i]]['echam']['echam'].aprc) * echam6_t63_slm_area.cell_area.values[None, :, :]).sum(axis=(1,2)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30
    ts_wiso_pre[expid[i]] = (((awi_esm_o[expid[i]]['wiso']['wiso'].wisoaprl + awi_esm_o[expid[i]]['wiso']['wiso'].wisoaprc) * echam6_t63_slm_area.cell_area.values[None, None, :, :]).sum(axis=(2, 3)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30


(ts_echam_pre[expid[0]] == ts_echam_pre[expid[2]]).all()

i = 1
(ts_echam_pre[expid[i]] == ts_wiso_pre[expid[i]][:, 0]).all()

(ts_wiso_pre[expid[i]][:, 3] + ts_wiso_pre[expid[i]][:, 4] + ts_wiso_pre[expid[i]][:, 5] + ts_wiso_pre[expid[i]][:, 6]) - ts_echam_pre[expid[i]]
'''
'''


ts_echam_evp = ((awi_esm_o[expid[i]]['echam']['echam'].evap * echam6_t63_slm_area.cell_area.values[None, :, :]).sum(axis=(1,2)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30

ts_wiso_evp = ((awi_esm_o[expid[i]]['wiso']['wiso'].wisoevap * echam6_t63_slm_area.cell_area.values[None, None, :, :]).sum(axis=(2, 3)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30

np.mean(ts_echam_pre + ts_echam_evp)
np.mean(ts_echam_pre)
# plot of global precipitation

fig, ax = plt.subplots(1, 1, figsize=np.array([14, 8]) / 2.54)

plt_line1, = ax.plot(
    np.arange(1, 48.1, 1),
    ts_echam_pre,
    linewidth=1, color='black', linestyle='--', marker = 'o', markersize=3.5,)

plt_line2, = ax.plot(
    np.arange(1, 48.1, 1),
    (ts_wiso_pre[:, 3] + ts_wiso_pre[:, 4] + ts_wiso_pre[:, 5] + ts_wiso_pre[:, 6]),
    linewidth=1, color='grey', linestyle='--', marker = 'o', markersize=3.5,)

ax.legend([plt_line1, plt_line2],
          ['Normal precipitation', 'Sum from four tag regions'],
          ncol=2)

ax.set_ylim(80, 90)
ax.set_xlim(0, 49)
ax.set_xticks(np.arange(1, 48.1, 12))
ax.grid(True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
ax.set_xlabel('Months in four simulation years')
ax.set_ylabel('Area-weighted global mean\nmonthly precipitation [$mm/mon$]')

fig.tight_layout()

# fig.savefig('figures/0_test/trial.png')
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.0_pi_final_qg_tag4_1y_0/6.0.0.6.0.0_area-weighted global mean monthly precipitation.png')


'''
ts_wiso_pre[:, 0] == ts_echam_pre
ts_echam_evp == ts_wiso_evp[:, 0]

(ts_wiso_pre[:, 3] + ts_wiso_pre[:, 4] + ts_wiso_pre[:, 5] + ts_wiso_pre[:, 6]) - ts_echam_pre
(ts_wiso_evp[:, 3] + ts_wiso_evp[:, 4] + ts_wiso_evp[:, 5] + ts_wiso_evp[:, 6]) - ts_echam_evp
'''
# endregion
# =============================================================================


# =============================================================================
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')


# calculate the water tracer fraction
tag_evp_frac = {}

for i in range(len(expid)):
    # i=2
    tag_evp_frac[expid[i]] = {}

    for j in range(awi_esm_o[expid[i]]["wiso"]["wiso_ann"].wisoevap.shape[1]-3):
        # j=0
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[3, j+3, :, :].values) / (awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :].values) * 100
        
        # set maximum effective evaporation to be 1.93e-8 mm/s or 0.05 mm/mon
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :]) <= 1.93e-8
        ] = np.nan

'''
i = 2
stats.describe(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :]), axis=None, nan_policy = 'omit')
stats.describe(abs(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[3, 0+3, :, :]), axis=None, nan_policy='omit')
stats.describe(tag_evp_frac[expid[i]]['region_1'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_2'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_3'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_4'], axis=None, nan_policy = 'omit')

stats.describe((tag_evp_frac[expid[i]]['region_1'] + tag_evp_frac[expid[i]]['region_2'] + tag_evp_frac[expid[i]]['region_3'] + tag_evp_frac[expid[i]]['region_4']), axis=None, nan_policy = 'omit')

'''


pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 0


for j in range(awi_esm_o[expid[i]]["wiso"]["wiso_ann"].wisoevap.shape[1]-3):
    # j=0
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_o[expid[i]]['echam']['echam_ann'].lon,
        awi_esm_o[expid[i]]['echam']['echam_ann'].lat,
        tag_evp_frac[expid[i]]['region_' + str(j+1)],
        norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
    
    ax.contour(
        tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='black',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)
    
    cbar = fig.colorbar(
        plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
        fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
        ticks=pltticks, extend="neither",)
    
    cbar.ax.set_xlabel(
        u'Fraction of evaporation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.1_check_fraction/6.0.0.6.1.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')
    print(str(j))


i = 2

pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_o[expid[i]]['echam']['echam_ann'].lon,
    awi_esm_o[expid[i]]['echam']['echam_ann'].lat,
    (tag_evp_frac[expid[i]]['region_1'] + tag_evp_frac[expid[i]]['region_2'] + tag_evp_frac[expid[i]]['region_3'] + tag_evp_frac[expid[i]]['region_4']),
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of evaporation from all tag regions [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.1_check_fraction/6.0.0.6.1.' +
            str(i) + '.6_' + expid[i] + '_tag_evp_frac_all_regions.png')

# endregion
# =============================================================================

