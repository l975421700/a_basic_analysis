

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
# =============================================================================
# region import monthly output pi_final_qg_tag5_1y_*

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    # 'pi_final_qg_tag5_1y_0_tag',
    # 'pi_final_qg_tag5_1y_1_mwtest1',
    # 'pi_final_qg_tag5_1y_2_mwtest2',
    # 'pi_final_qg_tag5_1y_3_qgtest1',
    # 'pi_final_qg_tag5_1y_4_qgtest2_notag',
    # 'pi_final_qg_tag5_1y_5_qgtest2',
    'pi_final_qg_tag5_1y_6_qgtest3',
    # 'pi_final_qg_tag5_1m_0_41code',
    # 'pi_final_qg_tag5_1m_5_43code',
    # 'pi_final_qg_tag5_1m_7_45code',
    ]

awi_esm_org_o = {}

for i in range(len(expid)):
    # i=0
    
    awi_esm_org_o[expid[i]] = {}
    
    ## echam
    awi_esm_org_o[expid[i]]['echam'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    awi_esm_org_o[expid[i]]['wiso'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


'''
i = 0
expid[i]
'''
# endregion
# =============================================================================


# =============================================================================
# region check tagging evaporation fraction

# calculate the water tracer fraction
tag_evp_frac = {}
imonth = 41

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = {}

    for j in range(awi_esm_org_o[expid[i]]["wiso"].wisoevap.shape[1]-3):
        # j=0
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = (awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, j+3, :, :].values) / (awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :].values) * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            abs(awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :]) <= 1.93e-8
        ] = np.nan


'''
i = 0
expid[i]
imonth = 11


# Absolute difference between normal evp and tagged evp from whole globe: less than 1.4551915e-11
np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :].values - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 7, :, :].values))

# Absolute difference between tagged evp from whole globe and from four regions: less than 6.346454e-08
np.nanmax(abs((awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :].values) - (awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 3, :, :].values + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 4, :, :].values + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 5, :, :].values + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 6, :, :].values)))


test = awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 7, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')

test = awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 7, :, :] - (awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 3, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 4, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 5, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 6, :, :])
# test.values[tagmap.tagmap[3, :, :] == 0] = np.nan
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')



i = 6
j = 4
stats.describe(tag_evp_frac[expid[i]]['region_' + str(j+1)], axis=None, nan_policy='omit')

i = 5

stats.describe(tag_evp_frac[expid[i]]['region_1'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_2'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_3'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_4'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_5'], axis=None, nan_policy = 'omit')

stats.describe((tag_evp_frac[expid[i]]['region_1'] + tag_evp_frac[expid[i]]['region_2'] + tag_evp_frac[expid[i]]['region_3'] + tag_evp_frac[expid[i]]['region_4']), axis=None, nan_policy = 'omit')

'''

pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 5


for j in range(awi_esm_org_o[expid[i]]["wiso"].wisoevap.shape[1]-3):
    # j=4
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]["wiso"].lon,
        awi_esm_org_o[expid[i]]["wiso"].lat,
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')
    print(str(j))


i = 5

pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]["wiso"].lon,
    awi_esm_org_o[expid[i]]["wiso"].lat,
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' +
            str(i) + '.5_' + expid[i] + '_tag_evp_frac_all_regions.png')


i = 5
j=4

pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]["wiso"].lon,
    awi_esm_org_o[expid[i]]["wiso"].lat,
    tag_evp_frac[expid[i]]['region_' + str(j+1)],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

ax.contour(
        tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='black',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of evaporation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' +
            str(i) + '.6_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region check tagging precipitation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

# calculate the water tracer fraction
tag_pre_frac = {}
imonth = 11

for i in range(len(expid)):
    # i=3
    tag_pre_frac[expid[i]] = {}

    for j in range(awi_esm_org_o[expid[i]]['wiso'].wisoaprl.shape[1]-3):
        # j=4
        
        tag_pre_frac[expid[i]]['region_' + str(j+1)] = \
            ((awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, j+3, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, j+3, :, :]
             ) / (awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :])).values * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_pre_frac[expid[i]]['region_' + str(j+1)][
            (awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :]) <= 1.93e-8
        ] = np.nan


'''
i = 5

# check difference between normal pre and tag pre from whole globe
# 2.3283064e-10
np.nanmax(abs((awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :]).values - (awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, 7, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, 7, :, :]).values))

# check difference between tag pre from whole globe and from four regions
# 9.747249e-08
np.mean(abs((awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :]).values - ((awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, 3, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, 3, :, :]).values + (awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, 4, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, 4, :, :]).values + (awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, 5, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, 5, :, :]).values + (awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, 6, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, 6, :, :]).values)))

stats.describe(tag_pre_frac[expid[i]]['region_' + str(4+1)], axis=None, nan_policy='omit')
99.98962 - 100.00129
'''


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

i = 5

for j in range(awi_esm_org_o[expid[i]]['wiso'].wisoaprl.shape[1]-3):
    # j=4
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]['echam'].lon,
        awi_esm_org_o[expid[i]]['echam'].lat,
        tag_pre_frac[expid[i]]['region_' + str(j+1)],
        norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
    
    ax.contour(
        tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='red',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)
    
    cbar = fig.colorbar(
        plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
        fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
        ticks=pltticks, extend="neither",)
    
    cbar.ax.set_xlabel(
        u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' + str(i) + '.1' + str(j) + '_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')
    print(str(j))


'''
stats.describe((tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']), axis=None, nan_policy='omit')
stats.describe(tag_pre_frac[expid[i]]['region_5'], axis=None, nan_policy='omit')
'''


# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from all tag regions [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' + str(i) + '.15_' + expid[i] + '_tag_pre_frac_all_regions.png')


# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

i = 5
j = 4

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    tag_pre_frac[expid[i]]['region_' + str(j+1)],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='red',
    levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag5_1y/6.0.0.6.2.' + str(i) + '.16_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')

'''
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import output pi_final_qg_tag5_1y_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_tag5_1y_3_qgtest1',
    'pi_final_qg_tag5_1y_4_qgtest2_notag',
    'pi_final_qg_tag5_1y_5_qgtest2',
    ]

awi_esm_o = {}

for i in range(len(expid)):
    # i=1
    
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

# endregion
# =============================================================================


# =============================================================================
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

tag_evp_frac = {}
iyear = 3

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = {}

    for j in range(awi_esm_o[expid[i]]["wiso"]["wiso_ann"].wisoevap.shape[1]-3):
        # j=0
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, j+3, :, :].values) / (awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :].values) * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :]) <= 1.93e-8
        ] = np.nan

'''
i = 2
iyear = 3
j = 4
# Absolute difference between normal evp and tagged evp from whole globe: less than 8.306955e-11
np.nanmax(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :].values - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 7, :, :].values))

# Absolute difference between tagged evp from whole globe and from four regions: less than 5.153015e-08
np.nanmax(abs(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 7, :, :].values - (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 5, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 6, :, :]).values))

test = awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 7, :, :] - (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 5, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 6, :, :])
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')


i = 2
j = 4
# 99.8477 to 100.26391
stats.describe(tag_evp_frac[expid[i]]['region_' + str(j+1)], axis=None, nan_policy='omit')

i = 2
# 20.720228 to 168.33777
stats.describe((tag_evp_frac[expid[i]]['region_1'] + tag_evp_frac[expid[i]]['region_2'] + tag_evp_frac[expid[i]]['region_3'] + tag_evp_frac[expid[i]]['region_4']), axis=None, nan_policy = 'omit')

stats.describe(tag_evp_frac[expid[i]]['region_1'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_2'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_3'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_4'], axis=None, nan_policy = 'omit')

'''

pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 2


for j in range(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap.shape[1]-3):
    # j=4
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' + str(i) + '.' +
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' +
            str(i) + '.5_' + expid[i] + '_tag_evp_frac_all_regions.png')


i = 2
j=4

pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel))


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
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of evaporation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' +
            str(i) + '.6_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')


# endregion
# =============================================================================


# =============================================================================
# region check tagging precipitation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

# calculate the water tracer fraction
tag_pre_frac = {}
iyear = 3

for i in range(len(expid)):
    # i=3
    tag_pre_frac[expid[i]] = {}

    for j in range(awi_esm_o[expid[i]]["wiso"]["wiso_ann"].wisoaprl.shape[1]-3):
        # j=4
        
        tag_pre_frac[expid[i]]['region_' + str(j+1)] = \
            ((awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, j+3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, j+3, :, :]) / (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :])).values * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_pre_frac[expid[i]]['region_' + str(j+1)][
            (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :]) <= 1.93e-8
        ] = np.nan


'''
i = 2

# check difference between normal pre and tag pre from whole globe
# 1.0913936e-09
np.nanmax(abs((awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :]).values - (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 7, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, 7, :, :]).values))

# check difference between tag pre from whole globe and from four regions
# 4.854984e-08
np.mean(abs((awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :]).values - (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, 3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, 4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 5, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, 5, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 6, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, 6, :, :]).values))

99.99666 - 100.00106
stats.describe(tag_pre_frac[expid[i]]['region_' + str(4+1)], axis=None, nan_policy='omit')

96.716774 - 105.52339
stats.describe((tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']), axis=None, nan_policy='omit')

'''

# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

i = 2

for j in range(awi_esm_o[expid[i]]["wiso"]["wiso_ann"].wisoaprl.shape[1]-3):
    # j=4
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_o[expid[i]]['echam']['echam_ann'].lon,
        awi_esm_o[expid[i]]['echam']['echam_ann'].lat,
        tag_pre_frac[expid[i]]['region_' + str(j+1)],
        norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
    
    ax.contour(
        tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='red',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)
    
    cbar = fig.colorbar(
        plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
        fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
        ticks=pltticks, extend="neither",)
    
    cbar.ax.set_xlabel(
        u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' + str(i) + '.1' + str(j) + '_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')
    print(str(j))



# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_o[expid[i]]['echam']['echam_ann'].lon,
    awi_esm_o[expid[i]]['echam']['echam_ann'].lat,
    tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from all tag regions [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' + str(i) + '.15_' + expid[i] + '_tag_pre_frac_all_regions.png')


# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

i = 2
j = 4

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    awi_esm_o[expid[i]]['echam']['echam_ann'].lon,
    awi_esm_o[expid[i]]['echam']['echam_ann'].lat,
    tag_pre_frac[expid[i]]['region_' + str(j+1)],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='red',
    levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3_pi_final_qg_tag5_1y_ann/6.0.0.6.3.' + str(i) + '.16_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')

# endregion
# =============================================================================


# =============================================================================
# region time series of pre, evp

echam6_t63_slm_area = xr.open_dataset('output/scratch/others/land_sea_masks/ECHAM6_T63_slm_area.nc')

gm_ts_mon = {}

for i in range(len(expid)):
    # i = 0
    
    gm_ts_mon[expid[i]] = {}
    
    gm_ts_mon[expid[i]]['echam'] = {}
    
    gm_ts_mon[expid[i]]['wiso'] = {}
    
    gm_ts_mon[expid[i]]['echam']['pre'] = (((awi_esm_o[expid[i]]['echam']['echam'].aprl + awi_esm_o[expid[i]]['echam']['echam'].aprc) * echam6_t63_slm_area.cell_area.values[None, :, :]).sum(axis=(1,2)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30
    
    gm_ts_mon[expid[i]]['wiso']['pre'] = (((awi_esm_o[expid[i]]['wiso']['wiso'].wisoaprl + awi_esm_o[expid[i]]['wiso']['wiso'].wisoaprc) * echam6_t63_slm_area.cell_area.values[None, None, :, :]).sum(axis=(2, 3)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30
    
    gm_ts_mon[expid[i]]['echam']['evp'] = ((awi_esm_o[expid[i]]['echam']['echam'].evap * echam6_t63_slm_area.cell_area.values[None, :, :]).sum(axis=(1,2)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30
    
    gm_ts_mon[expid[i]]['wiso']['evp'] = ((awi_esm_o[expid[i]]['wiso']['wiso'].wisoevap * echam6_t63_slm_area.cell_area.values[None, None, :, :]).sum(axis=(2, 3)) / echam6_t63_slm_area.cell_area.sum()).values * 86400 * 30


'''
(gm_ts_mon[expid[i]]['echam']['pre'] == gm_ts_mon[expid[i]]['wiso']['pre'][:, 0]).all()

# Difference between normal pre and tagged pre from the whole globe
# 0.0003818014172622952 mm/mon
np.nanmax(abs(gm_ts_mon[expid[i]]['echam']['pre'] - gm_ts_mon[expid[i]]['wiso']['pre'][:, 7])[3:])

# Difference between tagged pre from the whole globe and from four regions
# 0.0642279611319907 mm/mon
np.nanmax(abs(gm_ts_mon[expid[i]]['echam']['pre'] - (gm_ts_mon[expid[i]]['wiso']['pre'][:, 3] + gm_ts_mon[expid[i]]['wiso']['pre'][:, 4] + gm_ts_mon[expid[i]]['wiso']['pre'][:, 5] + gm_ts_mon[expid[i]]['wiso']['pre'][:, 6]))[3:])


(gm_ts_mon[expid[i]]['echam']['evp'] == gm_ts_mon[expid[i]]['wiso']['evp'][:, 0]).all()

# Difference between normal evp and tagged evp from the whole globe
# 3.689154510766457e-06 mm/mon
np.nanmax(abs(gm_ts_mon[expid[i]]['echam']['evp'] - gm_ts_mon[expid[i]]['wiso']['evp'][:, 7])[3:])

# Difference between tagged evp from the whole globe and from four regions
# 0.0007166950370702807 mm/mon
np.nanmax(abs(gm_ts_mon[expid[i]]['echam']['evp'] - (gm_ts_mon[expid[i]]['wiso']['evp'][:, 3] + gm_ts_mon[expid[i]]['wiso']['evp'][:, 4] + gm_ts_mon[expid[i]]['wiso']['evp'][:, 5] + gm_ts_mon[expid[i]]['wiso']['evp'][:, 6]))[3:])


np.mean(gm_ts_mon[expid[i]]['echam']['pre'] + gm_ts_mon[expid[i]]['echam']['evp'])
np.mean(ts_echam_pre)



'''
# why land region so much precipitation
# endregion
# =============================================================================


# =============================================================================
# region check original and modified simulations

i = 0
j = 2

(awi_esm_o[expid[i]]['echam']['echam'].evap == awi_esm_o[expid[j]]['echam']['echam'].evap).values.all()
(awi_esm_o[expid[i]]['echam']['echam_am'].evap == awi_esm_o[expid[j]]['echam']['echam_am'].evap).values.all()
(awi_esm_o[expid[i]]['echam']['echam_ann'].evap == awi_esm_o[expid[j]]['echam']['echam_ann'].evap).values.all()

(awi_esm_o[expid[i]]['wiso']['wiso'].wisoevap[:, 0:3, :, :].values == awi_esm_o[expid[j]]['wiso']['wiso'].wisoevap[:, 0:3, :, :].values).all()

kwiso = 0
np.max(abs(awi_esm_o[expid[i]]['wiso']['wiso'].wisoevap[:, kwiso, :, :].values - awi_esm_o[expid[j]]['wiso']['wiso'].wisoevap[:, kwiso, :, :].values))

imonth = 0
(awi_esm_o[expid[i]]['wiso']['wiso'].wisoevap[imonth, 0:3, :, :].values == awi_esm_o[expid[j]]['wiso']['wiso'].wisoevap[imonth, 0:3, :, :].values).all()

awi_esm_o[expid[i]]['wiso']['wiso_am']
awi_esm_o[expid[i]]['wiso']['wiso_ann']


# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import monthly output pi_final_qg_tag5_1m_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    # 'pi_final_qg_tag5_1m_0_41code',
    # 'pi_final_qg_tag5_1m_1_41code',
    # 'pi_final_qg_tag5_1m_2_41code_notag',
    # 'pi_final_qg_tag5_1m_3_wisocode_notag',
    # 'pi_final_qg_tag5_1m_4_wisocode_tag',
    # 'pi_final_qg_tag5_1m_5_43code',
    # 'pi_final_qg_tag5_1m_6_43code_notag',
    # 'pi_final_qg_tag5_1m_7_45code',
    # 'pi_final_qg_tag5_1m_8_45code_notag',
    'pi_final_qg_tag5_1y_5_qgtest2',
    'pi_final_qg_tag5_1y_6_qgtest3',
    'pi_final_qg_tag5_1y_7_qgtest5',
    ]

awi_esm_org_o = {}

for i in range(len(expid)):
    # i=7
    
    awi_esm_org_o[expid[i]] = {}
    
    ## echam
    awi_esm_org_o[expid[i]]['echam'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    awi_esm_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


'''
i = 7
expid[i]
awi_esm_org_o[expid[i]]

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')
stats.describe(tagmap.tagmap[7, :, :].values, axis=None)
stats.describe(tagmap.tagmap[3, :, :].values + tagmap.tagmap[4, :, :].values + tagmap.tagmap[5, :, :].values + tagmap.tagmap[6, :, :].values, axis=None)

(tagmap.tagmap[7, :, :].values == (tagmap.tagmap[3, :, :].values + tagmap.tagmap[4, :, :].values + tagmap.tagmap[5, :, :].values + tagmap.tagmap[6, :, :].values)).all()

'''
# endregion
# =============================================================================


# =============================================================================
# region check bit reproducibility

i = 1
j = 2

(awi_esm_org_o[expid[i]]['echam'].evap.values == awi_esm_org_o[expid[j]]['echam'].evap.values).all()
(awi_esm_org_o[expid[i]]['echam'].aprl.values == awi_esm_org_o[expid[j]]['echam'].aprl.values).all()
(awi_esm_org_o[expid[i]]['echam'].temp2.values == awi_esm_org_o[expid[j]]['echam'].temp2.values).all()
(awi_esm_org_o[expid[i]]['echam'].u10.values == awi_esm_org_o[expid[j]]['echam'].u10.values).all()
(awi_esm_org_o[expid[i]]['echam'].q2m.values == awi_esm_org_o[expid[j]]['echam'].q2m.values).all()

(awi_esm_org_o[expid[i]]['wiso'].wisoevap[0:12, :, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, :, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisoaprl[0:12, :, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoaprl[:, :, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisows[0:12, :, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisows[:, :, :, :].values).all()

(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :].values).all()

# (awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, :, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, :, :, :].values).all()



np.mean(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :].values - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :].values))

test = awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')


'''
'pi_final_qg_tag5_1m_0_41code' == 'pi_final_qg_tag5_1m_1_41code'
# Yes, it should be bit reproducible

'pi_final_qg_tag5_1m_2_41code_notag' == 'pi_final_qg_tag5_1m_3_wisocode_notag'
# If ntag = 0, tagging branch is the same as the wiso branch

'pi_final_qg_tag5_1m_0_41code' != 'pi_final_qg_tag5_1m_2_41code_notag'
# If ntag > 0, water isotope simulations are different: No!

'pi_final_qg_tag5_1m_3_wisocode_notag' != 'pi_final_qg_tag5_1m_4_wisocode_tag'
# Xiaoxu's codes already make the model bit unreproducible
# the difference is mainly around the coast

'pi_final_qg_tag5_1m_0_41code' != 'pi_final_qg_tag5_1m_4_wisocode_tag'
# The difference is mainly around the coast

'pi_final_qg_tag5_1m_0_41code' == 'pi_final_qg_tag5_1m_5_43code'
'pi_final_qg_tag5_1m_2_41code_notag' == 'pi_final_qg_tag5_1m_6_43code_notag'
# No difference between 41code and 43code

'pi_final_qg_tag5_1m_4_wisocode_tag' != 'pi_final_qg_tag5_1m_5_43code'

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check areas used for ECHAM6 T63

ECHAM6_T63_slm_area = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm_area.nc')

AWI_ESM_T63_areacella = xr.open_dataset('/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/fx/areacella/gn/v20200212/areacella_fx_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn.nc')

(ECHAM6_T63_slm_area.cell_area.values == AWI_ESM_T63_areacella.areacella.values).all()

ECHAM6_T63_slm_area.cell_area.values.sum()
AWI_ESM_T63_areacella.areacella.values.sum()

# endregion
# =============================================================================


