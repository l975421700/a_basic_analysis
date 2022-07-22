

# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import output pi_echam6_*

awi_esm_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    # 'pi_echam6_1d_127_3.48',
    # 'pi_echam6_1d_128_3.48',
    # 'pi_echam6_1d_129_3.48',
    # 'pi_echam6_1d_130_3.48',
    # 'pi_echam6_1d_132_3.48',
    'pi_echam6_1d_133_3.52',
    'pi_echam6_1d_134_3.52',
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

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagmap and SST bins

i = 0
expid[i]
i_ts=-1
i_level = 14

# tagmap, i_level-3 layer, last timestep
awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :]

# tsw, last timestep, tagmap == 1
np.max(awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values[awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :] == 1])
np.min(awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values[awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :] == 1])
# [289.155, 291.147]
# [289.15, 291.15]

################ plot tagmap, 14th layer, 191st timestep

pltlevel = np.arange(0, 1.001, 0.001)
pltticks = np.arange(0, 1.001, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]["wiso"].lon,
    awi_esm_org_o[expid[i]]["wiso"].lat,
    awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    u'Values of tag map [$-$] at layer ' + str(i_level-3) + ' and timestep ' + str(i_ts) + '\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0_' + expid[i] + '_layer' + str(i_level-3) + '_timestep' + str(i_ts) + '_tagmap_values.png')



################ plot tsw, 191st timestep

pltlevel = np.arange(-2, 30.01, 2)
pltticks = np.arange(-2, 30.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values - 273.15,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'tsw: surface temperature of water [$°C$] at timestep ' + str(i_ts) + '\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0_' + expid[i] + '_timestep' + str(i_ts) + '_tsw_values.png')


################ plot tsw, 191st timestep, tagmap == 1

tsw_tagmap1 = awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values - 273.15
tsw_tagmap1[awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :] != 1] = np.nan

pltlevel = np.arange(-2, 30.01, 2)
pltticks = np.arange(-2, 30.01, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('tab20c', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    tsw_tagmap1,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'tsw: surface temperature of water [$°C$] at timestep ' + str(i_ts) + ' where tagmap==1 for layer ' + str(i_level-3) + '\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0_' + expid[i] + '_layer' + str(i_level-3) + '_timestep' + str(i_ts) + '_tsw_values.png')


################ evaporation





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

# calculate the water tracer fraction
i = 0
expid[i]
i_ts=-1
i_level = 14

tag_evp_frac = {}

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = np.zeros((
        awi_esm_org_o[expid[i]]["wiso"].wisoevap.shape[1]-3,
        awi_esm_org_o[expid[i]]["wiso"].wisoevap.shape[2],
        awi_esm_org_o[expid[i]]["wiso"].wisoevap.shape[3],
    ))

    for j in range(tag_evp_frac[expid[i]].shape[0]):
        # j=0
        tag_evp_frac[expid[i]][j, :, :] = (awi_esm_org_o[expid[i]]['wiso'].wisoevap[i_ts, j+3, :, :].values) / (awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :].values) * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_evp_frac[expid[i]][j, :, :][
            abs(awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :]) <= 1.93e-8
        ] = np.nan

'''
i = 0
expid[i]
stats.describe(tag_evp_frac[expid[i]].sum(axis=0), axis=None, nan_policy='omit')
99.89331437, 100.13404826
'''

################ Tagged evaporation from all regions over normal evaporation

pltlevel = np.arange(99.8, 100.201, 0.001)
pltticks = np.arange(99.8, 100.201, 0.05)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]["wiso"].lon,
    awi_esm_org_o[expid[i]]["wiso"].lat,
    tag_evp_frac[expid[i]].sum(axis=0),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Tagged evaporation from all regions over normal evaporation [$\%$]\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.1_' + expid[i] + '_timestep' + str(i_ts) + '_fraction of evaporation from all tag regions.png')



################ Tagged evaporation from 14th regions over normal evaporation

pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


for j in range(tag_evp_frac[expid[i]].shape[0]):
    # j=i_level - 4
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]["wiso"].lon,
        awi_esm_org_o[expid[i]]["wiso"].lat,
        tag_evp_frac[expid[i]][j, :, :],
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    ax.contour(
        awi_esm_org_o[expid[i]]["wiso"].lon,
        awi_esm_org_o[expid[i]]["wiso"].lat,
        awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :],
        colors='black',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)
    
    cbar = fig.colorbar(
        plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
        fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
        ticks=pltticks, extend="neither",)
    
    cbar.ax.set_xlabel(
        u'Fraction of evaporation from the ' + str(i_level-3) + 'th tag region [$\%$]\necham-6.3.05p2-wiso: ' + expid[i],
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.1_' + expid[i] + '_timestep' + str(i_ts) + '_fraction of evaporation from ' + str(i_level-3) + 'th tag regions.png')
    print(str(j))







'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging precipitation fraction

# calculate the water tracer fraction
tag_pre_frac = {}
imonth = 119

for i in range(len(expid)):
    # i=3
    tag_pre_frac[expid[i]] = {}

    for j in range(awi_esm_org_o[expid[i]]['wiso'].wisoaprl.shape[1]-3):
        # j=4
        
        tag_pre_frac[expid[i]]['region_' + str(j+1)] = \
            ((awi_esm_org_o[expid[i]]['wiso'].wisoaprl[imonth, j+3, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[imonth, j+3, :, :]) / (awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :])).values * 100
        
        # 1.93e-8 mm/s -> 0.05 mm/mon
        tag_pre_frac[expid[i]]['region_' + str(j+1)][
            (awi_esm_org_o[expid[i]]['echam'].aprl[imonth, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[imonth, :, :]) <= 1.93e-8
        ] = np.nan


'''
i = 0
j = 4

stats.describe(tag_pre_frac[expid[i]]['region_' + str(j+1)], axis=None, nan_policy='omit')
# 99.98962 - 100.00129

stats.describe(tag_pre_frac[expid[i]]['region_1'], axis=None, nan_policy = 'omit')
stats.describe(tag_pre_frac[expid[i]]['region_2'], axis=None, nan_policy = 'omit')
stats.describe(tag_pre_frac[expid[i]]['region_3'], axis=None, nan_policy = 'omit')
stats.describe(tag_pre_frac[expid[i]]['region_4'], axis=None, nan_policy = 'omit')

stats.describe((tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']), axis=None, nan_policy = 'omit')
# 93.525375 - 144.75015
# -647.4625000000001 - 4475.015
'''


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

i = 0

for j in range(awi_esm_org_o[expid[i]]['wiso'].wisoaprl.shape[1]-3):
    # j=4
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]['echam'].lon,
        awi_esm_org_o[expid[i]]['echam'].lat,
        tag_pre_frac[expid[i]]['region_' + str(j+1)],
        norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
    
    
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
pltlevel = np.arange(-1, 1.001, 0.01)
pltticks = np.arange(-1, 1.001, 0.2)
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
    u'Deviation in fraction of precipitation from four tag regions from 1 [$0.1‰$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.15_' + expid[i] + '_tag_pre_frac_four_regions.png')


# plot the water tracer fraction
pltlevel = np.arange(-1, 1.001, 0.01)
pltticks = np.arange(-1, 1.001, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

i = 0
j = 4

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    tag_pre_frac[expid[i]]['region_' + str(j+1)],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Deviation in fraction of precipitation from the tag region from 1 [$0.1‰$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.16_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import yearly output pi_final_qg_tag5_1y_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_1y_1_qgtest2.1',
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

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check quantity conservation

i = 0
expid[i]
iyear = 9
iglobal = 7


#### check evaporation
# Absolute maximum difference between normal evp and tagged evp from whole globe: 3.0013325e-10
np.nanmax(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, iglobal, :, :]))
test = awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, iglobal, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')

# Absolute maximum difference between tagged evp from whole globe and from four regions: 8.519601e-08
np.nanmax(abs(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 3:7, :, :].sum(axis=0)))

test = awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[iyear, 3:7, :, :].sum(axis=0)
np.nanmax(abs(test.values))
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')


#### check precipitation
# Absolute maximum difference between normal and tagged pre from whole globe: 2.0904736e-08
np.nanmax(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, iglobal, :, :]))
test = awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, iglobal, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')

# Absolute maximum difference between tagged pre from whole globe and from four regions: 0.00020821579

np.nanmax(abs(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, iglobal, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 3:7, :, :].sum(axis=0) - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 3:7, :, :].sum(axis=0)))
test = awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, iglobal, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[iyear, iglobal, :, :] - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 3:7, :, :].sum(axis=0) - awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[iyear, 3:7, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')

# total precipitation
test = awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

tag_evp_frac = {}
iyear = 9

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
i = 0
j = 4
# 99.94175 to 100.05211
stats.describe(tag_evp_frac[expid[i]]['region_' + str(j+1)], axis=None, nan_policy='omit')

i = 0
# 55.02077 to 145.02907
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


i = 0


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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')
    print(str(j))


i = 0

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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' +
            str(i) + '.5_' + expid[i] + '_tag_evp_frac_all_regions.png')


i = 0
j=4

pltlevel = np.arange(99.95, 100.0501, 0.0001)
pltticks = np.arange(99.95, 100.06, 0.01)
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' +
            str(i) + '.6_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging precipitation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl_g.nc')

# calculate the water tracer fraction
tag_pre_frac = {}
iyear = 9

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
i = 0

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

i = 0

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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.1' + str(j) + '_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')
    print(str(j))



# plot the water tracer fraction
pltlevel = np.arange(95, 105.001, 0.01)
pltticks = np.arange(95, 105.001, 1)
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.15_' + expid[i] + '_tag_pre_frac_all_regions.png')


# plot the water tracer fraction
pltlevel = np.arange(99.95, 100.0501, 0.0001)
pltticks = np.arange(99.95, 100.06, 0.01)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

i = 0
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4_pi_final_qg_1y/6.0.0.6.4.' + str(i) + '.16_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check conservation of evaporation to precipitation

i = 0
expid[i]


AWI_ESM_T63_areacella = xr.open_dataset('/work/ollie/qigao001/output/scratch/others/land_sea_masks/areacella_fx_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn.nc')


iyear=3

total_evp = np.sum(AWI_ESM_T63_areacella.areacella.values * awi_esm_o[expid[i]]['echam']['echam_ann'].evap[iyear, :, :].values) / np.sum(AWI_ESM_T63_areacella.areacella.values)

total_pre = np.sum(AWI_ESM_T63_areacella.areacella.values * (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[iyear, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[iyear, :, :]).values) / np.sum(AWI_ESM_T63_areacella.areacella.values)

(abs(total_evp)/total_pre - 1) * 100

# 1.2 per mil deficit in the second year
# 2.4 per mil deficit in the second year
# 1.0 per mil deficit in the third year
# 1.9 per mil deficit in the fourth year


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import monthly output pi_final_qg_tag5_1m_*

# awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'
awi_esm_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    'pi_echam6_1d_132_3.48',
    'pi_echam6_1d_133_3.52',
    ]

awi_esm_org_o = {}

for i in range(len(expid)):
    # i=1
    
    awi_esm_org_o[expid[i]] = {}
    
    ## echam
    awi_esm_org_o[expid[i]]['echam'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    awi_esm_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


'''
'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check areas used for ECHAM6 T63

ECHAM6_T63_slm_area = xr.open_dataset('/home/users/qino/bas_palaeoclim_qino/others/land_sea_masks/ECHAM6_T63_slm_area.nc')

AWI_ESM_T63_areacella = xr.open_dataset('/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/fx/areacella/gn/v20200212/areacella_fx_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn.nc')

(ECHAM6_T63_slm_area.cell_area.values == AWI_ESM_T63_areacella.areacella.values).all()

ECHAM6_T63_slm_area.cell_area.values.sum()
AWI_ESM_T63_areacella.areacella.values.sum()


tagmap_ls_0 = xr.open_dataset('/work/ollie/qigao001/startdump/tagging/tagmap/tagmap_ls_0.nc')
output = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_33_3.21/analysis/echam/pi_echam6_1d_33_3.21.01_wiso.nc')
output1 = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_33_3.21/analysis/echam/pi_echam6_1d_33_3.21.01_echam.nc')

output1.tsw.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')

(output1.tsw[0, :, :].values[np.where(output.tagmap[-1, 3, :, :] - tagmap_ls_0.tagmap[3, :, :])] == 273.15).sum()


tagmap_ls_0.tagmap[3, :, :].sum()
output.tagmap[-1, 3, :, :].sum()
test = output.tagmap[-1, 3, :, :] - tagmap_ls_0.tagmap[3, :, :]
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


pltlevel = np.arange(0, 1.001, 0.001)
pltticks = np.arange(0, 1.001, 0.2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    test.lon, test.lat, test,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    u'Values of tagmap [$-$]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('/work/ollie/qigao001/0_backup/trial.png')



output = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_024_3.34/outdata/echam/pi_echam6_1d_024_3.34_200001.01_wiso', engine='cfgrib')

output1 = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_020_3.31/outdata/echam/pi_echam6_1d_020_3.31_200001.01_wiso', engine='cfgrib', decode_cf=True)
output1['p90.128']

m=6
n=9
np.max(output['p90.128'][:, m:n, :, :].sum(axis=1))
np.min(output['p90.128'][:, m:n, :, :].sum(axis=1))
# 1e-8
np.where( output['p90.128'][:, m:n, :, :].sum(axis=1) != 1)
i = 0
j = 0
k = 1
output['p90.128'][i, 6, j, k].values
output['p90.128'][i, 7, j, k].values
output['p90.128'][i, 8, j, k].values

output.data_vars

f = open('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_020_3.31/outdata/echam/pi_echam6_1d_020_3.31_200001.01_wiso', 'rb')

import cfgrib
ds = cfgrib.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1d_020_3.31/outdata/echam/pi_echam6_1d_020_3.31_200001.01_wiso')
ds['p90.128']

# all others passed

# endregion
# -----------------------------------------------------------------------------





