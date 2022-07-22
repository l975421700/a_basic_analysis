

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
# region import output: pi_final_qg_tag4

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = ['pi_final_qg_tag4', 'pi_final_qg_tag4_1y']

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging precipitation fraction


tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap4.nc')


# calculate the water tracer fraction
nregion = 3
tag_w_frac = {}

for i in range(len(expid)):
    # i=1
    tag_w_frac[expid[i]] = {}
    
    for j in range(nregion):
        # j=2
        tag_w_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprl[0, j+4, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprc[0, j+4, :, :]) / (awi_esm_o[expid[i]]['echam']['echam_am'].aprl[0, :, :] + awi_esm_o[expid[i]]['echam']['echam_am'].aprc[0, :, :]) * 100


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('viridis', len(pltlevel)).reversed()


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.0_pi_final_qg_tag4_tag_w_frac_region_3.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4_1y',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.1_pi_final_qg_tag4_1y_tag_w_frac_region_3.png')


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lon,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'].lat,
    tag_w_frac['pi_final_qg_tag4_1y']['region_3'] - tag_w_frac['pi_final_qg_tag4']['region_3'],
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
ax.contour(
    tagmap.lon, tagmap.lat, tagmap.tagmap[6, :, :], colors='black',
    levels=np.array([0.5]), linewidths=1, linestyles='dashed',)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="neither",)

cbar.ax.set_xlabel(
    'Fraction of precipitation from land surface in the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4 - pi_final_qg_tag4_1y',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2_pi_final_qg_tag4-pi_final_qg_tag4_1y_tag_w_frac_region_3.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output pi_final_qg_tag4_1y_nhsh_sl

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = ['pi_final_qg_tag4_1y_nhsh_sl']

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
# region check tagging precipitation fraction


tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')


# calculate the water tracer fraction
nregion = 4
tag_w_frac = {}

for i in range(len(expid)):
    # i=0
    tag_w_frac[expid[i]] = {}

    for j in range(nregion):
        # j=2
        tag_w_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprl[0, j+3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoaprc[0, j+3, :, :]
             ) / (awi_esm_o[expid[i]]['echam']['echam_am'].aprl[0, :, :] + awi_esm_o[expid[i]]['echam']['echam_am'].aprc[0, :, :]) * 100


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))


i = 0
for j in range(nregion):
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        tag_w_frac[expid[i]]['region_' + str(j+1)].lon,
        tag_w_frac[expid[i]]['region_' + str(j+1)].lat,
        tag_w_frac[expid[i]]['region_' + str(j+1)],
        norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)
    
    ax.contour(
        tagmap.lon, tagmap.lat, tagmap.tagmap[j+3, :, :], colors='black',
        levels=np.array([0.5]), linewidths=0.5, linestyles='solid',)
    
    cbar = fig.colorbar(
        plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
        fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
        ticks=pltticks, extend="neither",)
    
    cbar.ax.set_xlabel(
        u'Fraction of precipitation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4_1y_nhsh_sl',
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3.' +
                str(j) + '_pi_final_qg_tag4_tag_w_frac_region_' + str(j+1) + '.png')

stats.describe((tag_w_frac[expid[i]]['region_1'] + tag_w_frac[expid[i]]['region_2'] + tag_w_frac[expid[i]]['region_3'] + tag_w_frac[expid[i]]['region_4']).values, axis=None)


# plot the water tracer fraction
pltlevel = np.arange(40, 160.001, 0.01)
pltticks = np.arange(40, 160.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tag_w_frac[expid[i]]['region_3'].lon,
    tag_w_frac[expid[i]]['region_3'].lat,
    (tag_w_frac[expid[i]]['region_1'] + tag_w_frac[expid[i]]['region_2'] +
     tag_w_frac[expid[i]]['region_3'] + tag_w_frac[expid[i]]['region_4']).values,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from all tag regions [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4_1y_nhsh_sl',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3.4_pi_final_qg_tag4_tag_w_frac_allregion.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')


# calculate the water tracer fraction
nregion = 4
tag_evp_frac = {}

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = {}

    for j in range(nregion):
        # j=2
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_am'].wisoevap[0, j+3, :, :]) / \
            (awi_esm_o[expid[i]]['echam']['echam_am'].evap[0, :, :]) * 100


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 0
for j in range(nregion):
    # j=0
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        tag_evp_frac[expid[i]]['region_' + str(j+1)].lon,
        tag_evp_frac[expid[i]]['region_' + str(j+1)].lat,
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
        u'Fraction of evaporation from the tag region [$\%$]\nAWI-ESM-2-1-wiso: pi_final_qg_tag4_1y_nhsh_sl',
        linespacing=1.5)
    
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4.' +
                str(j) + '_pi_final_qg_tag4_tag_evp_frac_region_' + str(j+1) + '.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output pi_final_qg_tag1_1m_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_tag1_1m_0_org',
    'pi_final_qg_tag1_1m_1_tagging_l',
    ]

awi_esm_org_o = {}

for i in range(len(expid)):
    # i=0
    
    awi_esm_org_o[expid[i]] = {}
    
    ## echam
    awi_esm_org_o[expid[i]]['echam'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    awi_esm_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


'''
awi_esm_org_o['pi_final_qg_tag4_org_nhsh_sl_2']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

tagmap = xr.open_dataset(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nh_s.nc')


# calculate the water tracer fraction
nregion = 1
tag_evp_frac = {}

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = {}

    for j in range(nregion):
        # j=0
        # tag_evp_frac[expid[i]]['region_' + str(j+1)] = np.zeros(
        #     awi_esm_org_o[expid[i]]['wiso'].wisoevap[0, j+3, :, :].shape
        # )
        
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_org_o[expid[i]]['wiso'].wisoevap[0, j+3, :, :].values) / \
            (awi_esm_org_o[expid[i]]['echam'].evap[0, :, :].values) * 100
        
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            awi_esm_org_o[expid[i]]['echam'].evap[0, :, :] >= 0
        ] = 0


# stats.describe(
#     tag_evp_frac[expid[i]]['region_' + str(j+1)], axis=None,
#     nan_policy='omit')
# stats.describe(
#     awi_esm_org_o[expid[i]]['wiso'].wisoevap[0, j+3, :, :].values, axis=None,
#     nan_policy='omit')
# stats.describe(
#     awi_esm_org_o[expid[i]]['echam'].evap[0, :, :].values, axis=None,
#     nan_policy='omit')
# np.sum(awi_esm_org_o[expid[i]]['echam'].evap[0, :, :] >= 0)
# np.sum(awi_esm_org_o[expid[i]]['wiso'].wisoevap[0, j+3, :, :].values >= 0)
# np.nanmin(tag_evp_frac[expid[i]]['region_' + str(j+1)].values)

# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 1

# nrow = 2
# ncol = 2

# fig, axs = plt.subplots(
#     nrow, ncol, figsize=np.array([8.8*2*ncol, 11*nrow + 2]) / 2.54,
#     subplot_kw={'projection': ccrs.PlateCarree()},
#     gridspec_kw={'hspace': 0.12, 'wspace': 0.4},
# )

# for i in range(nrow):
#     for j in range(ncol):
#         axs[i, j] = framework_plot1(northextent=-60, ax_org=axs[i, j])


for j in range(nregion):
    # j=0
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]['echam'].lon,
        awi_esm_org_o[expid[i]]['echam'].lat,
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.1.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output pi_final_qg_tag4_1m_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_tag4_1m_0',
    ]

awi_esm_org_o = {}

for i in range(len(expid)):
    # i=2
    
    awi_esm_org_o[expid[i]] = {}
    
    ## echam
    awi_esm_org_o[expid[i]]['echam'] = xr.open_dataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    ## wiso
    awi_esm_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


'''
awi_esm_org_o['pi_final_qg_tag4_org_nhsh_sl_2']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')


# calculate the water tracer fraction
nregion = 4
tag_evp_frac = {}

for i in range(len(expid)):
    # i=3
    tag_evp_frac[expid[i]] = {}

    for j in range(nregion):
        # j=0
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_org_o[expid[i]]['wiso'].wisoevap[7, j+3, :, :].values) / \
            (awi_esm_org_o[expid[i]]['echam'].evap[7, :, :].values) * 100
        
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            awi_esm_org_o[expid[i]]['echam'].evap[7, :, :] >= 0
        ] = 0


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 0

# nrow = 2
# ncol = 2

# fig, axs = plt.subplots(
#     nrow, ncol, figsize=np.array([8.8*2*ncol, 11*nrow + 2]) / 2.54,
#     subplot_kw={'projection': ccrs.PlateCarree()},
#     gridspec_kw={'hspace': 0.12, 'wspace': 0.4},
# )

# for i in range(nrow):
#     for j in range(ncol):
#         axs[i, j] = framework_plot1(northextent=-60, ax_org=axs[i, j])


for j in range(nregion):
    # j=0
    fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)
    
    plt_cmp = ax.pcolormesh(
        awi_esm_org_o[expid[i]]['echam'].lon,
        awi_esm_org_o[expid[i]]['echam'].lat,
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.2.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging precipitation fraction

tagmap = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')

# calculate the water tracer fraction
nregion = 4
tag_pre_frac = {}

for i in range(len(expid)):
    # i=3
    tag_pre_frac[expid[i]] = {}

    for j in range(nregion):
        # j=0
        
        tag_pre_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_org_o[expid[i]]['wiso'].wisoaprl[7, j+3, :, :] + awi_esm_org_o[expid[i]]['wiso'].wisoaprc[7, j+3, :, :]
             ) / (awi_esm_org_o[expid[i]]['echam'].aprl[7, :, :] + awi_esm_org_o[expid[i]]['echam'].aprc[7, :, :]).values * 100


# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

i = 0

for j in range(nregion):
    # j=0
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')

stats.describe((tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']).values, axis=None)
stats.describe(tag_pre_frac[expid[i]]['region_4'].values, axis=None)


# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))


fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    awi_esm_org_o[expid[i]]['echam'].lon,
    awi_esm_org_o[expid[i]]['echam'].lat,
    (tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']).values,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Fraction of precipitation from all tag regions [$\%$]\nAWI-ESM-2-1-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.3.' + str(i) + '.4_' + expid[i] + '_tag_pre_frac_all_regions.png')

'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output pi_final_qg_tag4_1y_*

awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'

expid = [
    'pi_final_qg_tag4_1y_0',
    ]

awi_esm_o = {}

for i in range(len(expid)):
    # i=0
    
    awi_esm_o[expid[i]] = {}
    
    ## echam
    awi_esm_o[expid[i]]['echam'] = {}
    # awi_esm_o[expid[i]]['echam']['echam'] = xr.open_mfdataset(
    #     awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.nc')
    # awi_esm_o[expid[i]]['echam']['echam_am'] = xr.open_mfdataset(
    #     awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.am.nc')
    awi_esm_o[expid[i]]['echam']['echam_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_echam.ann.nc')
    
    ## wiso
    awi_esm_o[expid[i]]['wiso'] = {}
    # awi_esm_o[expid[i]]['wiso']['wiso'] = xr.open_mfdataset(
    #     awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.nc')
    # awi_esm_o[expid[i]]['wiso']['wiso_am'] = xr.open_mfdataset(
    #     awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.am.nc')
    awi_esm_o[expid[i]]['wiso']['wiso_ann'] = xr.open_mfdataset(
        awi_esm_odir + expid[i] + '/analysis/echam/' + expid[i] + '_*.01_wiso.ann.nc')


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging evaporation fraction

tagmap = xr.open_dataset('startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')


# calculate the water tracer fraction
nregion = 4
tag_evp_frac = {}

for i in range(len(expid)):
    # i=0
    tag_evp_frac[expid[i]] = {}

    for j in range(nregion):
        # j=0
        tag_evp_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[3, j+3, :, :].values) / (awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :].values) * 100
        
        # set maximum effective evaporation to be 1.93e-8 mm/s or 0.05 mm/mon
        tag_evp_frac[expid[i]]['region_' + str(j+1)][
            awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :] >= -1.93e-8
        ] = np.nan


'''
stats.describe((tag_evp_frac[expid[i]]['region_1'] + tag_evp_frac[expid[i]]['region_2'] + tag_evp_frac[expid[i]]['region_3'] + tag_evp_frac[expid[i]]['region_4']), axis=None, nan_policy = 'omit')


stats.describe(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :]), axis=None, nan_policy = 'omit')
stats.describe(abs(awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoevap[3, 0+3, :, :]), axis=None, nan_policy='omit')
stats.describe(tag_evp_frac[expid[i]]['region_1'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_2'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_3'], axis=None, nan_policy = 'omit')
stats.describe(tag_evp_frac[expid[i]]['region_4'], axis=None, nan_policy = 'omit')



'''
# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Greens', len(pltlevel))


i = 0


for j in range(nregion):
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_evp_frac_region_' + str(j+1) + '.png')


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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.4.' +
            str(i) + '.4_' + expid[i] + '_tag_evp_frac_all_regions.png')


'''
stats.describe(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :].values, axis=None)
stats.describe(abs(awi_esm_o[expid[i]]['echam']['echam_ann'].evap[3, :, :]).values, axis=None)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check tagging precipitation fraction

tagmap = xr.open_dataset(
    '/work/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_sl.nc')

# calculate the water tracer fraction
nregion = 4
tag_pre_frac = {}

for i in range(len(expid)):
    # i=3
    tag_pre_frac[expid[i]] = {}
    
    for j in range(nregion):
        # j=0
        
        tag_pre_frac[expid[i]]['region_' + str(j+1)] = \
            (awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[3, j+3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[3, j+3, :, :]
             ).values / (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[3, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[3, :, :]).values * 100
        
        # set the value of fraction to be missing values when precipitation is less than 0.5 mm/mon
        tag_pre_frac[expid[i]]['region_' + str(j+1)][
            (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[3, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[3, :, :]).values < 1.93e-8
        ] = np.nan

j=0
stats.describe((awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprl[3, j+3, :, :] + awi_esm_o[expid[i]]['wiso']['wiso_ann'].wisoaprc[3, j+3, :, :]).values, axis=None)
stats.describe((awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[3, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[3, :, :]).values, axis=None)

stats.describe(tag_pre_frac[expid[i]]['region_1'], axis=None, nan_policy='omit')
stats.describe(tag_pre_frac[expid[i]]['region_2'], axis=None, nan_policy='omit')
stats.describe(tag_pre_frac[expid[i]]['region_3'], axis=None, nan_policy='omit')
stats.describe(tag_pre_frac[expid[i]]['region_4'], axis=None, nan_policy='omit')
stats.describe((tag_pre_frac[expid[i]]['region_1'] + tag_pre_frac[expid[i]]['region_2'] + tag_pre_frac[expid[i]]['region_3'] + tag_pre_frac[expid[i]]['region_4']), axis=None, nan_policy='omit')

# plot the water tracer fraction
pltlevel = np.arange(0, 100.001, 0.5)
pltticks = np.arange(0, 100.001, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

i = 0

for j in range(nregion):
    # j=0
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
    fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.5.' + str(i) + '.' +
                str(j) + '_' + expid[i] + '_tag_pre_frac_region_' + str(j+1) + '.png')


# plot the water tracer fraction
pltlevel = np.arange(80, 120.001, 0.01)
pltticks = np.arange(80, 120.001, 5)
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
fig.savefig('figures/6_awi/6.0_awi-esm-2.1-wiso/6.0.0_pi_final/6.0.0.6_tagging_exp/6.0.0.6.5.' +
            str(i) + '.4_' + expid[i] + '_tag_pre_frac_all_regions.png')

'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check conservation of evaporation to precipitation

echam6_t63_slm_area = xr.open_dataset('/work/ollie/qigao001/output/scratch/others/land_sea_masks/ECHAM6_T63_slm_area.nc')

np.sum(echam6_t63_slm_area.cell_area.values * awi_esm_o[expid[i]]['echam']['echam_ann'].evap[0, :, :].values) / np.sum(echam6_t63_slm_area.cell_area.values)

np.sum(echam6_t63_slm_area.cell_area.values * (awi_esm_o[expid[i]]['echam']['echam_ann'].aprl[0, :, :] + awi_esm_o[expid[i]]['echam']['echam_ann'].aprc[0, :, :]).values) / np.sum(echam6_t63_slm_area.cell_area.values)


# 1.2 per mil deficit in the second year
# 2.4 per mil deficit in the second year
# 1.0 per mil deficit in the third year
# 1.9 per mil deficit in the fourth year


# endregion
# -----------------------------------------------------------------------------


