

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
# region import output pi_echam6_*

awi_esm_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    'pi_echam6_1m_3.4_15',
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
# =============================================================================


# =============================================================================
# region check quantity conservation

i = 0
expid[i]
i_ts=-1
np.max(awi_esm_org_o[expid[i]]['wiso'].tagmap[:, 3:, :, :].sum(axis=1))
np.min(awi_esm_org_o[expid[i]]['wiso'].tagmap[:, 3:, :, :].sum(axis=1))

np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[i_ts, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[i_ts, 3:, :, :].sum(axis=0)))



'''
    'TAG01',
    'TAG02',
    'pi_echam6_1m_3.0_8',
    'pi_echam6_1m_3.0_9',
    'pi_echam6_1m_3.0_10',
    'pi_echam6_1m_3.0_11',
    # 'pi_echam6_1m_3.0_12',
    'pi_echam6_1m_3.3_13',
    'pi_echam6_1y_3.3_2',
    
    'pi_echam6_1m_3.4_14',
    'pi_echam6_1m_3.4_15',
    'pi_echam6_1y_3.3_3',

i = 0
j = 1
expid[i] + ' : ' + expid[j]
iday = -1

# normal and tagged evp from nh/sh
# 'TAG01': 1.4551915228366852e-11 (1.2979247306551163e-12)
np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[iday, 3:5, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[iday, 3:5, :, :].sum(axis=0)))
test = awi_esm_org_o[expid[i]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[iday, 3:5, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/0_backup/test7.nc')

# # 'TAG02': 2.1827872842550278e-11 (2.65585084788553e-12)
np.nanmax(abs(awi_esm_org_o[expid[j]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[iday, 3:7, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[j]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[iday, 3:7, :, :].sum(axis=0)))
test = awi_esm_org_o[expid[j]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[iday, 3:7, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/0_backup/test8.nc')

k = 2
expid[k]
iday=-1
# 'pi_echam6_1m_3.0_8': 1.4551915228366852e-11 (1.429769882735046e-12)
np.nanmax(abs(awi_esm_org_o[expid[k]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[k]]['wiso'].wisoevap[iday, 3:5, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[k]]['echam'].evap[iday, :, :] - awi_esm_org_o[expid[k]]['wiso'].wisoevap[iday, 3:5, :, :].sum(axis=0)))


m = 3
expid[m]
imonth=-1
# 'pi_echam6_1m_3.0_9': 1.4551915228366852e-11 (3.516791796048185e-12)
np.nanmax(abs(awi_esm_org_o[expid[m]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[m]]['wiso'].wisoevap[imonth, 3:5, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[m]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[m]]['wiso'].wisoevap[imonth, 3:5, :, :].sum(axis=0)))

n = 4
expid[n]
imonth=-1
# 'pi_echam6_1m_3.0_10': 1.4379636370520643e-11 (2.960530794522301e-12)
np.nanmax(abs(awi_esm_org_o[expid[n]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[n]]['wiso'].wisoevap[imonth, 3:5, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[n]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[n]]['wiso'].wisoevap[imonth, 3:5, :, :].sum(axis=0)))

o = 5
expid[o]
imonth=-1
# 'pi_echam6_1m_3.0_11': 2.1827872842550278e-11 (4.0741730976555524e-12)
np.nanmax(abs(awi_esm_org_o[expid[o]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[o]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[o]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[o]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

p = 6
expid[p]
imonth=-1
# 'pi_echam6_1m_3.3_13': 1.0186340659856796e-10 (1.1922709733072325e-11)
np.nanmax(abs(awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[p]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[p]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

np.nanmax(abs(awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[p]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)) / awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :])
np.nanmean(abs(awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[p]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)) / awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :])
test = abs(awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[p]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)) / awi_esm_org_o[expid[p]]['echam'].evap[imonth, :, :]
test.to_netcdf('/work/ollie/qigao001/0_backup/test.nc')


q = 7
expid[q]
imonth=-1
# 'pi_echam6_1y_3.3_2': 1.4551915228366852e-11 (3.5440292675856552e-12)
np.nanmax(abs(awi_esm_org_o[expid[q]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[q]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[q]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[q]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

r = 8
expid[r]
imonth=-1
# 'pi_echam6_1m_3.4_14': 1.4551915228366852e-11 (3.4453427765078637e-12)
np.nanmax(abs(awi_esm_org_o[expid[r]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[r]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[r]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[r]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

imonth=-1
# 'pi_echam6_1m_3.4_15': 4.547473508864641e-11 (7.537378129048497e-12)
np.nanmax(abs(awi_esm_org_o[expid[9]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[9]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[9]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[9]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

imonth=-1
# 'pi_echam6_1y_3.3_3': 5.4569682106375694e-11 (8.06949568893995e-12)
np.nanmax(abs(awi_esm_org_o[expid[10]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[10]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[10]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[10]]['wiso'].wisoevap[imonth, 3:, :, :].sum(axis=0)))

'''


'''

i = 0
j = 1
expid[i] + ' ' + expid[j]
imonth = 0
iglobal = 3

#### global 0 tagmap
# qg: 5.624760213152058e-09 (1.4678977193675738e-11)
# mw: 5.350868081954729e-10 (2.43729498569945e-12)
np.nanmax(abs(awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, -2, :, :]))
np.nanmean(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, -2, :, :]))
test = awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, -2, :, :]
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test1.nc')
test = awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, -2, :, :]
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test2.nc')


#### normal and tagged evp from whole globe
# qg: 0 (0)
# mw: 9.285620762966573e-06 (2.08923571184035e-07)
# np.nanmax(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 3, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, -1, :, :]))
np.nanmax(abs(awi_esm_org_o[expid[j]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, iglobal, :, :]))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, iglobal, :, :]))
test = awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, iglobal, :, :]
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test3.nc')
test = awi_esm_org_o[expid[j]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, iglobal, :, :]
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test4.nc')


# normal and tagged evp from nh/sh
# qg: 7.414200808852911e-09 (5.3256362283112445e-11)
# mw: 9.285584383178502e-06 (2.0891927161099173e-07)
np.nanmax(abs(awi_esm_org_o[expid[j]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, 4:6, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 4:6, :, :].sum(axis=0)))
test = awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 4:6, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test5.nc')
test = awi_esm_org_o[expid[j]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, 4:6, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test6.nc')

# normal and tagged evp from wh/eh
# qg: 5.624315235763788e-09 (1.4609350766174128e-11)
# mw: 9.285595297114924e-06 (2.08921370080538e-07)
np.nanmax(abs(awi_esm_org_o[expid[j]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[imonth, 6:8, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 6:8, :, :].sum(axis=0)))


test = awi_esm_org_o[expid[i]]['echam'].evap[imonth, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[imonth, 6:8, :, :].sum(axis=0)
test.to_netcdf('/work/ollie/qigao001/output/0_backup/test.nc')
'''
# endregion
# =============================================================================


# =============================================================================
# region check tagmap and SST bins

i = 0
expid[i]
i_ts=-1
i_level = 17

# tagmap, 14th layer, 191st timestep
awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :]

# tsw, 191st timestep, tagmap == 1
np.max(awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values[awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :] == 1])
np.min(awi_esm_org_o[expid[i]]['echam'].tsw[i_ts, :, :].values[awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :] == 1])
# [295.15137, 297.14844]
# [295.15, 297.15]

# tagmap, 14th layer, 191st timestep, tsw [295.15, 297.15]
np.mean(awi_esm_org_o[expid[i]]['wiso'].tagmap[i_ts, i_level-1, :, :].values[np.where((awi_esm_org_o[expid[i]]['echam'].tsw[i_ts-1, :, :] > 295.15) & (awi_esm_org_o[expid[i]]['echam'].tsw[i_ts-1, :, :] <= 297.15))])


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
    u'Values of tagmap [$-$] at the 14th layer and the 191st timestep\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0.0_Values of tagmap at 14th layer and 191st timestep.png')



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
    'tsw: surface temperature of water [$°C$] at the 191st timestep\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0.1_tsw at the 191st timestep.png')


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
    'tsw: surface temperature of water [$°C$] at the 191st timestep where tagmap==1\necham-6.3.05p2-wiso: ' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.0.2_tsw at the 191st timestep with tagmap_1.png')


################ evaporation





# endregion
# =============================================================================


# =============================================================================
# region check tagging evaporation fraction

# calculate the water tracer fraction
i = 0
expid[i]
i_ts=-1
i_level = 17

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
99.57203865, 100.421875
'''

################ Tagged evaporation from all regions over normal evaporation

pltlevel = np.arange(99.5, 100.501, 0.001)
pltticks = np.arange(99.5, 100.501, 0.1)
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
fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.1.0_fraction of evaporation from all tag regions.png')

# 'pi_echam6_1d_3.6_4': 1.0186341e-10, 1.7560668e-11
np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[i_ts, 3:, :, :].sum(axis=0)))
np.nanmean(abs(awi_esm_org_o[expid[i]]['echam'].evap[i_ts, :, :] - awi_esm_org_o[expid[i]]['wiso'].wisoevap[i_ts, 3:, :, :].sum(axis=0)))


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
    fig.savefig('figures/6_awi/6.0_tagging_development/6.0.0_SST/6.0.0.1.1_fraction of evaporation from the ' + str(i_level-3) + 'th tag regions.png')
    print(str(j))







'''
'''
# endregion
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
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
# =============================================================================


# =============================================================================
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

# awi_esm_odir = '/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/'
awi_esm_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = [
    # 'TAG01',
    # 'TAG02',
    # 'pi_echam6_1m_3.0_2',
    # 'pi_echam6_1m_3.0_8',
    # 'pi_echam6_1m_3.0_9',
    # 'pi_echam6_1m_3.0_10',
    # 'pi_echam6_1m_3.0_11',
    # 'pi_echam6_1m_3.0_12',
    
    'pi_echam6_1d_3.4_1',
    # 'pi_echam6_1d_3.5_2',
    'pi_echam6_1d_3.5_3',
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
i = 1
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

i = 0
j = 1
expid[i] + '   ' + expid[j]
# normal climate variables
(awi_esm_org_o[expid[i]]['echam'].evap == awi_esm_org_o[expid[j]]['echam'].evap).all()
(awi_esm_org_o[expid[i]]['echam'].aprl == awi_esm_org_o[expid[j]]['echam'].aprl).all()
# np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].aprl - awi_esm_org_o[expid[j]]['echam'].aprl))
(awi_esm_org_o[expid[i]]['echam'].temp2 == awi_esm_org_o[expid[j]]['echam'].temp2).all()
# np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].temp2 - awi_esm_org_o[expid[j]]['echam'].temp2))
(awi_esm_org_o[expid[i]]['echam'].u10 == awi_esm_org_o[expid[j]]['echam'].u10).all()
(awi_esm_org_o[expid[i]]['echam'].q2m == awi_esm_org_o[expid[j]]['echam'].q2m).all()
# np.nanmax(abs(awi_esm_org_o[expid[i]]['echam'].q2m - awi_esm_org_o[expid[j]]['echam'].q2m))


# wiso variables

(awi_esm_org_o[expid[i]]['wiso'].wisoevap.values == awi_esm_org_o[expid[j]]['wiso'].wisoevap.values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisoaprl.values == awi_esm_org_o[expid[j]]['wiso'].wisoaprl.values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisows.values == awi_esm_org_o[expid[j]]['wiso'].wisows.values).all()

np.nanmax(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 3:6, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 3:6, :, :]))
test = awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 3:6, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 3:6, :, :]
test.to_netcdf('/work/ollie/qigao001/0_backup/test00.nc')


np.nanmax(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 3:6, :, :].values - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 3:6, :, :].values))
'''
(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisoaprl[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoaprl[:, 0:3, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisows[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisows[:, 0:3, :, :].values).all()

(awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 15:18, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 12:15, :, :].values).all()


'''

'''
(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisoaprl[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisoaprl[:, 0:3, :, :].values).all()
(awi_esm_org_o[expid[i]]['wiso'].wisows[:, 0:3, :, :].values == awi_esm_org_o[expid[j]]['wiso'].wisows[:, 0:3, :, :].values).all()
'''


np.mean(abs(awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :].values - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :].values))

test = awi_esm_org_o[expid[i]]['wiso'].wisoevap[:, 0:3, :, :] - awi_esm_org_o[expid[j]]['wiso'].wisoevap[:, 0:3, :, :]
test.to_netcdf('/work/ollie/qigao001/output/backup/test.nc')


'''
'pi_final_qg_tag5_1m_0_41code' == 'pi_final_qg_tag5_1m_1_41code'
# Yes, it should be bit reproducible

'pi_final_qg_tag5_1m_2_41code_notag' == 'pi_final_qg_tag5_1m_3_wisocode_notag'
# If ntag = 0, tagging branch is the same as the wiso branch

'pi_final_qg_tag5_1m_0_41code' != 'pi_final_qg_tag5_1m_2_41code_notag'
# If ntag > 0, water isotope simulations are different.

'pi_final_qg_tag5_1m_3_wisocode_notag' != 'pi_final_qg_tag5_1m_4_wisocode_tag'
# Xiaoxu's codes already make the model bit unreproducible
# the difference is mainly around the coast

'pi_final_qg_tag5_1m_0_41code' != 'pi_final_qg_tag5_1m_4_wisocode_tag'
# The difference is mainly around the coast

'pi_final_qg_tag5_1m_0_41code' == 'pi_final_qg_tag5_1m_5_43code'
'pi_final_qg_tag5_1m_2_41code_notag' == 'pi_final_qg_tag5_1m_6_43code_notag'
# No difference between 41code and 43code

'pi_final_qg_tag5_1m_4_wisocode_tag' != 'pi_final_qg_tag5_1m_5_43code'

'pi_final_qg_tag5_1y_5_qgtest2' == 'pi_final_qg_tag5_1y_6_qgtest3'
no improvement from echam6_qgtest2_20220401 to echam6_qgtest4_20220405

'pi_final_qg_tag5_1y_5_qgtest2' == 'pi_final_qg_tag5_1y_7_qgtest5'
no improvement from echam6_qgtest2_20220401 to echam6_qgtest5_20220405


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

# region check data format
outputdata = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_echam6_1m_28_2.5.18/outdata/echam/pi_echam6_1m_28_2.5.18_200001.01_wiso', engine='cfgrib')
outputdata.data_vars
outputdata.wisoevap[0, 0, 0].values

outputdata1 = xr.open_dataset('/work/ollie/cdanek/out/awicm-CMIP6/hu_svn471_ollie/fesom.2000.oce.mean.nc')
# outputdata1.tos[0, 0].values

# endregion
# =============================================================================


