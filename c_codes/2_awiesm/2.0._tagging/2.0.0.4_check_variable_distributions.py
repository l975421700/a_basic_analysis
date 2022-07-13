

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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
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
    zerok,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import data: variable distribution in ECHAM 9-day simulations


#---------------- import simulation

exp_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

expid = ['pi_echam6_1d_214_3.70',]

exp_org_o = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    exp_org_o[expid[i]]['echam'] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    # exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')


#---------------- import ESACCI SST
esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

cell_area = {}
cell_area['echam6_t63'] = xr.open_dataset('scratch/others/land_sea_masks/ECHAM6_T63_slm_area.nc')


'''
exp_org_o[expid[i]]['echam'].tsw
exp_org_o[expid[i]]['echam'].wind10
exp_org_o[expid[i]]['echam'].rh2m
'''
# endregion
# =============================================================================


# =============================================================================
# region oceanic SST distribution in a 9-day ECHAM6 simulations


#-------- get SST and cell area

oceanic_tsw = np.array([])
oceanic_cell_area = np.array([])

for itime in range(len(exp_org_o[expid[i]]['echam'].time)):
    # itime = 0
    
    oceanic_tsw = np.concatenate(
        (oceanic_tsw,
        exp_org_o[expid[i]]['echam'].tsw[itime, :, :].values[np.isfinite(analysed_sst)]),)
    
    oceanic_cell_area = np.concatenate(
        (oceanic_cell_area,
        cell_area['echam6_t63'].cell_area.values[np.isfinite(analysed_sst)]),)
    
    if (itime%20 == 0):
        print(str(itime) + '/' + str(len(exp_org_o[expid[i]]['echam'].time)))


#-------- plot it

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

plt_hist = plt.hist(
    x=(oceanic_tsw - zerok,),
    weights=(oceanic_cell_area,),
    color=['lightgray', ],
    bins=np.arange(-2, 32, 1),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(-2, 32, 2))
ax.set_xticklabels(np.arange(-2, 32, 2), size=8)
ax.set_xlabel('SST [$°C$]', size=10)

ax.set_yticks(np.arange(0, 0.081, 0.02))
ax.set_yticklabels(np.arange(0, 0.081, 0.02), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=0, linewidth=1, color='b', linestyle='-')
ax.axvline(x=28, linewidth=1, color='b', linestyle='-')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.97)

fig.savefig(
    'figures/6_awi/6.1_echam6/6.1.1_variable distribution/6.1.1.0_SST distribution_in_a_nine_day_ECHAM6_simulation.png',)


'''
(np.isfinite(analysed_sst)).sum()
(np.isnan(analysed_sst)).sum()

271.38 - 273.15 = -1.77
((oceanic_tsw - zerok) > 30).sum()
'''
# endregion
# =============================================================================


# =============================================================================
# region oceanic rh2m distribution in a 9-day ECHAM6 simulations


#-------- get rh2m and cell area

oceanic_rh2m = np.array([])
oceanic_cell_area = np.array([])

for itime in range(len(exp_org_o[expid[i]]['echam'].time)):
    # itime = 0
    
    oceanic_rh2m = np.concatenate(
        (oceanic_rh2m,
        exp_org_o[expid[i]]['echam'].rh2m[itime, :, :].values[np.isfinite(analysed_sst)]),)
    
    oceanic_cell_area = np.concatenate(
        (oceanic_cell_area,
        cell_area['echam6_t63'].cell_area.values[np.isfinite(analysed_sst)]),)
    
    if (itime%20 == 0):
        print(str(itime) + '/' + str(len(exp_org_o[expid[i]]['echam'].time)))


#-------- plot it

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

plt_hist = plt.hist(
    x=(oceanic_rh2m * 100,),
    weights=(oceanic_cell_area,),
    color=['lightgray', ],
    bins=np.arange(50, 120.1, 2.5),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(50, 125, 5))
ax.set_xticklabels(np.arange(50, 125, 5), size=8)
ax.set_xlabel('2 metre relative humidity [%]', size=10)

ax.set_yticks(np.arange(0, 0.051, 0.01))
ax.set_yticklabels(np.arange(0, 0.051, 0.01), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=55, linewidth=1, color='b', linestyle='-')
ax.axvline(x=105, linewidth=1, color='b', linestyle='-')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.97)

fig.savefig(
    'figures/6_awi/6.1_echam6/6.1.1_variable distribution/6.1.1.1_rh2m distribution_in_a_nine_day_ECHAM6_simulation.png',)

# endregion
# =============================================================================


# =============================================================================
# region oceanic wind10 distribution in a 9-day ECHAM6 simulations


#-------- get wind10 and cell area

oceanic_wind10 = np.array([])
oceanic_cell_area = np.array([])

for itime in range(len(exp_org_o[expid[i]]['echam'].time)):
    # itime = 0
    
    oceanic_wind10 = np.concatenate(
        (oceanic_wind10,
        exp_org_o[expid[i]]['echam'].wind10[itime, :, :].values[np.isfinite(analysed_sst)]),)
    
    oceanic_cell_area = np.concatenate(
        (oceanic_cell_area,
        cell_area['echam6_t63'].cell_area.values[np.isfinite(analysed_sst)]),)
    
    if (itime%20 == 0):
        print(str(itime) + '/' + str(len(exp_org_o[expid[i]]['echam'].time)))


#-------- plot it

fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8]) / 2.54)

plt_hist = plt.hist(
    x=(oceanic_wind10,),
    weights=(oceanic_cell_area,),
    color=['lightgray', ],
    bins=np.arange(0, 21, 1),
    density=True,
    rwidth=1,
)

ax.set_xticks(np.arange(0, 21, 1))
ax.set_xticklabels(np.arange(0, 21, 1), size=8)
ax.set_xlabel('10 metre wind speed [$m \; s^{-1}$]', size=10)

ax.set_yticks(np.arange(0, 0.121, 0.02))
ax.set_yticklabels(np.arange(0, 0.121, 0.02, dtype=np.float64), size=8)
ax.set_ylabel('Area-weighted frequency', size=10)

ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=1, linewidth=1, color='b', linestyle='-')
ax.axvline(x=16, linewidth=1, color='b', linestyle='-')
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.14, top=0.97)

fig.savefig(
    'figures/6_awi/6.1_echam6/6.1.1_variable distribution/6.1.1.2_wind10 distribution_in_a_nine_day_ECHAM6_simulation.png',)

'''
(oceanic_wind10 < 0).sum()
'''
# endregion
# =============================================================================


