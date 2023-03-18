

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
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
from haversine import haversine

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
    find_ilat_ilon,
    find_ilat_ilon_general,
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

pre_weighted_sst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sst.pkl', 'rb') as f:
    pre_weighted_sst[expid[i]] = pickle.load(f)

lon = pre_weighted_sst[expid[i]]['am'].lon.values
lat = pre_weighted_sst[expid[i]]['am'].lat.values
lon_2d, lat_2d = np.meshgrid(lon, lat,)

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

pre_weighted_lat = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

pre_weighted_lon = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[i]] = pickle.load(f)


#-------- import sst
amippi_sst = {}
amippi_sst['original'] = xr.open_dataset(
    'startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc')

T63GR15_jan_surf = xr.open_dataset(
    '/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_jan_surf.nc')
t63_slm = T63GR15_jan_surf.SLM
b_slm = np.broadcast_to(
    t63_slm.values == 1,
    amippi_sst['original'].sst.shape,)

amippi_sst['trimmed'] = amippi_sst['original'].sst.copy()
amippi_sst['trimmed'].values[b_slm] = np.nan
amippi_sst['trimmed'].values[:] = amippi_sst['trimmed'].values - zerok
amippi_sst['am'] = amippi_sst['trimmed'].mean(dim='time')




'''
amippi_sst['trimmed'].to_netcdf('scratch/test/test.nc')
amippi_sst['am'].to_netcdf('scratch/test/test.nc')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get sst at sources


sst_sources = {}
sst_sources[expid[i]] = {}
sst_sources[expid[i]]['am'] = pre_weighted_sst[expid[i]]['am'].copy()
sst_sources[expid[i]]['am'] = sst_sources[expid[i]]['am'].rename('sst_sources')
sst_sources[expid[i]]['am'].values[:] = 0

for ilat in range(len(lat)):
    # ilat = 87
    # print('#---------------- ' + str(ilat))
    
    for ilon in range(len(lon)):
        # ilon = 23
        # print('#-------- ' + str(ilon))
        
        try:
            source_lat = pre_weighted_lat[expid[i]]['am'][ilat, ilon].values
            source_lon = pre_weighted_lon[expid[i]]['am'][ilat, ilon].values
            
            sources_ind = find_ilat_ilon(source_lat, source_lon, lat, lon)
            
            sst_sources[expid[i]]['am'].values[ilat, ilon] = \
                amippi_sst['am'][sources_ind[0], sources_ind[1]].values
        except:
            print('no source lat/lon')
            sst_sources[expid[i]]['am'].values[ilat, ilon] = np.nan


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source sst and sst at sources


output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/6.1.3.2 ' + expid[i] + ' sst sources differences.png'

nrow = 1
ncol = 3

wspace = 0.02
hspace = 0.12
fm_left = 0.02
fm_bottom = hspace / nrow
fm_right = 0.98
fm_top = 0.98

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 7.8*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    )

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[jcol] = hemisphere_plot(
            northextent=-60, ax_org = axs[jcol])
        cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
        
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# source sst

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=8, cm_max=20, cm_interval1=1, cm_interval2=1, cmap='RdBu',)

# plt_mesh = axs[0].pcolormesh(
#     lon,
#     lat,
#     pre_weighted_sst[expid[i]]['am'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, pre_weighted_sst[expid[i]]['am'], axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[0].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[0], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Source SST [$°C$]', linespacing=1.5,)


# sst at sources


# plt_mesh = axs[1].pcolormesh(
#     lon,
#     lat,
#     sst_sources[expid[i]]['am'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, sst_sources[expid[i]]['am'], axs[1],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[1].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[1], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('SST at source lat & lon [$°C$]', linespacing=1.5,)


# source sst - sst at sources

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=0.5, cmap='PuOr',
    reversed=True, asymmetric=True,)

# plt_mesh = axs[2].pcolormesh(
#     lon,
#     lat,
#     pre_weighted_sst[expid[i]]['am'] - sst_sources[expid[i]]['am'],
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt_mesh = plot_t63_contourf(
    lon, lat, pre_weighted_sst[expid[i]]['am'] - sst_sources[expid[i]]['am'], axs[2],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
axs[2].add_feature(
	cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, ax=axs[2], aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.05,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Differences: (a) - (b) [$°C$]', linespacing=1.5,)

fig.subplots_adjust(
    left=fm_left, right = fm_right, bottom = fm_bottom, top = fm_top,
    wspace=wspace, hspace=hspace,
    )

fig.savefig(output_png)




# (pre_weighted_sst[expid[i]]['am'] - sst_sources[expid[i]]['am']).to_netcdf('scratch/test/test.nc')

# endregion
# -----------------------------------------------------------------------------


