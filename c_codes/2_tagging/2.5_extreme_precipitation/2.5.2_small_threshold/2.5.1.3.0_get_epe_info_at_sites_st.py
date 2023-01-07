

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
import gc

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
from scipy.stats import circstd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
    find_ilat_ilon,
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region import sites information

# import sites information
major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
Antarctic_stations = pd.read_csv('data_sources/others/Antarctic_stations.csv')
stations_sites = pd.concat(
    [major_ice_core_site[['Site', 'lon', 'lat']],
     Antarctic_stations[['Site', 'lon', 'lat']],],
    ignore_index=True,
    )

# import sites indices
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.t63_sites_indices.pkl',
    'rb') as f:
    t63_sites_indices = pickle.load(f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt_masked_bin_st at ice core sites

wisoaprt_masked_bin_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_bin_st.pkl',
    'rb') as f:
    wisoaprt_masked_bin_st[expid[i]] = pickle.load(f)


quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

wisoaprt_masked_bin_st_icores = {}
wisoaprt_masked_bin_st_icores[expid[i]] = {}


for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#-------- ' + icores)
    
    wisoaprt_masked_bin_st_icores[expid[i]][icores] = {}
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'] = {}
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'] = {}
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'] = {}
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
        # ialltime = 'mon'
        print('#---- ' + ialltime)
        
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'][ialltime] = {}
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'][ialltime] = {}
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime] = {}
        
        for iqtl in quantiles_bin.keys():
            # iqtl = '90%'
            # print('#---- ' + iqtl)
            
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'][ialltime][iqtl] = \
                wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'][ialltime][iqtl] = \
                wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime][iqtl] = \
                wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
    
    
    ialltime = 'am'
    print('#---- ' + ialltime)
    
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'][ialltime] = pd.DataFrame(
        columns=('iqtl', 'quantiles', 'am',))
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'][ialltime] = pd.DataFrame(
        columns=('iqtl', 'quantiles', 'am',))
    wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime] = pd.DataFrame(
        columns=('iqtl', 'quantiles', 'am',))
    
    for iqtl in quantiles_bin.keys():
        # iqtl = '90%'
        # print('#---- ' + iqtl)
        
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'][ialltime] = pd.concat([
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['mean'][ialltime],
            pd.DataFrame(data={
                'iqtl': iqtl,
                'quantiles': quantiles_bin[iqtl],
                'am': wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']].values,
                }, index=[0])],
            ignore_index=True,)
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'][ialltime] = pd.concat([
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['frc'][ialltime],
            pd.DataFrame(data={
                'iqtl': iqtl,
                'quantiles': quantiles_bin[iqtl],
                'am': wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']].values,
                }, index=[0])],
            ignore_index=True,)
        wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime] = pd.concat([
            wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime],
            pd.DataFrame(data={
                'iqtl': iqtl,
                'quantiles': quantiles_bin[iqtl],
                'am': wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']].values,
                }, index=[0])],
            ignore_index=True,)


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_masked_bin_st_icores.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_masked_bin_st_icores[expid[i]], f)




'''
#-------------------------------- check

#---- distance between data points

wisoaprt_masked_bin_st_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_masked_bin_st_icores.pkl',
    'rb') as f:
    wisoaprt_masked_bin_st_icores[expid[i]] = pickle.load(f)

iqtl = '90.5%'

from haversine import haversine
for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#----------------' + icores)
    
    slat = t63_sites_indices[icores]['lat']
    slon = t63_sites_indices[icores]['lon']
    glat = wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan']['ann'][iqtl].lat.values
    glon = wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan']['ann'][iqtl].lon.values
    
    distance = haversine([slat, slon], [glat, glon], normalize = True)
    
    if(distance > 100):
        print(distance)


#---- check values

wisoaprt_masked_bin_st_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_masked_bin_st_icores.pkl',
    'rb') as f:
    wisoaprt_masked_bin_st_icores[expid[i]] = pickle.load(f)

wisoaprt_masked_bin_st = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_bin_st.pkl',
    'rb') as f:
    wisoaprt_masked_bin_st[expid[i]] = pickle.load(f)

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))


for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#-------- ' + icores)
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
        # ialltime = 'mon'
        print('#---- ' + ialltime)
        
        for iqtl in quantiles_bin.keys():
            # iqtl = '90%'
            print('#---- ' + iqtl)
            
            d1=wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime][iqtl]
            d2=wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime][
                    :,
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']]
            # print((d1 == d2).all().values)
            print((d1[np.isfinite(d1)] == d2[np.isfinite(d2)]).all().values)
    
    for ialltime in ['am']:
        # ialltime = 'am'
        print('#---- ' + ialltime)
        
        for iqtl in quantiles_bin.keys():
            # iqtl = '90.5%'
            print('#---- ' + iqtl)
            
            d1=wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime][
                wisoaprt_masked_bin_st_icores[expid[i]][icores]['meannan'][ialltime].iqtl == iqtl
            ].am.values
            d2=wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime][
                    t63_sites_indices[icores]['ilat'],
                    t63_sites_indices[icores]['ilon']].values
            # print((d1 == d2).all())
            print(d1[np.isfinite(d1)] == d2[np.isfinite(d2)])





#-------------------------------- print annual mean values
iqtl = '90%'
for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#----------------' + icores)
    wisoaprt_epe_am_icores = \
        np.round(
            wisoaprt_epe_alltime_icores[expid[i]][icores][
                'am'][iqtl].values * 100, 1)
    wisoaprt_epe_annstd_icores = \
        np.round(
            (wisoaprt_epe_alltime_icores[expid[i]][icores][
                'ann'][iqtl] * 100).std(ddof=1).values,
            1)
    print(str(wisoaprt_epe_am_icores) + ' ± ' + str(wisoaprt_epe_annstd_icores))







'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get epe_st_binned sources

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

# set epe_st source files
source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10',
              'distance',
              ]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.epe_st_weighted_lat_binned.pkl',
    prefix + '.epe_st_weighted_lon_binned.pkl',
    prefix + '.epe_st_weighted_sst_binned.pkl',
    prefix + '.epe_st_weighted_rh2m_binned.pkl',
    prefix + '.epe_st_weighted_wind10_binned.pkl',
    prefix + '.transport_distance_epe_st_binned.pkl',
    ]

epe_st_sources_sites_binned = {}
epe_st_sources_sites_binned[expid[i]] = {}

for ivar, ifile in zip(source_var, source_var_files):
    # ivar = 'lat'
    # ifile = 'output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0/analysis/echam/pi_m_502_5.0.epe_st_weighted_lat_binned.pkl'
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f:
        epe_st_weighted_var = pickle.load(f)
    
    if (ivar == 'distance'):
        alltimes = ['ann']
    else:
        alltimes = ['mon', 'sea', 'ann', 'mm', 'sm']
    
    epe_st_sources_sites_binned[expid[i]][ivar] = {}
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        epe_st_sources_sites_binned[expid[i]][ivar][isite] = {}
        
        for ialltime in alltimes:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime] = {}
            
            for iqtl in epe_st_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime][iqtl] = \
                    epe_st_weighted_var[iqtl][ialltime][
                        :,
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].copy()
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime] = pd.DataFrame(
            columns=('iqtl', 'quantiles', 'am',))
        
        for iqtl in epe_st_weighted_var.keys():
            # iqtl = '90%'
            # print('#-- ' + iqtl)
            
            epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime] = pd.concat([
                epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime],
                pd.DataFrame(data={
                    'iqtl': iqtl,
                    'quantiles': quantiles_bin[iqtl],
                    'am': epe_st_weighted_var[iqtl][ialltime][
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].values,
                    }, index=[0])],
                ignore_index=True,)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_sources_sites_binned.pkl', 'wb') as f:
    pickle.dump(epe_st_sources_sites_binned[expid[i]], f)





'''
#-------------------------------- check
epe_st_sources_sites_binned = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_st_sources_sites_binned.pkl', 'rb') as f:
    epe_st_sources_sites_binned[expid[i]] = pickle.load(f)

source_var = [
    # 'lat', 'lon', 'sst', 'rh2m', 'wind10',
    'distance',
    ]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    # prefix + '.epe_st_weighted_lat_binned.pkl',
    # prefix + '.epe_st_weighted_lon_binned.pkl',
    # prefix + '.epe_st_weighted_sst_binned.pkl',
    # prefix + '.epe_st_weighted_rh2m_binned.pkl',
    # prefix + '.epe_st_weighted_wind10_binned.pkl',
    prefix + '.transport_distance_epe_st_binned.pkl',
    ]

for ivar, ifile in zip(source_var, source_var_files):
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f: epe_st_weighted_var = pickle.load(f)
    
    if (ivar == 'distance'):
        alltimes = ['ann']
    else:
        alltimes = ['mon', 'sea', 'ann', 'mm', 'sm']
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        for ialltime in alltimes:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            
            for iqtl in epe_st_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                data1 = epe_st_sources_sites_binned[expid[i]][ivar][isite][
                    ialltime][iqtl].values
                data2 = epe_st_weighted_var[iqtl][ialltime][
                    :,
                    t63_sites_indices[isite]['ilat'],
                    t63_sites_indices[isite]['ilon']].copy().values
                print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        for iqtl in epe_st_weighted_var.keys():
            # iqtl = '90.5%'
            # print('#-- ' + iqtl)
            
            data1 = epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime].loc[
                epe_st_sources_sites_binned[expid[i]][ivar][isite][ialltime].iqtl == iqtl
            ].am.values[0]
            
            data2 = epe_st_weighted_var[iqtl][ialltime][
                t63_sites_indices[isite]['ilat'],
                t63_sites_indices[isite]['ilon']].values
            print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
            # print(data1 == data2)
    
    del epe_st_weighted_var


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt_mask_bin_st at ice core sites

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_st.pkl',
    'rb') as f:
    wisoaprt_mask_bin_st = pickle.load(f)

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

wisoaprt_mask_bin_st_icores = {}
wisoaprt_mask_bin_st_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#-------- ' + icores)
    
    wisoaprt_mask_bin_st_icores[expid[i]][icores] = {}
    
    wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'] = {}
    
    for iqtl in quantiles_bin.keys():
        # iqtl = '90.5%'
        # print('#---- ' + iqtl)
        
        wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'][iqtl] = \
            wisoaprt_mask_bin_st[iqtl][
                :,
                t63_sites_indices[icores]['ilat'],
                t63_sites_indices[icores]['ilon']]
    
    wisoaprt_mask_bin_st_icores[expid[i]][icores]['am'] = pd.DataFrame(
        columns=('iqtl', 'quantiles', 'am',))
    
    for iqtl in quantiles_bin.keys():
        # iqtl = '90.5%'
        # print('#---- ' + iqtl)
        
        wisoaprt_mask_bin_st_icores[expid[i]][icores]['am'] = pd.concat([
            wisoaprt_mask_bin_st_icores[expid[i]][icores]['am'],
            pd.DataFrame(data={
                'iqtl': iqtl,
                'quantiles': quantiles_bin[iqtl],
                'am': wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'][
                    iqtl].mean().values * 365,
                }, index=[0])],
            ignore_index=True,)


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_mask_bin_st_icores.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_mask_bin_st_icores[expid[i]], f)




'''
#------------------------ check

wisoaprt_mask_bin_st_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_mask_bin_st_icores.pkl',
    'rb') as f:
    wisoaprt_mask_bin_st_icores[expid[i]] = pickle.load(f)

#---- distance between data points

iqtl = '90.5%'
from haversine import haversine

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#----------------' + icores)
    
    slat = t63_sites_indices[icores]['lat']
    slon = t63_sites_indices[icores]['lon']
    glat = wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'][iqtl].lat.values
    glon = wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'][iqtl].lon.values
    
    distance = haversine([slat, slon], [glat, glon], normalize = True)
    
    if(distance > 100):
        print(distance)

iqtl = '90.5%'
icores = 'EDC'
wisoaprt_mask_bin_st_icores[expid[i]][icores]['daily'][iqtl].mean() * 365
wisoaprt_mask_bin_st_icores[expid[i]][icores]['am'].am.values

'''
# endregion
# -----------------------------------------------------------------------------



