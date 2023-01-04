

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
# region get epe_nt_binned sources

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

# set epe_nt source files
source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10',
              'distance',
              ]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.epe_nt_weighted_lat_binned.pkl',
    prefix + '.epe_nt_weighted_lon_binned.pkl',
    prefix + '.epe_nt_weighted_sst_binned.pkl',
    prefix + '.epe_nt_weighted_rh2m_binned.pkl',
    prefix + '.epe_nt_weighted_wind10_binned.pkl',
    prefix + '.transport_distance_epe_nt_binned.pkl',
    ]

epe_nt_sources_sites_binned = {}
epe_nt_sources_sites_binned[expid[i]] = {}

for ivar, ifile in zip(source_var, source_var_files):
    # ivar = 'lat'
    # ifile = 'output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0/analysis/echam/pi_m_502_5.0.epe_nt_weighted_lat_binned.pkl'
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f:
        epe_nt_weighted_var = pickle.load(f)
    
    if (ivar == 'distance'):
        alltimes = ['ann']
    else:
        alltimes = ['mon', 'sea', 'ann', 'mm', 'sm']
    
    epe_nt_sources_sites_binned[expid[i]][ivar] = {}
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        epe_nt_sources_sites_binned[expid[i]][ivar][isite] = {}
        
        for ialltime in alltimes:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime] = {}
            
            for iqtl in epe_nt_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime][iqtl] = \
                    epe_nt_weighted_var[iqtl][ialltime][
                        :,
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].copy()
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime] = pd.DataFrame(
            columns=('iqtl', 'quantiles', 'am',))
        
        for iqtl in epe_nt_weighted_var.keys():
            # iqtl = '90%'
            # print('#-- ' + iqtl)
            
            epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime] = pd.concat([
                epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime],
                pd.DataFrame(data={
                    'iqtl': iqtl,
                    'quantiles': quantiles_bin[iqtl],
                    'am': epe_nt_weighted_var[iqtl][ialltime][
                        t63_sites_indices[isite]['ilat'],
                        t63_sites_indices[isite]['ilon']].values,
                    }, index=[0])],
                ignore_index=True,)

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_nt_sources_sites_binned.pkl', 'wb') as f:
    pickle.dump(epe_nt_sources_sites_binned[expid[i]], f)





'''
#-------------------------------- check
epe_nt_sources_sites_binned = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_nt_sources_sites_binned.pkl', 'rb') as f:
    epe_nt_sources_sites_binned[expid[i]] = pickle.load(f)

source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10',
            #   'distance',
              ]
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.epe_nt_weighted_lat_binned.pkl',
    prefix + '.epe_nt_weighted_lon_binned.pkl',
    prefix + '.epe_nt_weighted_sst_binned.pkl',
    prefix + '.epe_nt_weighted_rh2m_binned.pkl',
    prefix + '.epe_nt_weighted_wind10_binned.pkl',
    # prefix + '.transport_distance_epe_nt_binned.pkl',
    ]

for iind in np.arange(0, 5, 1):
    # iind = 0
    ivar = source_var[iind]
    ifile = source_var_files[iind]
    print('#------------ ' + ivar + ': ' + ifile)
    
    with open(ifile, 'rb') as f: epe_nt_weighted_var = pickle.load(f)
    
    for isite in stations_sites.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
            # ialltime = 'daily'
            print('#---- ' + ialltime)
            
            for iqtl in epe_nt_weighted_var.keys():
                # iqtl = '90%'
                # print('#-- ' + iqtl)
                data1 = epe_nt_sources_sites_binned[expid[i]][ivar][isite][
                    ialltime][iqtl].values
                data2 = epe_nt_weighted_var[iqtl][ialltime][
                    :,
                    t63_sites_indices[isite]['ilat'],
                    t63_sites_indices[isite]['ilon']].copy().values
                print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        
        ialltime = 'am'
        print('#---- ' + ialltime)
        for iqtl in epe_nt_weighted_var.keys():
            # iqtl = '90.5%'
            # print('#-- ' + iqtl)
            
            data1 = epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime].loc[
                epe_nt_sources_sites_binned[expid[i]][ivar][isite][ialltime].iqtl == iqtl
            ].am.values[0]
            
            data2 = epe_nt_weighted_var[iqtl][ialltime][
                t63_sites_indices[isite]['ilat'],
                t63_sites_indices[isite]['ilon']].values
            print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
            # print(data1 == data2)
    
    del epe_nt_weighted_var


'''
# endregion
# -----------------------------------------------------------------------------


