

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    ]


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs

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
import cartopy.feature as cfeature
from scipy.stats import pearsonr
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
    regrid,
    mean_over_ais,
    time_weighted_mean,
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
    cplot_ttest,
    xr_par_cor,
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


wisoaprt_alltime = {}
dO18_alltime = {}
dD_alltime = {}
d_ln_alltime = {}
d_excess_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)


source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_lon.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
        prefix + '.transport_distance.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)


temp2_alltime = {}
for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
        temp2_alltime[expid[i]] = pickle.load(f)


quantile_interval  = np.arange(10, 50 + 1e-4, 10, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))


sam_mon = {}
sam_posneg_ind = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    sam_mon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')
    
    sam_posneg_ind[expid[i]] = {}
    sam_posneg_ind[expid[i]]['pos'] = (sam_mon[expid[i]].sam > sam_mon[expid[i]].sam.std(ddof = 1)).values
    sam_posneg_ind[expid[i]]['neg'] = (sam_mon[expid[i]].sam < (-1 * sam_mon[expid[i]].sam.std(ddof = 1))).values



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get composite SAM and isotopes+sources+temp2


composite_sam_isotopes_sources_temp2 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    composite_sam_isotopes_sources_temp2[expid[i]] = {}
    
    for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',
                 'wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',
                 'temp2']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        composite_sam_isotopes_sources_temp2[expid[i]][ivar] = {}
        
        for ialltime in ['mon', 'mon_no_mm']:
            # ialltime = 'ann'
            print('#---- ' + ialltime)
            
            if (ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']):
                var = pre_weighted_var[expid[i]][ivar]['mon'].copy()
            elif (ivar == 'wisoaprt'):
                var = (wisoaprt_alltime[expid[i]]['mon'].sel(
                    wisotype=1) * seconds_per_d).compute()
            elif (ivar == 'dO18'):
                var = dO18_alltime[expid[i]]['mon'].copy()
            elif (ivar == 'dD'):
                var = dD_alltime[expid[i]]['mon'].copy()
            elif (ivar == 'd_ln'):
                var = d_ln_alltime[expid[i]]['mon'].copy()
            elif (ivar == 'd_excess'):
                var = d_excess_alltime[expid[i]]['mon'].copy()
            elif (ivar == 'temp2'):
                var = temp2_alltime[expid[i]]['mon'].copy()
                var['time'] = d_excess_alltime[expid[i]]['mon'].time
            
            if (ialltime == 'mon_no_mm'):
                var = var.groupby('time.month') - var.groupby('time.month').mean()
            
            composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime] = {}
            
            composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff'] = var[sam_posneg_ind[expid[i]]['pos']].mean(dim='time') - var[sam_posneg_ind[expid[i]]['neg']].mean(dim='time')
            
            composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['significance'] = ttest_fdr_control(
                var[sam_posneg_ind[expid[i]]['pos']],
                var[sam_posneg_ind[expid[i]]['neg']],)
            
            composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff_significant'] = composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff'].copy()
            
            composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff_significant'].values[composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['significance'] == False] = np.nan
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sam_isotopes_sources_temp2.pkl', 'wb') as f:
        pickle.dump(composite_sam_isotopes_sources_temp2[expid[i]], f)




'''
#-------------------------------- check
i = 0

composite_sam_isotopes_sources_temp2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sam_isotopes_sources_temp2.pkl', 'rb') as f:
    composite_sam_isotopes_sources_temp2[expid[i]] = pickle.load(f)

for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',
             'wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',
             'temp2']:
    # ivar = 'sst'
    print('#---------------- ' + ivar)
    
    for ialltime in ['mon', 'mon_no_mm']:
        # ialltime = 'ann'
        print('#---- ' + ialltime)
        
        if (ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']):
            var = pre_weighted_var[expid[i]][ivar]['mon'].copy()
        elif (ivar == 'wisoaprt'):
            var = (wisoaprt_alltime[expid[i]]['mon'].sel(
                wisotype=1) * seconds_per_d).compute()
        elif (ivar == 'dO18'):
            var = dO18_alltime[expid[i]]['mon'].copy()
        elif (ivar == 'dD'):
            var = dD_alltime[expid[i]]['mon'].copy()
        elif (ivar == 'd_ln'):
            var = d_ln_alltime[expid[i]]['mon'].copy()
        elif (ivar == 'd_excess'):
            var = d_excess_alltime[expid[i]]['mon'].copy()
        elif (ivar == 'temp2'):
            var = temp2_alltime[expid[i]]['mon'].copy()
            var['time'] = d_excess_alltime[expid[i]]['mon'].time
        
        if (ialltime == 'mon_no_mm'):
            var = var.groupby('time.month') - var.groupby('time.month').mean()
        
        data1 = (var[sam_posneg_ind[expid[i]]['pos']].mean(dim='time') - var[sam_posneg_ind[expid[i]]['neg']].mean(dim='time')).values
        data2 = composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff'].values
        print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        
        data3 = ttest_fdr_control(
            var[sam_posneg_ind[expid[i]]['pos']],
            var[sam_posneg_ind[expid[i]]['neg']],)
        data4 = composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['significance']
        print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())
        
        data5 = data1.copy()
        data5[data3 == False] = np.nan
        data6 = composite_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['diff_significant'].values
        print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())


'''
# endregion
# -----------------------------------------------------------------------------




