

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    'pi_601_5.1',
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

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get composite temp2 and isotopes

composite_temp2_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    composite_temp2_isotopes[expid[i]] = {}
    
    for ivar in ['temp2']:
        # ivar = 'temp2'
        print('#---------------- ' + ivar)
        
        composite_temp2_isotopes[expid[i]][ivar] = {}
        
        for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',]:
            # iisotopes = 'd_ln'
            print('#-------- ' + iisotopes)
            
            composite_temp2_isotopes[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['mon', 'sea', 'ann', 'mon_no_mm',]:
                # ialltime = 'ann'
                print('#---- ' + ialltime)
                
                composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime] = {}
                
                if (ialltime != 'mon_no_mm'):
                    ialltime2 = ialltime
                elif (ialltime == 'mon_no_mm'):
                    ialltime2 = 'mon'
                
                temp2var = temp2_alltime[expid[i]][ialltime2].copy()
                
                if (iisotopes == 'wisoaprt'):
                    isotopevar = (wisoaprt_alltime[expid[i]][ialltime2].sel(
                        wisotype=1) * seconds_per_d).compute()
                elif (iisotopes == 'dO18'):
                    isotopevar = (dO18_alltime[expid[i]][ialltime2]).copy()
                elif (iisotopes == 'dD'):
                    isotopevar = (dD_alltime[expid[i]][ialltime2]).copy()
                elif (iisotopes == 'd_ln'):
                    isotopevar = (d_ln_alltime[expid[i]][ialltime2]).copy()
                elif (iisotopes == 'd_excess'):
                    isotopevar = (d_excess_alltime[expid[i]][ialltime2]).copy()
                
                if (ialltime == 'mon_no_mm'):
                    temp2var = (temp2var.groupby('time.month') - temp2var.groupby('time.month').mean()).compute()
                    isotopevar = (isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean()).compute()
                
                for iqtl in quantiles.keys():
                    # iqtl = '10%'
                    print('#-- ' + iqtl + ': ' + str(quantiles[iqtl]))
                    
                    composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl] = np.zeros(d_ln_alltime[expid[i]]['am'].shape)
                    
                    for ilat in range(isotopevar.shape[1]):
                        # ilat = 2
                        for ilon in range(isotopevar.shape[2]):
                            # ilon = 2
                            
                            var1 = temp2var[:, ilat, ilon].values
                            var2 = isotopevar[:, ilat, ilon].values
                            
                            subset = (np.isfinite(var1) & np.isfinite(var2))
                            var1 = var1[subset]
                            var2 = var2[subset]
                            
                            if (len(var1) < 3):
                                composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl][ilat, ilon] = np.nan
                            else:
                                lower_qtl = np.quantile(var1, quantiles[iqtl])
                                upper_qtl = np.quantile(var1, 1-quantiles[iqtl])
                                
                                var1_pos = (var1 > upper_qtl)
                                var1_neg = (var1 < lower_qtl)
                                
                                var2_posmean = np.mean(var2[var1_pos])
                                var2_negmean = np.mean(var2[var1_neg])
                                
                                composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl][ilat, ilon] = var2_posmean - var2_negmean

    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_temp2_isotopes.pkl', 'wb') as f:
        pickle.dump(composite_temp2_isotopes[expid[i]], f)




'''
#-------------------------------- check
composite_temp2_isotopes = {}

for i in np.arange(0, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_temp2_isotopes.pkl', 'rb') as f:
        composite_temp2_isotopes[expid[i]] = pickle.load(f)

for i in np.arange(0, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in 'temp2':
        # ivar = 'temp2'
        print('#---------------- ' + ivar)
        
        for iqtl in ['10%']:
            # iqtl = '10%'
            print('#-------- ' + iqtl + ': ' + str(quantiles[iqtl]))
            
            for ialltime in ['daily', 'mon', 'sea', 'ann', 'mon_no_mm']:
                # ialltime = 'ann'
                print('#---- ' + ialltime)
                
                for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess', ]:
                    # iisotopes = 'd_ln'
                    print('#-- ' + iisotopes)
                    
                    if (ialltime != 'mon_no_mm'):
                        ialltime2 = ialltime
                    elif (ialltime == 'mon_no_mm'):
                        ialltime2 = 'mon'
                    
                    temp2var = (temp2_alltime[expid[i]][ialltime2]).copy()
                    if (iisotopes == 'wisoaprt'):
                        isotopevar = (wisoaprt_alltime[expid[i]][ialltime2].sel(
                            wisotype=1) * seconds_per_d).compute()
                    elif (iisotopes == 'dO18'):
                        isotopevar = (dO18_alltime[expid[i]][ialltime2]).copy()
                    elif (iisotopes == 'dD'):
                        isotopevar = (dD_alltime[expid[i]][ialltime2]).copy()
                    elif (iisotopes == 'd_ln'):
                        isotopevar = (d_ln_alltime[expid[i]][ialltime2]).copy()
                    elif (iisotopes == 'd_excess'):
                        isotopevar = (d_excess_alltime[expid[i]][ialltime2]).copy()
                    
                    if (ialltime == 'mon_no_mm'):
                        temp2var = (temp2var.groupby('time.month') - temp2var.groupby('time.month').mean()).compute()
                        isotopevar = (isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean()).compute()
                    
                    for ilat in np.arange(1, 96, 30):
                        for ilon in np.arange(1, 192, 60):
                            
                            var1 = temp2var[:, ilat, ilon].values
                            var2 = isotopevar[:, ilat, ilon].values
                            
                            subset = (np.isfinite(var1) & np.isfinite(var2))
                            var2 = var2[subset]
                            var1 = var1[subset]
                            
                            if (len(var1) < 3):
                                result = np.nan
                            else:
                                lower_qtl = np.quantile(var1, quantiles[iqtl])
                                upper_qtl = np.quantile(var1, 1-quantiles[iqtl])
                                
                                var1_pos = (var1 > upper_qtl)
                                var1_neg = (var1 < lower_qtl)
                                
                                var2_posmean = np.mean(var2[var1_pos])
                                var2_negmean = np.mean(var2[var1_neg])
                                
                                result = var2_posmean - var2_negmean
                            
                            # print(str(np.round(composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl][ilat, ilon], 2)))
                            # print(str(np.round(result, 2)))
                            
                            data1 = composite_temp2_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl][ilat, ilon]
                            data2 = result
                            
                            if (data1 != data2):
                                print(data1)
                                print(data2)


'''
# endregion
# -----------------------------------------------------------------------------


