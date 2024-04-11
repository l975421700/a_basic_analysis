

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_703_6.0_k52',
    'nudged_705_6.0',
    ]
i = 0

# -----------------------------------------------------------------------------
# region import packages

# management
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
import xskillscore as xs
from scipy.stats import pearsonr

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
)

from a_basic_analysis.b_module.statistics import (
    xr_par_cor,
    xr_par_cor_subset,
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


source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance', 'RHsst']
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
        prefix + '.pre_weighted_RHsst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)


# temp2_alltime = {}

# for i in range(len(expid)):
#     # i = 0
#     print(str(i) + ': ' + expid[i])
    
#     with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
#         temp2_alltime[expid[i]] = pickle.load(f)


# sam_mon = {}
# b_sam_mon = {}

# for i in range(len(expid)):
#     print(str(i) + ': ' + expid[i])
    
#     sam_mon[expid[i]] = xr.open_dataset(
#         exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')
    
#     b_sam_mon[expid[i]], _ = xr.broadcast(
#         sam_mon[expid[i]].sam,
#         d_ln_alltime[expid[i]]['mon'])

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and sources (RHsst & SST)

par_corr_sources_RHsst_SST_dthreshold = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sources_RHsst_SST_dthreshold[expid[i]] = {}
    
    for iisotopes in ['d_ln', 'd_excess',]:
        # ['d_ln', 'd_excess',]
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes] = {}
        
        for ivar in ['sst', 'RHsst',]:
            # ['sst', 'RHsst']
            # ivar = 'sst'
            print('#---------------- ' + ivar)
            
            par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes][ivar] = {}
            
            for ctr_var in list(set(['sst', 'RHsst']) - set([ivar])):
                # ctr_var = 'RHsst'
                print('#-------- ' + ctr_var)
                
                par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes][ivar][ctr_var] = {}
                
                for ialltime in ['daily',]:
                    # ialltime = 'daily'
                    print('#---- ' + ialltime)
                    
                    par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes][ivar][ctr_var][ialltime] = {}
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = d_ln_alltime[expid[i]][ialltime]
                    elif (iisotopes == 'd_excess'):
                        isotopevar = d_excess_alltime[expid[i]][ialltime]
                    
                    par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor_subset,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]][ctr_var][ialltime],
                        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) * seconds_per_d >= 0.02,
                        input_core_dims=[["time"], ["time"], ["time"],["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                    
                    par_corr_sources_RHsst_SST_dthreshold[expid[i]][iisotopes][ivar][ctr_var][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor_subset,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]][ctr_var][ialltime],
                        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) * seconds_per_d >= 0.02,
                        input_core_dims=[["time"], ["time"], ["time"],["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_RHsst_SST_dthreshold.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sources_RHsst_SST_dthreshold[expid[i]], f)




'''
'''
# endregion
# -----------------------------------------------------------------------------
