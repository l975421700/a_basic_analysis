

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_703_6.0_k52',
    'nudged_705_6.0',
    ]

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

from a_basic_analysis.b_module.namelist import (
    seconds_per_d,
)

from a_basic_analysis.b_module.statistics import (
    xr_par_cor,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

dO18_q_sfc_alltime = {}
dD_q_sfc_alltime = {}
d_ln_q_sfc_alltime = {}
d_excess_q_sfc_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
        dO18_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
        dD_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
        d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
        d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)


source_var = ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'RHsst']
# 'distance',
q_sfc_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_sfc_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_sfc_weighted_lat.pkl',
        prefix + '.q_sfc_weighted_lon.pkl',
        prefix + '.q_sfc_weighted_sst.pkl',
        prefix + '.q_sfc_weighted_rh2m.pkl',
        prefix + '.q_sfc_weighted_wind10.pkl',
        # prefix + '.transport_distance.pkl',
        prefix + '.q_sfc_weighted_RHsst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get Corr. isotopes and sources

corr_sources_isotopes_q_sfc = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_sources_isotopes_q_sfc[expid[i]] = {}
    
    for ivar in ['sst', 'RHsst']:
        # ivar = 'sst'
        # 'lat', 'lon', 'rh2m', 'wind10',
        print('#---------------- ' + ivar)
        
        corr_sources_isotopes_q_sfc[expid[i]][ivar] = {}
        
        for iisotopes in ['d_ln', 'd_excess',]:
            # iisotopes = 'd_ln'
            # 'wisoaprt', 'dO18', 'dD',
            print('#-------- ' + iisotopes)
            
            corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                if (iisotopes == 'dO18'):
                    isotopevar = dO18_q_sfc_alltime[expid[i]][ialltime]
                elif (iisotopes == 'dD'):
                    isotopevar = dD_q_sfc_alltime[expid[i]][ialltime]
                elif (iisotopes == 'd_ln'):
                    isotopevar = d_ln_q_sfc_alltime[expid[i]][ialltime]
                elif (iisotopes == 'd_excess'):
                    isotopevar = d_excess_q_sfc_alltime[expid[i]][ialltime]
                
                corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime] = {}
                
                corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'] = xr.corr(
                    isotopevar,
                    q_sfc_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['p'] = xs.pearson_r_eff_p_value(
                    isotopevar,
                    q_sfc_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r_significant'] = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'].copy()
                
                corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values[corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_sources_isotopes_q_sfc[expid[i]], f)



'''
#-------------------------------- check

i = 0

corr_sources_isotopes_q_sfc = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)

ivar = 'sst'
iisotopes = 'd_ln'
ialltime = 'mon'

isotopevar = d_ln_q_sfc_alltime[expid[i]][ialltime]

data1 = xr.corr(
    isotopevar,
    q_sfc_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data2 = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xs.pearson_r_eff_p_value(
    isotopevar,
    q_sfc_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data4 = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1
data5[data3 > 0.05] = np.nan
data6 = corr_sources_isotopes_q_sfc[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())


#-------------------------------- check

i = 0

corr_sources_isotopes_q_sfc = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_sfc.pkl', 'rb') as f:
    corr_sources_isotopes_q_sfc[expid[i]] = pickle.load(f)


(corr_sources_isotopes_q_sfc[expid[i]]['sst']['d_excess']['daily']['r'] ** 2).to_netcdf('scratch/test/test0.nc')
(corr_sources_isotopes_q_sfc[expid[i]]['RHsst']['d_excess']['daily']['r'] ** 2).to_netcdf('scratch/test/test1.nc')


'''
# endregion
# -----------------------------------------------------------------------------



