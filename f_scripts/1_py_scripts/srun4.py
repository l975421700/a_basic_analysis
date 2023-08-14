

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
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


sam_mon = {}
b_sam_mon = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    sam_mon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.sam_mon.nc')
    
    b_sam_mon[expid[i]], _ = xr.broadcast(
        sam_mon[expid[i]].sam,
        d_ln_alltime[expid[i]]['mon'])

'''
'''
# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region get partial Corr. source SST and isotopes, given d018, d_ln, dD etc.

par_corr_sst_isotopes2 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sst_isotopes2[expid[i]] = {}
    
    for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sst_isotopes2[expid[i]][iisotopes] = {}
        
        for ctr_iisotopes in list(set(['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']) - set([iisotopes])):
            # ctr_iisotopes = 'd_ln'
            print('#-------- ' + ctr_iisotopes)
            
            par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes] = {}
            
            for ialltime in ['mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'mon'
                print('#---- ' + ialltime)
                
                if (iisotopes == 'wisoaprt'):
                    isotopevar = wisoaprt_alltime[expid[i]][ialltime].sel(
                        wisotype=1) * seconds_per_d
                elif (iisotopes == 'dO18'):
                    isotopevar = dO18_alltime[expid[i]][ialltime]
                elif (iisotopes == 'dD'):
                    isotopevar = dD_alltime[expid[i]][ialltime]
                elif (iisotopes == 'd_ln'):
                    isotopevar = d_ln_alltime[expid[i]][ialltime]
                elif (iisotopes == 'd_excess'):
                    isotopevar = d_excess_alltime[expid[i]][ialltime]
                
                if (ctr_iisotopes == 'wisoaprt'):
                    ctr_var = wisoaprt_alltime[expid[i]][ialltime].sel(
                        wisotype=1) * seconds_per_d
                elif (ctr_iisotopes == 'dO18'):
                    ctr_var = dO18_alltime[expid[i]][ialltime]
                elif (ctr_iisotopes == 'dD'):
                    ctr_var = dD_alltime[expid[i]][ialltime]
                elif (ctr_iisotopes == 'd_ln'):
                    ctr_var = d_ln_alltime[expid[i]][ialltime]
                elif (ctr_iisotopes == 'd_excess'):
                    ctr_var = d_excess_alltime[expid[i]][ialltime]
                
                sst_var = pre_weighted_var[expid[i]]['sst'][ialltime]
                
                par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime] = {}
                
                par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        sst_var,
                        isotopevar,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        sst_var,
                        isotopevar,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'] = par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'].copy()
                
                par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'].values[par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'].values > 0.05] = np.nan
                
                # if (ialltime == 'mon'):
                    
                #     par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm'] = {}

                #     par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             sst_var.groupby('time.month') - sst_var.groupby('time.month').mean(),
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['p'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             sst_var.groupby('time.month') - sst_var.groupby('time.month').mean(),
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r_significant'] = par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r'].copy()

                #     par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r_significant'].values[par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sst_isotopes2.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sst_isotopes2[expid[i]], f)




'''
#-------------------------------- check
i = 0

par_corr_sst_isotopes2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sst_isotopes2.pkl', 'rb') as f:
    par_corr_sst_isotopes2[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ctr_iisotopes = 'dO18'
ialltime = 'mon'

isotopevar = d_ln_alltime[expid[i]][ialltime]
ctr_var = dO18_alltime[expid[i]][ialltime]
sst_var = pre_weighted_var[expid[i]]['sst'][ialltime]

data1 = xr.apply_ufunc(
    xr_par_cor,
    sst_var, isotopevar, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    ).values
data2 = par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xr.apply_ufunc(
    xr_par_cor,
    sst_var, isotopevar, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
    ).values
data4 = par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1.copy()
data5[data3 > 0.05] = np.nan
data6 = par_corr_sst_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())


'''
# endregion
# -----------------------------------------------------------------------------

