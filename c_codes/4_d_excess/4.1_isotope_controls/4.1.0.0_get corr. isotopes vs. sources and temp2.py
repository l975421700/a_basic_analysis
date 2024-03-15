

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
# region get Corr. isotopes and sources

corr_sources_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_sources_isotopes[expid[i]] = {}
    
    for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance', 'RHsst']:
        # ivar = 'sst'
        # 'lon',
        print('#---------------- ' + ivar)
        
        corr_sources_isotopes[expid[i]][ivar] = {}
        
        for iisotopes in ['d_ln', 'd_excess',]:
            # iisotopes = 'd_ln'
            # 'wisoaprt', 'dO18', 'dD',
            print('#-------- ' + iisotopes)
            
            corr_sources_isotopes[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'ann'
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
                
                corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime] = {}
                
                corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'] = xr.corr(
                    isotopevar,
                    pre_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'] = xs.pearson_r_eff_p_value(
                    isotopevar,
                    pre_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'] = corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'].copy()
                
                corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values[corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'].values > 0.05] = np.nan
                
                # if (ialltime == 'mon'):
                #     corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm'] = {}
                    
                #     corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r'] = xr.corr(
                #         isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #         pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month') - pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month').mean(),
                #         dim='time').compute()
                    
                #     corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['p'] = xs.pearson_r_eff_p_value(
                #         isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #         pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month') - pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month').mean(),
                #         dim='time').compute()
                    
                #     corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r_significant'] = corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r'].copy()
                    
                #     corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r_significant'].values[corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_sources_isotopes[expid[i]], f)




'''
#-------------------------------- check

i = 0

corr_sources_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl', 'rb') as f:
    corr_sources_isotopes[expid[i]] = pickle.load(f)

ivar = 'sst'
iisotopes = 'd_ln'
ialltime = 'mon'

isotopevar = d_ln_alltime[expid[i]][ialltime]

data1 = xr.corr(
    isotopevar,
    pre_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data2 = corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xs.pearson_r_eff_p_value(
    isotopevar,
    pre_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data4 = corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1
data5[data3 > 0.05] = np.nan
data6 = corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())


#-------------------------------- check with site estimation

i = 0
corr_sources_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl', 'rb') as f:
    corr_sources_isotopes[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

isotopes_alltime_icores = {}
pre_weighted_var_icores = {}
for i in range(len(expid)):
    print(i)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.isotopes_alltime_icores.pkl', 'rb') as f:
        isotopes_alltime_icores[expid[i]] = pickle.load(f)
    
    with open(
        exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.pre_weighted_var_icores.pkl', 'rb') as f:
        pre_weighted_var_icores[expid[i]] = pickle.load(f)

isite = 'Neumayer'
for ivar in ['sst', 'RHsst']:
    # ivar = 'sst'
    print('#-------------------------------- ' + ivar)
    
    for iisotopes in ['d_ln', 'd_excess']:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
            # ialltime = 'daily'
            print('#-------- ' + ialltime)
            
            print(np.round((corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'].sel(
                lat=t63_sites_indices[isite]['lat'],
                lon=t63_sites_indices[isite]['lon'],
                method='nearest',
            ).values) ** 2, 2))
            
            subset=np.isfinite(isotopes_alltime_icores[expid[i]][iisotopes][isite][ialltime]) & np.isfinite(pre_weighted_var_icores[expid[i]][isite][ivar][ialltime])
            print(np.round(pearsonr(
                isotopes_alltime_icores[expid[i]][iisotopes][isite][ialltime][subset],
                pre_weighted_var_icores[expid[i]][isite][ivar][ialltime][subset],
            ).statistic ** 2, 2))

ivar = 'RHsst'
iisotopes = 'd_excess'
ialltime = 'mon no mm'

corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'][t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon'],]

subset=np.isfinite(isotopes_alltime_icores[expid[i]][iisotopes][isite][ialltime]) & np.isfinite(pre_weighted_var_icores[expid[i]][isite][ivar][ialltime])
pearsonr(
    isotopes_alltime_icores[expid[i]][iisotopes][isite][ialltime][subset],
    pre_weighted_var_icores[expid[i]][isite][ivar][ialltime][subset],
    ).statistic

pearsonr(d_excess_alltime[expid[i]][ialltime][:, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon'],], pre_weighted_var[expid[i]][ivar][ialltime][:, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon'],])






'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and sources, given source SST

par_corr_sources_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sources_isotopes[expid[i]] = {}
    
    for ivar in ['lat', 'rh2m', 'wind10', 'distance', 'RHsst']:
        # ivar = 'lat'
        print('#---------------- ' + ivar)
        
        par_corr_sources_isotopes[expid[i]][ivar] = {}
        
        for iisotopes in ['d_ln', 'd_excess',]:
            # iisotopes = 'd_ln'
            print('#-------- ' + iisotopes)
            
            par_corr_sources_isotopes[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'ann'
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
                
                par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime] = {}
                
                par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]]['sst'][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]]['sst'][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'] = par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'].copy()
                
                par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values[par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'].values > 0.05] = np.nan
                
                # if (ialltime == 'mon'):
                #     par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm'] = {}
                    
                #     par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month') - pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month').mean(),
                #             pre_weighted_var[expid[i]]['sst'][ialltime].groupby('time.month') - pre_weighted_var[expid[i]]['sst'][ialltime].groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                #         )
                    
                #     par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['p'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month') - pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month').mean(),
                #             pre_weighted_var[expid[i]]['sst'][ialltime].groupby('time.month') - pre_weighted_var[expid[i]]['sst'][ialltime].groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                #         )
                    
                #     par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r_significant'] = par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r'].copy()
                    
                #     par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['r_significant'].values[par_corr_sources_isotopes[expid[i]][ivar][iisotopes]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sources_isotopes[expid[i]], f)




'''
#-------------------------------- check

i = 0

par_corr_sources_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes.pkl', 'rb') as f:
    par_corr_sources_isotopes[expid[i]] = pickle.load(f)

ivar = 'wind10'
iisotopes = 'dD'
ialltime = 'mon'

isotopevar = dD_alltime[expid[i]][ialltime]

data1 = xr.apply_ufunc(
    xr_par_cor,
    isotopevar,
    pre_weighted_var[expid[i]][ivar][ialltime],
    pre_weighted_var[expid[i]]['sst'][ialltime],
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    ).values
data2 = par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xr.apply_ufunc(
    xr_par_cor,
    isotopevar,
    pre_weighted_var[expid[i]][ivar][ialltime],
    pre_weighted_var[expid[i]]['sst'][ialltime],
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
    ).values
data4 = par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1.copy()
data5[data3 > 0.05] = np.nan
data6 = par_corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and sources (RHsst & SST)

par_corr_sources_RHsst_SST = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sources_RHsst_SST[expid[i]] = {}
    
    for iisotopes in ['d_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sources_RHsst_SST[expid[i]][iisotopes] = {}
        
        for ivar in ['sst', 'RHsst']:
            # ivar = 'sst'
            print('#---------------- ' + ivar)
            
            par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar] = {}
            
            for ctr_var in list(set(['sst', 'RHsst']) - set([ivar])):
                # ctr_var = 'RHsst'
                print('#-------- ' + ctr_var)
                
                par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var] = {}
                
                for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                    # ialltime = 'mon'
                    print('#---- ' + ialltime)
                    
                    par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var][ialltime] = {}
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = d_ln_alltime[expid[i]][ialltime]
                    elif (iisotopes == 'd_excess'):
                        isotopevar = d_excess_alltime[expid[i]][ialltime]
                    
                    par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                    
                    par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        pre_weighted_var[expid[i]][ivar][ialltime],
                        pre_weighted_var[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_RHsst_SST.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sources_RHsst_SST[expid[i]], f)




'''
#-------------------------------- check
i = 0
par_corr_sources_RHsst_SST = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_RHsst_SST.pkl', 'rb') as f:
    par_corr_sources_RHsst_SST[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann no am'

data1 = par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r']

data2 = xr.apply_ufunc(
    xr_par_cor,
    d_ln_alltime[expid[i]][ialltime],
    pre_weighted_var[expid[i]][ivar][ialltime],
    pre_weighted_var[expid[i]][ctr_var][ialltime],
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    )

print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
np.max(abs((data1.values[np.isfinite(data1.values)] - data2.values[np.isfinite(data2.values)]) / data2.values[np.isfinite(data2.values)]))


#-------------------------------- check site calculation

i = 0
par_corr_sources_RHsst_SST = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_RHsst_SST.pkl', 'rb') as f:
    par_corr_sources_RHsst_SST[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann no am'

isite = 'EDC'
with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

print(par_corr_sources_RHsst_SST[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']])

print(xr_par_cor(
    d_ln_alltime[expid[i]][ialltime][:, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
    pre_weighted_var[expid[i]][ivar][ialltime][:, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
    pre_weighted_var[expid[i]][ctr_var][ialltime][:, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get Corr. temp2 and isotopes

corr_temp2_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_temp2_isotopes[expid[i]] = {}
    
    for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#-------- ' + iisotopes)
        
        corr_temp2_isotopes[expid[i]][iisotopes] = {}
        
        for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
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
            
            temp2var = temp2_alltime[expid[i]][ialltime]
            # temp2var['time'] = isotopevar.time
            
            corr_temp2_isotopes[expid[i]][iisotopes][ialltime] = {}
            
            corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r'] = xr.corr(
                isotopevar,
                temp2var,
                dim='time').compute()
            
            corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['p'] = xs.pearson_r_eff_p_value(
                isotopevar,
                temp2var,
                dim='time').compute()
            
            corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r_significant'] = corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r'].copy()
            
            corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r_significant'].values[corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['p'].values > 0.05] = np.nan
            
            # if (ialltime == 'mon'):
            #     corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm'] = {}
                
            #     corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['r'] = xr.corr(
            #         isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
            #         temp2var.groupby('time.month') - temp2var.groupby('time.month').mean(),
            #         dim='time').compute()
                
            #     corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['p'] = xs.pearson_r_eff_p_value(
            #         isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
            #         temp2var.groupby('time.month') - temp2var.groupby('time.month').mean(),
            #         dim='time').compute()
                
            #     corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['r_significant'] = corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['r'].copy()
                
            #     corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['r_significant'].values[corr_temp2_isotopes[expid[i]][iisotopes]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_temp2_isotopes.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_temp2_isotopes[expid[i]], f)



'''
#-------------------------------- check

i = 0

corr_temp2_isotopes = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_temp2_isotopes.pkl', 'rb') as f:
    corr_temp2_isotopes[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ialltime = 'mon'

isotopevar = d_ln_alltime[expid[i]][ialltime]
temp2var = temp2_alltime[expid[i]][ialltime]
temp2var['time'] = isotopevar.time

data1 = xr.corr(isotopevar, temp2var, dim='time').compute().values
data2 = corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xs.pearson_r_eff_p_value(isotopevar, temp2var, dim='time').compute().values
data4 = corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1
data5[data3 > 0.05] = np.nan
data6 = corr_temp2_isotopes[expid[i]][iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. temp2 and isotopes, given d018, d_ln, dD etc.

par_corr_temp2_isotopes2 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_temp2_isotopes2[expid[i]] = {}
    
    for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_temp2_isotopes2[expid[i]][iisotopes] = {}
        
        for ctr_iisotopes in list(set(['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']) - set([iisotopes])):
            # ctr_iisotopes = 'd_ln'
            print('#-------- ' + ctr_iisotopes)
            
            par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes] = {}
            
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
                
                temp2var = temp2_alltime[expid[i]][ialltime]
                # temp2var['time'] = isotopevar.time
                
                par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime] = {}
                
                par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        temp2var,
                        isotopevar,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        temp2var,
                        isotopevar,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'] = par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'].copy()
                
                par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'].values[par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'].values > 0.05] = np.nan
                
                # if (ialltime == 'mon'):
                    
                #     par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm'] = {}

                #     par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             temp2var.groupby('time.month') - temp2var.groupby('time.month').mean(),
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['p'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             temp2var.groupby('time.month') - temp2var.groupby('time.month').mean(),
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r_significant'] = par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r'].copy()

                #     par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['r_significant'].values[par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_temp2_isotopes2.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_temp2_isotopes2[expid[i]], f)




'''
#-------------------------------- check
i = 0

par_corr_temp2_isotopes2 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_temp2_isotopes2.pkl', 'rb') as f:
    par_corr_temp2_isotopes2[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ctr_iisotopes = 'dO18'
ialltime = 'mon'

isotopevar = d_ln_alltime[expid[i]][ialltime]
ctr_var = dO18_alltime[expid[i]][ialltime]
temp2var = temp2_alltime[expid[i]][ialltime]
temp2var['time'] = isotopevar.time

data1 = xr.apply_ufunc(
    xr_par_cor,
    temp2var, isotopevar, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    ).values
data2 = par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xr.apply_ufunc(
    xr_par_cor,
    temp2var, isotopevar, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
    ).values
data4 = par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1.copy()
data5[data3 > 0.05] = np.nan
data6 = par_corr_temp2_isotopes2[expid[i]][iisotopes][ctr_iisotopes][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())


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
    
    for iisotopes in ['dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sst_isotopes2[expid[i]][iisotopes] = {}
        
        for ctr_iisotopes in list(set(['dO18', 'dD', 'd_ln', 'd_excess']) - set([iisotopes])):
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


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and temp2/sst, given temp2/sst

par_corr_isotopes_temp2_sst = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_isotopes_temp2_sst[expid[i]] = {}
    
    for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_isotopes_temp2_sst[expid[i]][iisotopes] = {}
        
        for ivar in ['temp2', 'sst']:
            # ivar = 'temp2'
            print('#-------- ' + ivar)
            
            par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar] = {}
            
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
                
                if (ivar == 'temp2'):
                    corr_var = temp2_alltime[expid[i]][ialltime]
                    # corr_var['time'] = isotopevar.time
                    
                    ctr_var = pre_weighted_var[expid[i]]['sst'][ialltime]
                elif (ivar == 'sst'):
                    corr_var = pre_weighted_var[expid[i]]['sst'][ialltime]
                    
                    ctr_var = temp2_alltime[expid[i]][ialltime]
                    # ctr_var['time'] = isotopevar.time
                
                par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime] = {}
                
                par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        corr_var,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotopevar,
                        corr_var,
                        ctr_var,
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
                
                par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r_significant'] = par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r'].copy()
                
                par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r_significant'].values[par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['p'].values > 0.05] = np.nan
                
                # if (ialltime == 'mon'):
                    
                #     par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm'] = {}

                #     par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['r'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             corr_var.groupby('time.month') - corr_var.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['p'] = xr.apply_ufunc(
                #             xr_par_cor,
                #             isotopevar.groupby('time.month') - isotopevar.groupby('time.month').mean(),
                #             corr_var.groupby('time.month') - corr_var.groupby('time.month').mean(),
                #             ctr_var.groupby('time.month') - ctr_var.groupby('time.month').mean(),
                #             input_core_dims=[["time"], ["time"], ["time"]],
                #             kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                #         )

                #     par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['r_significant'] = par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['r'].copy()

                #     par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['r_significant'].values[par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar]['mon_no_mm']['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_isotopes_temp2_sst.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_isotopes_temp2_sst[expid[i]], f)



'''
#-------------------------------- check

i = 0

par_corr_isotopes_temp2_sst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_isotopes_temp2_sst.pkl', 'rb') as f:
    par_corr_isotopes_temp2_sst[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ialltime = 'mon'
ivar = 'sst'

isotopevar = d_ln_alltime[expid[i]][ialltime]

corr_var = pre_weighted_var[expid[i]]['sst'][ialltime]

ctr_var = temp2_alltime[expid[i]][ialltime]
ctr_var['time'] = isotopevar.time


data1 = xr.apply_ufunc(
    xr_par_cor,
    isotopevar, corr_var, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    ).values
data2 = par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xr.apply_ufunc(
    xr_par_cor,
    isotopevar, corr_var, ctr_var,
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
    ).values
data4 = par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())

data5 = data1.copy()
data5[data3 > 0.05] = np.nan
data6 = par_corr_isotopes_temp2_sst[expid[i]][iisotopes][ivar][ialltime]['r_significant'].values
print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get Corr. SAM and isotopes+sources+temp2

corr_sam_isotopes_sources_temp2 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_sam_isotopes_sources_temp2[expid[i]] = {}
    
    for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',
                 'wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',
                 'temp2']:
        # ivar = 'sst'
        print('#---------------- ' + ivar)
        
        corr_sam_isotopes_sources_temp2[expid[i]][ivar] = {}
        
        for ialltime in ['mon', 'mon_no_mm']:
            # ialltime = 'ann'
            print('#---- ' + ialltime)
            
            if (ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']):
                var = pre_weighted_var[expid[i]][ivar]['mon']
            elif (ivar == 'wisoaprt'):
                var = wisoaprt_alltime[expid[i]]['mon'].sel(
                    wisotype=1) * seconds_per_d
            elif (ivar == 'dO18'):
                var = dO18_alltime[expid[i]]['mon']
            elif (ivar == 'dD'):
                var = dD_alltime[expid[i]]['mon']
            elif (ivar == 'd_ln'):
                var = d_ln_alltime[expid[i]]['mon']
            elif (ivar == 'd_excess'):
                var = d_excess_alltime[expid[i]]['mon']
            elif (ivar == 'temp2'):
                var = temp2_alltime[expid[i]]['mon']
                var['time'] = d_excess_alltime[expid[i]]['mon'].time
            
            if (ialltime == 'mon_no_mm'):
                var = var.groupby('time.month') - var.groupby('time.month').mean()
            
            corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime] = {}
            
            corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r'] = xr.corr(b_sam_mon[expid[i]], var, dim='time').compute()
            
            corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['p'] = xs.pearson_r_eff_p_value(b_sam_mon[expid[i]], var, dim='time').compute()
            
            corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r_significant'] = corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r'].copy()
            
            corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r_significant'].values[corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['p'].values > 0.05] = np.nan
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sam_isotopes_sources_temp2.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_sam_isotopes_sources_temp2[expid[i]], f)




'''
#-------------------------------- check

i = 0

corr_sam_isotopes_sources_temp2 = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sam_isotopes_sources_temp2.pkl', 'rb') as f:
    corr_sam_isotopes_sources_temp2[expid[i]] = pickle.load(f)

for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance',
             'wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess',
             'temp2']:
    # ivar = 'sst'
    print('#---------------- ' + ivar)
    
    for ialltime in ['mon', 'mon_no_mm']:
        # ialltime = 'ann'
        print('#---- ' + ialltime)
        
        if (ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']):
            var = pre_weighted_var[expid[i]][ivar]['mon']
        elif (ivar == 'wisoaprt'):
            var = wisoaprt_alltime[expid[i]]['mon'].sel(
                wisotype=1) * seconds_per_d
        elif (ivar == 'dO18'):
            var = dO18_alltime[expid[i]]['mon']
        elif (ivar == 'dD'):
            var = dD_alltime[expid[i]]['mon']
        elif (ivar == 'd_ln'):
            var = d_ln_alltime[expid[i]]['mon']
        elif (ivar == 'd_excess'):
            var = d_excess_alltime[expid[i]]['mon']
        elif (ivar == 'temp2'):
            var = temp2_alltime[expid[i]]['mon']
            var['time'] = d_excess_alltime[expid[i]]['mon'].time
        
        if (ialltime == 'mon_no_mm'):
            var = var.groupby('time.month') - var.groupby('time.month').mean()
        
        data1 = xr.corr(b_sam_mon[expid[i]], var, dim='time').compute().values
        data2 = corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r'].values
        print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        
        data3 = xs.pearson_r_eff_p_value(b_sam_mon[expid[i]], var, dim='time').compute().values
        data4 = corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['p'].values
        print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())
        
        data5 = data1.copy()
        data5[data3 > 0.05] = np.nan
        data6 = corr_sam_isotopes_sources_temp2[expid[i]][ivar][ialltime]['r_significant'].values
        print((data5[np.isfinite(data5)] == data6[np.isfinite(data6)]).all())

'''
# endregion
# -----------------------------------------------------------------------------

