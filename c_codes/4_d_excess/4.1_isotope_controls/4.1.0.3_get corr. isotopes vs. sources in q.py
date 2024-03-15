

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
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

isotope_q_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    isotope_q_alltime[expid[i]] = {}
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_alltime.pkl', 'rb') as f:
        isotope_q_alltime[expid[i]]['d_ln'] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_alltime.pkl', 'rb') as f:
        isotope_q_alltime[expid[i]]['d_xs'] = pickle.load(f)


source_var = ['sst', 'RHsst', ]
q_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_weighted_sst.pkl',
        prefix + '.q_weighted_RHsst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_weighted_var[expid[i]][ivar] = pickle.load(f)




'''
isotope_q_alltime[expid[i]]['d_xs']['ann']
q_weighted_var[expid[i]]['sst']['ann']


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get Corr. isotopes and sources in q

corr_sources_isotopes_q = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_sources_isotopes_q[expid[i]] = {}
    
    for ivar in ['sst', 'RHsst']:
        # ivar = 'sst'
        # 'lat', 'lon', 'rh2m', 'wind10',
        print('#---------------- ' + ivar)
        
        corr_sources_isotopes_q[expid[i]][ivar] = {}
        
        for iisotopes in ['d_ln', 'd_xs',]:
            # iisotopes = 'd_ln'
            # 'wisoaprt', 'dO18', 'dD',
            print('#-------- ' + iisotopes)
            
            corr_sources_isotopes_q[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime] = {}
                
                corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['r'] = xr.corr(
                    isotope_q_alltime[expid[i]][iisotopes][ialltime],
                    q_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['p'] = xs.pearson_r_eff_p_value(
                    isotope_q_alltime[expid[i]][iisotopes][ialltime],
                    q_weighted_var[expid[i]][ivar][ialltime],
                    dim='time').compute()
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_sources_isotopes_q[expid[i]], f)



'''
#-------------------------------- check

i = 0

corr_sources_isotopes_q = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q.pkl', 'rb') as f:
    corr_sources_isotopes_q[expid[i]] = pickle.load(f)

ivar = 'RHsst'
iisotopes = 'd_ln'
ialltime = 'ann'

data1 = xr.corr(
    isotope_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data2 = corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xs.pearson_r_eff_p_value(
    isotope_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var[expid[i]][ivar][ialltime],
    dim='time').compute().values
data4 = corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())


#-------------------------------- check site calculation

i = 0

corr_sources_isotopes_q = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q.pkl', 'rb') as f:
    corr_sources_isotopes_q[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

ivar = 'sst'
iisotopes = 'd_ln'
ialltime = 'ann'
ilev = 20

isite = 'EDC'
print(corr_sources_isotopes_q[expid[i]][ivar][iisotopes][ialltime]['r'][ilev, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']])

subset = np.isfinite(isotope_q_alltime[expid[i]][iisotopes][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']]) & \
            np.isfinite(q_weighted_var[expid[i]][ivar][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']])

print(pearsonr(
    isotope_q_alltime[expid[i]][iisotopes][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
    q_weighted_var[expid[i]][ivar][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and sources in q

par_corr_sources_isotopes_q = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sources_isotopes_q[expid[i]] = {}
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sources_isotopes_q[expid[i]][iisotopes] = {}
        
        for ivar in ['sst', 'RHsst']:
            # ivar = 'sst'
            print('#---------------- ' + ivar)
            
            par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar] = {}
            
            for ctr_var in list(set(['sst', 'RHsst']) - set([ivar])):
                # ctr_var = 'RHsst'
                print('#-------- ' + ctr_var)
                
                par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var] = {}
                
                for ialltime in ['mon', 'mon no mm', 'ann', 'ann no am']:
                    # ialltime = 'mon'
                    print('#---- ' + ialltime)
                    
                    par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime] = {}
                    
                    par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotope_q_alltime[expid[i]][iisotopes][ialltime],
                        q_weighted_var[expid[i]][ivar][ialltime],
                        q_weighted_var[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                    
                    par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotope_q_alltime[expid[i]][iisotopes][ialltime],
                        q_weighted_var[expid[i]][ivar][ialltime],
                        q_weighted_var[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sources_isotopes_q[expid[i]], f)




'''
#-------------------------------- check the calculation

par_corr_sources_isotopes_q={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q.pkl', 'rb') as f:
    par_corr_sources_isotopes_q[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

data1 = par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r']

data2 = xr.apply_ufunc(
    xr_par_cor,
    isotope_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var[expid[i]][ivar][ialltime],
    q_weighted_var[expid[i]][ctr_var][ialltime],
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    )

print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())

print(np.max(abs((data1.values[np.isfinite(data1.values)] - data2.values[np.isfinite(data2.values)]) / data2.values[np.isfinite(data2.values)])))

#-------------------------------- check site calculation

par_corr_sources_isotopes_q={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q.pkl', 'rb') as f:
    par_corr_sources_isotopes_q[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'
ilev = 25

isite = 'EDC'

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

print(par_corr_sources_isotopes_q[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][ilev, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']])

print(xr_par_cor(
    isotope_q_alltime[expid[i]][iisotopes][ialltime][:, ilev, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
    q_weighted_var[expid[i]][ivar][ialltime][:, ilev, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
    q_weighted_var[expid[i]][ctr_var][ialltime][:, ilev, t63_sites_indices[isite]['ilat'], t63_sites_indices[isite]['ilon']],
))



'''
# endregion
# -----------------------------------------------------------------------------


# zonal mean
# -----------------------------------------------------------------------------
# region import data


isotope_zm_q_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    isotope_zm_q_alltime[expid[i]] = {}
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_zm_q_alltime.pkl', 'rb') as f:
        isotope_zm_q_alltime[expid[i]]['d_ln'] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_zm_q_alltime.pkl', 'rb') as f:
        isotope_zm_q_alltime[expid[i]]['d_xs'] = pickle.load(f)

source_var = ['sst', 'RHsst', ]
q_weighted_var_zm = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    q_weighted_var_zm[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.q_weighted_sst_zm.pkl',
        prefix + '.q_weighted_RHsst_zm.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            q_weighted_var_zm[expid[i]][ivar] = pickle.load(f)




'''
isotope_zm_q_alltime[expid[i]]['d_xs']['ann']
q_weighted_var_zm[expid[i]]['sst']['ann']


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get Corr. isotopes and sources in q_zm

corr_sources_isotopes_q_zm = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    corr_sources_isotopes_q_zm[expid[i]] = {}
    
    for ivar in ['sst', 'RHsst']:
        # ivar = 'sst'
        # 'lat', 'lon', 'rh2m', 'wind10',
        print('#---------------- ' + ivar)
        
        corr_sources_isotopes_q_zm[expid[i]][ivar] = {}
        
        for iisotopes in ['d_ln', 'd_xs',]:
            # iisotopes = 'd_ln'
            # 'wisoaprt', 'dO18', 'dD',
            print('#-------- ' + iisotopes)
            
            corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes] = {}
            
            for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                # ialltime = 'daily'
                print('#---- ' + ialltime)
                
                corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime] = {}
                
                corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['r'] = xr.corr(
                    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
                    q_weighted_var_zm[expid[i]][ivar][ialltime],
                    dim='time').compute()
                
                corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['p'] = xs.pearson_r_eff_p_value(
                    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
                    q_weighted_var_zm[expid[i]][ivar][ialltime],
                    dim='time').compute()
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_zm.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(corr_sources_isotopes_q_zm[expid[i]], f)



'''
#-------------------------------- check

i = 0

corr_sources_isotopes_q_zm = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)

ivar = 'RHsst'
iisotopes = 'd_ln'
ialltime = 'ann'

data1 = xr.corr(
    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var_zm[expid[i]][ivar][ialltime],
    dim='time').compute().values
data2 = corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['r'].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data3 = xs.pearson_r_eff_p_value(
    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var_zm[expid[i]][ivar][ialltime],
    dim='time').compute().values
data4 = corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['p'].values
print((data3[np.isfinite(data3)] == data4[np.isfinite(data4)]).all())


#-------------------------------- check site calculation

i = 0

corr_sources_isotopes_q_zm = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

ivar = 'sst'
iisotopes = 'd_ln'
ialltime = 'ann'
ilev = 20

isite = 'EDC'
print(corr_sources_isotopes_q_zm[expid[i]][ivar][iisotopes][ialltime]['r'][ilev, t63_sites_indices[isite]['ilat']])

subset = np.isfinite(isotope_zm_q_alltime[expid[i]][iisotopes][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat']]) & \
            np.isfinite(q_weighted_var_zm[expid[i]][ivar][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat']])

print(pearsonr(
    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat']],
    q_weighted_var_zm[expid[i]][ivar][ialltime][
        :, ilev,
        t63_sites_indices[isite]['ilat']],))


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get partial Corr. isotopes and sources in q_zm

par_corr_sources_isotopes_q_zm = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    par_corr_sources_isotopes_q_zm[expid[i]] = {}
    
    for iisotopes in ['d_ln', 'd_xs',]:
        # iisotopes = 'd_ln'
        print('#---------------- ' + iisotopes)
        
        par_corr_sources_isotopes_q_zm[expid[i]][iisotopes] = {}
        
        for ivar in ['sst', 'RHsst']:
            # ivar = 'sst'
            print('#---------------- ' + ivar)
            
            par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar] = {}
            
            for ctr_var in list(set(['sst', 'RHsst']) - set([ivar])):
                # ctr_var = 'RHsst'
                print('#-------- ' + ctr_var)
                
                par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var] = {}
                
                for ialltime in ['daily', 'mon', 'mon no mm', 'ann', 'ann no am']:
                    # ialltime = 'mon'
                    print('#---- ' + ialltime)
                    
                    par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime] = {}
                    
                    par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
                        q_weighted_var_zm[expid[i]][ivar][ialltime],
                        q_weighted_var_zm[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
                    )
                    
                    par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime]['p'] = xr.apply_ufunc(
                        xr_par_cor,
                        isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
                        q_weighted_var_zm[expid[i]][ivar][ialltime],
                        q_weighted_var_zm[expid[i]][ctr_var][ialltime],
                        input_core_dims=[["time"], ["time"], ["time"]],
                        kwargs={'output': 'p'}, dask = 'allowed', vectorize = True
                    )
    
    output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_zm.pkl'
    
    if (os.path.isfile(output_file)):
        os.remove(output_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(par_corr_sources_isotopes_q_zm[expid[i]], f)




'''
#-------------------------------- check the calculation

par_corr_sources_isotopes_q_zm={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'

data1 = par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r']

data2 = xr.apply_ufunc(
    xr_par_cor,
    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime],
    q_weighted_var_zm[expid[i]][ivar][ialltime],
    q_weighted_var_zm[expid[i]][ctr_var][ialltime],
    input_core_dims=[["time"], ["time"], ["time"]],
    kwargs={'output': 'r'}, dask = 'allowed', vectorize = True
    )

print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())

print(np.max(abs((data1.values[np.isfinite(data1.values)] - data2.values[np.isfinite(data2.values)]) / data2.values[np.isfinite(data2.values)])))

#-------------------------------- check site calculation

par_corr_sources_isotopes_q_zm={}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.par_corr_sources_isotopes_q_zm.pkl', 'rb') as f:
    par_corr_sources_isotopes_q_zm[expid[i]] = pickle.load(f)

iisotopes = 'd_ln'
ivar = 'sst'
ctr_var = 'RHsst'
ialltime = 'ann'
ilev = 10

isite = 'EDC'

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)

print(par_corr_sources_isotopes_q_zm[expid[i]][iisotopes][ivar][ctr_var][ialltime]['r'][ilev, t63_sites_indices[isite]['ilat']])

print(xr_par_cor(
    isotope_zm_q_alltime[expid[i]][iisotopes][ialltime][:, ilev, t63_sites_indices[isite]['ilat']],
    q_weighted_var_zm[expid[i]][ivar][ialltime][:, ilev, t63_sites_indices[isite]['ilat']],
    q_weighted_var_zm[expid[i]][ctr_var][ialltime][:, ilev, t63_sites_indices[isite]['ilat']],
))



'''
# endregion
# -----------------------------------------------------------------------------


