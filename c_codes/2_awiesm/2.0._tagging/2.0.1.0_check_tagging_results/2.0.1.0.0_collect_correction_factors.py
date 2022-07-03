

expid = ['pi_echam6_1d_211_3.69',]
print('#---- ' + expid[0])

exp_odir = '/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/'

#-------- region import packages

# management
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import pickle

#-------- region import orignal model output

exp_org_o = {}
i = 0
exp_org_o[expid[i]] = {}
exp_org_o[expid[i]]['wiso'] = xr.open_dataset( exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc' )


#-------- region check Martin's corrections

var_names = np.array([var for var in exp_org_o[expid[i]]['wiso'].data_vars])

cor_prefix = ['phy_tagqte_c_', 'phy_tagxlte_c_', 'phy_tagxite_c_']
org_prefix = ['q_', 'xl_', 'xi_']

level = [47, 41, 34]

org_var = {}
cor_var = {}
rel_corr = {}

# loop along q/xl/xi
for l in range(len(cor_prefix)):
    # l = len(cor_prefix) - 1
    print('#----------------------------------------------------------------')
    print('#-------------------------------- ' + cor_prefix[l])
    
    #---- define variables to store results
    # (klev, kwiso, ktime, klat, klon)
    org_var[org_prefix[l][:-1]] = np.zeros((
        len(level),
        len(exp_org_o[expid[i]]['wiso'].wisotype) - 3,
        len(exp_org_o[expid[i]]['wiso'].time),
        len(exp_org_o[expid[i]]['wiso'].lat),
        len(exp_org_o[expid[i]]['wiso'].lon),
        ), dtype=np.float32)
    cor_var[org_prefix[l][:-1]] = np.zeros(org_var[org_prefix[l][:-1]].shape)
    rel_corr[org_prefix[l][:-1]] = np.zeros(org_var[org_prefix[l][:-1]].shape)
    
    cor_names = var_names[
            [var.startswith(cor_prefix[l]) for var in var_names]]
    
    for j in range(len(cor_names)): # loop along klev
        # j = len(cor_names) - 1
        print('#---------------- ' + cor_names[j])
        
        org_names = var_names[
            [var.startswith(org_prefix[l]) for var in var_names]]
        
        for k in range(len(org_names)): # loop along kwiso
            # k = len(org_names) - 1
            print('#-------- ' + org_names[k])
            
            org_var[org_prefix[l][:-1]][j, k, :, :, :] = exp_org_o[expid[i]]['wiso'][org_names[k]][
                :, level[j]-1, :, :]
            
            cor_var[org_prefix[l][:-1]][j, k, :, :, :] = exp_org_o[expid[i]]['wiso'][cor_names[j]][
                :, 3+k, :, :]
    
    rel_corr[org_prefix[l][:-1]] = abs(cor_var[org_prefix[l][:-1]] * 900 / org_var[org_prefix[l][:-1]])


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_org_var.pkl',
          'wb') as f:
    pickle.dump(org_var, f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_cor_var.pkl',
          'wb') as f:
    pickle.dump(cor_var, f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_rel_corr.pkl',
          'wb') as f:
    pickle.dump(rel_corr, f)



