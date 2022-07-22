

expid = ['pi_d_437_4.10',]
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


#-------- region

var_names = np.array([var for var in exp_org_o[expid[i]]['wiso'].data_vars])

org_prefix = ['q_', 'xl_', 'xi_']
sum_humidity = {'q': {}, 'xl': {}, 'xi': {}}

for l in range(len(org_prefix)):
    # l = len(org_prefix) - 1
    print('#----------------------------------------------------------------')
    print('#-------------------------------- ' + org_prefix[l])
    org_names = var_names[
            [var.startswith(org_prefix[l]) for var in var_names]]
    
    for j in range(len(org_names)):
        # j = len(org_names) - 1
        print('#---------------- ' + org_names[j])
        if (j == 0):
            sum_humidity[org_prefix[l][:-1]] = exp_org_o[expid[i]]['wiso'][org_names[j]].copy()
        else:
            sum_humidity[org_prefix[l][:-1]] += exp_org_o[expid[i]]['wiso'][org_names[j]].copy()

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_sum_humidity.pkl', 'wb') as f:
    pickle.dump(sum_humidity, f)

