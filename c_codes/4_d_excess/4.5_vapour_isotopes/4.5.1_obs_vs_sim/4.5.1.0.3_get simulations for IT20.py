
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
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
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from a_basic_analysis.b_module.basic_calculations import (
    find_ilat_ilon,
    find_ilat_ilon_general,
    find_gridvalue_at_site,
    find_multi_gridvalue_at_site,
    find_gridvalue_at_site_time,
    find_multi_gridvalue_at_site_time,
)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

#-------- import obs

with open('data_sources/water_isotopes/IT20/IT20_ACE.pkl', 'rb') as f:
    IT20_ACE = pickle.load(f)


#-------- import sim

wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

d_excess_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_sfc_alltime.pkl', 'rb') as f:
    d_excess_q_sfc_alltime[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)


lon = d_ln_q_sfc_alltime[expid[i]]['am'].lon
lat = d_ln_q_sfc_alltime[expid[i]]['am'].lat


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

IT20_ACE_1d_sim = {}

IT20_ACE_1d_sim[expid[i]] = IT20_ACE['1d'].copy()

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q',]:
    # var_name = 'd_ln'
    print('#-------- ' + var_name)
    
    if (var_name == 'dD'):
        ivar = dD_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd18O'):
        ivar = dO18_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd_xs'):
        ivar = d_excess_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd_ln'):
        ivar = d_ln_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'q'):
        ivar = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'].sel(lev=47)
    
    IT20_ACE_1d_sim[expid[i]][var_name + '_sim'] = \
        find_multi_gridvalue_at_site_time(
            IT20_ACE_1d_sim[expid[i]]['time'],
            IT20_ACE_1d_sim[expid[i]]['lat'],
            IT20_ACE_1d_sim[expid[i]]['lon'],
            ivar.time.values,
            ivar.lat.values,
            ivar.lon.values,
            ivar.values
        )

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.IT20_ACE_1d_sim.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(IT20_ACE_1d_sim[expid[i]], f)




'''
#-------------------------------- check
IT20_ACE_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.IT20_ACE_1d_sim.pkl', 'rb') as f:
    IT20_ACE_1d_sim[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)

ires = 20
stime = IT20_ACE_1d_sim[expid[i]]['time'][ires]
slat = IT20_ACE_1d_sim[expid[i]]['lat'][ires]
slon = IT20_ACE_1d_sim[expid[i]]['lon'][ires]

itime = np.argmin(abs(stime.asm8 - d_ln_q_sfc_alltime[expid[i]]['daily'].time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lat.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lon.values)


#-------- check d_ln
var_name = 'd_ln'
print(IT20_ACE_1d_sim[expid[i]][var_name + '_sim'][ires])
print(d_ln_q_sfc_alltime[expid[i]]['daily'][itime, ilat, ilon])

#-------- check q
var_name = 'q'
wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

print(IT20_ACE_1d_sim[expid[i]][var_name + '_sim'][ires])
print(wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'][itime, 0, ilat, ilon])

#-------- check dD
dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

var_name = 'dD'
print(IT20_ACE_1d_sim[expid[i]][var_name + '_sim'][ires])
print(dD_q_sfc_alltime[expid[i]]['daily'][itime, ilat, ilon])


'''
# endregion
# -----------------------------------------------------------------------------



