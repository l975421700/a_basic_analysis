
# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    
    # 'nudged_703_6.0_k52',
    # 'nudged_705_6.0',
    # 'nudged_706_6.0_k52_88',
    # 'nudged_707_6.0_k43',
    # 'nudged_708_6.0_I01',
    # 'nudged_709_6.0_I03',
    # 'nudged_710_6.0_S3',
    'nudged_711_6.0_S6',
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

with open('data_sources/water_isotopes/MC16/MC16_Dome_C.pkl', 'rb') as f:
    MC16_Dome_C = pickle.load(f)

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

# temp2_alltime = {}
# with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
#     temp2_alltime[expid[i]] = pickle.load(f)
# temp2_alltime[expid[i]]['daily']['time'] = temp2_alltime[expid[i]]['daily']['time'].dt.floor('D').rename('time')


lon = d_ln_q_sfc_alltime[expid[i]]['am'].lon
lat = d_ln_q_sfc_alltime[expid[i]]['am'].lat



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

MC16_Dome_C_1d_sim = {}

MC16_Dome_C_1d_sim[expid[i]] = MC16_Dome_C['1d'].copy()

for var_name in ['dD', 'd18O', 'd_xs', 'd_ln', 'q']:
    # var_name = 'd_ln'
    # 't_3m',
    print('#-------- ' + var_name)
    
    if (var_name == 'dD'):
        ivar = dD_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd18O'):
        ivar = dO18_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd_xs'):
        ivar = d_excess_q_sfc_alltime[expid[i]]['daily']
    elif (var_name == 'd_ln'):
        ivar = d_ln_q_sfc_alltime[expid[i]]['daily']
    # elif (var_name == 't_3m'):
    #     ivar = temp2_alltime[expid[i]]['daily']
    elif (var_name == 'q'):
        ivar = wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'].sel(lev=47)
    
    MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'] = \
        find_multi_gridvalue_at_site_time(
            MC16_Dome_C_1d_sim[expid[i]]['time'],
            MC16_Dome_C_1d_sim[expid[i]]['lat'],
            MC16_Dome_C_1d_sim[expid[i]]['lon'],
            ivar.time.values,
            ivar.lat.values,
            ivar.lon.values,
            ivar.values,
        )

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.MC16_Dome_C_1d_sim.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(MC16_Dome_C_1d_sim[expid[i]], f)






'''
#-------------------------------- check
MC16_Dome_C_1d_sim = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.MC16_Dome_C_1d_sim.pkl', 'rb') as f:
    MC16_Dome_C_1d_sim[expid[i]] = pickle.load(f)

d_ln_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_sfc_alltime.pkl', 'rb') as f:
    d_ln_q_sfc_alltime[expid[i]] = pickle.load(f)

ires = 12
stime = MC16_Dome_C_1d_sim[expid[i]]['time'][ires]
slat = MC16_Dome_C_1d_sim[expid[i]]['lat'][ires]
slon = MC16_Dome_C_1d_sim[expid[i]]['lon'][ires]

itime = np.argmin(abs(stime.asm8 - d_ln_q_sfc_alltime[expid[i]]['daily'].time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lat.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lon.values)


#-------- check q

var_name = 'q'
wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'][ires]
wiso_q_6h_sfc_alltime[expid[i]]['q16o']['daily'][itime, 0, ilat, ilon]

#-------- check d_ln
var_name = 'd_ln'
MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'][ires]
d_ln_q_sfc_alltime[expid[i]]['daily'][itime, ilat, ilon]

#-------- check temp2
temp2_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
    temp2_alltime[expid[i]] = pickle.load(f)
temp2_alltime[expid[i]]['daily']['time'] = temp2_alltime[expid[i]]['daily']['time'].dt.floor('D').rename('time')

var_name = 't_3m'
MC16_Dome_C_1d_sim[expid[i]][var_name + '_sim'][ires]
temp2_alltime[expid[i]]['daily'][itime, ilat, ilon]

'''
# endregion
# -----------------------------------------------------------------------------



