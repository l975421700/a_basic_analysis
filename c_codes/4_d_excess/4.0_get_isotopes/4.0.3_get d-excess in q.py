

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_703_6.0_k52',
    'nudged_705_6.0',
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

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


dO18_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl', 'rb') as f:
    dO18_q_alltime[expid[i]] = pickle.load(f)

dD_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
    dD_q_alltime[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_xs

d_excess_q_alltime = {}
d_excess_q_alltime[expid[i]] = {}

for ialltime in ['mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    d_excess_q_alltime[expid[i]][ialltime] = \
        dD_q_alltime[expid[i]][ialltime] - 8 * dO18_q_alltime[expid[i]][ialltime]

#-------- monthly without monthly mean
d_excess_q_alltime[expid[i]]['mon no mm'] = (d_excess_q_alltime[expid[i]]['mon'].groupby('time.month') - d_excess_q_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_excess_q_alltime[expid[i]]['ann no am'] = (d_excess_q_alltime[expid[i]]['ann'] - d_excess_q_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_excess_q_alltime[expid[i]], f)



'''
#-------------------------------- check

d_excess_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_q_alltime.pkl', 'rb') as f:
    d_excess_q_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

itime = -1
iplev = 0
ilat = 40
ilon = 90

aa = dD_q_alltime[expid[i]][ialltime][itime, iplev, ilat, ilon].values
bb = dO18_q_alltime[expid[i]][ialltime][itime, iplev, ilat, ilon].values
cc = d_excess_q_alltime[expid[i]][ialltime][itime, iplev, ilat, ilon].values

aa - 8 * bb
cc
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_ln

d_ln_q_alltime = {}
d_ln_q_alltime[expid[i]] = {}

for ialltime in ['mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    # ialltime = 'sm'
    
    ln_dD = 1000 * np.log(1 + dD_q_alltime[expid[i]][ialltime] / 1000)
    ln_d18O = 1000 * np.log(1 + dO18_q_alltime[expid[i]][ialltime] / 1000)
    
    d_ln_q_alltime[expid[i]][ialltime] = \
        ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

#-------- monthly without monthly mean
d_ln_q_alltime[expid[i]]['mon no mm'] = (d_ln_q_alltime[expid[i]]['mon'].groupby('time.month') - d_ln_q_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_ln_q_alltime[expid[i]]['ann no am'] = (d_ln_q_alltime[expid[i]]['ann'] - d_ln_q_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_ln_q_alltime[expid[i]], f)



'''
#-------------------------------- check

d_ln_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_q_alltime.pkl', 'rb') as f:
    d_ln_q_alltime[expid[i]] = pickle.load(f)

dO18_q_alltime = {}
dD_q_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_alltime.pkl', 'rb') as f:
        dO18_q_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_alltime.pkl', 'rb') as f:
        dD_q_alltime[expid[i]] = pickle.load(f)

iplev = 0

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    for ilat in np.arange(1, 96, 30):
        for ilon in np.arange(1, 192, 60):
            # i = 0; ilat = 40; ilon = 90
            
            for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
                # ialltime = 'ann'
                if (ialltime != 'am'):
                    dO18_q = dO18_q_alltime[expid[i]][ialltime][-1, iplev, ilat, ilon]
                    dD_q = dD_q_alltime[expid[i]][ialltime][-1, iplev, ilat, ilon]
                    d_ln_q = d_ln_q_alltime[expid[i]][ialltime][-1, iplev, ilat, ilon].values
                else:
                    dO18_q = dO18_q_alltime[expid[i]][ialltime][iplev, ilat, ilon]
                    dD_q = dD_q_alltime[expid[i]][ialltime][iplev, ilat, ilon]
                    d_ln_q = d_ln_q_alltime[expid[i]][ialltime][iplev, ilat, ilon]
                
                d_ln_new = (1000 * np.log(1 + dD_q / 1000) - \
                    8.47 * 1000 * np.log(1 + dO18_q / 1000) + \
                        0.0285 * (1000 * np.log(1 + dO18_q / 1000)) ** 2).values
                
                # print(np.round(d_ln_q, 2))
                # print(np.round(d_ln_new, 2))
                if (((d_ln_q - d_ln_new) / d_ln_q) > 0.001):
                    print(d_ln_q)
                    print(d_ln_new)





'''
# endregion
# -----------------------------------------------------------------------------


# Zonal mean
# -----------------------------------------------------------------------------
# region import data


dO18_zm_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_zm_q_alltime.pkl', 'rb') as f:
    dO18_zm_q_alltime[expid[i]] = pickle.load(f)

dD_zm_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_zm_q_alltime.pkl', 'rb') as f:
    dD_zm_q_alltime[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_xs_zm

d_excess_zm_q_alltime = {}
d_excess_zm_q_alltime[expid[i]] = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    d_excess_zm_q_alltime[expid[i]][ialltime] = \
        dD_zm_q_alltime[expid[i]][ialltime] - 8 * dO18_zm_q_alltime[expid[i]][ialltime]

#-------- monthly without monthly mean
d_excess_zm_q_alltime[expid[i]]['mon no mm'] = (d_excess_zm_q_alltime[expid[i]]['mon'].groupby('time.month') - d_excess_zm_q_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_excess_zm_q_alltime[expid[i]]['ann no am'] = (d_excess_zm_q_alltime[expid[i]]['ann'] - d_excess_zm_q_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_zm_q_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_excess_zm_q_alltime[expid[i]], f)



'''
#-------------------------------- check

d_excess_zm_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_zm_q_alltime.pkl', 'rb') as f:
    d_excess_zm_q_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

itime = -1
iplev = 0
ilat = 40

aa = dD_zm_q_alltime[expid[i]][ialltime][itime, iplev, ilat,].values
bb = dO18_zm_q_alltime[expid[i]][ialltime][itime, iplev, ilat,].values
cc = d_excess_zm_q_alltime[expid[i]][ialltime][itime, iplev, ilat,].values

aa - 8 * bb
cc
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_ln_zm

d_ln_zm_q_alltime = {}
d_ln_zm_q_alltime[expid[i]] = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    # ialltime = 'sm'
    
    ln_dD = 1000 * np.log(1 + dD_zm_q_alltime[expid[i]][ialltime] / 1000)
    ln_d18O = 1000 * np.log(1 + dO18_zm_q_alltime[expid[i]][ialltime] / 1000)
    
    d_ln_zm_q_alltime[expid[i]][ialltime] = \
        ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

#-------- monthly without monthly mean
d_ln_zm_q_alltime[expid[i]]['mon no mm'] = (d_ln_zm_q_alltime[expid[i]]['mon'].groupby('time.month') - d_ln_zm_q_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_ln_zm_q_alltime[expid[i]]['ann no am'] = (d_ln_zm_q_alltime[expid[i]]['ann'] - d_ln_zm_q_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_zm_q_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_ln_zm_q_alltime[expid[i]], f)



'''
#-------------------------------- check

d_ln_zm_q_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_zm_q_alltime.pkl', 'rb') as f:
    d_ln_zm_q_alltime[expid[i]] = pickle.load(f)

dO18_zm_q_alltime = {}
dD_zm_q_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_zm_q_alltime.pkl', 'rb') as f:
        dO18_zm_q_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_zm_q_alltime.pkl', 'rb') as f:
        dD_zm_q_alltime[expid[i]] = pickle.load(f)

iplev = 0

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    for ilat in np.arange(1, 96, 30):
            # i = 0; ilat = 40; ilon = 90
            
            for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
                # ialltime = 'ann'
                if (ialltime != 'am'):
                    dO18_q = dO18_zm_q_alltime[expid[i]][ialltime][-1, iplev, ilat,]
                    dD_q = dD_zm_q_alltime[expid[i]][ialltime][-1, iplev, ilat,]
                    d_ln_q = d_ln_zm_q_alltime[expid[i]][ialltime][-1, iplev, ilat,].values
                else:
                    dO18_q = dO18_zm_q_alltime[expid[i]][ialltime][iplev, ilat,]
                    dD_q = dD_zm_q_alltime[expid[i]][ialltime][iplev, ilat,]
                    d_ln_q = d_ln_zm_q_alltime[expid[i]][ialltime][iplev, ilat,]
                
                d_ln_new = (1000 * np.log(1 + dD_q / 1000) - \
                    8.47 * 1000 * np.log(1 + dO18_q / 1000) + \
                        0.0285 * (1000 * np.log(1 + dO18_q / 1000)) ** 2).values
                
                print(np.round(d_ln_q, 2))
                print(np.round(d_ln_new, 2))
                if (((d_ln_q - d_ln_new) / d_ln_q) > 0.001):
                    print(d_ln_q)
                    print(d_ln_new)





'''
# endregion
# -----------------------------------------------------------------------------




