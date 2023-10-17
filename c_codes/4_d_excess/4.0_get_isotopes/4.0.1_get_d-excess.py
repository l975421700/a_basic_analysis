

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    'nudged_701_5.0',
    ]
i = 0


# -----------------------------------------------------------------------------
# region import packages

# management
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
import os

# data analysis
import numpy as np
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_xs

d_excess_alltime = {}
d_excess_alltime[expid[i]] = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    d_excess_alltime[expid[i]][ialltime] = \
        dD_alltime[expid[i]][ialltime] - 8 * dO18_alltime[expid[i]][ialltime]

#-------- monthly without monthly mean
d_excess_alltime[expid[i]]['mon no mm'] = (d_excess_alltime[expid[i]]['mon'].groupby('time.month') - d_excess_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_excess_alltime[expid[i]]['ann no am'] = (d_excess_alltime[expid[i]]['ann'] - d_excess_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_excess_alltime[expid[i]], f)


'''
#-------------------------------- check

d_excess_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    d_excess_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

itime = -1
ilat = 40
ilon = 90

aa = dD_alltime[expid[i]][ialltime][itime, ilat, ilon].values
bb = dO18_alltime[expid[i]][ialltime][itime, ilat, ilon].values
cc = d_excess_alltime[expid[i]][ialltime][itime, ilat, ilon].values

print(aa - 8 * bb)
print(cc)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_ln

d_ln_alltime = {}
d_ln_alltime[expid[i]] = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    # ialltime = 'sm'
    
    ln_dD = 1000 * np.log(1 + dD_alltime[expid[i]][ialltime] / 1000)
    ln_d18O = 1000 * np.log(1 + dO18_alltime[expid[i]][ialltime] / 1000)
    
    d_ln_alltime[expid[i]][ialltime] = \
        ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

#-------- monthly without monthly mean
d_ln_alltime[expid[i]]['mon no mm'] = (d_ln_alltime[expid[i]]['mon'].groupby('time.month') - d_ln_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
d_ln_alltime[expid[i]]['ann no am'] = (d_ln_alltime[expid[i]]['ann'] - d_ln_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(d_ln_alltime[expid[i]], f)


'''
#-------------------------------- check

d_ln_alltime = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    d_ln_alltime[expid[i]] = pickle.load(f)

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    for ilat in np.arange(1, 96, 30):
        for ilon in np.arange(1, 192, 60):
            # i = 0; ilat = 40; ilon = 90
            
            for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm', 'am']:
                # ialltime = 'ann'
                if (ialltime != 'am'):
                    dO18 = dO18_alltime[expid[i]][ialltime][-1, ilat, ilon]
                    dD = dD_alltime[expid[i]][ialltime][-1, ilat, ilon]
                    d_ln = d_ln_alltime[expid[i]][ialltime][-1, ilat, ilon].values
                else:
                    dO18 = dO18_alltime[expid[i]][ialltime][ilat, ilon]
                    dD = dD_alltime[expid[i]][ialltime][ilat, ilon]
                    d_ln = d_ln_alltime[expid[i]][ialltime][ilat, ilon]
                
                d_ln_new = (1000 * np.log(1 + dD / 1000) - \
                    8.47 * 1000 * np.log(1 + dO18 / 1000) + \
                        0.0285 * (1000 * np.log(1 + dO18 / 1000)) ** 2).values
                
                # print(np.round(d_ln, 2))
                # print(np.round(d_ln_new, 2))
                if (((d_ln - d_ln_new) / d_ln) > 0.000001):
                    print(d_ln)
                    print(d_ln_new)

'''
# endregion
# -----------------------------------------------------------------------------

