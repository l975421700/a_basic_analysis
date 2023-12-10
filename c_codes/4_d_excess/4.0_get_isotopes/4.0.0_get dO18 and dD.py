

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


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
    # 'nudged_701_5.0',
    
    'nudged_703_6.0_k52',
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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

# setting
VSMOW_O18 = 0.22279967
VSMOW_D   = 0.3288266

# threshold to calculate dO18 and dD
wiso_calc_min = 0.05 / 2.628e6

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann dO18

dO18_alltime = {}
dO18_alltime[expid[i]] = {}

for ialltime in ['6h', 'daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'am'
    # ialltime = 'daily'
    print(ialltime)
    
    dO18_alltime[expid[i]][ialltime] = \
        (((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=2) / \
            wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)) / \
                VSMOW_O18 - 1) * 1000).compute()
    
    dO18_alltime[expid[i]][ialltime].values[
        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min
        ] = np.nan
    
    dO18_alltime[expid[i]][ialltime] = \
        dO18_alltime[expid[i]][ialltime].rename('dO18')

#-------- monthly without monthly mean
dO18_alltime[expid[i]]['mon no mm'] = (dO18_alltime[expid[i]]['mon'].groupby('time.month') - dO18_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
dO18_alltime[expid[i]]['ann no am'] = (dO18_alltime[expid[i]]['ann'] - dO18_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(dO18_alltime[expid[i]], f)


'''
#-------------------------------- check
dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

ialltime = 'ann'

ccc = dO18_alltime[expid[i]][ialltime].values
ddd = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=2) / wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) / VSMOW_O18 - 1) * 1000).compute().values

ddd[wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min] = np.nan

(ccc[np.isfinite(ccc)] == ddd[np.isfinite(ddd)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann dD

dD_alltime = {}
dD_alltime[expid[i]] = {}

for ialltime in ['6h', 'daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    # ialltime = 'am'
    # ialltime = 'daily'
    print(ialltime)
    
    dD_alltime[expid[i]][ialltime] = \
        (((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=3) / \
            wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)) / \
                VSMOW_D - 1) * 1000).compute()
    
    dD_alltime[expid[i]][ialltime].values[
        wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min
        ] = np.nan
    
    dD_alltime[expid[i]][ialltime] = \
        dD_alltime[expid[i]][ialltime].rename('dD')

#-------- monthly without monthly mean
dD_alltime[expid[i]]['mon no mm'] = (dD_alltime[expid[i]]['mon'].groupby('time.month') - dD_alltime[expid[i]]['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
dD_alltime[expid[i]]['ann no am'] = (dD_alltime[expid[i]]['ann'] - dD_alltime[expid[i]]['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(dD_alltime[expid[i]], f)




'''
#-------------------------------- check
dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

ccc = dD_alltime[expid[i]][ialltime].values
ddd = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=3) / wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1) / VSMOW_D - 1) * 1000).compute().values

ddd[wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).values < wiso_calc_min] = np.nan

(ccc[np.isfinite(ccc)] == ddd[np.isfinite(ddd)]).all()


'''
# endregion
# -----------------------------------------------------------------------------

