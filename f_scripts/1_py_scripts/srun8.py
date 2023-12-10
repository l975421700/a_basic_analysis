

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


#SBATCH --time=00:30:00


exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'nudged_701_5.0',
    'nudged_703_6.0_k52',
    # 'nudged_705_6.0',
    ]
i = 0

ifile_start = 0 #12 #0 #120
ifile_end   = 528 #528 #516 #1740 #840

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0,  3, 0]

# var_name  = 'sst'
# itag      = 7
# min_sf    = 268.15
# max_sf    = 318.15

# var_name  = 'lat'
# itag      = 5
# min_sf    = -90
# max_sf    = 90

# var_name  = 'rh2m'
# itag      = 8
# min_sf    = 0
# max_sf    = 1.6

# var_name  = 'wind10'
# itag      = 9
# min_sf    = 0
# max_sf    = 28

# var_name  = 'sinlon'
# itag      = 11
# min_sf    = -1
# max_sf    = 1

var_name  = 'coslon'
itag      = 12
min_sf    = -1
max_sf    = 1


# var_name  = 'RHsst'
# itag      = 14
# min_sf    = 0
# max_sf    = 1.4

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
import os

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle
from scipy import stats

from a_basic_analysis.b_module.source_properties import (
    source_properties,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region set indices

kwiso2 = 0

kstart = kwiso2 + sum(ntags[:itag])

str_ind1 = str(kstart + 2)
str_ind2 = str(kstart + 3)

if (len(str_ind1) == 1):
    str_ind1 = '0' + str_ind1
if (len(str_ind2) == 1):
    str_ind2 = '0' + str_ind2

print(kstart); print(str_ind1); print(str_ind2)


'''
exp_out_wiso_q_1m['q_' + str_ind1]
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
exp_org_o[expid[i]]['wiso_q_6h_sfc'] = xr.open_mfdataset(
    filenames_wiso_q_6h_sfc[ifile_start:ifile_end],
    )



'''

ncfile1 = xr.open_dataset('/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/unknown/nudged_703_6.0_k52_199906.01_wiso.nc')
ncfile2 = xr.open_dataset('/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/unknown/nudged_703_6.0_k52_199906.01_wiso_q_6h_org.nc')

ncfile2['time'] = ncfile1['time']

ncfile2.to_netcdf('/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/unknown/nudged_703_6.0_k52_199906.01_wiso_q_6h.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate atmospheric source var


#-------- aggregate atmospheric water

ocean_q = (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str_ind1] + \
    exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str_ind2] + \
        exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str_ind1] + \
            exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str_ind2] + \
                exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str_ind1] + \
                    exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str_ind2]
        ).sel(lev=47).compute()

var_scaled_q = (exp_org_o[expid[i]]['wiso_q_6h_sfc']['q_' + str_ind1] + \
    exp_org_o[expid[i]]['wiso_q_6h_sfc']['xl_' + str_ind1] + \
        exp_org_o[expid[i]]['wiso_q_6h_sfc']['xi_' + str_ind1]
        ).sel(lev=47).compute()


#-------- mon_sea_ann

ocean_q_alltime = mon_sea_ann(var_6hourly=ocean_q)
var_scaled_q_alltime = mon_sea_ann(var_6hourly=var_scaled_q)

#-------- q-weighted var

q_sfc_weighted_var = {}

for ialltime in ['6h', 'daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    q_sfc_weighted_var[ialltime] = source_properties(
        var_scaled_q_alltime[ialltime],
        ocean_q_alltime[ialltime],
        min_sf, max_sf,
        var_name,
        prefix = 'q_sfc_weighted_', threshold = 0,
    )

#-------- monthly without monthly mean
q_sfc_weighted_var['mon no mm'] = (q_sfc_weighted_var['mon'].groupby('time.month') - q_sfc_weighted_var['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
q_sfc_weighted_var['ann no am'] = (q_sfc_weighted_var['ann'] - q_sfc_weighted_var['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + var_name + '.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_sfc_weighted_var, f)




'''
#-------------------------------- check calculation of q_weighted_var

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_sfc_weighted_' + var_name + '.pkl',
          'rb') as f:
    q_sfc_weighted_var = pickle.load(f)

filenames_wiso_q_6h_sfc = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_6h_sfc.nc'))
ifile = -1
ncfile2 = xr.open_dataset(filenames_wiso_q_6h_sfc[ifile_start:ifile_end][ifile])

ocean_q = (ncfile2['q_' + str_ind1] + \
    ncfile2['q_' + str_ind2] + \
        ncfile2['xl_' + str_ind1] + \
            ncfile2['xl_' + str_ind2] + \
                ncfile2['xi_' + str_ind1] + \
                    ncfile2['xi_' + str_ind2]
        ).sel(lev=47).compute()
var_scaled_q = (ncfile2['q_' + str_ind1] + \
    ncfile2['xl_' + str_ind1] + \
        ncfile2['xi_' + str_ind1]
        ).sel(lev=47).compute()

data1 = q_sfc_weighted_var['6h'][-124:].values
data2 = source_properties(
    var_scaled_q,
    ocean_q,
    min_sf, max_sf,
    var_name,
    prefix = 'q_sfc_weighted_', threshold = 0,
).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = q_sfc_weighted_var['mon'][-1].values
data2 = source_properties(
    var_scaled_q.resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False),
    ocean_q.resample({'time': '1d'}).mean(skipna=False).resample({'time': '1M'}).mean(skipna=False),
    min_sf, max_sf,
    var_name,
    prefix = 'q_sfc_weighted_', threshold = 0,
).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


'''
# endregion
# -----------------------------------------------------------------------------




