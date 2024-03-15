#SBATCH --time=00:30:00


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_703_6.0_k52',
    ]
i = 0

ifile_start = 0 #0 #120
ifile_end   = 528 #1740 #840

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

var_name  = 'sinlon'
itag      = 11
min_sf    = -1
max_sf    = 1

# var_name  = 'coslon'
# itag      = 12
# min_sf    = -1
# max_sf    = 1

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


fl_wiso_q_plev = sorted(glob.glob(
    exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.daily_wiso_q_plev.nc'
        ))

exp_out_wiso_q_plev = xr.open_mfdataset(
    fl_wiso_q_plev[ifile_start:ifile_end],
    )


'''
fl_wiso_q_plev = sorted(glob.glob(
    exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'
        ))


#-------- check with echam output with p level q

fl_wiso_q_plev = sorted(glob.glob(
    exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'
        ))
fl_uvq_plev = sorted(glob.glob(
    exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_uvq_plev.nc'
        ))

ifile = -1
print(fl_uvq_plev[ifile_start:ifile_end][ifile])
print(fl_wiso_q_plev[ifile_start:ifile_end][ifile])
ncfile = xr.open_dataset(fl_uvq_plev[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(fl_wiso_q_plev[ifile_start:ifile_end][ifile])

(ncfile.q.values[np.isfinite(ncfile.q.values)] == ncfile2.q16o.values[np.isfinite(ncfile2.q16o.values)]).all()
np.max(abs(ncfile.q.values[np.isfinite(ncfile.q.values)] - ncfile2.q16o.values[np.isfinite(ncfile2.q16o.values)]))
test = ncfile.q.values[np.isfinite(ncfile.q.values)] - ncfile2.q16o.values[np.isfinite(ncfile2.q16o.values)]
wheremax = np.where(test == np.max(abs(test)))

print(test[wheremax])
print(np.max(abs(test)))
ncfile.q.values[np.isfinite(ncfile.q.values)][wheremax]
ncfile2.q16o.values[np.isfinite(ncfile2.q16o.values)][wheremax]




#-------- check with echam output with model level q

fl_gl_1m = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_gl_1m.nc'
        ))

ifile = -1
fl_wiso_q_1m[ifile_start:ifile_end][ifile]
fl_gl_1m[ifile_start:ifile_end][ifile]
ncfile = xr.open_dataset(fl_gl_1m[ifile_start:ifile_end][ifile])

(ncfile.xi.squeeze().values == exp_out_wiso_q_1m.xi16o[ifile, ].values).all()
# np.max(abs(ncfile.xi.squeeze().values - exp_out_wiso_q_1m.xi16o[ifile, ].values))

ncfile2 = xr.open_dataset(fl_wiso_q_1m[ifile_start:ifile_end][ifile])
(ncfile2.xi_24.squeeze().values == exp_out_wiso_q_1m.xi_24[ifile, ].values).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate atmospheric source var zm


#-------- aggregate atmospheric water

ocean_q_zm = (exp_out_wiso_q_plev['q_' + str_ind1] + \
    exp_out_wiso_q_plev['q_' + str_ind2] + \
        exp_out_wiso_q_plev['xl_' + str_ind1] + \
            exp_out_wiso_q_plev['xl_' + str_ind2] + \
                exp_out_wiso_q_plev['xi_' + str_ind1] + \
                    exp_out_wiso_q_plev['xi_' + str_ind2]
        ).mean(dim='lon').compute()

var_scaled_q_zm = (exp_out_wiso_q_plev['q_' + str_ind1] + \
    exp_out_wiso_q_plev['xl_' + str_ind1] + \
        exp_out_wiso_q_plev['xi_' + str_ind1]
        ).mean(dim='lon').compute()


#-------- mon_sea_ann

ocean_q_zm_alltime = mon_sea_ann(var_daily=ocean_q_zm)
var_scaled_q_zm_alltime = mon_sea_ann(var_daily=var_scaled_q_zm)

#-------- q-weighted var

q_weighted_var_zm = {}

for ialltime in ['daily', 'mon', 'mm', 'sea', 'sm', 'ann', 'am']:
    print(ialltime)
    
    q_weighted_var_zm[ialltime] = source_properties(
        var_scaled_q_zm_alltime[ialltime],
        ocean_q_zm_alltime[ialltime],
        min_sf, max_sf,
        var_name,
        prefix = 'q_weighted_', threshold = 0,
    )

#-------- monthly without monthly mean
q_weighted_var_zm['mon no mm'] = (q_weighted_var_zm['mon'].groupby('time.month') - q_weighted_var_zm['mon'].groupby('time.month').mean(skipna=True)).compute()

#-------- annual without annual mean
q_weighted_var_zm['ann no am'] = (q_weighted_var_zm['ann'] - q_weighted_var_zm['ann'].mean(dim='time', skipna=True)).compute()

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_' + var_name + '_zm.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(q_weighted_var_zm, f)


'''
#-------------------------------- check calculation of q_weighted_var

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.q_weighted_' + var_name + '_zm.pkl',
          'rb') as f:
    q_weighted_var_zm = pickle.load(f)


fl_wiso_q_plev = sorted(glob.glob(
    exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.monthly_wiso_q_plev.nc'
        ))

ifile = -30
print(fl_wiso_q_plev[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(fl_wiso_q_plev[ifile_start:ifile_end][ifile])

ocean_q_zm = (ncfile2['q_' + str_ind1] + \
    ncfile2['q_' + str_ind2] + \
        ncfile2['xl_' + str_ind1] + \
            ncfile2['xl_' + str_ind2] + \
                ncfile2['xi_' + str_ind1] + \
                    ncfile2['xi_' + str_ind2]
        ).mean(dim='lon').compute()
var_scaled_q_zm = (ncfile2['q_' + str_ind1] + \
    ncfile2['xl_' + str_ind1] + \
        ncfile2['xi_' + str_ind1]
        ).mean(dim='lon').compute()

plev = 0
ilat = 45
ilon = 90

sq = var_scaled_q_zm[0, plev, ilat].values
oq = ocean_q_zm[0, plev, ilat].values
q_var_new = (sq / oq) * (max_sf - min_sf) + min_sf

if (var_name == 'sst'):
    q_var_new = q_var_new - 273.15

if (var_name == 'rh2m'):
    q_var_new = q_var_new * 100

q_var = q_weighted_var['mon'][ifile, plev, ilat].values

print(q_var)
print(q_var_new)




stats.describe(ocean_q, axis=None, nan_policy='omit')
stats.describe(
    ocean_q.sel(plev=slice(1e+5, 2e+4)), axis=None, nan_policy='omit')
'''
# endregion
# -----------------------------------------------------------------------------




