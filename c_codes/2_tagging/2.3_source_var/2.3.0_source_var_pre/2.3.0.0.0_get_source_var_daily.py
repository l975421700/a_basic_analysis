

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_502_5.0',]
i=0

output_dir = exp_odir + expid[i] + '/analysis/echam/'

ifile_start = 120
ifile_end   = 720

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]

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


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle

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
# region import data

fl_wiso_daily = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'
        ))

exp_out_wiso_daily = xr.open_mfdataset(
    fl_wiso_daily[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region set indices

kwiso2 = 3

kstart = kwiso2 + sum(ntags[:itag])
kend   = kwiso2 + sum(ntags[:(itag+1)])

print(kstart); print(kend)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source var

#-------------------------------- precipitation

ocean_pre = (
    exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) + \
        exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))
        ).sum(dim='wisotype').compute()
var_scaled_pre = (
    exp_out_wiso_daily.wisoaprl.sel(wisotype=kstart+2) + \
        exp_out_wiso_daily.wisoaprc.sel(wisotype=kstart+2)).compute()

var_scaled_pre.values[ocean_pre.values < 2e-8] = 0
ocean_pre.values[ocean_pre.values < 2e-8] = 0


#-------- monthly/seasonal/annual (mean) values

ocean_pre_alltime      = mon_sea_ann(ocean_pre)
var_scaled_pre_alltime = mon_sea_ann(var_scaled_pre)

#-------------------------------- pre-weighted var

pre_weighted_var = {}

for ialltime in ocean_pre_alltime.keys():
    print(ialltime)
    
    pre_weighted_var[ialltime] = source_properties(
        var_scaled_pre_alltime[ialltime],
        ocean_pre_alltime[ialltime],
        min_sf, max_sf,
        var_name,
    )

with open(output_dir + expid[i] + '.pre_weighted_' + var_name + '.pkl',
          'wb') as f:
    pickle.dump(pre_weighted_var, f)


'''

#-------- other checks
((exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))).sum(dim='wisotype').compute() == (exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) +  exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3)).values).sum(dim='wisotype').compute()).all()

print(np.max(var_scaled_pre.values[ocean_pre.values < 2e-8]))
print(np.max(ocean_pre.values[ocean_pre.values < 2e-8]))


#-------- import data
pre_weighted_lat = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.pkl', 'rb') as f:
    pre_weighted_lat[expid[i]] = pickle.load(f)

#-------- check consistency

# with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_pre_alltime.pkl', 'wb') as f:
#     pickle.dump(ocean_pre_alltime, f)


'''
# endregion
# -----------------------------------------------------------------------------


