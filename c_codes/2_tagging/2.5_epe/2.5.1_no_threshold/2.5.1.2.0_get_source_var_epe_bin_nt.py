

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
i = 0
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

print(var_name)
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

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_nt.pkl',
    'rb') as f:
    wisoaprt_mask_bin_nt = pickle.load(f)


quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

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
# region calculate epe_nt source var

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

epe_nt_var_scaled_pre = {}
epe_nt_ocean_pre = {}
epe_nt_var_scaled_pre_alltime = {}
epe_nt_ocean_pre_alltime = {}
epe_nt_weighted_var = {}

for iqtl in quantiles_bin.keys():
    print(iqtl)
    
    epe_nt_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
        wisoaprt_mask_bin_nt[iqtl],
        other=0,
    )
    epe_nt_ocean_pre[iqtl] = ocean_pre.copy().where(
        wisoaprt_mask_bin_nt[iqtl],
        other=0,
    )
    
    #-------- mon_sea_ann values
    epe_nt_var_scaled_pre_alltime[iqtl] = mon_sea_ann(epe_nt_var_scaled_pre[iqtl])
    epe_nt_ocean_pre_alltime[iqtl]      = mon_sea_ann(epe_nt_ocean_pre[iqtl])
    
    
    #-------------------------------- pre-weighted var
    
    epe_nt_weighted_var[iqtl] = {}
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
        print(ialltime)
        epe_nt_weighted_var[iqtl][ialltime] = source_properties(
            epe_nt_var_scaled_pre_alltime[iqtl][ialltime],
            epe_nt_ocean_pre_alltime[iqtl][ialltime],
            min_sf, max_sf,
            var_name, prefix = 'epe_nt_weighted_', threshold = 0,
        )
    
    del epe_nt_var_scaled_pre[iqtl], epe_nt_ocean_pre[iqtl], epe_nt_var_scaled_pre_alltime[iqtl], epe_nt_ocean_pre_alltime[iqtl]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_nt_weighted_' + var_name + '_binned.pkl',
          'wb') as f:
    pickle.dump(epe_nt_weighted_var, f)


'''
#-------------------------------- check

var_names = ['sst', 'lat', 'rh2m', 'wind10', 'sinlon', 'coslon']
itags     = [7, 5, 8, 9, 11, 12]
min_sfs   = [268.15, -90, 0, 0, -1, -1]
max_sfs   = [318.15, 90, 1.6, 28, 1, 1]

fl_wiso_daily = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'
        ))

for ivar in np.arange(2, 6, 1):
    # ivar = 1
    print(var_names[ivar] + ': ' + str(itags[ivar]) + ', ' + \
        str(min_sfs[ivar]) + ' to ' + str(max_sfs[ivar]))
    
    kwiso2 = 3
    kstart = kwiso2 + sum(ntags[:itags[ivar]])
    
    exp_out_wiso_daily = xr.open_mfdataset(
        fl_wiso_daily[ifile_start:ifile_end],
        data_vars='minimal', coords='minimal', parallel=True)
    
    ocean_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))
            ).sum(dim='wisotype').compute()
    var_scaled_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=kstart+2) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=kstart+2)).compute()
    
    var_scaled_pre.values[ocean_pre.values < 2e-8] = 0
    ocean_pre.values[ocean_pre.values < 2e-8] = 0
    
    del exp_out_wiso_daily
    
    with open(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_nt.pkl',
        'rb') as f:
        wisoaprt_mask_bin_nt = pickle.load(f)
    
    epe_nt_var_scaled_pre = {}
    epe_nt_ocean_pre = {}
    epe_nt_var_scaled_pre_alltime = {}
    epe_nt_ocean_pre_alltime = {}
    
    for iqtl in ['90.5%']: # quantiles_bin.keys():
        # iqtl = '90.5%'
        print(iqtl)
        
        epe_nt_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
            wisoaprt_mask_bin_nt[iqtl],
            other=0,
        )
        epe_nt_ocean_pre[iqtl] = ocean_pre.copy().where(
            wisoaprt_mask_bin_nt[iqtl],
            other=0,
        )
        
        #---- check
        print((epe_nt_var_scaled_pre[iqtl].values[wisoaprt_mask_bin_nt[iqtl]] == var_scaled_pre.values[wisoaprt_mask_bin_nt[iqtl]]).all())
        print((epe_nt_var_scaled_pre[iqtl].values[wisoaprt_mask_bin_nt[iqtl] == False] == 0).all())
        print((epe_nt_ocean_pre[iqtl].values[wisoaprt_mask_bin_nt[iqtl]] == ocean_pre.values[wisoaprt_mask_bin_nt[iqtl]]).all())
        print((epe_nt_ocean_pre[iqtl].values[wisoaprt_mask_bin_nt[iqtl] == False] == 0).all())
        
        del wisoaprt_mask_bin_nt
        
        #-------- mon_sea_ann values
        epe_nt_var_scaled_pre_alltime[iqtl] = mon_sea_ann(epe_nt_var_scaled_pre[iqtl])
        epe_nt_ocean_pre_alltime[iqtl]      = mon_sea_ann(epe_nt_ocean_pre[iqtl])
        
        #---- check
        print((epe_nt_var_scaled_pre_alltime[iqtl]['daily'] == epe_nt_var_scaled_pre[iqtl]).all().values)
        print((epe_nt_ocean_pre_alltime[iqtl]['daily'] == epe_nt_ocean_pre[iqtl]).all().values)
        
        for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
            # ialltime = 'am'
            print(ialltime)
            res01 = source_properties(
                epe_nt_var_scaled_pre_alltime[iqtl][ialltime],
                epe_nt_ocean_pre_alltime[iqtl][ialltime],
                min_sfs[ivar], max_sfs[ivar],
                var_names[ivar], prefix = 'epe_nt_weighted_', threshold = 0,
            ).values
            
            with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_nt_weighted_' + var_names[ivar] + '_binned.pkl',
                      'rb') as f:
                epe_nt_weighted_var = pickle.load(f)
            
            res02 = epe_nt_weighted_var[iqtl][ialltime].values
            
            print((res01[np.isfinite(res01)] == res02[np.isfinite(res02)]).all())
            # print(np.nanmax(abs((res01 - res02) / res01)))
            
    del epe_nt_weighted_var, ocean_pre, var_scaled_pre, epe_nt_var_scaled_pre, epe_nt_ocean_pre, epe_nt_var_scaled_pre_alltime, epe_nt_ocean_pre_alltime, res01, res02





import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))
'''
# endregion
# -----------------------------------------------------------------------------

