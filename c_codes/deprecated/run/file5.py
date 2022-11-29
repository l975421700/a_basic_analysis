

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
i = 0
ifile_start = 120
ifile_end   = 720 # 1080

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

var_name  = 'sinlon'
itag      = 11
min_sf    = -1
max_sf    = 1

# var_name  = 'coslon'
# itag      = 12
# min_sf    = -1
# max_sf    = 1

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
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin.pkl',
    'rb') as f:
    wisoaprt_mask_bin = pickle.load(f)


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
# region calculate epe source var

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

epe_var_scaled_pre = {}
epe_ocean_pre = {}
epe_var_scaled_pre_alltime = {}
epe_ocean_pre_alltime = {}
epe_weighted_var = {}

for iqtl in quantiles_bin.keys():
    print(iqtl)
    
    epe_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
        wisoaprt_mask_bin[iqtl],
        other=0,
    )
    epe_ocean_pre[iqtl] = ocean_pre.copy().where(
        wisoaprt_mask_bin[iqtl],
        other=0,
    )
    
    #-------- mon_sea_ann values
    epe_var_scaled_pre_alltime[iqtl] = mon_sea_ann(epe_var_scaled_pre[iqtl])
    epe_ocean_pre_alltime[iqtl]      = mon_sea_ann(epe_ocean_pre[iqtl])
    
    
    #-------------------------------- pre-weighted var
    
    epe_weighted_var[iqtl] = {}
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
        print(ialltime)
        epe_weighted_var[iqtl][ialltime] = source_properties(
            epe_var_scaled_pre_alltime[iqtl][ialltime],
            epe_ocean_pre_alltime[iqtl][ialltime],
            min_sf, max_sf,
            var_name, prefix = 'epe_weighted_',
        )
    
    del epe_var_scaled_pre[iqtl], epe_ocean_pre[iqtl], epe_var_scaled_pre_alltime[iqtl], epe_ocean_pre_alltime[iqtl]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_weighted_' + var_name + '_binned.pkl',
          'wb') as f:
    pickle.dump(epe_weighted_var, f)


'''
#-------------------------------- check

var_names = ['sst', 'lat', 'rh2m', 'wind10', 'sinlon', 'coslon']
itags     = [7, 5, 8, 9, 11, 12]
min_sfs   = [268.15, -90, 0, 0, -1, -1]
max_sfs   = [318.15, 90, 1.6, 28, 1, 1]


for ivar in range(6):
    # ivar = 0
    print(var_names[ivar] + ': ' + str(itags[ivar]) + ': ' + \
        str(min_sfs[ivar]) + ': ' + str(max_sfs[ivar]))
    
    kwiso2 = 3
    
    kstart = kwiso2 + sum(ntags[:itags[ivar]])
    kend   = kwiso2 + sum(ntags[:(itags[ivar]+1)])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.epe_weighted_' + var_names[ivar] + '.pkl',
          'rb') as f:
        epe_weighted_var = pickle.load(f)
    
    ocean_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))
            ).sum(dim='wisotype').compute()
    var_scaled_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=kstart+2) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=kstart+2)).compute()
    
    epe_var_scaled_pre = {}
    epe_ocean_pre = {}
    epe_var_scaled_pre_alltime = {}
    epe_ocean_pre_alltime = {}
    
    for iqtl in ['90%']: # quantiles.keys():
        # iqtl = '90%'
        print(iqtl)
        
        epe_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
            wisoaprt_epe[expid[i]]['mask'][iqtl],
            other=0,
        )
        epe_ocean_pre[iqtl] = ocean_pre.copy().where(
            wisoaprt_epe[expid[i]]['mask'][iqtl],
            other=0,
        )
        
        #---- check
        print((epe_var_scaled_pre[iqtl].values[wisoaprt_epe[expid[i]]['mask'][iqtl]] == var_scaled_pre.values[wisoaprt_epe[expid[i]]['mask'][iqtl]]).all())
        print((epe_var_scaled_pre[iqtl].values[wisoaprt_epe[expid[i]]['mask'][iqtl] == False] == 0).all())
        print((epe_ocean_pre[iqtl].values[wisoaprt_epe[expid[i]]['mask'][iqtl]] == ocean_pre.values[wisoaprt_epe[expid[i]]['mask'][iqtl]]).all())
        print((epe_ocean_pre[iqtl].values[wisoaprt_epe[expid[i]]['mask'][iqtl] == False] == 0).all())
        
        #-------- mon_sea_ann values
        epe_var_scaled_pre_alltime[iqtl] = mon_sea_ann(epe_var_scaled_pre[iqtl])
        epe_ocean_pre_alltime[iqtl]      = mon_sea_ann(epe_ocean_pre[iqtl])
        
        #---- check
        print((epe_var_scaled_pre_alltime[iqtl]['daily'] == epe_var_scaled_pre[iqtl]).all().values)
        print((epe_ocean_pre_alltime[iqtl]['daily'] == epe_ocean_pre[iqtl]).all().values)
        
        for ialltime in epe_ocean_pre_alltime[iqtl].keys():
            # ialltime = 'am'
            print(ialltime)
            res01 = source_properties(
                epe_var_scaled_pre_alltime[iqtl][ialltime],
                epe_ocean_pre_alltime[iqtl][ialltime],
                min_sfs[ivar], max_sfs[ivar],
                var_names[ivar], prefix = 'epe_weighted_',
            ).values
            
            res02 = epe_weighted_var[iqtl][ialltime].values
            
            print((res01[np.isfinite(res01)] == res02[np.isfinite(res02)]).all())
            # print(np.nanmax(abs((res01 - res02) / res01)))
            
    del epe_weighted_var





import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))
'''
# endregion
# -----------------------------------------------------------------------------

