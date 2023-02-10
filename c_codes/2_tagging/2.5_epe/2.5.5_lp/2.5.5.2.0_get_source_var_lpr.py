

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

wisoaprt_cum_qtl = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_cum_qtl.pkl',
    'rb') as f:
    wisoaprt_cum_qtl[expid[i]] = pickle.load(f)

# quantile_interval  = np.arange(1, 99 + 1e-4, 1, dtype=np.int64)
quantile_interval  = np.arange(10, 50 + 1e-4, 10, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
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
# region calculate lpr source var

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

lpr_var_scaled_pre = {}
lpr_ocean_pre = {}
lpr_var_scaled_pre_alltime = {}
lpr_ocean_pre_alltime = {}
lpr_weighted_var = {}

for iqtl in quantiles.keys():
    print(iqtl)
    
    lpr_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
        wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False,
        other=0,
    )
    lpr_ocean_pre[iqtl] = ocean_pre.copy().where(
        wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False,
        other=0,
    )
    
    #-------- mon_sea_ann values
    lpr_var_scaled_pre_alltime[iqtl] = mon_sea_ann(lpr_var_scaled_pre[iqtl])
    lpr_ocean_pre_alltime[iqtl]      = mon_sea_ann(lpr_ocean_pre[iqtl])
    
    
    #-------------------------------- pre-weighted var
    
    lpr_weighted_var[iqtl] = {}
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
        print(ialltime)
        lpr_weighted_var[iqtl][ialltime] = source_properties(
            lpr_var_scaled_pre_alltime[iqtl][ialltime],
            lpr_ocean_pre_alltime[iqtl][ialltime],
            min_sf, max_sf,
            var_name, prefix = 'lpr_weighted_', threshold = 0,
        )
    
    del lpr_var_scaled_pre[iqtl], lpr_ocean_pre[iqtl], lpr_var_scaled_pre_alltime[iqtl], lpr_ocean_pre_alltime[iqtl]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lpr_weighted_' + var_name + '.pkl',
          'wb') as f:
    pickle.dump(lpr_weighted_var, f)


'''
# 22 min to run
#SBATCH --time=00:30:00
#SBATCH --partition=fat

'''
#-------------------------------- check

var_names = ['sst', 'lat', 'rh2m', 'wind10', 'sinlon', 'coslon']
itags     = [7, 5, 8, 9, 11, 12]
min_sfs   = [268.15, -90, 0, 0, -1, -1]
max_sfs   = [318.15, 90, 1.6, 28, 1, 1]


for ivar in np.arange(1, 6, 1):
    # ivar = 0
    print(var_names[ivar] + ': ' + str(itags[ivar]) + ': ' + \
        str(min_sfs[ivar]) + ': ' + str(max_sfs[ivar]))
    
    kwiso2 = 3
    
    kstart = kwiso2 + sum(ntags[:itags[ivar]])
    kend   = kwiso2 + sum(ntags[:(itags[ivar]+1)])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.lpr_weighted_' + var_names[ivar] + '.pkl',
          'rb') as f:
        lpr_weighted_var = pickle.load(f)
    
    ocean_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kstart+3)) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kstart+3))
            ).sum(dim='wisotype').compute()
    var_scaled_pre = (
        exp_out_wiso_daily.wisoaprl.sel(wisotype=kstart+2) + \
            exp_out_wiso_daily.wisoaprc.sel(wisotype=kstart+2)).compute()
    
    var_scaled_pre.values[ocean_pre.values < 2e-8] = 0
    ocean_pre.values[ocean_pre.values < 2e-8] = 0
    
    lpr_var_scaled_pre = {}
    lpr_ocean_pre = {}
    lpr_var_scaled_pre_alltime = {}
    lpr_ocean_pre_alltime = {}
    
    for iqtl in ['10%']: # quantiles.keys():
        # iqtl = '10%'
        print(iqtl)
        
        lpr_var_scaled_pre[iqtl] = var_scaled_pre.copy().where(
            wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False,
            other=0,
        )
        lpr_ocean_pre[iqtl] = ocean_pre.copy().where(
            wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False,
            other=0,
        )
        
        #---- check
        print((lpr_var_scaled_pre[iqtl].values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False] == var_scaled_pre.values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False]).all())
        print((lpr_var_scaled_pre[iqtl].values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl]] == 0).all())
        print((lpr_ocean_pre[iqtl].values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False] == ocean_pre.values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl] == False]).all())
        print((lpr_ocean_pre[iqtl].values[wisoaprt_cum_qtl[expid[i]]['mask'][iqtl]] == 0).all())
        
        #-------- mon_sea_ann values
        lpr_var_scaled_pre_alltime[iqtl] = mon_sea_ann(lpr_var_scaled_pre[iqtl])
        lpr_ocean_pre_alltime[iqtl]      = mon_sea_ann(lpr_ocean_pre[iqtl])
        
        #---- check
        print((lpr_var_scaled_pre_alltime[iqtl]['daily'] == lpr_var_scaled_pre[iqtl]).all().values)
        print((lpr_ocean_pre_alltime[iqtl]['daily'] == lpr_ocean_pre[iqtl]).all().values)
        
        for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
            # ialltime = 'am'
            print(ialltime)
            res01 = source_properties(
                lpr_var_scaled_pre_alltime[iqtl][ialltime],
                lpr_ocean_pre_alltime[iqtl][ialltime],
                min_sfs[ivar], max_sfs[ivar],
                var_names[ivar], prefix = 'lpr_weighted_', threshold = 0,
            ).values
            
            res02 = lpr_weighted_var[iqtl][ialltime].values
            
            print((res01[np.isfinite(res01)] == res02[np.isfinite(res02)]).all())
            # print(np.nanmax(abs((res01 - res02) / res01)))
            
    del lpr_weighted_var





import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))
# endregion
# -----------------------------------------------------------------------------

