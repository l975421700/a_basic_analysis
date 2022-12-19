

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    find_ilat_ilon,
    regrid,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region define functions


def combined_preprocessing(ds_in):
    
    ds=ds_in.copy()
    ds=cpp.rename_cmip6(ds)
    ds=cpp.broadcast_lonlat(ds)
    # ds=cpp.correct_lon(ds)
    # ds=cpp.parse_lon_lat_bounds(ds)
    # ds=cpp.maybe_convert_vertex_to_bounds(ds)
    # ds=cpp.maybe_convert_bounds_to_vertex(ds)
    
    return ds


def get_var_LIG(var):
    var_dic={}
    for model in models:
        print(model)
        try:
            files=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/'+var+'/'+var+'_Omon_'+model+'_lig127k_*.nc')
            if not files:
                print(model+' LIG data not avaialbe')
                continue
            if any("_gr_" in filename for filename in files):
                print('LIG of '+model+' in native grid')
            if not any("r1i1p1f1" in filename for filename in files):
                index=files[0].index('_lig127k_')+9
                ens_name=files[0][index:index+9]
                print('LIG of '+model+' ensemble is '+ens_name)
            
            ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
            var_dic[model]=combined_preprocessing(
                ds.isel(time=slice(-1200,None)))
        except OSError as err:
            print('LIG of '+model+' not readable' , err)
            continue
        
    return var_dic


def get_var_PI(var):
    var_dic={}
    for model in models:
        print(model)
        files_LIG=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/'+var+'/'+var+'_Omon_'+model+'_lig127k_*.nc')
        try:
            index=files_LIG[0].index('_lig127k_')+9
            ens=files_LIG[0][index:index+8]
        except:
            print(model+'no LIG, trying r1i1p1f1')
            ens='r1i1p1f1'
        try:
            files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_Omon_'+model+'_piControl_'+ens+'*.nc')
            if not files:
                print(model+' PI data ensemble is not same as LIG')
                files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_Omon_'+model+'_piControl_*.nc')
                if not files:
                    print(model+' PI data not avaialbe')
                    continue
            if any("_gr_" in filename for filename in files):
                print('PI of '+model+' in native grid')
            # if not any("r1i1p1f1" in filename for filename in files):
            #     index=files[0].index('_piControl_')+11
            #     ens_name=files[0][index:index+8]
            #     print('PI of '+model+' ensemble is '+ens_name)
            ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
            # ds = cpp.parse_lon_lat_bounds(ds)
            # ds = cpp.maybe_convert_bounds_to_vertex(ds)
            # ds = cpp.maybe_convert_vertex_to_bounds(ds)
            var_dic[model]=combined_preprocessing(
                ds.isel(time=slice(-1200,None)))
        except Exception as err:
            print(err,'PI of '+model+'in CEDA not readable' )
    return var_dic

def regrid_rahul(ds_in,variable):
    var=ds_in[variable]
    ds_out = xe.util.grid_global(1,1)
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear',periodic=True,unmapped_to_nan=True,ignore_degenerate=True,extrap_method='nearest_s2d')
    return regridder(var)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get multiple simulations

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',
    ]
# 'INM-CM4-8',

lig_sst=get_var_LIG('tos')
pi_sst=get_var_PI('tos')

# HadGEM3 from vittorias LIG Simulation
model = 'HadGEM3-GC31-LL'
file='/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/uba937_ThetaSo_18502050.nc'
ds = xr.open_dataset(file)
# change time indices
sst=ds.thetao.isel(time=slice(-1200,None),deptht=0,drop=True)
sst=sst.assign_coords(time = lig_sst['IPSL-CM6A-LR'].time)
# ds = ds.assign_coords(time = lig_sst['IPSL-CM6A-LR'].time)
lig_sst[model]=combined_preprocessing(
    sst.to_dataset().rename(dict(thetao='tos')))
lig_sst[model].tos.values[lig_sst[model].tos.values==9.9692100e+36] = np.nan

with open('scratch/cmip6/lig/sst/lig_sst.pkl', 'wb') as f:
    pickle.dump(lig_sst, f)
with open('scratch/cmip6/lig/sst/pi_sst.pkl', 'wb') as f:
    pickle.dump(pi_sst, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

#---------------- check time length

for imodel in lig_sst.keys():
    print('#-------- ' + imodel)
    print('#---- LIG')
    
    print(len(lig_sst[imodel].time))
    # print(lig_sst[imodel].lon)
    # print(lig_sst[imodel].lat)
    
    print('#---- PI')
    
    print(len(pi_sst[imodel].time))
    # print(pi_sst[imodel].lon)
    # print(pi_sst[imodel].lat)


#---------------- check 'HadGEM3-GC31-LL'

ds=xr.open_dataset('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/uba937_ThetaSo_18502050.nc')

data1 = ds.thetao.isel(time=slice(-1200,None),deptht=0,drop=True).to_dataset().rename(dict(thetao='tos')).tos.values
# (data1 == 9.9692100e+36).sum()
data2 = lig_sst['HadGEM3-GC31-LL'].tos.values
(data1[data1 != 9.9692100e+36] == data2[np.isfinite(data2)]).all()


#---------------- check 'AWI-ESM-1-1-LR'
model = 'AWI-ESM-1-1-LR'
ds = xr.open_mfdataset('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/tos/tos_Omon_'+model+'_lig127k_*.nc')

(ds.tos.values == lig_sst[model].tos.values).all()



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regrid AWI sst

! cdo -remapcon,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/tos/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc

! cdo -remapcon,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/tos/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc

# -remapbil does not work
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded simulations

with open('scratch/cmip6/lig/sst/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

lig_sst_regrid = {}
pi_sst_regrid = {}

models=sorted(lig_sst.keys())

for model in models:
    print(model)
    if (model != 'AWI-ESM-1-1-LR'):
        lig_sst_regrid[model] = regrid(lig_sst[model])
        pi_sst_regrid[model] = regrid(pi_sst[model])
    elif (model == 'AWI-ESM-1-1-LR'):
        # model = 'AWI-ESM-1-1-LR'
        lig_ds = xr.open_dataset('scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
        pi_ds = xr.open_dataset('scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')
        
        lig_ds = combined_preprocessing(lig_ds)
        pi_ds = combined_preprocessing(pi_ds)
        lig_ds['lon'] = lig_ds.lon.transpose()
        lig_ds['lat'] = lig_ds.lat.transpose()
        pi_ds['lon'] = pi_ds.lon.transpose()
        pi_ds['lat'] = pi_ds.lat.transpose()
        
        lig_sst_regrid[model] = lig_ds
        pi_sst_regrid[model] = pi_ds

with open('scratch/cmip6/lig/sst/lig_sst_regrid.pkl', 'wb') as f:
    pickle.dump(lig_sst_regrid, f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid.pkl', 'wb') as f:
    pickle.dump(pi_sst_regrid, f)



'''
#-------------------------------- check
with open('scratch/cmip6/lig/sst/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid.pkl', 'rb') as f:
    pi_sst_regrid = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

model = 'AWI-ESM-1-1-LR'
lig_ds = xr.open_dataset('scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
pi_ds = xr.open_dataset('scratch/cmip6/lig/sst/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')


data1 = lig_ds.tos.values
data2 = lig_sst_regrid[model].tos.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = pi_ds.tos.values
data2 = pi_sst_regrid[model].tos.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---- check lon/lat/tos dimension

for imodel in lig_sst_regrid.keys():
    print('#-------- ' + imodel)
    print('#---- LIG')
    
    print(lig_sst_regrid[imodel].lon.shape)
    # print(lig_sst_regrid[imodel].lat.shape)
    print(lig_sst_regrid[imodel].tos.shape)
    
    print('#---- PI')
    
    print(pi_sst_regrid[imodel].lon.shape)
    # print(pi_sst_regrid[imodel].lat.shape)
    print(pi_sst_regrid[imodel].tos.shape)


#---- check grids of two regridding methods

print((lig_sst_regrid['NorESM2-LM'].lon.values == \
    lig_sst_regrid['AWI-ESM-1-1-LR'].lon.values).all())
print((lig_sst_regrid['NorESM2-LM'].lat.values == \
    lig_sst_regrid['AWI-ESM-1-1-LR'].lat.values).all())

print((pi_sst_regrid['NorESM2-LM'].lon.values == \
    pi_sst_regrid['AWI-ESM-1-1-LR'].lon.values).all())
print((pi_sst_regrid['NorESM2-LM'].lat.values == \
    pi_sst_regrid['AWI-ESM-1-1-LR'].lat.values).all())


#---- check two regridding methods qg vs rh
model = 'NorESM2-LM'
test = regrid(lig_sst[model])
test1 = regrid_rahul(lig_sst[model], 'tos')
print((test.tos.values[np.isfinite(test.tos.values)] == lig_sst_regrid[model].tos.values[np.isfinite(lig_sst_regrid[model].tos.values)]).all())
print((test1.values[np.isfinite(test1.values)] == lig_sst_regrid[model].tos.values[np.isfinite(lig_sst_regrid[model].tos.values)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded sst

with open('scratch/cmip6/lig/sst/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid.pkl', 'rb') as f:
    pi_sst_regrid = pickle.load(f)

models=sorted(lig_sst_regrid.keys())

lig_sst_regrid_alltime = {}
pi_sst_regrid_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    
    lig_sst_regrid_alltime[model] = mon_sea_ann(
        var_monthly = lig_sst_regrid[model].tos, seasons = 'Q-MAR',)
    pi_sst_regrid_alltime[model] = mon_sea_ann(
        var_monthly = pi_sst_regrid[model].tos, seasons = 'Q-MAR',)
    
    lig_sst_regrid_alltime[model]['mm'] = \
        lig_sst_regrid_alltime[model]['mm'].rename({'month': 'time'})
    lig_sst_regrid_alltime[model]['sm'] = \
        lig_sst_regrid_alltime[model]['sm'].rename({'month': 'time'})
    lig_sst_regrid_alltime[model]['am'] = \
        lig_sst_regrid_alltime[model]['am'].expand_dims('time', axis=0)
    
    pi_sst_regrid_alltime[model]['mm'] = \
        pi_sst_regrid_alltime[model]['mm'].rename({'month': 'time'})
    pi_sst_regrid_alltime[model]['sm'] = \
        pi_sst_regrid_alltime[model]['sm'].rename({'month': 'time'})
    pi_sst_regrid_alltime[model]['am'] = \
        pi_sst_regrid_alltime[model]['am'].expand_dims('time', axis=0)


with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sst_regrid_alltime, f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sst_regrid_alltime, f)




'''
with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid.pkl', 'rb') as f:
    pi_sst_regrid = pickle.load(f)

#-------------------------------- check

model = 'CESM2'
data1 = lig_sst_regrid_alltime[model]['mon'].values
data2 = lig_sst_regrid[model].tos.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = pi_sst_regrid_alltime[model]['mon'].values
data2 = pi_sst_regrid[model].tos.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


lig_sst_regrid_alltime[models[0]]['mm'].rename({'month': 'time'})
lig_sst_regrid_alltime[models[0]]['sm'].rename({'month': 'time'})
lig_sst_regrid_alltime[models[0]]['am'].expand_dims('time', axis=0)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region concatenate ensembles

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

models=sorted(lig_sst_regrid_alltime.keys())

pi_sst_regrid_alltime_list = {}
pi_sst_regrid_alltime_ens = {}
lig_sst_regrid_alltime_list = {}
lig_sst_regrid_alltime_ens = {}

for ialltime in lig_sst_regrid_alltime[models[0]].keys():
    # ialltime = 'mon'
    print(ialltime)
    
    # LIG
    time = lig_sst_regrid_alltime[models[0]][ialltime].time
    
    lig_sst_regrid_alltime_list[ialltime] = [
        lig_sst_regrid_alltime[model][ialltime].
            expand_dims('ensemble', axis=0).
            assign_coords(ensemble=[model]).
            assign_coords(time=time) for model in models
    ]
    
    lig_sst_regrid_alltime_ens[ialltime] = xr.concat(
        lig_sst_regrid_alltime_list[ialltime], dim='ensemble').compute()
    
    # PI
    time = pi_sst_regrid_alltime[models[0]][ialltime].time
    
    pi_sst_regrid_alltime_list[ialltime] = [
        pi_sst_regrid_alltime[model][ialltime].
            expand_dims('ensemble', axis=0).
            assign_coords(ensemble=[model]).
            assign_coords(time=time) for model in models
    ]
    
    pi_sst_regrid_alltime_ens[ialltime] = xr.concat(
        pi_sst_regrid_alltime_list[ialltime], dim='ensemble').compute()

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(lig_sst_regrid_alltime_ens, f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(pi_sst_regrid_alltime_ens, f)




'''
#-------------------------------- check
with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sst_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sst_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

models=sorted(lig_sst_regrid_alltime.keys())
model = models[3]

data1 = lig_sst_regrid_alltime[model]['am'].values
data2 = lig_sst_regrid_alltime_ens['am'].sel(ensemble=model).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = pi_sst_regrid_alltime[model]['am'].values
data2 = pi_sst_regrid_alltime_ens['am'].sel(ensemble=model).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())







# too slow
xr.merge(lig_sst_list[ialltime])
# time dimension differs
xr.concat(
    [lig_sst_regrid_alltime[model][ialltime] for model in models], dim='ensemble')\
        .assign_coords({"ensemble": models})\
            .to_dataset()

ialltime = 'mon'
imodel = 11
data1 = lig_sst_list[ialltime][imodel].values[0]
data2 = lig_sst_regrid_alltime[models[imodel]][ialltime].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get ensemble statistics

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sst_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sst_regrid_alltime_ens = pickle.load(f)

sst_regrid_alltime_ens_stats = {}
sst_regrid_alltime_ens_stats['lig'] = {}
sst_regrid_alltime_ens_stats['pi'] = {}
sst_regrid_alltime_ens_stats['lig_pi'] = {}

for ialltime in lig_sst_regrid_alltime_ens.keys():
    # ialltime = 'mon'
    print(ialltime)
    
    sst_regrid_alltime_ens_stats['lig'][ialltime] = {}
    sst_regrid_alltime_ens_stats['pi'][ialltime] = {}
    sst_regrid_alltime_ens_stats['lig_pi'][ialltime] = {}
    
    sst_regrid_alltime_ens_stats['lig'][ialltime]['mean'] = \
        lig_sst_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sst_regrid_alltime_ens_stats['lig'][ialltime]['std'] = \
        lig_sst_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sst_regrid_alltime_ens_stats['pi'][ialltime]['mean'] = \
        pi_sst_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sst_regrid_alltime_ens_stats['pi'][ialltime]['std'] = \
        pi_sst_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sst_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'] = \
        (lig_sst_regrid_alltime_ens[ialltime] - \
            pi_sst_regrid_alltime_ens[ialltime].values).mean(
                dim='ensemble', skipna=True,).compute()
    sst_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'] = \
        (lig_sst_regrid_alltime_ens[ialltime] - \
            pi_sst_regrid_alltime_ens[ialltime].values).std(
                dim='ensemble', skipna=True, ddof=1).compute()

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'wb') as f:
    pickle.dump(sst_regrid_alltime_ens_stats, f)




'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sst_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sst_regrid_alltime_ens = pickle.load(f)

ialltime = 'mon'

data1 = sst_regrid_alltime_ens_stats['lig'][ialltime]['mean'].values
data2 = lig_sst_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sst_regrid_alltime_ens_stats['pi'][ialltime]['mean'].values
data2 = pi_sst_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data1 = sst_regrid_alltime_ens_stats['lig'][ialltime]['std'].values
data2 = lig_sst_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sst_regrid_alltime_ens_stats['pi'][ialltime]['std'].values
data2 = pi_sst_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = sst_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'].values
data2 = (lig_sst_regrid_alltime_ens[ialltime] - \
    pi_sst_regrid_alltime_ens[ialltime].values).mean(
        dim='ensemble', skipna=True,).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sst_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'].values
data2 = (lig_sst_regrid_alltime_ens[ialltime] - \
    pi_sst_regrid_alltime_ens[ialltime].values).std(
        dim='ensemble', skipna=True,ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get lig sst alltime anomalies

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

lig_pi_sst_regrid_alltime = {}

for imodel in lig_sst_regrid_alltime.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    lig_pi_sst_regrid_alltime[imodel] = {}
    
    for ialltime in lig_sst_regrid_alltime[imodel].keys():
        # ialltime = 'am'
        print('#-------- ' + ialltime)
        
        lig_pi_sst_regrid_alltime[imodel][ialltime] = \
            (lig_sst_regrid_alltime[imodel][ialltime] - \
                pi_sst_regrid_alltime[imodel][ialltime].values).compute()


with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_pi_sst_regrid_alltime, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)


ilat = 48
ilon = 90

for imodel in lig_sst_regrid_alltime.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    for ialltime in lig_sst_regrid_alltime[imodel].keys():
        # ialltime = 'am'
        print('#-------- ' + ialltime)
        
        data1 = lig_pi_sst_regrid_alltime[imodel][ialltime][:, ilat, ilon].values
        data2 = lig_sst_regrid_alltime[imodel][ialltime][:, ilat, ilon].values - pi_sst_regrid_alltime[imodel][ialltime][:, ilat, ilon].values
        
        print((data1 == data2).all())




'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get lig sst ensemble anomalies

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sst_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sst_regrid_alltime_ens = pickle.load(f)

lig_pi_sst_regrid_alltime_ens = {}

for ialltime in lig_sst_regrid_alltime_ens.keys():
    # ialltime = 'am'
    print('#-------- ' + ialltime)
    
    lig_pi_sst_regrid_alltime_ens[ialltime] = \
        (lig_sst_regrid_alltime_ens[ialltime] - \
            pi_sst_regrid_alltime_ens[ialltime].values).compute()

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(lig_pi_sst_regrid_alltime_ens, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sst/lig_pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sst_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sst/lig_sst_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sst_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sst_regrid_alltime_ens = pickle.load(f)


imodel = 2
ilat = 48
ilon = 90

for ialltime in lig_sst_regrid_alltime_ens.keys():
    # ialltime = 'am'
    print('#-------- ' + ialltime)
    
    data1 = lig_pi_sst_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values
    data2 = lig_sst_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values - pi_sst_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values
    
    print((data1 == data2).all())


'''
# endregion
# -----------------------------------------------------------------------------

