

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
            files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/'+var+'/'+var+'_Omon_'+model+'_lig127k_*.nc')
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
                ds.isel(time=slice(-2400,None)))
        except OSError as err:
            print('LIG of '+model+' not readable' , err)
            continue
        
    return var_dic


def get_var_PI(var):
    var_dic={}
    for model in models:
        print(model)
        files_LIG=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/'+var+'/'+var+'_Omon_'+model+'_lig127k_*.nc')
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
                ds.isel(time=slice(-2400,None)))
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
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','INM-CM4-8','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',
    ]

lig_sst=get_var_LIG('tos')
pi_sst=get_var_PI('tos')

# HadGEM3 from vittorias LIG Simulation
model = 'HadGEM3-GC31-LL'
file='/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/uba937_ThetaSo_18502050.nc'
ds = xr.open_dataset(file)
# change time indices
ds = ds.assign_coords(time = lig_sst['IPSL-CM6A-LR'].time)
sst=ds.thetao.isel(time=slice(-2400,None),deptht=0,drop=True)
lig_sst[model]=sst.to_dataset().rename(dict(thetao='tos'))
lig_sst[model].tos.values[lig_sst[model].tos.values==9.9692100e+36] = np.nan

with open('scratch/cmip6/lig/lig_sst.pkl', 'wb') as f:
    pickle.dump(lig_sst, f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'wb') as f:
    pickle.dump(pi_sst, f)


'''
#-------------------------------- check

with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

#---------------- 'HadGEM3-GC31-LL'
ds=xr.open_dataset('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/uba937_ThetaSo_18502050.nc')
ds.thetao.isel(time=slice(-2400,None),deptht=0,drop=True).to_dataset().rename(dict(thetao='tos')).tos.values == 9.9692100e+36

ds.time
lig_sst['HadGEM3-GC31-LL'].time

data1 = ds.thetao.isel(time=slice(-2400,None),deptht=0,drop=True).values
data2 = lig_sst['HadGEM3-GC31-LL'].tos.values
(data1[data1 != 9.9692100e+36] == data2[np.isfinite(data2)]).all()


#---------------- check model length
# models=[
#     'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
#     'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','INM-CM4-8','IPSL-CM6A-LR',
#     'MIROC-ES2L','NESM3','NorESM2-LM']


# others
# models=sorted(lig_sst.keys())
# models.remove('AWI-ESM-1-1-LR')

# lig_mean_regrid={}
# for model in models:
#     print(model)
#     LIG_mean=lig_sst[model].mean(dim='time').compute()
#     lig_mean_regrid[model]=regrid(LIG_mean,'tos')

# len(lig_sst.keys())
# len(pi_sst.keys())
'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann sst

with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

models=sorted(lig_sst.keys())

lig_sst_alltime = {}
pi_sst_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sst_alltime[model] = mon_sea_ann(var_monthly = lig_sst[model].tos)
    pi_sst_alltime[model] = mon_sea_ann(var_monthly = pi_sst[model].tos)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sst_alltime, f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sst_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

models=sorted(lig_sst.keys())

#---- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_sst[model].tos.values
    data2 = lig_sst_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_sst[model].tos.values
    data2 = pi_sst_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

#---- check storage
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    print(lig_sst[model].tos.nbytes / 2**30)
    
    # print(lig_sst_alltime[model].keys())
    
    for ialltime in lig_sst_alltime[model].keys():
        print(lig_sst_alltime[model][ialltime].nbytes / 2**30)

#---- check 'NorESM2-LM'
with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)

files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/tos/tos_Omon_NorESM2-LM_lig127k_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds_pp=combined_preprocessing(ds.isel(time=slice(-2400,None)))

data1 = ds_pp.tos.values
data2 = lig_sst_alltime['NorESM2-LM']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regrid AWI sst

! cdo -P 4 -remapcon,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/tos/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc

! cdo -P 4 -remapcon,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/tos/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded simulations

with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
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
        lig_sst_regrid[model] = xr.open_dataset('scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
        pi_sst_regrid[model] = xr.open_dataset('scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')

with open('scratch/cmip6/lig/lig_sst_regrid.pkl', 'wb') as f:
    pickle.dump(lig_sst_regrid, f)
with open('scratch/cmip6/lig/pi_sst_regrid.pkl', 'wb') as f:
    pickle.dump(pi_sst_regrid, f)



'''
#---- check grids of two regridding methods
with open('scratch/cmip6/lig/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_regrid.pkl', 'rb') as f:
    pi_sst_regrid = pickle.load(f)

awiesm_regrid_lig = xr.open_dataset('scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
awiesm_regrid_pi = xr.open_dataset('scratch/cmip6/lig/tos_Omon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')

lig_sst_regrid['NorESM2-LM'].lon
awiesm_regrid_lig.lon
pi_sst_regrid['NorESM2-LM'].lon
awiesm_regrid_pi.lon
lig_sst_regrid['NorESM2-LM'].lat
awiesm_regrid_lig.lat
pi_sst_regrid['NorESM2-LM'].lat
awiesm_regrid_pi.lat

test = regrid(lig_sst[model])
test1 = regrid_rahul(lig_sst[model], 'tos')
(test.tos.values[np.isfinite(test.tos.values)] == lig_sst[model].tos.values[np.isfinite(lig_sst[model].tos.values)]).all()
(test1.values[np.isfinite(test1.values)] == lig_sst[model].tos.values[np.isfinite(lig_sst[model].tos.values)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded sst

with open('scratch/cmip6/lig/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_regrid.pkl', 'rb') as f:
    pi_sst_regrid = pickle.load(f)

models=sorted(lig_sst_regrid.keys())

lig_sst_regrid_alltime = {}
pi_sst_regrid_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    
    lig_sst_regrid_alltime[model] = mon_sea_ann(
        var_monthly = lig_sst_regrid[model].tos)
    pi_sst_regrid_alltime[model] = mon_sea_ann(
        var_monthly = pi_sst_regrid[model].tos)

with open('scratch/cmip6/lig/lig_sst_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sst_regrid_alltime, f)
with open('scratch/cmip6/lig/pi_sst_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sst_regrid_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)



#---- while no awi-esm
    if (model != 'AWI-ESM-1-1-LR'):
    elif (model == 'AWI-ESM-1-1-LR'):
        lig_sst_regrid_alltime[model] = {}
        pi_sst_regrid_alltime[model] = {}

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded AMIP SST

pi_sst_file = 'startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc'
boundary_conditions = {}
boundary_conditions['sst'] = {}
boundary_conditions['sst']['pi'] = xr.open_dataset(pi_sst_file)

# set land points as np.nan
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    boundary_conditions['sst']['pi'].sst.shape,)
boundary_conditions['sst']['pi'].sst.values[b_slm] = np.nan
boundary_conditions['sst']['pi'].sst.values -= zerok

amip_pi_sst_regrid = {}
amip_pi_sst_regrid['mm'] = regrid(boundary_conditions['sst']['pi'].sst)
amip_pi_sst_regrid['am'] = time_weighted_mean(amip_pi_sst_regrid['mm'])
amip_pi_sst_regrid['sm'] = amip_pi_sst_regrid['mm'].groupby(
    'time.season').map(time_weighted_mean)

with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'wb') as f:
    pickle.dump(amip_pi_sst_regrid, f)

'''
with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'rb') as f:
    amip_pi_sst_regrid = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------

