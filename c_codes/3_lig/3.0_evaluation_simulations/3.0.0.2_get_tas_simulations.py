

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
            files=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/atmos/'+var+'/'+var+'_Amon_'+model+'_lig127k_*.nc')
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
        files_LIG=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/atmos/'+var+'/'+var+'_Amon_'+model+'_lig127k_*.nc')
        try:
            index=files_LIG[0].index('_lig127k_')+9
            ens=files_LIG[0][index:index+8]
        except:
            print(model+'no LIG, trying r1i1p1f1')
            ens='r1i1p1f1'
        try:
            files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_Amon_'+model+'_piControl_'+ens+'*.nc')
            if not files:
                print(model+' PI data ensemble is not same as LIG')
                files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_Amon_'+model+'_piControl_*.nc')
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

lig_tas=get_var_LIG('tas')
pi_tas=get_var_PI('tas')
pi_tas.pop('INM-CM4-8')

with open('scratch/cmip6/lig/lig_tas.pkl', 'wb') as f:
    pickle.dump(lig_tas, f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'wb') as f:
    pickle.dump(pi_tas, f)


'''
#-------------------------------- check

with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

#---------------- 'AWI-ESM'
ds = xr.open_mfdataset(
    '/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/atmos/tas/tas_Amon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc', use_cftime=True,parallel=True
)
(ds.tas.values == lig_tas['AWI-ESM-1-1-LR'].tas.values).all()
'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann tas

with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

models=sorted(lig_tas.keys())


lig_tas_alltime = {}
pi_tas_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_tas_alltime[model] = mon_sea_ann(var_monthly = lig_tas[model].tas)
    pi_tas_alltime[model] = mon_sea_ann(var_monthly = pi_tas[model].tas)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'wb') as f:
    pickle.dump(lig_tas_alltime, f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'wb') as f:
    pickle.dump(pi_tas_alltime, f)


'''
for model in models:
    print('#------------------------' + model)
    print(lig_tas[model].lon.shape)
    print(lig_tas[model].lat.shape)
    print(lig_tas[model].tas.shape)


with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'rb') as f:
    pi_tas_alltime = pickle.load(f)

models=sorted(lig_tas.keys())

#---- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_tas[model].tas.values
    data2 = lig_tas_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_tas[model].tas.values
    data2 = pi_tas_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---- check 'AWI-ESM-1-1-LR'
ds = xr.open_mfdataset(
    '/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/atmos/tas/tas_Amon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc', use_cftime=True,parallel=True
)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)

data1 = ds.tas.values
data2 = lig_tas_alltime['AWI-ESM-1-1-LR']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded simulations

with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

lig_tas_regrid = {}
pi_tas_regrid = {}

models=sorted(lig_tas.keys())

for model in models:
    print(model)
    lig_tas_regrid[model] = regrid(lig_tas[model])
    pi_tas_regrid[model] = regrid(pi_tas[model])

with open('scratch/cmip6/lig/lig_tas_regrid.pkl', 'wb') as f:
    pickle.dump(lig_tas_regrid, f)
with open('scratch/cmip6/lig/pi_tas_regrid.pkl', 'wb') as f:
    pickle.dump(pi_tas_regrid, f)



'''
#---- check grids of two regridding methods
with open('scratch/cmip6/lig/lig_tas_regrid.pkl', 'rb') as f:
    lig_tas_regrid = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_regrid.pkl', 'rb') as f:
    pi_tas_regrid = pickle.load(f)

lig_tas_regrid.keys()
pi_tas_regrid.keys()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded tas

with open('scratch/cmip6/lig/lig_tas_regrid.pkl', 'rb') as f:
    lig_tas_regrid = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_regrid.pkl', 'rb') as f:
    pi_tas_regrid = pickle.load(f)

models=sorted(lig_tas_regrid.keys())

lig_tas_regrid_alltime = {}
pi_tas_regrid_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    
    lig_tas_regrid_alltime[model] = mon_sea_ann(
        var_monthly = lig_tas_regrid[model].tas)
    pi_tas_regrid_alltime[model] = mon_sea_ann(
        var_monthly = pi_tas_regrid[model].tas)

with open('scratch/cmip6/lig/lig_tas_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_tas_regrid_alltime, f)
with open('scratch/cmip6/lig/pi_tas_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(pi_tas_regrid_alltime, f)


'''
with open('scratch/cmip6/lig/lig_tas_regrid_alltime.pkl', 'rb') as f:
    lig_tas_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_regrid_alltime.pkl', 'rb') as f:
    pi_tas_regrid_alltime = pickle.load(f)


'''
# endregion
# -----------------------------------------------------------------------------


