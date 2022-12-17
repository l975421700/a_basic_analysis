

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
            files=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/'+var+'/'+var+'_SImon_'+model+'_lig127k_*.nc')
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
        files_LIG=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/'+var+'/'+var+'_SImon_'+model+'_lig127k_*.nc')
        try:
            index=files_LIG[0].index('_lig127k_')+9
            ens=files_LIG[0][index:index+8]
        except:
            print(model+'no LIG, trying r1i1p1f1')
            ens='r1i1p1f1'
        try:
            files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_SImon_'+model+'_piControl_'+ens+'*.nc')
            if not files:
                print(model+' PI data ensemble is not same as LIG')
                files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/'+var+'/'+var+'_SImon_'+model+'_piControl_*.nc')
                if not files:
                    print(model+' PI data not avaialbe')
                    continue
            if any("_gr_" in filename for filename in files):
                print('PI of '+model+' in native grid')
            ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
            var_dic[model]=combined_preprocessing(
                ds.isel(time=slice(-1200,None)))
        except Exception as err:
            print(err,'PI of '+model+'in CEDA not readable' )
    return var_dic


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
lig_sic=get_var_LIG('siconc')
pi_sic=get_var_PI('siconc')
# pi_sic.pop('INM-CM4-8')

# GISS-E2-1-G, LIG
model = 'GISS-E2-1-G'
files = glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconca/siconca_SImon_GISS-E2-1-G_lig127k_r1i1p1f1_gn_*.nc')
ds = xr.open_mfdataset(
    paths=files,use_cftime=True,parallel=True).rename(dict(siconca='siconc'))
lig_sic[model] = combined_preprocessing(ds.isel(time=slice(-1200,None))).compute()

# HadGEM3 from vittorias LIG Simulation
model = 'HadGEM3-GC31-LL'
files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/seaice_monthly_uba937_*.nc')
ds = xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
siconc = ds.aice.to_dataset().rename(dict(aice='siconc')).isel(time=slice(-1200,None))
lig_sic[model] = combined_preprocessing(siconc).compute()
lig_sic[model].siconc.values[lig_sic[model].siconc.values==9.9692100e+36] = np.nan
print((lig_sic[model].siconc.values==9.9692100e+36).sum())

# siconc.siconc.values[siconc.siconc.values==9.9692100e+36] = np.nan
# print((siconc.siconc.values==9.9692100e+36).sum())


# GISS-E2-1-G, PI
model = 'GISS-E2-1-G'
files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconca/siconca_SImon_GISS-E2-1-G_piControl_r101i1p1f1_gn_*.nc')
ds = xr.open_mfdataset(
    paths=files,use_cftime=True,parallel=True).rename(dict(siconca='siconc'))
pi_sic[model] = combined_preprocessing(ds.isel(time=slice(-1200,None))).compute()


with open('scratch/cmip6/lig/sic/lig_sic.pkl', 'wb') as f:
    pickle.dump(lig_sic, f)
with open('scratch/cmip6/lig/sic/pi_sic.pkl', 'wb') as f:
    pickle.dump(pi_sic, f)


'''
#-------------------------------- check

with open('scratch/cmip6/lig/sic/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

#---------------- check time length

for imodel in lig_sic.keys():
    print('#-------- ' + imodel)
    print('#---- LIG')
    
    print(len(lig_sic[imodel].time))
    # print(lig_sic[imodel].lon)
    # print(lig_sic[imodel].lat)
    
    print('#---- PI')
    
    print(len(pi_sic[imodel].time))
    # print(pi_sic[imodel].lon)
    # print(pi_sic[imodel].lat)

#---------------- check 'HadGEM3-GC31-LL'
ds=xr.open_mfdataset('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/seaice_monthly_uba937_*.nc',use_cftime=True,parallel=True)
data1 = ds.aice.isel(time=slice(-1200,None)).values
data2 = lig_sic['HadGEM3-GC31-LL'].siconc.values
print((data1[data1 != 9.9692100e+36] == data2[np.isfinite(data2)]).all())


#---------------- check 'GISS-E2-1-G' LIG
ds = xr.open_mfdataset(
    paths='/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconca/siconca_SImon_GISS-E2-1-G_lig127k_r1i1p1f1_gn_*.nc',
    use_cftime=True,parallel=True).rename(dict(siconca='siconc'))
data1 = ds.siconc.isel(time=slice(-1200,None)).values
data2 = lig_sic['GISS-E2-1-G'].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

#---------------- check 'GISS-E2-1-G' PI

ds = xr.open_mfdataset(
    paths='/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconca/siconca_SImon_GISS-E2-1-G_piControl_r101i1p1f1_gn_*.nc',
    use_cftime=True,parallel=True).rename(dict(siconca='siconc'))

data1 = ds.siconc.isel(time=slice(-1200,None)).values
data2 = pi_sic['GISS-E2-1-G'].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---------------- check 'AWI-ESM-1-1-LR'
model = 'AWI-ESM-1-1-LR'
ds = xr.open_mfdataset('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_'+model+'_lig127k_*.nc')

data1 = ds.siconc.isel(time=slice(-1200,None)).values
data2 = lig_sic[model].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regrid AWI/Nor/CESM sic

#---- regrid AWI-ESM-1-1-LR

! cdo -remapcon,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc

! cdo -remapcon,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc


#---- regrid NorESM2-LM

! cdo -remapbil,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_210101-220012.nc

! cdo -remapbil,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_160001-210012.nc


#---- regrid CESM2

! cdo -remapbil,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_050101-070012.nc

! cdo -remapbil,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/sic/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_000101-120012.nc



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded simulations

with open('scratch/cmip6/lig/sic/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

lig_sic_regrid = {}
pi_sic_regrid = {}

models=sorted(lig_sic.keys())

for model in models:
    # model = 'ACCESS-ESM1-5'
    print(model)
    if not (model in ['AWI-ESM-1-1-LR', 'NorESM2-LM', 'CESM2']):
        lig_sic_regrid[model] = regrid(lig_sic[model])
        pi_sic_regrid[model] = regrid(pi_sic[model])
    elif (model == 'AWI-ESM-1-1-LR'):
        lig_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')
    elif (model == 'NorESM2-LM'):
        lig_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_210101-220012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_160001-210012.nc')
    elif (model == 'CESM2'):
        lig_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_050101-070012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_000101-120012.nc')
    
    if (model in ['AWI-ESM-1-1-LR', 'NorESM2-LM', 'CESM2']):
        
        lig_sim = combined_preprocessing(
            lig_sim.isel(time=slice(-1200,None))).compute()
        pi_sim = combined_preprocessing(
            pi_sim.isel(time=slice(-1200,None))).compute()
        lig_sim['lon'] = lig_sim.lon.transpose()
        lig_sim['lat'] = lig_sim.lat.transpose()
        pi_sim['lon'] = pi_sim.lon.transpose()
        pi_sim['lat'] = pi_sim.lat.transpose()
        
        lig_sic_regrid[model] = lig_sim
        pi_sic_regrid[model] = pi_sim

with open('scratch/cmip6/lig/sic/lig_sic_regrid.pkl', 'wb') as f:
    pickle.dump(lig_sic_regrid, f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid.pkl', 'wb') as f:
    pickle.dump(pi_sic_regrid, f)





'''
#-------------------------------- check
with open('scratch/cmip6/lig/sic/lig_sic_regrid.pkl', 'rb') as f:
    lig_sic_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid.pkl', 'rb') as f:
    pi_sic_regrid = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

#---- check lon/lat/siconc dimension

for imodel in lig_sic_regrid.keys():
    print('#-------- ' + imodel)
    print('#---- LIG')
    
    print(lig_sic_regrid[imodel].lon.shape)
    print(lig_sic_regrid[imodel].lat.shape)
    print(lig_sic_regrid[imodel].siconc.shape)
    
    print('#---- PI')
    
    print(pi_sic_regrid[imodel].lon.shape)
    print(pi_sic_regrid[imodel].lat.shape)
    print(pi_sic_regrid[imodel].siconc.shape)

#---- check single model

model = 'AWI-ESM-1-1-LR'
lig_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
pi_sim = xr.open_dataset('scratch/cmip6/lig/sic/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')

data1 = lig_sim.siconc.values
data2 = lig_sic_regrid[model].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = pi_sim.siconc.values
data2 = pi_sic_regrid[model].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---- check regrid of NorESM2-LM
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

pi_sic['NorESM2-LM'].siconc[0].to_netcdf('scratch/test/test3.nc')
test = regrid(pi_sic['NorESM2-LM'].isel(time=slice(0, 1)), method="conservative")
test.to_netcdf('scratch/test/test2.nc')



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded sic

with open('scratch/cmip6/lig/sic/lig_sic_regrid.pkl', 'rb') as f:
    lig_sic_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid.pkl', 'rb') as f:
    pi_sic_regrid = pickle.load(f)

models=sorted(lig_sic_regrid.keys())

lig_sic_regrid_alltime = {}
pi_sic_regrid_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sic_regrid_alltime[model] = mon_sea_ann(
        var_monthly = lig_sic_regrid[model].siconc, seasons = 'Q-MAR',)
    pi_sic_regrid_alltime[model] = mon_sea_ann(
        var_monthly = pi_sic_regrid[model].siconc, seasons = 'Q-MAR',)
    
    lig_sic_regrid_alltime[model]['mm'] = \
        lig_sic_regrid_alltime[model]['mm'].rename({'month': 'time'})
    lig_sic_regrid_alltime[model]['sm'] = \
        lig_sic_regrid_alltime[model]['sm'].rename({'month': 'time'})
    lig_sic_regrid_alltime[model]['am'] = \
        lig_sic_regrid_alltime[model]['am'].expand_dims('time', axis=0)
    
    pi_sic_regrid_alltime[model]['mm'] = \
        pi_sic_regrid_alltime[model]['mm'].rename({'month': 'time'})
    pi_sic_regrid_alltime[model]['sm'] = \
        pi_sic_regrid_alltime[model]['sm'].rename({'month': 'time'})
    pi_sic_regrid_alltime[model]['am'] = \
        pi_sic_regrid_alltime[model]['am'].expand_dims('time', axis=0)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sic_regrid_alltime, f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sic_regrid_alltime, f)


'''
#-------------------------------- check
with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid.pkl', 'rb') as f:
    lig_sic_regrid = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid.pkl', 'rb') as f:
    pi_sic_regrid = pickle.load(f)


model = 'CESM2'
data1 = lig_sic_regrid_alltime[model]['mon'].values
data2 = lig_sic_regrid[model].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = pi_sic_regrid_alltime[model]['mon'].values
data2 = pi_sic_regrid[model].siconc.values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region concatenate ensembles

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

models=sorted(lig_sic_regrid_alltime.keys())

pi_sic_regrid_alltime_list = {}
pi_sic_regrid_alltime_ens = {}
lig_sic_regrid_alltime_list = {}
lig_sic_regrid_alltime_ens = {}

for ialltime in lig_sic_regrid_alltime[models[0]].keys():
    # ialltime = 'mon'
    print(ialltime)
    
    # LIG
    time = lig_sic_regrid_alltime[models[0]][ialltime].time
    
    lig_sic_regrid_alltime_list[ialltime] = [
        lig_sic_regrid_alltime[model][ialltime].
            expand_dims('ensemble', axis=0).
            assign_coords(ensemble=[model]).
            assign_coords(time=time) for model in models
    ]
    
    lig_sic_regrid_alltime_ens[ialltime] = xr.concat(
        lig_sic_regrid_alltime_list[ialltime], dim='ensemble').compute()
    
    # PI
    time = pi_sic_regrid_alltime[models[0]][ialltime].time
    
    pi_sic_regrid_alltime_list[ialltime] = [
        pi_sic_regrid_alltime[model][ialltime].
            expand_dims('ensemble', axis=0).
            assign_coords(ensemble=[model]).
            assign_coords(time=time) for model in models
    ]
    
    pi_sic_regrid_alltime_ens[ialltime] = xr.concat(
        pi_sic_regrid_alltime_list[ialltime], dim='ensemble').compute()

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(lig_sic_regrid_alltime_ens, f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(pi_sic_regrid_alltime_ens, f)




'''
#-------------------------------- check
with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

models=sorted(lig_sic_regrid_alltime.keys())
model = models[3]

data1 = lig_sic_regrid_alltime[model]['mon'].values
data2 = lig_sic_regrid_alltime_ens['mon'].sel(ensemble=model).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = pi_sic_regrid_alltime[model]['mon'].values
data2 = pi_sic_regrid_alltime_ens['mon'].sel(ensemble=model).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get ensemble statistics

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

sic_regrid_alltime_ens_stats = {}
sic_regrid_alltime_ens_stats['lig'] = {}
sic_regrid_alltime_ens_stats['pi'] = {}
sic_regrid_alltime_ens_stats['lig_pi'] = {}

for ialltime in lig_sic_regrid_alltime_ens.keys():
    # ialltime = 'mon'
    print(ialltime)
    
    sic_regrid_alltime_ens_stats['lig'][ialltime] = {}
    sic_regrid_alltime_ens_stats['pi'][ialltime] = {}
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime] = {}
    
    sic_regrid_alltime_ens_stats['lig'][ialltime]['mean'] = \
        lig_sic_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sic_regrid_alltime_ens_stats['lig'][ialltime]['std'] = \
        lig_sic_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sic_regrid_alltime_ens_stats['pi'][ialltime]['mean'] = \
        pi_sic_regrid_alltime_ens[ialltime].mean(
            dim='ensemble', skipna=True).compute()
    sic_regrid_alltime_ens_stats['pi'][ialltime]['std'] = \
        pi_sic_regrid_alltime_ens[ialltime].std(
            dim='ensemble', skipna=True, ddof=1).compute()
    
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'] = \
        (lig_sic_regrid_alltime_ens[ialltime] - \
            pi_sic_regrid_alltime_ens[ialltime].values).mean(
                dim='ensemble', skipna=True,).compute()
    sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'] = \
        (lig_sic_regrid_alltime_ens[ialltime] - \
            pi_sic_regrid_alltime_ens[ialltime].values).std(
                dim='ensemble', skipna=True, ddof=1).compute()

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'wb') as f:
    pickle.dump(sic_regrid_alltime_ens_stats, f)




'''
#-------------------------------- check
with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sic/sic_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sic_regrid_alltime_ens_stats = pickle.load(f)

ialltime = 'am'

data1 = sic_regrid_alltime_ens_stats['lig'][ialltime]['mean'].values
data2 = lig_sic_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['pi'][ialltime]['mean'].values
data2 = pi_sic_regrid_alltime_ens[ialltime].mean(
    dim='ensemble', skipna=True).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

data1 = sic_regrid_alltime_ens_stats['lig'][ialltime]['std'].values
data2 = lig_sic_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['pi'][ialltime]['std'].values
data2 = pi_sic_regrid_alltime_ens[ialltime].std(
    dim='ensemble', skipna=True, ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


data1 = sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['mean'].values
data2 = (lig_sic_regrid_alltime_ens[ialltime] - \
    pi_sic_regrid_alltime_ens[ialltime].values).mean(
        dim='ensemble', skipna=True,).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
data1 = sic_regrid_alltime_ens_stats['lig_pi'][ialltime]['std'].values
data2 = (lig_sic_regrid_alltime_ens[ialltime] - \
    pi_sic_regrid_alltime_ens[ialltime].values).std(
        dim='ensemble', skipna=True,ddof=1).compute().values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get lig sic alltime anomalies

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

lig_pi_sic_regrid_alltime = {}

for imodel in lig_sic_regrid_alltime.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    lig_pi_sic_regrid_alltime[imodel] = {}
    
    for ialltime in lig_sic_regrid_alltime[imodel].keys():
        # ialltime = 'am'
        print('#-------- ' + ialltime)
        
        lig_pi_sic_regrid_alltime[imodel][ialltime] = \
            (lig_sic_regrid_alltime[imodel][ialltime] - \
                pi_sic_regrid_alltime[imodel][ialltime].values).compute()


with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_pi_sic_regrid_alltime, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)


ilat = 48
ilon = 90

for imodel in lig_sic_regrid_alltime.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    for ialltime in lig_sic_regrid_alltime[imodel].keys():
        # ialltime = 'am'
        print('#-------- ' + ialltime)
        
        data1 = lig_pi_sic_regrid_alltime[imodel][ialltime][:, ilat, ilon].values
        data2 = lig_sic_regrid_alltime[imodel][ialltime][:, ilat, ilon].values - pi_sic_regrid_alltime[imodel][ialltime][:, ilat, ilon].values
        
        print((data1 == data2).all())




'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get lig sic ensemble anomalies

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)

lig_pi_sic_regrid_alltime_ens = {}

for ialltime in lig_sic_regrid_alltime_ens.keys():
    # ialltime = 'am'
    print('#-------- ' + ialltime)
    
    lig_pi_sic_regrid_alltime_ens[ialltime] = \
        (lig_sic_regrid_alltime_ens[ialltime] - \
            pi_sic_regrid_alltime_ens[ialltime].values).compute()

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime_ens.pkl', 'wb') as f:
    pickle.dump(lig_pi_sic_regrid_alltime_ens, f)



'''
#-------------------------------- check

with open('scratch/cmip6/lig/sic/lig_pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_pi_sic_regrid_alltime_ens = pickle.load(f)

with open('scratch/cmip6/lig/sic/lig_sic_regrid_alltime_ens.pkl', 'rb') as f:
    lig_sic_regrid_alltime_ens = pickle.load(f)
with open('scratch/cmip6/lig/sic/pi_sic_regrid_alltime_ens.pkl', 'rb') as f:
    pi_sic_regrid_alltime_ens = pickle.load(f)


imodel = 2
ilat = 48
ilon = 90

for ialltime in lig_sic_regrid_alltime_ens.keys():
    # ialltime = 'am'
    print('#-------- ' + ialltime)
    
    data1 = lig_pi_sic_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values
    data2 = lig_sic_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values - pi_sic_regrid_alltime_ens[ialltime][imodel, :, ilat, ilon].values
    
    print((data1 == data2).all())


'''
# endregion
# -----------------------------------------------------------------------------

