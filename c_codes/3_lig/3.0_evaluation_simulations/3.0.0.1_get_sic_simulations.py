

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
                ds.isel(time=slice(-2400,None)))
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
                ds.isel(time=slice(-2400,None)))
        except Exception as err:
            print(err,'PI of '+model+'in CEDA not readable' )
    return var_dic


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get multiple simulations

models=[
    'ACCESS-ESM1-5','AWI-ESM-1-1-LR','CESM2','CNRM-CM6-1','EC-Earth3-LR',
    'FGOALS-g3','GISS-E2-1-G', 'HadGEM3-GC31-LL','INM-CM4-8','IPSL-CM6A-LR',
    'MIROC-ES2L','NESM3','NorESM2-LM',
    ]

lig_sic=get_var_LIG('siconc')
pi_sic=get_var_PI('siconc')
pi_sic.pop('INM-CM4-8')

# GISS-E2-1-G, LIG
model = 'GISS-E2-1-G'
files = glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconca/siconca_SImon_GISS-E2-1-G_lig127k_r1i1p1f1_gn_*.nc')
ds = xr.open_mfdataset(
    paths=files,use_cftime=True,parallel=True).rename(dict(siconca='siconc'))
lig_sic[model] = combined_preprocessing(ds.isel(time=slice(-2400,None)))

# HadGEM3 from vittorias LIG Simulation
model = 'HadGEM3-GC31-LL'
files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/seaice_monthly_uba937_*.nc')
ds = xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
siconc = ds.aice.to_dataset().rename(dict(aice='siconc'))
siconc.siconc.values[siconc.siconc.values==9.9692100e+36] = np.nan
lig_sic[model] = combined_preprocessing(siconc)

# GISS-E2-1-G, PI
model = 'GISS-E2-1-G'
files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconca/siconca_SImon_GISS-E2-1-G_piControl_r101i1p1f1_gn_*.nc')
ds = xr.open_mfdataset(
    paths=files,use_cftime=True,parallel=True).rename(dict(siconca='siconc'))
pi_sic[model] = combined_preprocessing(ds.isel(time=slice(-2400,None)))


with open('scratch/cmip6/lig/lig_sic.pkl', 'wb') as f:
    pickle.dump(lig_sic, f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'wb') as f:
    pickle.dump(pi_sic, f)


'''
#-------------------------------- check

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann sic

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

models=sorted(lig_sic.keys())

lig_sic_alltime = {}
pi_sic_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sic_alltime[model] = mon_sea_ann(var_monthly = lig_sic[model].siconc)
    pi_sic_alltime[model] = mon_sea_ann(var_monthly = pi_sic[model].siconc)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sic_alltime, f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sic_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'rb') as f:
    pi_sic_alltime = pickle.load(f)

models=sorted(lig_sic.keys())

#-------------------------------- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_sic[model].siconc.values
    data2 = lig_sic_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_sic[model].siconc.values
    data2 = pi_sic_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#-------------------------------- check one model
with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)

model = 'NorESM2-LM'
files = glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_'+model+'_lig127k_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds_pp=combined_preprocessing(ds.isel(time=slice(-2400,None)))

data1 = ds_pp.siconc.values
data2 = lig_sic_alltime['NorESM2-LM']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regrid AWI/Nor/CESM sic

#---- regrid AWI-ESM-1-1-LR

! cdo -P 4 -remapcon,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc

! cdo -P 4 -remapcon,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc


#---- regrid NorESM2-LM
# okay
! cdo -P 4 -remapbil,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_210101-220012.nc

# okay
! cdo -P 4 -remapbil,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_160001-210012.nc


#---- regrid CESM2
# okay
! cdo -P 4 -remapbil,global_1 -mergetime /gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_050101-070012.nc

# okay
! cdo -P 4 -remapbil,global_1 -mergetime /home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/siconc/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_*.nc scratch/cmip6/lig/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_000101-120012.nc



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded simulations

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
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
        lig_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-310012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_AWI-ESM-1-1-LR_piControl_r1i1p1f1_gn_185501-195412.nc')
    elif (model == 'NorESM2-LM'):
        lig_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_NorESM2-LM_lig127k_r1i1p1f1_gn_210101-220012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_NorESM2-LM_piControl_r1i1p1f1_gn_160001-210012.nc')
    elif (model == 'CESM2'):
        lig_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_CESM2_lig127k_r1i1p1f1_gn_050101-070012.nc')
        pi_sim = xr.open_dataset('scratch/cmip6/lig/siconc_SImon_CESM2_piControl_r1i1p1f1_gn_000101-120012.nc')
    
    if (model in ['AWI-ESM-1-1-LR', 'NorESM2-LM', 'CESM2']):
        lig_sic_regrid[model] = combined_preprocessing(
            lig_sim.isel(time=slice(-2400,None))).compute()
        pi_sic_regrid[model] = combined_preprocessing(
            pi_sim.isel(time=slice(-2400,None))).compute()

with open('scratch/cmip6/lig/lig_sic_regrid.pkl', 'wb') as f:
    pickle.dump(lig_sic_regrid, f)
with open('scratch/cmip6/lig/pi_sic_regrid.pkl', 'wb') as f:
    pickle.dump(pi_sic_regrid, f)





'''

with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

pi_sic['NorESM2-LM'].siconc[0].to_netcdf('scratch/test/test3.nc')
test = regrid(pi_sic['NorESM2-LM'].isel(time=slice(0, 1)), method="conservative")
test.to_netcdf('scratch/test/test2.nc')

with open('scratch/cmip6/lig/lig_sic_regrid.pkl', 'rb') as f:
    lig_sic_regrid = pickle.load(f)

with open('scratch/cmip6/lig/pi_sic_regrid.pkl', 'rb') as f:
    pi_sic_regrid = pickle.load(f)

pi_sic_regrid['NorESM2-LM'].siconc[0].to_netcdf('scratch/test/test3.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded sic

with open('scratch/cmip6/lig/lig_sic_regrid.pkl', 'rb') as f:
    lig_sic_regrid = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_regrid.pkl', 'rb') as f:
    pi_sic_regrid = pickle.load(f)

models=sorted(lig_sic_regrid.keys())

lig_sic_regrid_alltime = {}
pi_sic_regrid_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sic_regrid_alltime[model] = mon_sea_ann(
        var_monthly = lig_sic_regrid[model].siconc)
    pi_sic_regrid_alltime[model] = mon_sea_ann(
        var_monthly = pi_sic_regrid[model].siconc)

with open('scratch/cmip6/lig/lig_sic_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sic_regrid_alltime, f)
with open('scratch/cmip6/lig/pi_sic_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sic_regrid_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_regrid_alltime.pkl', 'rb') as f:
    pi_sic_regrid_alltime = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'rb') as f:
    pi_sic_alltime = pickle.load(f)

models=sorted(lig_sic_regrid_alltime.keys())

model = 'HadGEM3-GC31-LL'
lig_sic_regrid_alltime[model]['am'].to_netcdf('scratch/test/test0.nc')
lig_sic_alltime[model]['am'].to_netcdf('scratch/test/test1.nc')

pi_sic_regrid_alltime[model]['am'].to_netcdf('scratch/test/test2.nc')
pi_sic_alltime[model]['am'].to_netcdf('scratch/test/test3.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded AMIP SIC

amip_pi_sic = xr.open_dataset(
    'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')

# set land points as np.nan
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    amip_pi_sic.sic.shape,)
amip_pi_sic.sic.values[b_slm] = np.nan

amip_pi_sic_regrid = {}
amip_pi_sic_regrid['mm'] = regrid(amip_pi_sic.sic)
amip_pi_sic_regrid['am'] = time_weighted_mean(amip_pi_sic_regrid['mm'])
amip_pi_sic_regrid['sm'] = amip_pi_sic_regrid['mm'].groupby(
    'time.season').map(time_weighted_mean)

with open('scratch/cmip6/lig/amip_pi_sic_regrid.pkl', 'wb') as f:
    pickle.dump(amip_pi_sic_regrid, f)


'''
with open('scratch/cmip6/lig/amip_pi_sic_regrid.pkl', 'rb') as f:
    amip_pi_sic_regrid = pickle.load(f)
amip_pi_sic_regrid['am']
'''
# endregion
# -----------------------------------------------------------------------------



