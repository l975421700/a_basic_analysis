

# =============================================================================
# region import packages


# basic library
import numpy as np
import xarray as xr
import datetime
from datetime import timedelta
import glob
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')
import netCDF4
import cftime
import time

# plot
import matplotlib.path as mpath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
mpl.rcParams['figure.dpi'] = 600
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'data_source/bg_cartopy'
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from cartopy.mpl.geoaxes import GeoAxes
import xesmf as xe
import proplot as pplt
import cmip6_preprocessing.preprocessing as cpp

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from geopy import distance

# add ellipse
from scipy import linalg
from scipy import stats
from sklearn import mixture
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# self defined
from a00_basic_analysis.b_module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot1,
    hemisphere_plot,
)

# endregion
# =============================================================================


# =============================================================================
# region define namelist and functions

# models = ['AWI-ESM-1-1-LR', 'HadGEM3-GC31-LL',]
models = ['ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CESM2', 'CNRM-CM6-1', 'EC-Earth3-LR', 'FGOALS-g3', 'GISS-E2-1-G',
          'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'NESM3', 'NorESM2-LM']


def combined_preprocessing(ds):
    ds = cpp.rename_cmip6(ds)
    ds = cpp.broadcast_lonlat(ds)
    #     ds=cpp.correct_lon(ds)
    #     ds=cpp.parse_lon_lat_bounds(ds)
    #     ds=cpp.maybe_convert_vertex_to_bounds(ds)
    #     ds=cpp.maybe_convert_bounds_to_vertex(ds)
    return ds

def get_var_LIG(var):
    var_dic={}
    for model in models:
        print(model)
        try:
            files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/' +
                              var+'/'+var+'_SImon_'+model+'_lig127k_*.nc')
            if not files:
                print(model+' LIG data not avaialbe')
                continue
            if any("_gr_" in filename for filename in files):
                print('LIG of '+model+' in native grid')
            
            if not any("r1i1p1f1" in filename for filename in files):
                index = files[0].index('_lig127k_')+9
                ens_name = files[0][index:index+9]
                print('LIG of '+model+' ensemble is '+ens_name)
            
            ds = xr.open_mfdataset(paths=files, use_cftime=True, parallel=True)
            var_dic[model] = combined_preprocessing(
                ds.isel(time=slice(-2400, None)))
        except OSError as err:
            print('LIG of '+model+' not readable', err)
            continue
    
    return var_dic


def get_var_PI(var):
    var_dic = {}
    for model in models:
        print(model)
        files_LIG = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/' +
                              var+'/'+var+'_SImon_'+model+'_lig127k_*.nc')
        try:
            index = files_LIG[0].index('_lig127k_')+9
            ens = files_LIG[0][index:index+8]
        except:
            print(model+'no LIG, trying r1i1p1f1')
            ens = 'r1i1p1f1'
        try:
            files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/' +
                              var+'/'+var+'_SImon_'+model+'_piControl_'+ens+'*.nc')
            
            if not files:
                print(model+' PI data ensemble is not same as LIG')
                files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/' +
                                  var+'/'+var+'_SImon_'+model+'_piControl_*.nc')
                if not files:
                    print(model+' PI data not avaialbe')
                    continue
            
            if any("_gr_" in filename for filename in files):
                print('PI of '+model+' in native grid')
            
            ds = xr.open_mfdataset(paths=files, use_cftime=True, parallel=True)
            var_dic[model] = combined_preprocessing(
                ds.isel(time=slice(-2400, None)))
        except:
            print('PI of '+model+'in CEDA not readable')
            continue
    
    return var_dic


def monthly_clim(var, monthnum):
    monthly_data = {}
    monthly_obj = var.groupby('time.month')
    for sname, sdata in monthly_obj:
        monthly_data[sname] = sdata
    monthly_clim = monthly_data[monthnum].mean(dim='time')
    return monthly_clim


def replace_CESMlatlon(ds_in):
    ds = ds_in.copy()
    gfile = xr.open_dataset(
        '/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/areacello/areacello_Ofx_CESM2_historical_r1i1p1f1_gn.nc')
    lat = gfile.lat
    lon = gfile.lon
    #     lat_masked=xr.where(abs(lat)<=90,lat,np.nan)
    #     newlat=lat_masked.interpolate_na(dim='y',method='linear')
    ds.lat.values = lat.values
    #     lon_masked=xr.where(abs(lon)<=360,lon,np.nan)
    #     newlon=lon_masked.interpolate_na(dim='x',method='linear')
    ds.lon.values = lon.values
    return ds


def regrid(ds):
    ds_in = ds.copy()
    var = ds.siconc
    ds_out = xe.util.grid_global(1, 1)
    
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear',
                             ignore_degenerate=True, periodic=True)
    return regridder(var)


def seasonal_ave(var,season):
    if model == 'IPSL-CM6A-LR':
        var=xr.where(var=='nan',1e+20,var)
    season_data={}
    season_obj=var.groupby('time.season')
    for sname,sdata in season_obj:
        season_data[sname]=sdata
    
    ####shift December data to next year######
    season_data['DJF']['time']=season_data['DJF'].time + timedelta(days=30) ### for monthly data
    #########average over year############################
    season_av=season_data[season].groupby('time.year').mean('time')
    ##########assign proper time axis################
    yearlist=[cftime.Datetime360Day(year ,6, 16) for year in season_av.year]
    season_av=season_av.rename({'year':'time'})
    season_av['time']=yearlist
    ############################################
    
    return season_av



# endregion
# =============================================================================


# =============================================================================
# region input data
siconc_LIG=get_var_LIG('siconc')

# HADCM3 from vittorias LIG Simulation
model = 'HadGEM3-GC31-LL'
files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/Vittoria_LIG_run_links/seaice_monthly_uba937_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds=ds.where(ds.aice<1000)
ds=ds.rename(dict(aice='siconc'))
ds['siconc']=ds.siconc*100

siconc_LIG[model]=ds.isel(time=slice(-2400,None))

###GISS read siconca in ATM grid##############
model='GISS-E2-1-G'
var='siconca'
files=glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/'+var+'/'+var+'_SImon_'+model+'_lig127k_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds=ds.rename(dict(siconca='siconc'))
siconc_LIG[model]=combined_preprocessing(ds.isel(time=slice(-2400,None)))


siconc_PI = get_var_PI('siconc')
model = 'GISS-E2-1-G'
var = 'siconca'
files = glob.glob('/home/users/rahuls/LOUISE/PMIP_LIG/piControl_CEDA_Links/' +
                  var+'/'+var+'_SImon_'+model+'_piControl_*.nc')
ds = xr.open_mfdataset(paths=files, use_cftime=True, parallel=True)
ds = ds.rename(dict(siconca='siconc'))
siconc_PI[model] = combined_preprocessing(ds.isel(time=slice(-2400, None)))


# JJA clim
# LIG
siconc_JJAclim_LIG={}
for i,model in enumerate(siconc_LIG.keys()):
    print(model)
    jjaclim=seasonal_ave(siconc_LIG[model],'JJA').mean(dim='time')
    jjaclim=jjaclim.ffill('x',3)
    jjaclim=jjaclim.fillna(0)
    if model == 'CESM2':
        jjaclim=replace_CESMlatlon(jjaclim)
    
    jjaclim_regrid=regrid(jjaclim)
    
    siconc_JJAclim_LIG[model]=jjaclim_regrid.compute()

#PI
siconc_JJAclim_PI={}
for model in (siconc_PI.keys()):
    print(model)
    jjaclim=seasonal_ave(siconc_PI[model],'JJA').mean(dim='time')
    jjaclim=jjaclim.ffill('x',3)
    jjaclim=jjaclim.fillna(0)
    
    if model == 'CESM2':
        jjaclim=replace_CESMlatlon(jjaclim)
    if model == 'NorESM2-LM':
        jjaclim['lat']=siconc_LIG[model].lat
        jjaclim['lon']=siconc_LIG[model].lon
    jjaclim_regrid=regrid(jjaclim)
    
    siconc_JJAclim_PI[model]=jjaclim_regrid.compute()

# MME
for i, model in enumerate(models):
    print(model)
    if i == 0:
        mme_LIG = siconc_JJAclim_LIG[model].copy()
        mme_PI = siconc_JJAclim_PI[model].copy()
    else:
        mme_LIG += siconc_JJAclim_LIG[model]
        mme_PI += siconc_JJAclim_PI[model]

siconc_JJAclim_LIG['MME'] = mme_LIG/(len(models))
siconc_JJAclim_PI['MME'] = mme_PI/(len(models))


model_list = ['MME']+models


fig=pplt.figure()
pplt.rc['cartopy.circular'] = True
pplt.rc['font.large']=True
idx=1

for model in model_list:
    print(model)
    ax=fig.add_subplot(4,4,idx,proj='npstere')
    #     ax.format(grid=True,coast=True, latlines=10, lonlines=30, lonlabels=True, labels=True, boundinglat=60)
    
    cmap='Blues_r'#plt.get_cmap('RdBu_r')
    
    ax.coastlines()
    ax.add_feature(ctp.feature.LAND,zorder=100,facecolor='gray',edgecolor='k')
    
    levs=[15,30,45,60,75,90]
    
    val=siconc_JJAclim_PI[model]
    data=val.values
    lon=val.lon[0]
    lat=val.lat[:,0]
    
    data,lon=add_cyclic_point(data, coord=lon)
    #     norm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=8)
    norm = pplt.Norm('segmented',levels=levs)
    
    map=ax.contourf(lon,lat,data,transform=ccrs.PlateCarree(),\
        cmap=cmap,levels=levs,norm=norm,add_colorbar=False,extend='both')
    
    #     for rows in df_points.iterrows():
    #         lon=rows[1]['LON']
    #         lat=rows[1]['LAT']
    #         colval=rows[1]['OBS_MEAN']
    #         marker=rows[1]['marker']
    #         ax.scatter(lon,lat,c=colval,transform=ccrs.PlateCarree(),\
    #                 marker=marker,markersize=100,\
    #                 cmap=cmap,levels=levs,norm=norm,edgecolors='black',\
    #                   linewidth=2,extend='both')
    ax.set_extent((-180, 180, 90, 60), crs=ccrs.PlateCarree())
    ax.set_title(model,fontsize=14)
    idx=idx+1

# ax.legend(handles=legend_elements,loc='r',ncol=1)

cbar=fig.colorbar(map,loc='b',cols=(2,4),length=0.6,width=0.5,space=-12.5)
fig.save('PI_siconc_north_JJA_mean.png',dpi=600)

# endregion
# =============================================================================


