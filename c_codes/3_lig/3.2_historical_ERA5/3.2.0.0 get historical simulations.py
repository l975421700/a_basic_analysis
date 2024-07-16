

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/home/users/qino')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!
import intake
import fsspec
from dask_gateway import Gateway
from dask.distributed import Client
from collections import defaultdict
from cdo import Cdo
cdo=Cdo()

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
import cartopy.feature as cfeature
from scipy.stats import pearsonr

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
    regrid,
    find_ilat_ilon,
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
# region import data, define functions

def combined_preprocessing(ds_in):
    
    ds=ds_in.copy()
    ds=cpp.rename_cmip6(ds)
    ds=cpp.broadcast_lonlat(ds)
    
    return ds
# .pipe(combined_preprocessing)


def drop_all_bounds(ds):
    """Drop coordinates like 'time_bounds' from datasets,
    which can lead to issues when merging."""
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)
# .pipe(drop_all_bounds)


def open_dsets(df):
    """Open datasets from cloud storage and return xarray dataset."""
    dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=True, use_cftime=True) for ds_url in df.zstore]
    try:
        ds = xr.merge(dsets, join='exact')
        return ds
    except ValueError:
        return None

def open_delayed(df):
    """A dask.delayed wrapper around `open_dsets`.
    Allows us to open many datasets in parallel."""
    return dask.delayed(open_dsets)(df)


pi_sst_regrid_alltime = pd.read_pickle('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl')

cmip_info = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

esm_datastore = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

'''
cmip_info.head()
cmip_info['institution_id'].unique()

print(esm_datastore)

https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/ECS_Gregory_method.html
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region download data using intake_esm

query = dict(
    experiment_id=['historical'],
    table_id='Omon',
    variable_id=['tos'],
    source_id=list(pi_sst_regrid_alltime.keys()) + ['EC-Earth3'],
)

esm_data_subset = esm_datastore.search(**query)

dsets = defaultdict(dict)

for group, df in esm_data_subset.df.groupby("source_id").first().groupby(by=['source_id']):
    dsets[group[0]] = open_delayed(df)

hist_tos = dask.compute(dict(dsets))[0]

with open('scratch/cmip6/hist/tos/hist_tos.pkl', 'wb') as f:
    pickle.dump(hist_tos, f)




'''
#-------------------------------- check

with open('scratch/cmip6/hist/tos/hist_tos.pkl', 'rb') as f:
    hist_tos = pickle.load(f)

open_dsets(esm_data_subset.df.groupby("source_id").first().loc[['AWI-ESM-1-1-LR']])
hist_tos['AWI-ESM-1-1-LR']

for imodel in hist_tos.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#-------------------------------- ' + imodel)
    
    # print(hist_tos[imodel].tos.shape)
    # print(hist_tos[imodel])
    # print(len(hist_tos[imodel].time))
    # print(hist_tos[imodel].time[0])
    # print(hist_tos[imodel].attrs['parent_source_id'])
    # print(hist_tos[imodel].attrs['institution'])
    
    print(hist_tos[imodel].lon)


print(esm_data_subset)
esm_data_subset.df.groupby("source_id").first()

%time open_dsets(df)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region regrid data

with open('scratch/cmip6/hist/tos/hist_tos.pkl', 'rb') as f:
    hist_tos = pickle.load(f)

hist_tos_regrid = {}

for imodel in hist_tos.keys():
    # imodel = 'ACCESS-ESM1-5'
    # imodel = 'AWI-ESM-1-1-LR'
    print('#-------------------------------- ' + imodel)
    
    if (imodel != 'AWI-ESM-1-1-LR'):
        regridded_data = regrid(hist_tos[imodel])
    elif (imodel == 'AWI-ESM-1-1-LR'):
        hist_tos[imodel].to_netcdf('calculate.nc')
        cdo.remapcon('global_1',input='calculate.nc',output='calculate1.nc')
        regridded_data = regrid(xr.open_dataset('calculate1.nc', use_cftime=True))
        os.remove("calculate.nc")
        os.remove("calculate1.nc")
    
    hist_tos_regrid[imodel] = regridded_data.pipe(combined_preprocessing).pipe(drop_all_bounds).isel(time=slice(-396, None))

with open('scratch/cmip6/hist/tos/hist_tos_regrid.pkl', 'wb') as f:
    pickle.dump(hist_tos_regrid, f)




'''
#-------------------------------- check

with open('scratch/cmip6/hist/tos/hist_tos.pkl', 'rb') as f:
    hist_tos = pickle.load(f)
with open('scratch/cmip6/hist/tos/hist_tos_regrid.pkl', 'rb') as f:
    hist_tos_regrid = pickle.load(f)

for imodel in hist_tos_regrid.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#-------------------------------- ' + imodel)
    
    # print(hist_tos[imodel])
    # print(hist_tos_regrid[imodel])
    
    print(hist_tos[imodel].tos.shape)
    print(hist_tos_regrid[imodel].tos.shape)



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded tos

with open('scratch/cmip6/hist/tos/hist_tos_regrid.pkl', 'rb') as f:
    hist_tos_regrid = pickle.load(f)

hist_tos_regrid_alltime = {}

for imodel in hist_tos_regrid.keys():
    # imodel = 'AWI-ESM-1-1-LR'
    print('#-------------------------------- ' + imodel)
    
    hist_tos_regrid_alltime[imodel] = mon_sea_ann(
        var_monthly=hist_tos_regrid[imodel].tos, seasons = 'Q-MAR',)
    
    hist_tos_regrid_alltime[imodel]['mm'] = \
        hist_tos_regrid_alltime[imodel]['mm'].rename({'month': 'time'})
    hist_tos_regrid_alltime[imodel]['sm'] = \
        hist_tos_regrid_alltime[imodel]['sm'].rename({'month': 'time'})
    hist_tos_regrid_alltime[imodel]['am'] = \
        hist_tos_regrid_alltime[imodel]['am'].expand_dims('time', axis=0)

with open('scratch/cmip6/hist/tos/hist_tos_regrid_alltime.pkl', 'wb') as f:
    pickle.dump(hist_tos_regrid_alltime, f)



'''
with open('scratch/cmip6/hist/tos/hist_tos_regrid_alltime.pkl', 'rb') as f:
    hist_tos_regrid_alltime = pickle.load(f)

for imodel in hist_tos_regrid_alltime.keys():
    # imodel = 'AWI-ESM-1-1-LR'
    print('#-------------------------------- ' + imodel)
    
    print(hist_tos_regrid_alltime[imodel]['am'])
'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region get mon_sea_ann regridded esacci sst

esacci_sst = xr.open_dataset('scratch/cmip6/hist/sst/sst_mon_ESACCI-2.1_198201_201612_rg1.nc', use_cftime=True)

esacci_sst_alltime = mon_sea_ann(
    var_monthly=(regrid(esacci_sst.analysed_sst.isel(time=slice(0, -24))) - zerok).compute(),
    seasons='Q-MAR',)

esacci_sst_alltime['mm'] = \
    esacci_sst_alltime['mm'].rename({'month': 'time'})
esacci_sst_alltime['sm'] = \
    esacci_sst_alltime['sm'].rename({'month': 'time'})
esacci_sst_alltime['am'] = \
    esacci_sst_alltime['am'].expand_dims('time', axis=0)

with open('scratch/cmip6/hist/tos/esacci_sst_alltime.pkl', 'wb') as f:
    pickle.dump(esacci_sst_alltime, f)



'''
with open('scratch/cmip6/hist/tos/esacci_sst_alltime.pkl', 'rb') as f:
    esacci_sst_alltime = pickle.load(f)

esacci_sst_alltime['mon']
esacci_sst_alltime['am']

'''
# endregion
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# region Search using ESGF API

from __future__ import print_function
import requests
import xml.etree.ElementTree as ET
import numpy

# Author: Unknown
# I got the original version from a word document published by ESGF
# https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing

# API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken
    
    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)


with open('scratch/cmip6/lig/sst/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

imodel = list(pi_sst_regrid_alltime.keys())[0]

result = esgf_search(
    activity_id='CMIP', experiment_id='historical', member_id="r1i1p1f1",
    source_id=imodel, table_id='Omon', variable_id='tos',
    )

print(result[0])


for imodel in list(pi_sst_regrid_alltime.keys()):
    print('#-------------------------------- ' + imodel)
    
    if (imodel == 'EC-Earth3-LR'):
        imodel = 'EC-Earth3'
    
    result = esgf_search(
        activity_id='CMIP', experiment_id='historical',
        source_id=imodel, table_id='Omon', variable_id='tos',
        #  member_id="r1i1p1f1",
        )
    
    print(len(result))



'''

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann era5 sst

era5_mon_sst = xr.open_dataset('scratch/cmip6/hist/tos/era5_mon_sst_1940_2023.nc', use_cftime=True)

era5_mon_sst_alltime = mon_sea_ann(
    var_monthly=(regrid(era5_mon_sst.sst.isel(time=slice(-522, -104))) - zerok).compute(),
    seasons='Q-MAR',)

era5_mon_sst_alltime['mm'] = \
    era5_mon_sst_alltime['mm'].rename({'month': 'time'})
era5_mon_sst_alltime['sm'] = \
    era5_mon_sst_alltime['sm'].rename({'month': 'time'})
era5_mon_sst_alltime['am'] = \
    era5_mon_sst_alltime['am'].expand_dims('time', axis=0)

with open('scratch/cmip6/hist/tos/era5_mon_sst_alltime.pkl', 'wb') as f:
    pickle.dump(era5_mon_sst_alltime, f)



# endregion
# -----------------------------------------------------------------------------


