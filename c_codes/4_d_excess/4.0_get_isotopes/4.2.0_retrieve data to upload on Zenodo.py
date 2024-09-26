

# salloc --account=paleodyn.paleodyn --qos=12h --time=12:00:00 --nodes=1 --mem=120GB
# source ${HOME}/miniconda3/bin/activate deepice
# ipython


import xarray as xr
import numpy as np
import pickle


exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['nudged_705_6.0',]
i = 0


wisoaprt_alltime = {}
dO18_alltime = {}
dD_alltime = {}
temp2_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
        temp2_alltime[expid[i]] = pickle.load(f)


source_var = ['lat', 'lon', 'sst', 'RHsst']
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_lon.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_RHsst.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)



wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/precipitation.nc')
dO18_alltime[expid[i]]['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/dO18.nc')
dD_alltime[expid[i]]['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/dD.nc')
pre_weighted_var[expid[i]]['lat']['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/pre_weighted_lat.nc')
pre_weighted_var[expid[i]]['lon']['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/pre_weighted_lon.nc')
pre_weighted_var[expid[i]]['sst']['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/pre_weighted_sst.nc')
pre_weighted_var[expid[i]]['RHsst']['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/pre_weighted_RHsst.nc')
temp2_alltime[expid[i]]['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/temp2.nc')


