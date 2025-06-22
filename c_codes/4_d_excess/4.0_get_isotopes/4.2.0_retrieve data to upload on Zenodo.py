

# salloc --account=paleodyn.paleodyn --qos=12h --time=5:00:00 --nodes=1 --mem=120GB
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




wiso_q_6h_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_q_6h_sfc_alltime.pkl', 'rb') as f:
    wiso_q_6h_sfc_alltime[expid[i]] = pickle.load(f)

wiso_q_6h_sfc_alltime[expid[i]]['q16o']['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/q_sfc.nc')

dO18_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_q_sfc_alltime.pkl', 'rb') as f:
    dO18_q_sfc_alltime[expid[i]] = pickle.load(f)

dO18_q_sfc_alltime[expid[i]]['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/dO18_q_sfc.nc')


dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

dD_q_sfc_alltime[expid[i]]['mon'].to_netcdf('output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/dD_q_sfc.nc')


source_var = ['lat', 'lon', 'sst', 'RHsst']
q_sfc_weighted_var = {}
q_sfc_weighted_var[expid[i]] = {}
prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
source_var_files = [
    prefix + '.q_sfc_weighted_lat.pkl',
    prefix + '.q_sfc_weighted_lon.pkl',
    prefix + '.q_sfc_weighted_sst.pkl',
    prefix + '.q_sfc_weighted_RHsst.pkl',
    ]
for ivar, ifile in zip(source_var, source_var_files):
    # ivar=source_var[0]; ifile = source_var_files[0]
    print(ivar + ':    ' + ifile)
    with open(ifile, 'rb') as f:
        q_sfc_weighted_var[expid[i]][ivar] = pickle.load(f)
    
    q_sfc_weighted_var[expid[i]][ivar]['mon'].to_netcdf(f'output/echam-6.3.05p2-wiso/pi/nudged_705_6.0/analysis/echam/upload_to_Zenodo/q_sfc_weighted_{ivar}.nc')






