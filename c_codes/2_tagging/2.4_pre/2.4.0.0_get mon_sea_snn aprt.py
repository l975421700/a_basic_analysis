

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    'pi_610_5.8',
    ]
i = 0

ifile_start = 120
ifile_end   = 840 # 1080

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')
import psutil

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


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns

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
    mean_over_ais,
    time_weighted_mean,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    zerok,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(
    filenames_wiso[ifile_start:ifile_end],
    )

'''
#-------- check pre
filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
nc1 = xr.open_dataset(filenames_wiso[ifile])
nc2 = xr.open_dataset(filenames_echam[ifile])

np.max(abs(nc1.wisoaprl[:, 0].mean(dim='time').values - nc2.aprl[0].values))


#-------- input previous files

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        # filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        # exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)
        
        # filenames_wiso_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_wiso.nc'))
        # exp_org_o[expid[i]]['wiso_daily'] = xr.open_mfdataset(filenames_wiso_daily, data_vars='minimal', coords='minimal', parallel=True)
        
        filenames_echam_daily = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*daily.01_echam.nc'))
        exp_org_o[expid[i]]['echam_daily'] = xr.open_mfdataset(filenames_echam_daily[120:], data_vars='minimal', coords='minimal', parallel=True)

'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region get mon/sea/ann wisoaprt

wisoaprt = {}
wisoaprt[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl[:, :3] + \
        exp_org_o[expid[i]]['wiso'].wisoaprc[:, :3].values).compute()

wisoaprt[expid[i]] = wisoaprt[expid[i]].rename('wisoaprt')

wisoaprt_alltime = {}
wisoaprt_alltime[expid[i]] = mon_sea_ann(wisoaprt[expid[i]])


with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'wb') as f:
    pickle.dump(wisoaprt_alltime[expid[i]], f)


'''
#-------- check calculation
wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[120:1080][ifile])

(wisoaprt_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values)).all()

(wisoaprt_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values).mean(dim='time')).all()


#-------- check simulation of aprt and wisoaprt
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:180], data_vars='minimal', coords='minimal', parallel=True)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl.values + \
    exp_org_o[expid[i]]['echam'].aprc.values - \
        wisoaprt_alltime[expid[i]]['mon'][:60, 0].values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon/sea/ann wisoaprt averaged over AIS


wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

wisoaprt_mean_over_ais = {}
wisoaprt_mean_over_ais[expid[i]] = {}

for ialltime in wisoaprt_alltime[expid[i]].keys():
    # ialltime = 'mm'
    if (ialltime != 'am'):
        wisoaprt_mean_over_ais[expid[i]][ialltime] = mean_over_ais(
            wisoaprt_alltime[expid[i]][ialltime][:, 0],
            echam6_t63_ais_mask['mask']['AIS'],
            echam6_t63_cellarea.cell_area.values,
            )
    else:
        # ialltime = 'am'
        wisoaprt_mean_over_ais[expid[i]][ialltime] = mean_over_ais(
            wisoaprt_alltime[expid[i]][ialltime][0].expand_dims(dim='time'),
            echam6_t63_ais_mask['mask']['AIS'],
            echam6_t63_cellarea.cell_area.values,
            )
    print(ialltime)

wisoaprt_mean_over_ais[expid[i]]['mon_std'] = wisoaprt_mean_over_ais[expid[i]]['mon'].groupby('time.month').std(skipna=True).compute()
wisoaprt_mean_over_ais[expid[i]]['ann_std'] = wisoaprt_mean_over_ais[expid[i]]['ann'].std(skipna=True).compute()

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl',
          'wb') as f:
    pickle.dump(wisoaprt_mean_over_ais[expid[i]], f)



'''
#-------------------------------- check
wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

wisoaprt_mean_over_ais = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl', 'rb') as f:
    wisoaprt_mean_over_ais[expid[i]] = pickle.load(f)

#-------- check mm
imonth = 5
np.average(
    wisoaprt_alltime[expid[i]]['mm'][imonth, 0].values[
        echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[
        echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['mm'][imonth].values

#-------- check mon
imon = 30
np.average(
    wisoaprt_alltime[expid[i]]['mon'][imon, 0].values[
        echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[
        echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['mon'][imon].values

#-------- check am
np.average(
    wisoaprt_alltime[expid[i]]['am'][0].values[echam6_t63_ais_mask['mask']['AIS']],
    weights = echam6_t63_cellarea.cell_area.values[echam6_t63_ais_mask['mask']['AIS']],
)
wisoaprt_mean_over_ais[expid[i]]['am'].values

(wisoaprt_mean_over_ais[expid[i]]['mm'] == mean_over_ais(
    wisoaprt_alltime[expid[i]]['mm'][:, 0],
    echam6_t63_ais_mask['mask']['AIS'],
    echam6_t63_cellarea.cell_area.values
    )).all().values

(wisoaprt_mean_over_ais[expid[i]]['mon'] == mean_over_ais(
    wisoaprt_alltime[expid[i]]['mon'][:, 0],
    echam6_t63_ais_mask['mask']['AIS'],
    echam6_t63_cellarea.cell_area.values,
    )).all().values


'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region get mon/sea/ann aprt_geo7


aprt_geo7 = {}
aprt_geo7[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(16, 22)) + \
        exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(16, 22))
    ).compute()

aprt_geo7[expid[i]] = aprt_geo7[expid[i]].rename('aprt_geo7')

aprt_geo7_alltime = {}
aprt_geo7_alltime[expid[i]] = mon_sea_ann(aprt_geo7[expid[i]], lcopy = False,)

aprt_geo7_alltime[expid[i]]['sum'] = {}
for ialltime in aprt_geo7_alltime[expid[i]].keys():
    # ialltime = 'daily'
    if (ialltime != 'sum'):
        print(ialltime)
        aprt_geo7_alltime[expid[i]]['sum'][ialltime] = \
            aprt_geo7_alltime[expid[i]][ialltime].sum(dim='wisotype')

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'wb') as f:
    pickle.dump(aprt_geo7_alltime[expid[i]], f)



'''
#-------- check calculation
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[ifile_start:ifile_end][ifile])

(aprt_geo7_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, 15:22] + ncfile.wisoaprc[:, 15:22].values)).all().values

(aprt_geo7_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:,15:22] + ncfile.wisoaprc[:,15:22].values).mean(dim='time')).all().values


#-------- check calculation of sum over wisotypes

print((aprt_geo7_alltime[expid[i]]['sum']['daily'] == aprt_geo7_alltime[expid[i]]['daily'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mon'] == aprt_geo7_alltime[expid[i]]['mon'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sea'] == aprt_geo7_alltime[expid[i]]['sea'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['ann'] == aprt_geo7_alltime[expid[i]]['ann'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mm'] == aprt_geo7_alltime[expid[i]]['mm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sm'] == aprt_geo7_alltime[expid[i]]['sm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['am'] == aprt_geo7_alltime[expid[i]]['am'].sum(dim='wisotype')).all().values)


#-------- check am values
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

aprt_geo7_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon/sea/ann aprt_geo7 averaged over AIS/WAIS/EAIS/AP

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

aprt_geo7_spave = {}
for imask in echam6_t63_ais_mask['mask'].keys():
    # imask = 'EAIS'
    aprt_geo7_spave[imask] = {}
    
    for ialltime in aprt_geo7_alltime[expid[i]].keys():
        # ialltime = 'sm'
        if (ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']):
            print(imask + ': ' + ialltime)
            
            lentime = aprt_geo7_alltime[expid[i]][ialltime].shape[0]
            lenwisotype = len(aprt_geo7_alltime[expid[i]][ialltime].wisotype)
            
            if (ialltime in ['mon', 'sea', 'ann']):
                aprt_geo7_spave[imask][ialltime] = xr.DataArray(
                    data = np.zeros((lentime, lenwisotype)),
                    coords = {
                        'time': aprt_geo7_alltime[expid[i]][ialltime].time,
                        'wisotype': aprt_geo7_alltime[expid[i]][ialltime].wisotype,
                    }
                )
            elif (ialltime == 'mm'):
                # ialltime = 'mm'
                aprt_geo7_spave[imask][ialltime] = xr.DataArray(
                    data = np.zeros((lentime, lenwisotype)),
                    coords = {
                        'month': aprt_geo7_alltime[expid[i]][ialltime].month,
                        'wisotype': aprt_geo7_alltime[expid[i]][ialltime].wisotype,
                    }
                )
            elif (ialltime == 'sm'):
                # ialltime = 'sm'
                aprt_geo7_spave[imask][ialltime] = xr.DataArray(
                    data = np.zeros((lentime, lenwisotype)),
                    coords = {
                        'season': aprt_geo7_alltime[expid[i]][ialltime].season,
                        'wisotype': aprt_geo7_alltime[expid[i]][ialltime].wisotype,
                    }
                )
            
            for itime in range(lentime):
                # itime = 0
                for iwisotype in range(lenwisotype):
                    # iwisotype = 0
                    aprt_data = aprt_geo7_alltime[expid[i]][ialltime][
                        itime, iwisotype, :, :].values
                    aprt_geo7_spave[imask][ialltime].values[itime, iwisotype] = \
                        np.average(
                            aprt_data[echam6_t63_ais_mask['mask'][imask]],
                            weights=echam6_t63_cellarea.cell_area.values[
                                echam6_t63_ais_mask['mask'][imask]
                            ]
                        )
            
        elif (ialltime == 'am'):
            # ialltime = 'am'
            print(imask + ': ' + ialltime)
            
            lenwisotype = len(aprt_geo7_alltime[expid[i]][ialltime].wisotype)
            aprt_geo7_spave[imask][ialltime] = xr.DataArray(
                data = np.zeros((lenwisotype)),
                coords = {
                    'wisotype': aprt_geo7_alltime[expid[i]][ialltime].wisotype,
                }
                )
            
            for iwisotype in range(lenwisotype):
                # iwisotype = 0
                aprt_data = aprt_geo7_alltime[expid[i]][ialltime][
                    iwisotype, :, :].values
                aprt_geo7_spave[imask][ialltime].values[iwisotype] = \
                        np.average(
                            aprt_data[echam6_t63_ais_mask['mask'][imask]],
                            weights=echam6_t63_cellarea.cell_area.values[
                                echam6_t63_ais_mask['mask'][imask]
                            ]
                        )

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'wb') as f:
    pickle.dump(aprt_geo7_spave, f)



'''
#-------------------------------- check
# import data

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
    echam6_t63_ais_mask = pickle.load(f)
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

#-------- check but 'am'

imask = 'AP'
ialltime = 'ann'
itime = -2
iwisotype = 3
aprt_data = aprt_geo7_alltime[expid[i]][ialltime][itime, iwisotype].values
res1 = aprt_geo7_spave[imask][ialltime][itime, iwisotype].values
res2 = np.average(
    aprt_data[echam6_t63_ais_mask['mask'][imask]],
    weights = echam6_t63_cellarea.cell_area.values[
        echam6_t63_ais_mask['mask'][imask]],)
res1 == res2

for imask in ['EAIS', 'WAIS', 'AP', 'AIS']:
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
        itime = 3
        iwisotype = 3
        aprt_data = aprt_geo7_alltime[expid[i]][ialltime][itime, iwisotype].values
        res1 = aprt_geo7_spave[imask][ialltime][itime, iwisotype].values
        res2 = np.average(
            aprt_data[echam6_t63_ais_mask['mask'][imask]],
            weights = echam6_t63_cellarea.cell_area.values[
                echam6_t63_ais_mask['mask'][imask]],)
        print(res1 == res2)

imask = 'AP'
ialltime = 'am'
iwisotype = 3
aprt_data = aprt_geo7_alltime[expid[i]][ialltime][iwisotype].values
res1 = aprt_geo7_spave[imask][ialltime][iwisotype].values
res2 = np.average(
    aprt_data[echam6_t63_ais_mask['mask'][imask]],
    weights = echam6_t63_cellarea.cell_area.values[
        echam6_t63_ais_mask['mask'][imask]],)
print(res1 == res2)

for imask in ['EAIS', 'WAIS', 'AP', 'AIS']:
    ialltime = 'am'
    iwisotype = 3
    aprt_data = aprt_geo7_alltime[expid[i]][ialltime][iwisotype].values
    res1 = aprt_geo7_spave[imask][ialltime][iwisotype].values
    res2 = np.average(
        aprt_data[echam6_t63_ais_mask['mask'][imask]],
        weights = echam6_t63_cellarea.cell_area.values[
            echam6_t63_ais_mask['mask'][imask]],)
    print(res1 == res2)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann aprt_geo7 frac over AIS

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'AIS', 'Land excl. AIS', 'Atlantic Ocean',
    'Indian Ocean', 'Pacific Ocean', 'SH seaice', 'Southern Ocean']
wisotypes = {'AIS': 16, 'Land excl. AIS': 17,
             'Atlantic Ocean': 18, 'Indian Ocean': 19, 'Pacific Ocean': 20,
             'SH seaice': 21, 'Southern Ocean': 22}

aprt_frc_AIS_alltime = {}

for imask in aprt_geo7_spave.keys():
    # imask = 'AIS'
    print(imask)
    
    aprt_frc_AIS_alltime[imask] = {}
    
    for ialltime in aprt_geo7_spave[imask].keys():
        # ialltime = 'mm'
        print(ialltime)
        
        aprt_AIS_ialltime = aprt_geo7_spave[imask][ialltime].sum(dim='wisotype').values
        
        aprt_frc_AIS_alltime[imask][ialltime] = {}
        
        if ialltime in ['mon', 'sea', 'ann']:
            time = aprt_geo7_spave[imask][ialltime].time
        elif ialltime in ['mm']:
            time = aprt_geo7_spave[imask]['mm'].month
        elif ialltime in ['sm']:
            time = aprt_geo7_spave[imask]['sm'].season
        elif ialltime in ['am']:
            time = ['am']
        
        for iwisotype in wisotypes.keys():
            # iwisotype = 'AIS'
            print(iwisotype)
            aprt_frc_AIS_alltime[imask][ialltime][iwisotype] = \
                pd.DataFrame(data={
                    'time': time,
                    'frc_AIS': (aprt_geo7_spave[imask][ialltime].sel(
                        wisotype=slice(16, wisotypes[iwisotype])).sum(dim='wisotype').values / \
                                aprt_AIS_ialltime) * 100
                    })

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'wb') as f:
    pickle.dump(aprt_frc_AIS_alltime, f)


'''
#-------------------------------- check initial mm calculation passed

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'rb') as f:
    aprt_frc_AIS_alltime = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

aprt_frc_AIS = {}


for imask in aprt_geo7_spave.keys():
    # imask = 'EAIS'
    print(imask)
    
    aprt_mm_AIS = aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values
    
    aprt_frc_AIS[imask] = {}
    
    aprt_frc_AIS[imask]['Open ocean'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['SH sea ice'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['Land excl. Antarctica'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                wisotypes['NHland'], wisotypes['SHland'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['Antarctica'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                wisotypes['NHland'], wisotypes['SHland'],
                wisotypes['Antarctica'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })

for imask in aprt_frc_AIS_alltime.keys():
    # imask = 'AIS'
    print(imask)
    for isource in aprt_frc_AIS_alltime[imask]['mm'].keys():
        # isource = 'Open ocean'
        print(isource)
        print((aprt_frc_AIS_alltime[imask]['mm'][isource].frc_AIS.values == aprt_frc_AIS[imask][isource].frc_AIS.values).all())

#-------------------------------- check 'mm'

#-------- import data
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'rb') as f:
    aprt_frc_AIS_alltime = pickle.load(f)

imask = 'AIS'
iregion = 'Open ocean'
res1 = aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 21)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()


iregion = 'SH sea ice'
res1 = aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 22)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()

iregion = 'Land excl. Antarctica'
res1 = aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=[16, 17, 19, 20, 21, 22]).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()
np.max(abs(res1 - res2))

iregion = 'Antarctica'
aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS.values



aprt_frc_AIS_alltime[imask]['mm']['SH sea ice'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['Open ocean'].frc_AIS
aprt_frc_AIS_alltime[imask]['mm']['Land excl. Antarctica'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['SH sea ice'].frc_AIS
(aprt_frc_AIS_alltime[imask]['mm']['Antarctica'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['Land excl. Antarctica'].frc_AIS).values
aprt_geo7_spave['AIS']['mm'].sel(wisotype=18).values / aprt_geo7_spave['AIS']['mm'].sum(dim='wisotype').values * 100

aprt_geo7_spave['AIS']['mm'].sel(wisotype=20).values / aprt_geo7_spave['AIS']['mm'].sum(dim='wisotype').values

#-------------------------------- check 'ann'

#-------- import data
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'rb') as f:
    aprt_frc_AIS_alltime = pickle.load(f)

imask = 'AIS'
iregion = 'Open ocean'
res1 = aprt_frc_AIS_alltime[imask]['ann'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['ann'].sel(
    wisotype=slice(19, 21)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['ann'].sum(dim='wisotype').values) * 100
(res1 == res2).all()


iregion = 'SH sea ice'
res1 = aprt_frc_AIS_alltime[imask]['ann'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['ann'].sel(
    wisotype=slice(19, 22)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['ann'].sum(dim='wisotype').values) * 100
(res1 == res2).all()

iregion = 'Land excl. Antarctica'
res1 = aprt_frc_AIS_alltime[imask]['ann'][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['ann'].sel(
    wisotype=[16, 17, 19, 20, 21, 22]).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['ann'].sum(dim='wisotype').values) * 100
(res1 == res2).all()
np.max(abs(res1 - res2))

iregion = 'Antarctica'
aprt_frc_AIS_alltime[imask]['ann'][iregion].frc_AIS.values


# SH sea ice
(aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Open ocean'].frc_AIS).values

# other land
(aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS).values

# Antarctica
(aprt_frc_AIS_alltime[imask]['ann']['Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS).values

aprt_geo7_spave['AIS']['ann'].sel(wisotype=18).values / \
    aprt_geo7_spave['AIS']['ann'].sum(dim='wisotype').values * 100

# # NH sea ice
# aprt_geo7_spave['AIS']['ann'].sel(wisotype=20).values / \
#     aprt_geo7_spave['AIS']['ann'].sum(dim='wisotype').values

        
        # aprt_frc_AIS_alltime[imask][ialltime]['Open ocean'] = \
        #     pd.DataFrame(data={
        #         'time': time,
        #         'frc_AIS': (aprt_geo7_spave[imask][ialltime].sel(
        #             wisotype=[
        #                 wisotypes['NHocean'], wisotypes['NHseaice'],
        #                 wisotypes['SHocean'],
        #                 ]).sum(dim='wisotype').values / aprt_AIS_ialltime) * 100
        #         })
        
        # aprt_frc_AIS_alltime[imask][ialltime]['SH sea ice'] = \
        #     pd.DataFrame(data={
        #         'time': time,
        #         'frc_AIS': (aprt_geo7_spave[imask][ialltime].sel(
        #             wisotype=[
        #                 wisotypes['NHocean'], wisotypes['NHseaice'],
        #                 wisotypes['SHocean'], wisotypes['SHseaice'],
        #                 ]).sum(dim='wisotype').values / aprt_AIS_ialltime) * 100
        #         })
        
        # aprt_frc_AIS_alltime[imask][ialltime]['Land excl. Antarctica'] = \
        #     pd.DataFrame(data={
        #         'time': time,
        #         'frc_AIS': (aprt_geo7_spave[imask][ialltime].sel(
        #             wisotype=[
        #                 wisotypes['NHocean'], wisotypes['NHseaice'],
        #                 wisotypes['SHocean'], wisotypes['SHseaice'],
        #                 wisotypes['NHland'], wisotypes['SHland'],
        #                 ]).sum(dim='wisotype').values / aprt_AIS_ialltime) * 100
        #         })
        
        # aprt_frc_AIS_alltime[imask][ialltime]['Antarctica'] = \
        #     pd.DataFrame(data={
        #         'time': time,
        #         'frc_AIS': (aprt_geo7_spave[imask][ialltime].sel(
        #             wisotype=[
        #                 wisotypes['NHocean'], wisotypes['NHseaice'],
        #                 wisotypes['SHocean'], wisotypes['SHseaice'],
        #                 wisotypes['NHland'], wisotypes['SHland'],
        #                 wisotypes['Antarctica'],
        #                 ]).sum(dim='wisotype').values / aprt_AIS_ialltime) * 100
        #         })

aprt_frc_AIS_alltime['AIS']['am']

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ERA5 pre

era5_mon_tp_1979_2021 = xr.open_dataset(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021.nc')

# change units to mm/d
era5_mon_tp_1979_2021_alltime = mon_sea_ann(
    var_monthly = (era5_mon_tp_1979_2021.tp * 1000).compute())

with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'wb') as f:
    pickle.dump(era5_mon_tp_1979_2021_alltime, f)


'''
#-------- check
era5_mon_tp_1979_2021 = xr.open_dataset(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021.nc')
with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_tp_1979_2021_alltime = pickle.load(f)

(era5_mon_tp_1979_2021.tp * 1000 == era5_mon_tp_1979_2021_alltime['mon']).all()


(era5_mon_tp_1979_2021_alltime['mon'].resample({'time': '1Y'}).map(time_weighted_mean).compute() == era5_mon_tp_1979_2021_alltime['ann']).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann ERA5 averaged over AIS

with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_tp_1979_2021_alltime = pickle.load(f)

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

tp_era5_mean_over_ais = {}

for ialltime in era5_mon_tp_1979_2021_alltime.keys():
    # ialltime = 'daily'
    if (ialltime != 'am'):
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            era5_mon_tp_1979_2021_alltime[ialltime],
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    else:
        # ialltime = 'am'
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            era5_mon_tp_1979_2021_alltime[ialltime].expand_dims(dim='time'),
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    print(ialltime)

tp_era5_mean_over_ais['mon_std'] = tp_era5_mean_over_ais['mon'].groupby('time.month').std(skipna=True).compute()
tp_era5_mean_over_ais['ann_std'] = tp_era5_mean_over_ais['ann'].std(skipna=True).compute()

with open('scratch/products/era5/pre/tp_era5_mean_over_ais.pkl', 'wb') as f:
    pickle.dump(tp_era5_mean_over_ais, f)



'''
#-------------------------------- check

with open(
    'scratch/products/era5/pre/era5_mon_tp_1979_2021_alltime.pkl', 'rb') as f:
    era5_mon_tp_1979_2021_alltime = pickle.load(f)

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

with open('scratch/products/era5/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)

#-------- check mm
itime = 8
np.average(
    era5_mon_tp_1979_2021_alltime['mm'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mm'][itime].values

#-------- check mon
itime = 30
np.average(
    era5_mon_tp_1979_2021_alltime['mon'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mon'][itime].values

#-------- check am
np.average(
    era5_mon_tp_1979_2021_alltime['am'].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['am'].values


(tp_era5_mean_over_ais['mm'] == mean_over_ais(
    era5_mon_tp_1979_2021_alltime['mm'],
    era5_ais_mask['mask']['AIS'],
    era5_cellarea.cell_area.values,
    )).all().values
(tp_era5_mean_over_ais['mon'] == mean_over_ais(
    era5_mon_tp_1979_2021_alltime['mon'],
    era5_ais_mask['mask']['AIS'],
    era5_cellarea.cell_area.values,
    )).all().values

# check consistency

wisoaprt_mean_over_ais = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mean_over_ais.pkl', 'rb') as f:
    wisoaprt_mean_over_ais[expid[i]] = pickle.load(f)

(wisoaprt_mean_over_ais[expid[i]]['mm'] * 3600 * 24 * month_days).sum()
(tp_era5_mean_over_ais['mm'] * month_days).sum()

wisoaprt_mean_over_ais[expid[i]]['am'].values * 3600 * 24 * 365
tp_era5_mean_over_ais['am'].values * 365

wisoaprt_mean_over_ais[expid[i]]['ann_std'] * 3600 * 24 * 365
tp_era5_mean_over_ais['ann_std'] * 365

# ECHAM6:   160.58 ± 7.65 mm/yr,
# ERA5:     174.66 ± 8.84 mm/yr
'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=12:00:00
# -----------------------------------------------------------------------------
# region get mon_sea_ann ocean_aprt

time = exp_org_o[expid[i]]['wiso'].time
lon  = exp_org_o[expid[i]]['wiso'].lon
lat  = exp_org_o[expid[i]]['wiso'].lat

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 3
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon', 'geo7']
itags = [5, 7, 8, 9, 11, 12]

ocean_aprt = {}
ocean_aprt[expid[i]] = xr.DataArray(
    data = np.zeros(
        (len(time), len(var_names), len(lat), len(lon)),
        dtype=np.float32),
    coords={
        'time':         time,
        'var_names':    var_names,
        'lat':          lat,
        'lon':          lon,
    }
)

for count,var_name in enumerate(var_names[:-1]):
    # count = 0; var_name = 'lat'
    kstart = kwiso2 + sum(ntags[:itags[count]])
    
    print(str(count) + ' : ' + var_name + ' : ' + str(itags[count]) + \
        ' : ' + str(kstart))
    
    ocean_aprt[expid[i]].sel(var_names=var_name)[:] = \
        (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(
            wisotype=slice(kstart+2, kstart+3)) + \
                exp_org_o[expid[i]]['wiso'].wisoaprc.sel(
                    wisotype=slice(kstart+2, kstart+3))
                ).sum(dim='wisotype')

ocean_aprt[expid[i]].sel(var_names='geo7')[:] = \
    (exp_org_o[expid[i]]['wiso'].wisoaprl.sel(
        wisotype=[19, 21]) + \
            exp_org_o[expid[i]]['wiso'].wisoaprc.sel(
                wisotype=[19, 21])
            ).sum(dim='wisotype')

ocean_aprt_alltime = {}
ocean_aprt_alltime[expid[i]] = mon_sea_ann(
    ocean_aprt[expid[i]], lcopy = False,)

print(psutil.Process().memory_info().rss / (2 ** 30))

del ocean_aprt[expid[i]]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'wb') as f:
    pickle.dump(ocean_aprt_alltime[expid[i]], f)






'''

#-------------------------------- check ocean_aprt

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

ocean_aprt = {}
ocean_aprt[expid[i]] = ocean_aprt_alltime[expid[i]]['daily']

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[ifile_start:ifile_end][ifile])

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]
kwiso2 = 3
var_names = ['lat', 'sst', 'rh2m', 'wind10', 'sinlon', 'coslon']
itags = [5, 7, 8, 9, 11, 12]

ilat = 48
ilon = 90

for count in range(6):
    # count = 5
    print(count)
    
    kstart = kwiso2 + sum(ntags[:itags[count]])

    res1 = ocean_aprt[expid[i]][-31:, :, ilat, ilon].sel(
        var_names = var_names[count])

    res2 = ncfile.wisoaprl[:, :, ilat, ilon].sel(
        wisotype=[kstart+2, kstart+3]).sum(dim='wisotype') + \
            ncfile.wisoaprc[:, :, ilat, ilon].sel(
        wisotype=[kstart+2, kstart+3]).sum(dim='wisotype')

    print(np.max(abs(res1 - res2)).values)

# check 'geo7'
res1 = ocean_aprt[expid[i]][-31:, :, ilat, ilon].sel(var_names = 'geo7')
res2 = ncfile.wisoaprl[:, :, ilat, ilon].sel(
    wisotype=[19, 21]).sum(dim='wisotype') + \
        ncfile.wisoaprc[:, :, ilat, ilon].sel(
    wisotype=[19, 21]).sum(dim='wisotype')
print(np.max(abs(res1 - res2)).values)


#-------------------------------- check alltime calculation
ocean_aprt_alltime[expid[i]].keys()
(ocean_aprt_alltime[expid[i]]['daily'] == ocean_aprt[expid[i]]).all().values

(ocean_aprt_alltime[expid[i]]['mon'] == ocean_aprt_alltime[expid[i]]['daily'].resample({'time': '1M'}).mean()).all()

#-------------------------------- check ocean pre consistency
np.max(abs((ocean_aprt_alltime[expid[i]]['mon'][:, 5] - \
    ocean_aprt_alltime[expid[i]]['mon'][:, 0]) / \
        ocean_aprt_alltime[expid[i]]['mon'][:, 5]))
np.mean(abs(ocean_aprt_alltime[expid[i]]['mon'][:, 5] - ocean_aprt_alltime[expid[i]]['mon'][:, 0]))
np.mean(abs(ocean_aprt_alltime[expid[i]]['mon'][:, 5]))

np.max(abs((ocean_aprt_alltime[expid[i]]['am'][5] - \
    ocean_aprt_alltime[expid[i]]['am'][0]) / \
        ocean_aprt_alltime[expid[i]]['am'][5]))

'''
# endregion
# -----------------------------------------------------------------------------

