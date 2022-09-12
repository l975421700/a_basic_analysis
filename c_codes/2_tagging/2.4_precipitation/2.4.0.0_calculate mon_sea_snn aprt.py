

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0


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
exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso[120:720], data_vars='minimal', coords='minimal', parallel=True)



'''
#-------- check pre
filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:720], data_vars='minimal', coords='minimal', parallel=True)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl[-1].values - exp_org_o[expid[i]]['wiso'].wisoaprl[-31:, 0].mean(dim='time').values))


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


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann wisoaprt

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
i = 0
wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'))
ifile = -1
ncfile = xr.open_dataset(filenames_wiso[120:720][ifile])

(wisoaprt_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values)).all()

(wisoaprt_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:, :3] + ncfile.wisoaprc[:, :3].values).mean(dim='time')).all()



#-------- check simulation of aprt and wisoaprt
i = 0
exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))
exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam[120:180], data_vars='minimal', coords='minimal', parallel=True)

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

np.max(abs(exp_org_o[expid[i]]['echam'].aprl.values + exp_org_o[expid[i]]['echam'].aprc.values - wisoaprt_alltime[expid[i]]['mon'][:60, 0].values))

#-------- memory profiling

sys.getsizeof(wisoaprt[expid[i]])

import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 2**30)

import gc
gc.collect()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann wisoaprt averaged over AIS


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


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann aprt_geo7


aprt_geo7 = {}
aprt_geo7[expid[i]] = (
    exp_org_o[expid[i]]['wiso'].wisoaprl.sel(wisotype=slice(16, 22)) + \
        exp_org_o[expid[i]]['wiso'].wisoaprc.sel(wisotype=slice(16, 22)).values
    ).compute()

aprt_geo7[expid[i]] = aprt_geo7[expid[i]].rename('aprt_geo7')

aprt_geo7_alltime = {}
aprt_geo7_alltime[expid[i]] = mon_sea_ann(aprt_geo7[expid[i]])

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
ncfile = xr.open_dataset(filenames_wiso[120:720][ifile])

(aprt_geo7_alltime[expid[i]]['daily'][-31:,] == \
    (ncfile.wisoaprl[:, 15:22] + ncfile.wisoaprc[:, 15:22].values)).all()

(aprt_geo7_alltime[expid[i]]['mon'][ifile,] == \
    (ncfile.wisoaprl[:,15:22] + ncfile.wisoaprc[:,15:22].values).mean(dim='time')).all()


#-------- check calculation of sum over wisotypes
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

print((aprt_geo7_alltime[expid[i]]['sum']['daily'] == aprt_geo7_alltime[expid[i]]['daily'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mon'] == aprt_geo7_alltime[expid[i]]['mon'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sea'] == aprt_geo7_alltime[expid[i]]['sea'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['ann'] == aprt_geo7_alltime[expid[i]]['ann'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['mm'] == aprt_geo7_alltime[expid[i]]['mm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['sm'] == aprt_geo7_alltime[expid[i]]['sm'].sum(dim='wisotype')).all().values)
print((aprt_geo7_alltime[expid[i]]['sum']['am'] == aprt_geo7_alltime[expid[i]]['am'].sum(dim='wisotype')).all().values)


#-------- check am values
i = 0
aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

aprt_geo7_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon/sea/ann aprt_geo7 averaged over AIS/WAIS/EAIS/AP

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
        if (ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']):
            print(imask + ': ' + ialltime)
            
            lentime = aprt_geo7_alltime[expid[i]][ialltime].shape[0]
            lenwisotype = len(aprt_geo7_alltime[expid[i]][ialltime].wisotype)
            
            if (ialltime in ['daily', 'mon', 'sea', 'ann']):
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

#-------- check daily

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
    for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm']:
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
# region calculate mon_sea_ann ERA5 pre

tp_era5_79_14 = xr.open_dataset(
    'scratch/cmip6/hist/pre/tp_ERA5_mon_sl_197901_201412.nc')

# change units to mm/d
tp_era5_79_14_alltime = mon_sea_ann(
    var_monthly = (tp_era5_79_14.tp * 1000).compute())

with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'wb') as f:
    pickle.dump(tp_era5_79_14_alltime, f)



'''
sh_pre_level = np.concatenate(
    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
sh_pre_ticks = np.concatenate(
    (np.arange(0, 100, 20), np.arange(100, 1600.01,300)))
sh_pre_norm = BoundaryNorm(sh_pre_level, ncolors=len(sh_pre_level))
sh_pre_cmp = cm.get_cmap('RdBu', len(sh_pre_level))


fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([8.8, 9.8]) / 2.54,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.04, fm_top=0.99,
    )

plt_cmp = ax.pcolormesh(
    tp_era5_79_14_alltime['am'].longitude,
    tp_era5_79_14_alltime['am'].latitude,
    tp_era5_79_14_alltime['am'] * 365,
    norm=sh_pre_norm, cmap=sh_pre_cmp, transform=ccrs.PlateCarree(),
)
plt_pre_cbar = fig.colorbar(
    cm.ScalarMappable(norm=sh_pre_norm, cmap=sh_pre_cmp), ax=ax,
    orientation="horizontal",pad=0.02,shrink=0.9,aspect=40,extend='max',
    # anchor=(-0.2, -0.35), fraction=0.12, panchor=(0.5, 0),
    ticks=sh_pre_ticks)
plt_pre_cbar.ax.set_xlabel('Annual mean precipitation [$mm\;yr^{-1}$]\nERA5, 1979-2014', linespacing=1.5)

fig.savefig('figures/trial.png',)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann ERA5 averaged over AIS

with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'rb') as f:
    tp_era5_79_14_alltime = pickle.load(f)

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

tp_era5_mean_over_ais = {}

for ialltime in tp_era5_79_14_alltime.keys():
    # ialltime = 'daily'
    if (ialltime != 'am'):
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            tp_era5_79_14_alltime[ialltime],
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    else:
        # ialltime = 'am'
        tp_era5_mean_over_ais[ialltime] = mean_over_ais(
            tp_era5_79_14_alltime[ialltime].expand_dims(dim='time'),
            era5_ais_mask['mask']['AIS'],
            era5_cellarea.cell_area.values,
            )
    print(ialltime)

tp_era5_mean_over_ais['mon_std'] = tp_era5_mean_over_ais['mon'].groupby('time.month').std(skipna=True).compute()
tp_era5_mean_over_ais['ann_std'] = tp_era5_mean_over_ais['ann'].std(skipna=True).compute()

with open('scratch/cmip6/hist/pre/tp_era5_mean_over_ais.pkl', 'wb') as f:
    pickle.dump(tp_era5_mean_over_ais, f)



'''
#-------------------------------- check

with open('scratch/cmip6/hist/pre/tp_era5_79_14_alltime.pkl', 'rb') as f:
    tp_era5_79_14_alltime = pickle.load(f)

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

with open('scratch/cmip6/hist/pre/tp_era5_mean_over_ais.pkl', 'rb') as f:
    tp_era5_mean_over_ais = pickle.load(f)

#-------- check mm
itime = 8
np.average(
    tp_era5_79_14_alltime['mm'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mm'][itime].values

#-------- check mon
itime = 30
np.average(
    tp_era5_79_14_alltime['mon'][itime].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['mon'][itime].values

#-------- check am
np.average(
    tp_era5_79_14_alltime['am'].values[era5_ais_mask['mask']['AIS']],
    weights=era5_cellarea.cell_area.values[era5_ais_mask['mask']['AIS']]
)
tp_era5_mean_over_ais['am'].values


(tp_era5_mean_over_ais['mm'] == mean_over_ais(
    tp_era5_79_14_alltime['mm'],
    era5_ais_mask['mask']['AIS'],
    era5_cellarea.cell_area.values,
    )).all().values
(tp_era5_mean_over_ais['mon'] == mean_over_ais(
    tp_era5_79_14_alltime['mon'],
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

# ECHAM6:   160.3 ± 7.7 mm/yr,
# ERA5:     173.6 ± 8.7 mm/yr
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate annual circle of aprt frac over AIS

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

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS.pkl', 'wb') as f:
    pickle.dump(aprt_frc_AIS, f)


'''
#-------------------------------- check

#-------- import data
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS.pkl', 'rb') as f:
    aprt_frc_AIS = pickle.load(f)

imask = 'EAIS'
iregion = 'Open ocean'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 21)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()


iregion = 'SH sea ice'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 22)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()

iregion = 'Land excl. Antarctica'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=[16, 17, 19, 20, 21, 22]).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()
np.max(abs(res1 - res2))

iregion = 'Antarctica'
aprt_frc_AIS[imask][iregion].frc_AIS.values



aprt_frc_AIS[imask]['SH sea ice'].frc_AIS - aprt_frc_AIS[imask]['Open ocean'].frc_AIS
aprt_frc_AIS[imask]['Land excl. Antarctica'].frc_AIS - aprt_frc_AIS[imask]['SH sea ice'].frc_AIS
aprt_frc_AIS[imask]['Antarctica'].frc_AIS - aprt_frc_AIS[imask]['Land excl. Antarctica'].frc_AIS

aprt_geo7_spave['AIS']['mm'].sel(wisotype=20).values / aprt_mm_AIS.values

'''
# endregion
# -----------------------------------------------------------------------------

