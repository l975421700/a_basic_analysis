

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_1d_803_6.0', 'pi_1d_804_6.1']
i=0

output_dir = exp_odir + expid[i] + '/analysis/echam/'

ifile_start = 12 #0 #120
ifile_end   = 24 # 516 #1740 #840

ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0,  0, 30]
itag=15

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
    mon_sea_ann,
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
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

fl_wiso_daily = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso.nc'
        ))

exp_out_wiso_daily = xr.open_mfdataset(
    fl_wiso_daily[ifile_start:ifile_end],
    )

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region set indices

kwiso2 = 3

kstart = kwiso2 + sum(ntags[:itag])
kend   = kwiso2 + sum(ntags[:(itag+1)])

print(kstart); print(kend)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate source RHsst - binning tagmap with RHsst

RHsstbins = np.concatenate((np.arange(0, 1.401 + 1e-4, 0.05), np.array([2])))
RHsstbins_mid = np.arange(0.025, 1.425 + 1e-4, 0.05)

RHsst_binned_pre = (exp_out_wiso_daily.wisoaprl.sel(wisotype=slice(kstart+2, kend)) + exp_out_wiso_daily.wisoaprc.sel(wisotype=slice(kstart+2, kend))).compute()
ocean_pre = RHsst_binned_pre.sum(dim='wisotype').compute()

ocean_pre_alltime      = mon_sea_ann(ocean_pre)
RHsst_binned_pre_alltime = mon_sea_ann(RHsst_binned_pre)

pre_weighted_RHsst = {}

for ialltime in ['daily', 'mon', 'sea', 'ann',]:
    print(ialltime)
    print(RHsst_binned_pre_alltime[ialltime].shape)
    pre_weighted_RHsst[ialltime] = ((RHsst_binned_pre_alltime[ialltime] * RHsstbins_mid[None, :, None, None]).sum(dim='wisotype') / ocean_pre_alltime[ialltime]).compute()
    pre_weighted_RHsst[ialltime].values[ocean_pre_alltime[ialltime].values < 1e-9] = np.nan
    pre_weighted_RHsst[ialltime] = pre_weighted_RHsst[ialltime].rename('pre_weighted_RHsst')

output_file = exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_RHsst.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(pre_weighted_RHsst, f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region cross check source RHsst

pre_weighted_RHsst = {}

with open(exp_odir + expid[0] + '/analysis/echam/' + expid[0] + '.pre_weighted_RHsst.pkl', 'rb') as f:
    pre_weighted_RHsst[expid[0]] = pickle.load(f)

with open(exp_odir + expid[1] + '/analysis/echam/' + expid[1] + '.pre_weighted_RHsst.pkl', 'rb') as f:
    pre_weighted_RHsst[expid[1]] = pickle.load(f)


output_png = 'figures/test/test.png'

lon = pre_weighted_RHsst[expid[0]]['am'].lon
lat = pre_weighted_RHsst[expid[0]]['am'].lat

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=60, cm_max=80, cm_interval1=1, cm_interval2=2, cmap='viridis',)
pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
    cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1, cmap='PiYG',)

cbar_label1 = 'Source RHsst [$\%$]'
cbar_label2 = 'Differences [$\%$]'

nrow = 1
ncol = 3
fm_right = 1 - 4 / (8.8*ncol + 4)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol + 4, 5*nrow]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol], add_grid_labels=False)

plt_mesh1 = plot_t63_contourf(
    lon, lat, pre_weighted_RHsst[expid[0]]['ann'][0] * 100, axs[0],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh2 = plot_t63_contourf(
    lon, lat, pre_weighted_RHsst[expid[1]]['ann'][0] * 100, axs[1],
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

plt_mesh3 = plot_t63_contourf(
    lon, lat, (pre_weighted_RHsst[expid[0]]['ann'][0] - \
        pre_weighted_RHsst[expid[1]]['ann'][0]) * 100, axs[2],
    pltlevel2, 'both', pltnorm2, pltcmp2, ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, '(a) Scaled-flux water tracing',
    transform=axs[0].transAxes, ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, '(b) Prescribed-region water tracing (5$\%$ RHsst bins)',
    transform=axs[1].transAxes, ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, '(c) Differences: (a) - (b)',
    transform=axs[2].transAxes, ha='center', va='center', rotation='horizontal')

cbar2 = fig.colorbar(
    plt_mesh3, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(0.8, 0.5),
    ticks=pltticks2)
cbar2.ax.set_ylabel(cbar_label2, linespacing=1.5)

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="vertical",shrink=1.2,aspect=40,extend='both',
    anchor=(3.2, 0.5),
    ticks=pltticks)
cbar1.ax.set_ylabel(cbar_label1, linespacing=1.5)

fig.subplots_adjust(left=0.005, right = fm_right, bottom = 0.005, top = 0.93)
fig.savefig(output_png)


echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

diff = abs(pre_weighted_RHsst[expid[0]]['ann'][0] - \
        pre_weighted_RHsst[expid[1]]['ann'][0]).values
np.nanmax(diff)
np.nanmean(diff)
np.average(
    diff[np.isfinite(diff)],
    weights = echam6_t63_cellarea.cell_area.values[np.isfinite(diff)],
)



'''
pre_weighted_RHsst['pi_1d_803_6.0']['ann'].to_netcdf('scratch/test/test0.nc')
pre_weighted_RHsst['pi_1d_804_6.1']['ann'].to_netcdf('scratch/test/test1.nc')
((pre_weighted_RHsst['pi_1d_803_6.0']['ann'] - pre_weighted_RHsst['pi_1d_804_6.1']['ann']).compute()).to_netcdf('scratch/test/test2.nc')
'''
# endregion
# -----------------------------------------------------------------------------
