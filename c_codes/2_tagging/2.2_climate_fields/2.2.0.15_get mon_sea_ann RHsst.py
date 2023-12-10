

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'hist_700_5.0',
    # 'nudged_701_5.0',
    # 'pi_1d_803_6.0',
    # 'nudged_705_6.0',
    'nudged_703_6.0_k52',
    ]
i=0


ifile_start = 0 #12 #0 #120
ifile_end   = 480 #528 #24 # 516 #1740 #840


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
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from a_basic_analysis.b_module.namelist import (
    zerok,
)

from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_surf = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '_??????.01_surf.nc'))

exp_org_o[expid[i]]['surf'] = xr.open_mfdataset(
    filenames_surf[ifile_start:ifile_end],
    )


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann RHsst

RHsst = (exp_org_o[expid[i]]['surf'].zqklevw / exp_org_o[expid[i]]['surf'].zqsw * 100).compute()

RHsst_alltime = {}
RHsst_alltime[expid[i]] = mon_sea_ann(var_daily=RHsst)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.RHsst_alltime.pkl', 'wb') as f:
    pickle.dump(RHsst_alltime[expid[i]], f)


'''
RHsst_am = (exp_org_o[expid[i]]['surf'].zqklevw / exp_org_o[expid[i]]['surf'].zqsw).mean(dim='time').compute()

# plot it

output_png = 'figures/test/test1.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=50, cm_max=90, cm_interval1=4, cm_interval2=8, cmap='viridis',)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)

plt_mesh1 = plot_t63_contourf(
    RHsst_am.lon, RHsst_am.lat,
    RHsst_am * 100, ax,
    pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('RHsst [$\%$]', linespacing=1.5,)
fig.savefig(output_png, dpi=600)


'''
# endregion
# -----------------------------------------------------------------------------
