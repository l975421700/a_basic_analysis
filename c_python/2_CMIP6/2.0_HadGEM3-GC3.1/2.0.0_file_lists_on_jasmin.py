

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')


# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
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
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})


# self defined
from a00_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
)


# endregion
# =============================================================================


# =============================================================================
# region file list

#### monthly sea ice concentration, HadGEM3-GC31-LL, piControl,
fl_siconc_simon_hg3_ll_pi = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
    )))

siconc_simon_hg3_ll_pi = xr.open_mfdataset(
    fl_siconc_simon_hg3_ll_pi, data_vars='minimal',
    coords='minimal', compat='override',
    )

# fig, ax = framework_plot1("global")
# fig.savefig('figures/00_test/trial.png', dpi=600)

pltlevel = np.arange(0, 1.01, 0.01)
pltticks = np.arange(0, 1.01, 0.2)

fig, ax = hemisphere_plot(northextent=-30, sb_length=2000, sb_barheight=200,)

plt_cmp = ax.pcolormesh(
    siconc_simon_hg3_ll_pi.longitude,
    siconc_simon_hg3_ll_pi.latitude,
    siconc_simon_hg3_ll_pi.siconc[0, :, :],
    norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
    transform=ccrs.PlateCarree(),)
cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=pltticks, extend='neither',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "sicon in HadGEM3-GC31-LL, piControl on " + \
    str(siconc_simon_hg3_ll_pi.siconc[0, :, :].time.values)[0:10])

fig.savefig('figures/00_test/trial.png')

# stats.describe(siconc_simon_hg3_ll_pi.latitude.values, axis=None)


'''
#### MIPs
# /badc/cmip6/data/CMIP6/CMIP
# /badc/cmip6/data/CMIP6/ScenarioMIP
# /badc/cmip6/data/CMIP6/PMIP


#### Model centres in CMIP
# /badc/cmip6/data/CMIP6/CMIP/MOHC
# /badc/cmip6/data/CMIP6/CMIP/AWI


#### Model configuarations
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM


#### Experiments
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/1pctCO2
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/abrupt-4xCO2
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/amip
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical


#### Variant
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1


#### Variables and resolutions
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc


#### Variables
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest


#### Example files
# /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_185001-194912.nc

'''
# endregion
# =============================================================================





