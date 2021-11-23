

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
from scipy import stats

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

#### HadGEM3-GC31-LL


### piControl

## r1i1p1f1, (only one)

# siconc, Sea-Ice Area Percentage (Ocean Grid), [%]
siconc_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))

# tas, Near-Surface Air Temperature, [K]
tas_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Amon/tas/gn/latest/tas_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))

# psl, Sea Level Pressure, [Pa]
psl_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Amon/psl/gn/latest/psl_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))

# pr, Precipitation, [kg m-2 s-1]
pr_hg3_ll_pi_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Amon/pr/gn/latest/pr_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_*.nc',
)))


### historical

## r1i1p1f3  (r2i1p1f3  r3i1p1f3  r4i1p1f3  r5i1p1f3)
# siconc, Sea-Ice Area Percentage (Ocean Grid), [%]
siconc_hg3_ll_hi_r1_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc',
)))

## (r1i1p1f3  r2i1p1f3)  r3i1p1f3  (r4i1p1f3  r5i1p1f3)
# siconc, Sea-Ice Area Percentage (Ocean Grid), [%]
siconc_hg3_ll_hi_r3_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r3i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r3i1p1f3_gn_*.nc',
)))

## (r1i1p1f3  r2i1p1f3  r3i1p1f3)  r4i1p1f3  (r5i1p1f3)
# siconc, Sea-Ice Area Percentage (Ocean Grid), [%]
siconc_hg3_ll_hi_r4_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r4i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r4i1p1f3_gn_*.nc',
)))

## (r1i1p1f3  r2i1p1f3  r3i1p1f3  r4i1p1f3) r5i1p1f3
# siconc, Sea-Ice Area Percentage (Ocean Grid), [%]
siconc_hg3_ll_hi_r5_fl = np.array(sorted(glob.glob(
    '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r5i1p1f3/SImon/siconc/gn/latest/siconc_SImon_HadGEM3-GC31-LL_historical_r5i1p1f3_gn_*.nc',
)))


# fig, ax = framework_plot1("global")
# fig.savefig('figures/00_test/trial.png', dpi=600)


# stats.describe(siconc_simon_hg3_ll_pi.latitude.values, axis=None)


'''
Directory structure = <mip_era>/<activity_id>/<institution_id>/<source_id>/<experiment_id>/<member_id>/<table_id>/<variable_id>/<grid_label>/<version>

# siconc_hg3_ll_pi = xr.open_mfdataset(
    #     siconc_hg3_ll_pi_fl, data_vars='minimal',
    #     coords='minimal', compat='override',
    # )
# pltlevel = np.arange(0, 1.01, 0.01)
# pltticks = np.arange(0, 1.01, 0.2)

# fig, ax = hemisphere_plot(northextent=-30, sb_length=2000, sb_barheight=200,)

# plt_cmp = ax.pcolormesh(
#     siconc_simon_hg3_ll_pi.longitude,
#     siconc_simon_hg3_ll_pi.latitude,
#     siconc_simon_hg3_ll_pi.siconc[0, :, :],
#     norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
#     cmap=cm.get_cmap('Blues', len(pltlevel)), rasterized=True,
#     transform=ccrs.PlateCarree(),)
# cbar = fig.colorbar(
#     plt_cmp, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
#     shrink=1, aspect=25, ticks=pltticks, extend='neither',
#     anchor=(0.5, 1), panchor=(0.5, 0))
# cbar.ax.set_xlabel(
#     "sicon in HadGEM3-GC31-LL, piControl on " + \
#     str(siconc_simon_hg3_ll_pi.siconc[0, :, :].time.values)[0:10])

# fig.savefig('figures/00_test/trial.png')


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


# trying baspy
import sys  # print(sys.path)
sys.path.append('/home/users/qino/git_repo')
import baspy as bp
## Retrieve a filtered version of the CMIP5 catalogue as a Pandas DataFrame
df = bp.catalogue(
    dataset='cmip6', Model='HadGEM3-GC31-LL', RunID='r1i1p1f1', CMOR='SImon',
    Experiment='piControl', Var=['siconc'])

import re
re.split(';', str(df.DataFiles.values)[2:-2])

'''
# endregion
# =============================================================================





