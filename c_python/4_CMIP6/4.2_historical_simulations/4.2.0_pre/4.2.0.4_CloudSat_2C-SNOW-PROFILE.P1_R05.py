

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
from scipy import stats
import xesmf as xe

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
    rb_colormap,
    quick_var_plot,
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

# endregion
# =============================================================================


# =============================================================================
# region test CloudSat derived data

cloudsat_mon_pre = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE/CloudSat_2C-SNOW-PROFILE_Antarctica_20072010_SRSinf4.nc')
cloudsat_mon_pre_2nd = xr.open_dataset(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE/CloudSat_2C-SNOW-PROFILE_Antarctica_20072010_SRSinf4_1binexcluded.nc')


quick_var_plot(
    var=cloudsat_mon_pre.Snowfall_rate * 24 * 365,
    varname='pre', whicharea='SH',
    xlabel='Annual mean precipitation [$mm\;yr^{-1}$]\nCloudSat 2C-SNOW-PROFILE derived, 2007-2010',
    lon=cloudsat_mon_pre.longitude, lat=cloudsat_mon_pre.latitude,
    outputfile='figures/4_cmip6/4.1_historical/4.1.0_precipitation/4.1.0.0_am_pre/4.1.0.0 SH am_pre CloudSat derived, 2007-2010.png'
)


# endregion
# =============================================================================


