


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
import pandas as pd

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
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
    awi_esm_odir,
)

# endregion
# =============================================================================


# =============================================================================
# region import output

expid = 'pi_final_qg_tag3'
yrstart = 2000
yrend = 2001


awi_esm_o = {}

#### pi_final_qg_tag3

awi_esm_o[expid] = {}

## echam
awi_esm_o[expid]['echam'] = {}

awi_esm_o[expid]['echam']['echam'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.nc')

awi_esm_o[expid]['echam']['echam_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.am.nc')

awi_esm_o[expid]['echam']['echam_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_echam.ann.nc')

## wiso
awi_esm_o[expid]['wiso'] = {}

awi_esm_o[expid]['wiso']['wiso'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.nc')

awi_esm_o[expid]['wiso']['wiso_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.am.nc')

awi_esm_o[expid]['wiso']['wiso_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso.ann.nc')

## wiso_d
awi_esm_o[expid]['wiso_d'] = {}

awi_esm_o[expid]['wiso_d']['wiso_d'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.nc')

awi_esm_o[expid]['wiso_d']['wiso_d_am'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.am.nc')

awi_esm_o[expid]['wiso_d']['wiso_d_ann'] = xr.open_dataset(
    awi_esm_odir + expid + '/analysis/echam/' + expid + '_' + str(yrstart) + '_' + str(yrend) + '.01_wiso_d.ann.nc')


# endregion
# =============================================================================


# =============================================================================
# region check water tagging

awi_esm_o[expid]['wiso']['wiso']

# endregion
# =============================================================================




