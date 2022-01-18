

# =============================================================================
# region import packages

# management
from calendar import monthcalendar
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
import pickle

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
    create_ais_mask,
)

from a00_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion

# region input AIS masks, 1d grid area

grid_area_1d = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')

with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)

ais_area = {}
ais_area['eais'] = grid_area_1d.cell_area.values[ais_masks['eais_mask']].sum()
ais_area['wais'] = grid_area_1d.cell_area.values[ais_masks['wais_mask']].sum()
ais_area['ap'] = grid_area_1d.cell_area.values[ais_masks['ap_mask']].sum()

month_days_79_14 = np.tile(month_days, 36)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import cdo regridded data

mon_pre_cdorg1 = {}
mon_pre_79_14 = {}
mon_pre_spa = {}

#### ERA5
mon_pre_cdorg1['era5'] = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/era5_mon_sl_79_21_pre_cdorg1.nc')
mon_pre_79_14['era5'] = mon_pre_cdorg1['era5'].tp[:, 0, :, :].sel(time=slice(
    '1979-01-01', '2014-12-30')) * 1000

mon_pre_spa['era5'] = {}
mon_pre_spa['era5']['eais'] = \
    (mon_pre_79_14['era5'] * grid_area_1d.cell_area * ais_masks['eais_mask01']
     ).sum(axis=(1, 2)) / ais_area['eais'] * month_days_79_14

mon_pre_spa['era5']['wais'] = \
    (mon_pre_79_14['era5'] * grid_area_1d.cell_area * ais_masks['wais_mask01']
     ).sum(axis=(1, 2)) / ais_area['wais'] * month_days_79_14

mon_pre_spa['era5']['ap'] = \
    (mon_pre_79_14['era5'] * grid_area_1d.cell_area * ais_masks['ap_mask01']
     ).sum(axis=(1, 2)) / ais_area['ap'] * month_days_79_14


# #### HadGEM3-GC31-LL, historical, r1i1p1f3
# mon_pre_cdorg1['hg3_ll_hist1'] = xr.open_dataset(
#     'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/hg3_ll_hi_r1_mon_pre_cdorg1.nc')


# #### AWI-ESM-1-1-LR, historical, r1i1p1f1
# mon_pre_cdorg1['awe_lr_hist1'] = xr.open_dataset(
#     'bas_palaeoclim_qino/scratch/cmip6/historical/mon_pre/awe_lr_hi_r1_mon_pre_cdorg1.nc')


# endregion
# =============================================================================




