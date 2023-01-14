

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_502_5.0',
    ]
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
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ

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
from matplotlib.path import Path

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
)

from a_basic_analysis.b_module.namelist import (
    month,
    month_num,
    month_dec,
    month_dec_num,
    seasons,
    seasons_last_num,
    hours,
    months,
    month_days,
    zerok,
    panel_labels,
    seconds_per_d,
)

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

from a_basic_analysis.b_module.statistics import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

# T63 slm and slf
T63GR15_jan_surf = xr.open_dataset(
    '/work/ollie/pool/ECHAM6/input/r0007/T63/T63GR15_jan_surf.nc')
t63_slm = T63GR15_jan_surf.SLM
lon = T63GR15_jan_surf.lon.values
lat = T63GR15_jan_surf.lat.values

#-------------------------------- get ocean basin divisions

lon2, lat2 = np.meshgrid(lon, lat)
coors = np.hstack((lon2.reshape(-1, 1), lat2.reshape(-1, 1)))

atlantic_path1 = Path([
    (0, 90), (100, 90), (100, 30), (20, 30), (20, -90), (0, -90), (0, 90)])
atlantic_path2 = Path([
    (360, 90), (360, -90), (290, -90), (290, 8), (279, 8), (270, 15),
    (260, 20), (260, 90), (360, 90)])
pacific_path = Path([
    (100, 90), (260, 90), (260, 20), (270, 15), (279, 8), (290, 8), (290, -90),
    (140, -90), (140, -30), (130, -30), (130, -10), (100, 0), (100, 30),
    (100, 90)])
indiano_path = Path([
    (100, 30), (100, 0), (130, -10), (130, -30), (140, -30), (140, -90),
    (20, -90), (20, 30), (100, 30)])

atlantic_mask1 = atlantic_path1.contains_points(coors, radius = -0.5).reshape(lon2.shape)
atlantic_mask2 = atlantic_path2.contains_points(coors).reshape(lon2.shape)
atlantic_mask = atlantic_mask1 | atlantic_mask2
pacific_mask = pacific_path.contains_points(coors).reshape(lon2.shape)
indiano_mask = indiano_path.contains_points(coors).reshape(lon2.shape)

t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate area

atlantic_area = t63_cellarea.cell_area.values[
    (t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & atlantic_mask].sum()
pacific_area = t63_cellarea.cell_area.values[
    (t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & pacific_mask].sum()
indiano_area = t63_cellarea.cell_area.values[
    (t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & indiano_mask].sum()

print(indiano_area / atlantic_area)
# 1.44
print(pacific_area / atlantic_area)
# 2.30

'''
((t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & atlantic_mask).to_netcdf('scratch/test/test1.nc')
((t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & pacific_mask).to_netcdf('scratch/test/test.nc')
((t63_slm == 0) & (lat2 < 0) & (lat2 > -50) & indiano_mask).to_netcdf('scratch/test/test.nc')

'''
# endregion
# -----------------------------------------------------------------------------

