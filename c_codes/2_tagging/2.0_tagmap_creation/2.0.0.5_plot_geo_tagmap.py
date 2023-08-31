

# -----------------------------------------------------------------------------
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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.path import Path

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
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
    zerok,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

#-------------------------------- get land sea mask

# T63 slm and slf
T63GR15_jan_surf = xr.open_dataset(
    'output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0/input/echam/unit.24')
# 1 means land
t63_slf = T63GR15_jan_surf.SLF.values
# 1 means land
t63_slm = T63GR15_jan_surf.SLM.values

#-------------------------------- get model output from the 1st time step
t63_1st_output = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_d_500_wiso/unknown/pi_d_500_wiso_200001.01_echam.nc')

pi_geo_tagmap = xr.open_dataset('startdump/tagging/tagmap/pi_geo_tagmap.nc')


'''
# trimmed slm
esacci_echam6_t63_trim = xr.open_dataset(
    'startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
lon = esacci_echam6_t63_trim.lon.values
lat = esacci_echam6_t63_trim.lat.values
analysed_sst = esacci_echam6_t63_trim.analysed_sst.values

# True means land
# t63_slm_trimmed = np.isnan(analysed_sst)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot pi_geo_tagmap

colors = [
    'salmon', 'darkviolet',
    'royalblue', 'deepskyblue', 'lightblue',
    'darkorange', 'bisque']

pltlevel_o = np.arange(1 - 0.5, 7 +1.5, 1)
pltnorm_o = BoundaryNorm(pltlevel_o, ncolors=len(pltlevel_o), clip=False)
pltcmp_o = cm.get_cmap('PRGn', len(pltlevel_o))

fig, ax = globe_plot(
    figsize=np.array([12.8, 6.4]) / 2.54, add_grid_labels=False,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.01, fm_top=0.99,)

for i in [0, 1, 2, 3, 4, 6, 5]:
    print(i)
    mask_data = pi_geo_tagmap.tagmap.sel(level=i+4).copy().values
    mask_data[mask_data == 0] = np.nan
    mask_data = mask_data * (i + 1)
    
    ax.pcolormesh(
        pi_geo_tagmap.lon, pi_geo_tagmap.lat, mask_data,
        norm=pltnorm_o, cmap=pltcmp_o,
        transform=ccrs.PlateCarree(),)

fig.savefig('figures/test/test.png')




# endregion
# -----------------------------------------------------------------------------



