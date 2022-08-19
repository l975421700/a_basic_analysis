

# -----------------------------------------------------------------------------
# region import packages

import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import numpy as np
import geopandas as gpd
import pickle
from a_basic_analysis.b_module.basic_calculations import (
    create_ais_mask,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create echam6_t63_ais_mask

echam6_t63_slm = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_slm.nc')
echam6_t63_cellarea = xr.open_dataset('scratch/others/land_sea_masks/echam6_t63_cellarea.nc')
ais_shpfile = gpd.read_file('data_sources/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')


echam6_t63_ais_mask = create_ais_mask(
    echam6_t63_slm.lon.values, echam6_t63_slm.lat.values,
    ais_shpfile, echam6_t63_cellarea.cell_area.values
)

with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'wb') as f:
    pickle.dump(echam6_t63_ais_mask, f)


# with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
#     echam6_t63_ais_mask = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region create era5_ais_mask

era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')
ais_shpfile = gpd.read_file('data_sources/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

era5_ais_mask = create_ais_mask(
    era5_cellarea.longitude.values,
    era5_cellarea.latitude.values,
    ais_shpfile,
    era5_cellarea.cell_area.values
)

with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'wb') as f:
    pickle.dump(era5_ais_mask, f)


'''
#-------------------------------- check

# with open('scratch/others/land_sea_masks/echam6_t63_ais_mask.pkl', 'rb') as f:
#     echam6_t63_ais_mask = pickle.load(f)
with open('scratch/others/land_sea_masks/era5_ais_mask.pkl', 'rb') as f:
    era5_ais_mask = pickle.load(f)
era5_cellarea = xr.open_dataset('scratch/cmip6/constants/ERA5_gridarea.nc')

from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,)
from matplotlib.colors import BoundaryNorm
from matplotlib import cm, pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
mpl.rcParams['axes.linewidth'] = 0.2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import xarray as xr
import numpy as np

#-------- cell area is on the right order
era5_ais_mask['cellarea']
# era5: 9742831.164838282 + 2094134.2730679102 + 225967.7696354244 = 12062933.207541618
# echam_t63: 9759740.151912227 + 2144474.2285643886 + 265318.13031548966 = 12169532.510792103

#-------- mask01 looks right

fig, ax = hemisphere_plot(northextent=-60)

# ax.contour(
#     era5_cellarea.longitude.values,
#     era5_cellarea.latitude.values,
#     era5_ais_mask['mask01']['EAIS'],
#     colors='blue', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
# ax.contour(
#     era5_cellarea.longitude.values,
#     era5_cellarea.latitude.values,
#     era5_ais_mask['mask01']['WAIS'],
#     colors='red', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
# ax.contour(
#     era5_cellarea.longitude.values,
#     era5_cellarea.latitude.values,
#     era5_ais_mask['mask01']['AP'],
#     colors='yellow', levels=np.array([0.5]),
#     transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    era5_cellarea.longitude.values,
    era5_cellarea.latitude.values,
    era5_ais_mask['mask01']['AIS'],
    colors='m', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/trial.png')


((era5_ais_mask['mask01']['EAIS'] == 1) == era5_ais_mask['mask']['EAIS']).all()
((era5_ais_mask['mask01']['WAIS'] == 1) == era5_ais_mask['mask']['WAIS']).all()
((era5_ais_mask['mask01']['AP'] == 1) == era5_ais_mask['mask']['AP']).all()
((era5_ais_mask['mask01']['AIS'] == 1) == era5_ais_mask['mask']['AIS']).all()

'''
# endregion
# -----------------------------------------------------------------------------
