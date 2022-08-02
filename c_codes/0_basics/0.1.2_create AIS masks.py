

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
