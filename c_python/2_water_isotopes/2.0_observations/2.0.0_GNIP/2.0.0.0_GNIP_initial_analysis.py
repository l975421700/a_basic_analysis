

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
from datetime import datetime

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
)

from a00_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a00_basic_analysis.b_module.namelist import (
    month_days,
)

# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region import GNIP station data and cleaning

gnip_data = pd.read_excel(
    'bas_palaeoclim_qino/observations/products/GNIP/2021-12-01_GNIP_Snapshot.xlsx',
)

gnip_dropna = gnip_data.dropna(subset=['O18', 'H2']).copy()

gnip_site_count = gnip_dropna.groupby(['Site'], as_index=False).size()

gnip_site_loc = gnip_dropna.groupby(['Site'], as_index=False).nth(0)
# gnip_snapshot.groupby(['Site'], as_index=False).nth()

gnip_site = pd.merge(
    gnip_site_loc[['Site', 'Longitude', 'Latitude']],
    gnip_site_count, on='Site', how='left')

'''
gnip_snapshot.dtypes

for i in gnip_snapshot.columns:
    print(str(i) + ':\n')
    if gnip_snapshot[i].dtype == 'O':
        print(gnip_snapshot[i].value_counts())
    else:
        print(stats.describe(gnip_snapshot[i], nan_policy = 'omit'))
    print('\n')

stats.describe(gnip_snapshot['O18'], nan_policy = 'omit') # 79209/141154
stats.describe(gnip_snapshot['H2'], nan_policy = 'omit') # 71800/141154
dropna: 70747/141154
stats.describe(gnip_snapshot_dropna['O18'])
stats.describe(gnip_snapshot_dropna['H2'])

len(np.unique(gnip_snapshot_dropna['Site']))
len(np.unique(gnip_snapshot['Site']))
'''
# endregion
# =============================================================================


# =============================================================================
# region plot GNIP stations with 018 and H2 records

fig, ax = framework_plot1("global")

plt_se1_center = ax.scatter(
    gnip_site['Longitude'], gnip_site['Latitude'],
    s=1.5, c=np.log(gnip_site['size']),
    cmap='Blues', edgecolors='r', linewidths=0.15, zorder=2)

fig.savefig(
    'figures/5_water_isotopes/5.0_observations/5.0.0_GNIP/5.0.0.0_global distribution of GNIP stations.png')

'''
stats.describe(gnip_site['size'])
'''

# endregion
# =============================================================================
