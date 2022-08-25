

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]
i = 0

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import os

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
import pickle

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

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
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

from a_basic_analysis.b_module.source_properties import (
    source_properties,
    calc_lon_diff,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

pre_weighted_sst = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sst.pkl', 'rb') as f:
    pre_weighted_sst[expid[i]] = pickle.load(f)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean values

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/6.1.3.2 ' + expid[i] + ' pre_weighted_sst am Antarctica.png'

pltlevel = np.arange(8, 18 + 1e-4, 1)
pltticks = np.arange(8, 18 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7]) / 2.54)

plt1 = ax.pcolormesh(
    pre_weighted_sst[expid[i]]['am'].lon,
    pre_weighted_sst[expid[i]]['am'].lat,
    pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Source SST [$°C$]\n ', linespacing=2)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------

