

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

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
from scipy.stats import pearsonr
from scipy.stats import linregress
import metpy.calc as mpcalc
from metpy.units import units

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
import cartopy.feature as cfeature
from matplotlib.ticker import AutoMinorLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.path as mpath
from metpy.calc import pressure_to_height_std
from metpy.units import units
from windrose import WindroseAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    remove_trailing_zero_pos_abs,
    ticks_labels,
    hemisphere_conic_plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
    regrid,
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
    plot_labels,
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
    plot_t63_contourf,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get hourly data

filelist_AWS = sorted(glob.glob('data_sources/AWS/Climantartide_6577cf0dbf2f3/Concordia_*_TPHVD.txt'))
AWS_Dome_C = {}

for iyear, ifile in zip(np.arange(2005, 2023, 1), range(len(filelist_AWS))):
    print('#-------------------------------- ' + str(ifile) + ': ' + str(iyear))
    # print(filelist_AWS[ifile])
    
    AWS_Dome_C[str(iyear)] = pd.read_csv(
        filelist_AWS[ifile], sep = '\s+', header=0, na_values='Null',
        dtype={'Temp': np.float64, 'Pres': np.float64, 'Rh': np.float64,
               'Vel': np.float64, 'Dir': np.float64})
    
    # print(AWS_Dome_C[str(iyear)])

AWS_Dome_C['1h'] = AWS_Dome_C['2005'].copy()

for iyear in np.arange(2006, 2023, 1):
    # iyear = 2006
    print('#-------------------------------- ' + str(iyear))
    
    AWS_Dome_C['1h'] = pd.concat([
        AWS_Dome_C['1h'],
        AWS_Dome_C[str(iyear)]
    ], ignore_index=True)

AWS_Dome_C['1h']['time'] = pd.to_datetime([date + 'T' + time for date,time in zip(AWS_Dome_C['1h']['DateTime'], AWS_Dome_C['1h']['UTC'])])

AWS_Dome_C['1h'] = AWS_Dome_C['1h'].drop(columns=['DateTime', 'UTC'])


output_file = 'data_sources/AWS/Climantartide_6577cf0dbf2f3/AWS_Dome_C.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(AWS_Dome_C, f)




'''
with open('data_sources/AWS/Climantartide_6577cf0dbf2f3/AWS_Dome_C.pkl', 'rb') as f:
    AWS_Dome_C = pickle.load(f)

AWS_Dome_C['1h'].resample('1d', on='time').mean().reset_index()

# for iyear in np.arange(2005, 2023, 1):
#     print('#-------------------------------- ' + str(iyear))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get daily data

with open('data_sources/AWS/Climantartide_6577cf0dbf2f3/AWS_Dome_C.pkl', 'rb') as f:
    AWS_Dome_C = pickle.load(f)

wind_u, wind_v = mpcalc.wind_components(AWS_Dome_C['1h']['Vel'].values * units('m/s'), AWS_Dome_C['1h']['Dir'].values * units.deg)

AWS_Dome_C['1h']['wind_u'] = wind_u.magnitude
AWS_Dome_C['1h']['wind_v'] = wind_v.magnitude

AWS_Dome_C['1d'] = AWS_Dome_C['1h'].resample('1d', on='time').mean().reset_index()



# endregion
# -----------------------------------------------------------------------------


