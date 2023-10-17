

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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs

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
from scipy.stats import pearsonr
from matplotlib.ticker import AutoMinorLocator

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
    regrid,
    mean_over_ais,
    time_weighted_mean,
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
    cplot_ttest,
    xr_par_cor,
)

from a_basic_analysis.b_module.component_plot import (
    cplot_ice_cores,
    plt_mesh_pars,
    plot_t63_contourf,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region clean NK16 data

data_files = {
    '13summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0001.txt',
    '14summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0002.txt',
    '15summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0003.txt',
    '16summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0004.txt',
    '17summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0005.txt',
    '18summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0006.txt',
    '19summer': 'data_sources/water_isotopes/NK16/nipr_parc_364_0007.txt',
}

NK16_Australia_Syowa = {}

for iperiod in data_files.keys():
    # iperiod = '13summer'
    print('#---------------- ' + iperiod)
    print(data_files[iperiod])
    
    NK16_Australia_Syowa[iperiod] = pd.read_csv(
        data_files[iperiod],
        sep = '\s+', header=0, skiprows=92,
    )
    
    NK16_Australia_Syowa[iperiod]['#yy-mm-ddThh:mm'] = NK16_Australia_Syowa[iperiod]['#yy-mm-ddThh:mm'].astype('datetime64[ns]')
    NK16_Australia_Syowa[iperiod]['yy-mm-ddThh:mm'] = NK16_Australia_Syowa[iperiod]['yy-mm-ddThh:mm'].astype('datetime64[ns]')
    
    NK16_Australia_Syowa[iperiod]['lat'] = (NK16_Australia_Syowa[iperiod]['lat[N]'] + NK16_Australia_Syowa[iperiod]['lat[N].1']) / 2
    NK16_Australia_Syowa[iperiod]['lon'] = (NK16_Australia_Syowa[iperiod]['lon[E]'] + NK16_Australia_Syowa[iperiod]['lon[E].1']) / 2
    
    NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod].rename(columns={
        '#yy-mm-ddThh:mm': 'time',
        'Ps': 'pressure',
        'Ta': 't_air',
        'RH': 'rh',
        'RHsst': 'rh_sst',
        'Tsst': 'sst',
        'H2O': 'humidity',
        'Deu': 'dD',
        'Oxy': 'd18O',
    })
    
    NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][[
        'time', 'lat', 'lon', 'dD', 'd18O', 't_air', 'sst', 'rh',
        'rh_sst', 'humidity']]
    
    if (iperiod == '13summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][21:-14]
    elif (iperiod == '14summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][22:-1]
    elif (iperiod == '15summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][22:-22]
    elif (iperiod == '16summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][22:]
    elif (iperiod == '17summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][22:-2]
    elif (iperiod == '18summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][6:]
    elif (iperiod == '19summer'):
        NK16_Australia_Syowa[iperiod] = NK16_Australia_Syowa[iperiod][7:-15]
    
    NK16_Australia_Syowa[iperiod + '6h'] = NK16_Australia_Syowa[iperiod].resample('6h', on='time').mean().reset_index()
    NK16_Australia_Syowa[iperiod + '1d'] = NK16_Australia_Syowa[iperiod].resample('1d', on='time').mean().reset_index()


NK16_Australia_Syowa['1h'] = pd.concat(
    [NK16_Australia_Syowa['13summer'],
     NK16_Australia_Syowa['14summer'],
     NK16_Australia_Syowa['15summer'],
     NK16_Australia_Syowa['16summer'],
     NK16_Australia_Syowa['17summer'],
     NK16_Australia_Syowa['18summer'],
     NK16_Australia_Syowa['19summer'],],
    ignore_index=True
)

NK16_Australia_Syowa['6h'] = pd.concat(
    [NK16_Australia_Syowa['13summer6h'],
     NK16_Australia_Syowa['14summer6h'],
     NK16_Australia_Syowa['15summer6h'],
     NK16_Australia_Syowa['16summer6h'],
     NK16_Australia_Syowa['17summer6h'],
     NK16_Australia_Syowa['18summer6h'],
     NK16_Australia_Syowa['19summer6h'],],
    ignore_index=True
)

NK16_Australia_Syowa['1d'] = pd.concat(
    [NK16_Australia_Syowa['13summer1d'],
     NK16_Australia_Syowa['14summer1d'],
     NK16_Australia_Syowa['15summer1d'],
     NK16_Australia_Syowa['16summer1d'],
     NK16_Australia_Syowa['17summer1d'],
     NK16_Australia_Syowa['18summer1d'],
     NK16_Australia_Syowa['19summer1d'],],
    ignore_index=True
)

for ialltime in ['1h', '6h', '1d']:
    # ialltime = '1h'
    print('#------------------------ ' + ialltime)
    
    NK16_Australia_Syowa[ialltime]['d_xs'] = NK16_Australia_Syowa[ialltime]['dD'] - 8 * NK16_Australia_Syowa[ialltime]['d18O']
    
    ln_dD = 1000 * np.log(1 + NK16_Australia_Syowa[ialltime]['dD'] / 1000)
    ln_d18O = 1000 * np.log(1 + NK16_Australia_Syowa[ialltime]['d18O'] / 1000)
    
    NK16_Australia_Syowa[ialltime]['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

output_file = 'data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl'

if (os.path.isfile(output_file)):
    os.remove(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(NK16_Australia_Syowa, f)




with open('data_sources/water_isotopes/NK16/NK16_Australia_Syowa.pkl', 'rb') as f:
    NK16_Australia_Syowa = pickle.load(f)


'''
iperiod = '13summer'
NK16_Australia_Syowa[iperiod].columns

#---------------- check time intervals
print(np.max(NK16_Australia_Syowa[iperiod]['yy-mm-ddThh:mm'] - NK16_Australia_Syowa[iperiod]['#yy-mm-ddThh:mm']))

#---------------- check column types
iperiod = '13summer'

for icol in NK16_Australia_Syowa[iperiod].columns:
    print(icol)
    
    print(NK16_Australia_Syowa[iperiod][icol])

#---------------- print imported data
for iperiod in data_files.keys():
    print('#---------------- ' + iperiod)
    print(NK16_Australia_Syowa[iperiod])

for ialltime in ['1h', '6h', '1d']:
    print('#---------------- ' + ialltime)
    print(NK16_Australia_Syowa[ialltime])


#---------------- import separately
NK16_Australia_Syowa['13summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0001.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['14summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0002.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['15summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0003.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['16summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0004.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['17summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0005.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['18summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0006.txt',
    sep = '\s+', header=0, skiprows=92,
)

NK16_Australia_Syowa['19summer'] = pd.read_csv(
    'data_sources/water_isotopes/NK16/nipr_parc_364_0007.txt',
    sep = '\s+', header=0, skiprows=92,
)

'''
# endregion
# -----------------------------------------------------------------------------




