

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
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
from scipy import stats
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
import pycircstat as circ
from scipy.stats import circstd
import cmip6_preprocessing.preprocessing as cpp

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
import cartopy.feature as cfeature

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
    find_ilat_ilon,
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


chadwick2021 = pd.read_csv(
    'data_sources/LIG/Chadwick-etal_2021.tab', sep='\t', header=0, skiprows=43)



'''
chadwick2021.Event.iloc[
    np.where(np.floor(
        chadwick2021['Age [ka BP] (Age model, EDC3 (EPICA Ice Do...)']
        ) == 127)]

# 7 cores
with open('scratch/cmip6/lig/lig_sic_regrid_alltime.pkl', 'rb') as f:
    lig_sic_regrid_alltime = pickle.load(f)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region linearly interpolate reconstructions

chadwick_interp = pd.DataFrame(
    data = {
        'sites': chadwick2021.Event.unique(),
        'lat': 0.,
        'lon': 0.,
        'age [ka]': 127,
        'sic_sep': 0.,
        'sst_sum': 0.,
    }
)

for isite in chadwick2021.Event.unique():
    # isite = 'TPC288'
    print(isite)
    
    edc3_ages = chadwick2021.loc[
        chadwick2021.Event == isite,
        'Age [ka BP] (Age model, EDC3 (EPICA Ice Do...)',
        ]
    
    sic_sep = chadwick2021.loc[
        chadwick2021.Event == isite,
        'Sea ice con (9) [%] (Modern analog technique (MAT))',
        ]
    
    sst_sum = chadwick2021.loc[
        chadwick2021.Event == isite,
        'SST sum [°C] (Modern analog technique (MAT))',
        ]
    
    chadwick_interp.loc[chadwick_interp.sites == isite, 'lat'] = \
        chadwick2021.loc[chadwick2021.Event == isite, 'Latitude'].iloc[0]
    chadwick_interp.loc[chadwick_interp.sites == isite, 'lon'] = \
        chadwick2021.loc[chadwick2021.Event == isite, 'Longitude'].iloc[0]
    
    chadwick_interp.loc[chadwick_interp.sites == isite, 'sic_sep'] = \
        np.interp(127, edc3_ages, sic_sep)
    
    chadwick_interp.loc[chadwick_interp.sites == isite, 'sst_sum'] = \
        np.interp(127, edc3_ages, sst_sum)

with open('scratch/cmip6/lig/chadwick_interp.pkl', 'wb') as f:
    pickle.dump(chadwick_interp, f)

'''
with open('scratch/cmip6/lig/chadwick_interp.pkl', 'rb') as f:
    chadwick_interp = pickle.load(f)

chadwick_interp

'''
# endregion
# -----------------------------------------------------------------------------

