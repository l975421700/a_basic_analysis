

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
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

# self defined
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
    quick_var_plot,
    mesh2plot,
    framework_plot1,
    remove_trailing_zero,
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
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

lon = aprt_geo7_alltime[expid[i]]['am'].lon
lat = aprt_geo7_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get aprt_frc

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

aprt_frc = {}
for iregion in geo_regions:
    aprt_frc[iregion] = {}

aprt_frc['Otherland']  = {}
aprt_frc['Otherocean']  = {}

for ialltime in aprt_geo7_alltime[expid[i]].keys():
    if (ialltime != 'sum'):
        # ialltime = 'daily'
        print('#---- ' + ialltime)

        for iregion in geo_regions:
            # iregion = 'NHland'
            print(iregion + ': ' + str(wisotypes[iregion]))

            aprt_frc[iregion][ialltime] = \
                100 * (aprt_geo7_alltime[expid[i]][ialltime].sel(
                    wisotype=wisotypes[iregion]) / \
                        aprt_geo7_alltime[expid[i]]['sum'][ialltime])

for ialltime in aprt_frc['Antarctica'].keys():
    aprt_frc['Otherland'][ialltime] = \
        aprt_frc['NHland'][ialltime] + aprt_frc['SHland'][ialltime]
    aprt_frc['Otherocean'][ialltime] = \
        aprt_frc['NHocean'][ialltime] + aprt_frc['NHseaice'][ialltime] + \
            aprt_frc['SHocean'][ialltime]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc.pkl', 'wb') as f:
    pickle.dump(aprt_frc, f)


'''
#-------- check calculation
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc.pkl', 'rb') as f:
    aprt_frc = pickle.load(f)

print((aprt_frc['SHseaice']['am'] == 100 * (aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=22) / aprt_geo7_alltime[expid[i]]['sum']['am'])).all().values)

print((aprt_frc['SHocean']['mon'] == 100 * (aprt_geo7_alltime[expid[i]]['mon'].sel(wisotype=21) / aprt_geo7_alltime[expid[i]]['sum']['mon'])).all().values)

print((aprt_frc['Antarctica']['ann'] == 100 * (aprt_geo7_alltime[expid[i]]['ann'].sel(wisotype=18) / aprt_geo7_alltime[expid[i]]['sum']['ann'])).all().values)

print((aprt_frc['Otherland']['mon'] == (aprt_frc['NHland']['mon'] + aprt_frc['SHland']['mon'])).all().values)

print((aprt_frc['Otherocean']['ann'] == (aprt_frc['NHocean']['ann'] + aprt_frc['NHseaice']['ann'] + aprt_frc['SHocean']['ann'])).all().values)


(100 * (aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=22) / aprt_geo7_alltime[expid[i]]['sum']['am']) == (100 * aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=22) / aprt_geo7_alltime[expid[i]]['sum']['am'])).all()
np.max(abs(100 * (aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=22) / aprt_geo7_alltime[expid[i]]['sum']['am']) - (100 * aprt_geo7_alltime[expid[i]]['am'].sel(wisotype=22) / aprt_geo7_alltime[expid[i]]['sum']['am'])))
'''
# endregion
# -----------------------------------------------------------------------------


