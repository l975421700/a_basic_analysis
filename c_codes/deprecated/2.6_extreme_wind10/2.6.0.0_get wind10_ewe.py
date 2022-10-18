

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

wind10_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_alltime.pkl', 'rb') as f:
    wind10_alltime[expid[i]] = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

lon = wind10_alltime[expid[i]]['am'].lon
lat = wind10_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

quantiles = {'90%': 0.9, '95%': 0.95, '99%': 0.99}


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wind10_ewe

wind10_ewe = {}
wind10_ewe[expid[i]] = {}
wind10_ewe[expid[i]]['quantiles'] = {}
wind10_ewe[expid[i]]['mask'] = {}

for iqtl in quantiles.keys():
    # iqtl = '90%'
    print(iqtl + ': ' + str(quantiles[iqtl]))
    
    #-------- calculate quantiles
    wind10_ewe[expid[i]]['quantiles'][iqtl] = \
        wind10_alltime[expid[i]]['daily'].copy().quantile(
            quantiles[iqtl], dim='time', skipna=True)
    
    #-------- get mask
    wind10_ewe[expid[i]]['mask'][iqtl] = \
        (wind10_alltime[expid[i]]['daily'].copy() >= \
            wind10_ewe[expid[i]]['quantiles'][iqtl])

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_ewe.pkl',
    'wb') as f:
    pickle.dump(wind10_ewe[expid[i]], f)


'''
#-------------------------------- check

wind10_ewe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wind10_ewe.pkl',
    'rb') as f:
    wind10_ewe[expid[i]] = pickle.load(f)

ilat=48
ilon=90

#-------- check ['masked_data']['original']
res001 = wind10_ewe[expid[i]]['masked_data']['original'][:, ilat, ilon]
res002 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].copy().where(
    wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= (0.02 / seconds_per_d),
    other=np.nan,)
print((res001[np.isfinite(res001)] == res002[np.isfinite(res002)]).all().values)

for iqtl in quantiles.keys():
    # iqtl = '95%'
    #-------- check ['quantiles'][iqtl]
    res01 = wind10_ewe[expid[i]]['quantiles'][iqtl][ilat, ilon].values
    res02 = np.nanquantile(res002, quantiles[iqtl],)
    print(res01 == res02)
    
    #-------- check ['mask'][iqtl]
    res11 = wind10_ewe[expid[i]]['mask'][iqtl][:, ilat, ilon]
    res12 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= res01
    print((res11 == res12).all().values)
    
    #-------- check ['masked_data'][iqtl]
    res21 = wind10_ewe[expid[i]]['masked_data'][iqtl][:, ilat, ilon]
    res22 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].where(
        res11.values, other=0)
    print((res21 == res22).all().values)
    
    #-------- check ['masked_data_ocean'][iqtl]
    res23 = wind10_ewe[expid[i]]['masked_data_ocean'][iqtl][:, ilat, ilon]
    res24 = ocean_aprt_alltime[expid[i]]['daily'].sel(var_names='lat')[
        :, ilat, ilon].where(res11.values, other=0)
    print((res23 == res24).all().values)
    
    #-------- check ['sum_aprt'][ialltime]
    #---- am
    res301 = (wind10_ewe[expid[i]]['sum_aprt']['am'][iqtl][ilat, ilon]).compute()
    res302 = (res21.sum(skipna=True)).compute()
    print((res301 == res302).values)
    print(((res301 - res302) / res301).values)
    
    #---- sm
    res311 = wind10_ewe[expid[i]]['sum_aprt']['sm'][iqtl][:, ilat, ilon]
    res312 = res21.groupby('time.season').sum()
    print((res311 == res312).all().values)
    print(np.nanmax(abs(((res311 - res312) / res311).values)))
    
    #---- mm
    res321 = wind10_ewe[expid[i]]['sum_aprt']['mm'][iqtl][:, ilat, ilon]
    res322 = res21.groupby('time.month').sum()
    print((res321 == res322).all().values)
    print(np.nanmax(abs(((res321 - res322) / res321).values)))
    
    #---- ann
    res331 = wind10_ewe[expid[i]]['sum_aprt']['ann'][iqtl][:, ilat, ilon]
    res332 = res21.resample({'time': '1Y'}).sum()
    print((res331 == res332).all().values)
    print(np.nanmax(abs(((res331 - res332) / res331).values)))
    
    #---- sea
    res341 = wind10_ewe[expid[i]]['sum_aprt']['sea'][iqtl][:, ilat, ilon]
    res342 = res21.resample({'time': 'Q-FEB'}).sum()[1:-1]
    print((res341 == res342).all().values)
    print(np.nanmax(abs(((res341 - res342) / res341).values)))
    
    #---- mon
    res351 = wind10_ewe[expid[i]]['sum_aprt']['mon'][iqtl][:, ilat, ilon]
    res352 = res21.resample({'time': '1M'}).sum()
    print((res351 == res352).all().values)
    print(np.nanmax(abs(((res351 - res352) / res351).values)))
    
    #-------- check ['sum_aprt_ocean'][ialltime]
    #---- am
    res301 = wind10_ewe[expid[i]]['sum_aprt_ocean']['am'][iqtl][ilat, ilon]
    res302 = (res23.sum(skipna=True)).compute()
    print((res301 == res302).values)
    print(((res301 - res302) / res301).values)
    
    #---- sm
    res311 = wind10_ewe[expid[i]]['sum_aprt_ocean']['sm'][iqtl][:, ilat, ilon]
    res312 = res23.groupby('time.season').sum()
    print((res311 == res312).all().values)
    print(np.nanmax(abs(((res311 - res312) / res311).values)))
    
    #---- mm
    res321 = wind10_ewe[expid[i]]['sum_aprt_ocean']['mm'][iqtl][:, ilat, ilon]
    res322 = res23.groupby('time.month').sum()
    print((res321 == res322).all().values)
    print(np.nanmax(abs(((res321 - res322) / res321).values)))
    
    #---- ann
    res331 = wind10_ewe[expid[i]]['sum_aprt_ocean']['ann'][iqtl][:, ilat, ilon]
    res332 = res23.resample({'time': '1Y'}).sum()
    print((res331 == res332).all().values)
    print(np.nanmax(abs(((res331 - res332) / res331).values)))
    
    #---- sea
    res341 = wind10_ewe[expid[i]]['sum_aprt_ocean']['sea'][iqtl][:, ilat, ilon]
    res342 = res23.resample({'time': 'Q-FEB'}).sum()[1:-1]
    print((res341 == res342).all().values)
    print(np.nanmax(abs(((res341 - res342) / res341).values)))
    
    #---- mon
    res351 = wind10_ewe[expid[i]]['sum_aprt_ocean']['mon'][iqtl][:, ilat, ilon]
    res352 = res23.resample({'time': '1M'}).sum()
    print((res351 == res352).all().values)
    print(np.nanmax(abs(((res351 - res352) / res351).values)))
    
    #-------- check fraction of pre
    
    res401 = wind10_ewe[expid[i]]['frc_aprt']['am'][iqtl][ilat, ilon]
    res402 = wind10_ewe[expid[i]]['sum_aprt']['am'][iqtl][ilat, ilon] / \
        wind10_ewe[expid[i]]['sum_aprt']['am']['original'][ilat, ilon]
    print((res401 == res402).all().values)
    
    for ialltime in ['sm', 'mm', 'ann', 'sea', 'mon']:
        # ialltime = 'sm'
        res411 = wind10_ewe[expid[i]]['frc_aprt'][ialltime][iqtl][
            :, ilat, ilon]
        res412 = wind10_ewe[expid[i]]['sum_aprt'][ialltime][iqtl][
            :, ilat, ilon] / \
            wind10_ewe[expid[i]]['sum_aprt'][ialltime]['original'][
                :, ilat, ilon]
        print((res411 == res412).all().values)








import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))

# the same
(wind10_ewe[expid[i]]['masked_data'][iqtl].sum(dim='time') == wind10_ewe[expid[i]]['masked_data'][iqtl].sum(dim='time', skipna=True)).all()

'''
# endregion
# -----------------------------------------------------------------------------

