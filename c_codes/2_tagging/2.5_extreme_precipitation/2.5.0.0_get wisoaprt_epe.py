

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_416_4.9',
    'pi_m_502_5.0',
    ]
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

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

quantile_interval  = np.arange(50, 99 + 1e-4, 1, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

'''
# quantiles = {'90%': 0.9, '95%': 0.95, '99%': 0.99}

ocean_aprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.ocean_aprt_alltime.pkl', 'rb') as f:
    ocean_aprt_alltime[expid[i]] = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt_epe

wisoaprt_epe = {}
wisoaprt_epe[expid[i]] = {}
wisoaprt_epe[expid[i]]['quantiles'] = {}
wisoaprt_epe[expid[i]]['mask'] = {}
wisoaprt_epe[expid[i]]['masked_data'] = {}
# wisoaprt_epe[expid[i]]['masked_data_ocean'] = {}
wisoaprt_epe[expid[i]]['sum_aprt'] = {}
# wisoaprt_epe[expid[i]]['sum_aprt_ocean'] = {}
wisoaprt_epe[expid[i]]['frc_aprt'] = {}

# set a threshold of 0.02 mm/d
wisoaprt_epe[expid[i]]['masked_data']['original'] = \
    wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy().where(
        wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1) >= (0.02 / seconds_per_d),
        other=np.nan,
    ).compute()

# wisoaprt_epe[expid[i]]['masked_data_ocean']['original'] = \
#     ocean_aprt_alltime[expid[i]]['daily'].sel(var_names='lat').copy().where(
#         wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1) >= (0.02 / seconds_per_d),
#         other=np.nan,
#     )

for iqtl in quantiles.keys():
    # iqtl = '90%'
    print(iqtl + ': ' + str(quantiles[iqtl]))
    
    #-------- calculate quantiles
    wisoaprt_epe[expid[i]]['quantiles'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data']['original'].quantile(
            quantiles[iqtl], dim='time', skipna=True).compute()
    
    #-------- get mask
    wisoaprt_epe[expid[i]]['mask'][iqtl] = \
        (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy() >= \
            wisoaprt_epe[expid[i]]['quantiles'][iqtl]).compute()
    
    #-------- filtered below-threshold aprt
    if (iqtl == '90%'):
        wisoaprt_epe[expid[i]]['masked_data'][iqtl] = \
            wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy().where(
                wisoaprt_epe[expid[i]]['mask'][iqtl],
                other=0,
            ).compute()
    # #-------- filtered below-threshold ocean aprt
    # wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl] = \
    #     ocean_aprt_alltime[expid[i]]['daily'].sel(var_names='lat').copy().where(
    #         wisoaprt_epe[expid[i]]['mask'][iqtl],
    #         other=0,
    #     )

#-------- calculate sum of pre

for ialltime in ['am', 'sm', 'mm', 'ann', 'sea', 'mon']:
    wisoaprt_epe[expid[i]]['sum_aprt'][ialltime] = {}
    # wisoaprt_epe[expid[i]]['sum_aprt_ocean'][ialltime] = {}

for iqtl in ['90%', 'original']: # wisoaprt_epe[expid[i]]['masked_data'].keys():
    #-------- sum of wisoaprt_epe[expid[i]]['masked_data']
    # am
    wisoaprt_epe[expid[i]]['sum_aprt']['am'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].mean(
            dim='time', skipna=True).compute()
    # sm
    wisoaprt_epe[expid[i]]['sum_aprt']['sm'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].groupby(
            'time.season').mean(skipna=True).compute()
    # mm
    wisoaprt_epe[expid[i]]['sum_aprt']['mm'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].groupby(
            'time.month').mean(skipna=True).compute()
    # ann
    wisoaprt_epe[expid[i]]['sum_aprt']['ann'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].resample(
            {'time': '1Y'}).mean(skipna=True).compute()
    # sea
    wisoaprt_epe[expid[i]]['sum_aprt']['sea'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].resample(
            {'time': 'Q-FEB'}).mean(skipna=True)[1:-1].compute()
    # mon
    wisoaprt_epe[expid[i]]['sum_aprt']['mon'][iqtl] = \
        wisoaprt_epe[expid[i]]['masked_data'][iqtl].resample(
            {'time': '1M'}).mean(skipna=True).compute()

# for iqtl in wisoaprt_epe[expid[i]]['masked_data_ocean'].keys():
#     #-------- sum of wisoaprt_epe[expid[i]]['masked_data_ocean']
#     # am
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['am'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].sum(
#             dim='time', skipna=True).compute()
#     # sm
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['sm'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].groupby(
#             'time.season').sum(skipna=True).compute()
#     # mm
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['mm'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].groupby(
#             'time.month').sum(skipna=True).compute()
#     # ann
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['ann'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].resample(
#             {'time': '1Y'}).sum(skipna=True).compute()
#     # sea
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['sea'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].resample(
#             {'time': 'Q-FEB'}).sum(skipna=True)[1:-1].compute()
#     # mon
#     wisoaprt_epe[expid[i]]['sum_aprt_ocean']['mon'][iqtl] = \
#         wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl].resample(
#             {'time': '1M'}).sum(skipna=True).compute()

#-------- calculate fraction of pre

for ialltime in ['am', 'sm', 'mm', 'ann', 'sea', 'mon']:
    wisoaprt_epe[expid[i]]['frc_aprt'][ialltime] = {}
    
    for iqtl in ['90%']: # quantiles.keys():
        # ialltime = 'am'; iqtl = '90%'
        wisoaprt_epe[expid[i]]['frc_aprt'][ialltime][iqtl] = \
            wisoaprt_epe[expid[i]]['sum_aprt'][ialltime][iqtl] / \
                wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)
                # wisoaprt_epe[expid[i]]['sum_aprt'][ialltime]['original']

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_epe[expid[i]], f)


'''
#-------------------------------- check

wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)

ilat=48
ilon=90

#-------- check ['masked_data']['original']
res001 = wisoaprt_epe[expid[i]]['masked_data']['original'][:, ilat, ilon]
res002 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].copy().where(
    wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= (0.02 / seconds_per_d),
    other=np.nan,)
print((res001[np.isfinite(res001)] == res002[np.isfinite(res002)]).all().values)

for iqtl in quantiles.keys():
    print(iqtl)
    # iqtl = '90%'
    #-------- check ['quantiles'][iqtl]
    res01 = wisoaprt_epe[expid[i]]['quantiles'][iqtl][ilat, ilon].values
    res02 = np.nanquantile(res002, quantiles[iqtl],)
    print(res01 == res02)
    #-------- check ['mask'][iqtl]
    res11 = wisoaprt_epe[expid[i]]['mask'][iqtl][:, ilat, ilon]
    res12 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon] >= res01
    print((res11 == res12).all().values)
    #-------- check ['masked_data'][iqtl]
    if (iqtl == '90%'):
        res21 = wisoaprt_epe[expid[i]]['masked_data'][iqtl][:, ilat, ilon]
        res22 = wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].where(
            res11.values, other=0)
        print((res21 == res22).all().values)
    # #-------- check ['masked_data_ocean'][iqtl]
    # res23 = wisoaprt_epe[expid[i]]['masked_data_ocean'][iqtl][:, ilat, ilon]
    # res24 = ocean_aprt_alltime[expid[i]]['daily'].sel(var_names='lat')[
    #     :, ilat, ilon].where(res11.values, other=0)
    # print((res23 == res24).all().values)
    #-------- check ['sum_aprt'][ialltime]
    if (iqtl == '90%'):
        #---- am
        res301 = (wisoaprt_epe[expid[i]]['sum_aprt']['am'][iqtl][ilat, ilon]).compute()
        res302 = (res21.mean(skipna=True)).compute()
        print((res301 == res302).values)
        print(((res301 - res302) / res301).values)
        #---- sm
        res311 = wisoaprt_epe[expid[i]]['sum_aprt']['sm'][iqtl][:, ilat, ilon]
        res312 = res21.groupby('time.season').mean()
        print((res311 == res312).all().values)
        print(np.nanmax(abs(((res311 - res312) / res311).values)))
        #---- mm
        res321 = wisoaprt_epe[expid[i]]['sum_aprt']['mm'][iqtl][:, ilat, ilon]
        res322 = res21.groupby('time.month').mean()
        print((res321 == res322).all().values)
        print(np.nanmax(abs(((res321 - res322) / res321).values)))
        #---- ann
        res331 = wisoaprt_epe[expid[i]]['sum_aprt']['ann'][iqtl][:, ilat, ilon]
        res332 = res21.resample({'time': '1Y'}).mean()
        print((res331 == res332).all().values)
        print(np.nanmax(abs(((res331 - res332) / res331).values)))
        #---- sea
        res341 = wisoaprt_epe[expid[i]]['sum_aprt']['sea'][iqtl][:, ilat, ilon]
        res342 = res21.resample({'time': 'Q-FEB'}).mean()[1:-1]
        print((res341 == res342).all().values)
        print(np.nanmax(abs(((res341 - res342) / res341).values)))

        #---- mon
        res351 = wisoaprt_epe[expid[i]]['sum_aprt']['mon'][iqtl][:, ilat, ilon]
        res352 = res21.resample({'time': '1M'}).mean()
        print((res351 == res352).all().values)
        print(np.nanmax(abs(((res351 - res352) / res351).values)))
    # #-------- check ['sum_aprt_ocean'][ialltime]
    # #---- am
    # res301 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['am'][iqtl][ilat, ilon]
    # res302 = (res23.sum(skipna=True)).compute()
    # print((res301 == res302).values)
    # print(((res301 - res302) / res301).values)
    # #---- sm
    # res311 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['sm'][iqtl][:, ilat, ilon]
    # res312 = res23.groupby('time.season').sum()
    # print((res311 == res312).all().values)
    # print(np.nanmax(abs(((res311 - res312) / res311).values)))
    # #---- mm
    # res321 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['mm'][iqtl][:, ilat, ilon]
    # res322 = res23.groupby('time.month').sum()
    # print((res321 == res322).all().values)
    # print(np.nanmax(abs(((res321 - res322) / res321).values)))
    # #---- ann
    # res331 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['ann'][iqtl][:, ilat, ilon]
    # res332 = res23.resample({'time': '1Y'}).sum()
    # print((res331 == res332).all().values)
    # print(np.nanmax(abs(((res331 - res332) / res331).values)))
    # #---- sea
    # res341 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['sea'][iqtl][:, ilat, ilon]
    # res342 = res23.resample({'time': 'Q-FEB'}).sum()[1:-1]
    # print((res341 == res342).all().values)
    # print(np.nanmax(abs(((res341 - res342) / res341).values)))
    # #---- mon
    # res351 = wisoaprt_epe[expid[i]]['sum_aprt_ocean']['mon'][iqtl][:, ilat, ilon]
    # res352 = res23.resample({'time': '1M'}).sum()
    # print((res351 == res352).all().values)
    # print(np.nanmax(abs(((res351 - res352) / res351).values)))
    #-------- check fraction of pre
        res401 = wisoaprt_epe[expid[i]]['frc_aprt']['am'][iqtl][ilat, ilon]
        res402 = wisoaprt_epe[expid[i]]['sum_aprt']['am'][iqtl][ilat, ilon] / \
            wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1)[ilat, ilon]
        # wisoaprt_epe[expid[i]]['sum_aprt']['am']['original'][ilat, ilon]
        print((res401 == res402).all().values)
        for ialltime in ['sm', 'mm', 'ann', 'sea', 'mon']:
            print(ialltime)
            # ialltime = 'sm'
            res411 = wisoaprt_epe[expid[i]]['frc_aprt'][ialltime][iqtl][
                :, ilat, ilon]
            res412 = wisoaprt_epe[expid[i]]['sum_aprt'][ialltime][iqtl][
                :, ilat, ilon] / \
                    wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)[:, ilat, ilon]
                # wisoaprt_epe[expid[i]]['sum_aprt'][ialltime]['original'][
                # :, ilat, ilon]
            print((res411 == res412).all().values)









import psutil
print(psutil.Process().memory_info().rss / (2 ** 30))

# the same
(wisoaprt_epe[expid[i]]['masked_data'][iqtl].sum(dim='time') == wisoaprt_epe[expid[i]]['masked_data'][iqtl].sum(dim='time', skipna=True)).all()

'''
# endregion
# -----------------------------------------------------------------------------

