

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
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


with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_st.pkl',
    'rb') as f:
    wisoaprt_mask_bin_st = pickle.load(f)


quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann masked wisoaprt

wisoaprt_masked_bin_st = {}
wisoaprt_masked_bin_st[expid[i]] = {}
wisoaprt_masked_bin_st[expid[i]]['mean'] = {}
wisoaprt_masked_bin_st[expid[i]]['frc']  = {}
wisoaprt_masked_bin_st[expid[i]]['meannan'] = {}

for iqtl in quantiles_bin.keys():
    # iqtl = '90.5%'
    print(iqtl + ': ' + str(quantiles_bin[iqtl]))
    
    masked_data = \
        wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy().where(
            wisoaprt_mask_bin_st[iqtl],
            other=0,
        ).compute()
    
    wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl] = mon_sea_ann(masked_data)
    wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl].pop('daily')
    # wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl].pop('mon')
    
    wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl] = {}
    
    for ialltime in wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl].keys():
        wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl][ialltime] = \
            (wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl][ialltime] / \
                wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1)).compute()
    
    masked_data_nan = \
        wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy().where(
            wisoaprt_mask_bin_st[iqtl],
            other=np.nan,
        ).compute()
    
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl] = {}
    # am
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['am'] = masked_data_nan.mean(
        dim='time', skipna=True).compute()
    # sm
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['sm'] = masked_data_nan.groupby(
        'time.season').mean(skipna=True).compute()
    # mm
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['mm'] = masked_data_nan.groupby(
        'time.month').mean(skipna=True).compute()
    # ann
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['ann'] = masked_data_nan.resample({'time': '1Y'}).mean(skipna=True).compute()
    # sea
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['sea'] = masked_data_nan.resample({'time': 'Q-FEB'}).mean(skipna=True)[1:-1].compute()
    # mon
    wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl]['mon'] = masked_data_nan.resample({'time': '1M'}).mean(skipna=True).compute()

with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_bin_st.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_masked_bin_st[expid[i]], f)






'''
# 28 min to run
#SBATCH --time=10:00:00
#SBATCH --partition=fat

#-------------------------------- check

wisoaprt_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
    wisoaprt_alltime[expid[i]] = pickle.load(f)

quantile_interval_bin = np.arange(0.5, 99.5 + 1e-4, 1, dtype=np.float64)
quantiles_bin = dict(zip(
    [str(x) + '%' for x in quantile_interval_bin],
    [x for x in quantile_interval_bin],
    ))

ilat=45
ilon=90

# get masked data
for iqtl in quantiles_bin.keys():
    # iqtl = '90.5%'
    print(iqtl + ': ' + str(quantiles_bin[iqtl]))
    
    with open(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_mask_bin_st.pkl',
        'rb') as f:
        wisoaprt_mask_bin_st = pickle.load(f)
    
    masked_data = \
        wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].copy().where(
            wisoaprt_mask_bin_st[iqtl][:, ilat, ilon],
            other=0,
        ).compute()
    masked_data_nan = \
        wisoaprt_alltime[expid[i]]['daily'][:, 0, ilat, ilon].copy().where(
            wisoaprt_mask_bin_st[iqtl][:, ilat, ilon],
            other=np.nan,
        ).compute()
    
    mean_values = mon_sea_ann(masked_data)
    mean_values.pop('daily')
    
    frc_values = {}
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
        frc_values[ialltime] = (mean_values[ialltime] / \
            wisoaprt_alltime[expid[i]][ialltime][:, 0, ilat, ilon].values
            ).compute()
    ialltime = 'am'
    frc_values[ialltime] = (mean_values[ialltime] / \
        wisoaprt_alltime[expid[i]][ialltime][0, ilat, ilon].values
        ).compute()
    
    meannan_values = {}
    meannan_values['mon'] = masked_data_nan.resample({'time': '1M'}).mean(skipna=True).compute()
    meannan_values['sea'] = masked_data_nan.resample({'time': 'Q-FEB'}).mean(skipna=True)[1:-1].compute()
    meannan_values['ann'] = masked_data_nan.resample({'time': '1Y'}).mean(skipna=True).compute()
    meannan_values['mm'] = masked_data_nan.groupby('time.month').mean(skipna=True).compute()
    meannan_values['sm'] = masked_data_nan.groupby('time.season').mean(skipna=True).compute()
    meannan_values['am'] = masked_data_nan.mean(dim='time', skipna=True).compute()
    
    del wisoaprt_mask_bin_st
    
    wisoaprt_masked_bin_st = {}
    with open(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_masked_bin_st.pkl',
        'rb') as f:
        wisoaprt_masked_bin_st[expid[i]] = pickle.load(f)
    
    for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
        # ialltime = 'mon'
        # check mean_values
        data1 = mean_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl][ialltime].values[
            :, ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
        
        data1 = frc_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl][ialltime].values[
            :, ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
        
        data1 = meannan_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime].values[
            :, ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
    
    for ialltime in ['am']:
        # ialltime = 'mon'
        # check mean_values
        data1 = mean_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['mean'][iqtl][ialltime].values[
            ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
        
        data1 = frc_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['frc'][iqtl][ialltime].values[
            ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
        
        data1 = meannan_values[ialltime].values
        data2 = wisoaprt_masked_bin_st[expid[i]]['meannan'][iqtl][ialltime].values[
            ilat, ilon
        ]
        # print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
        print(np.max(data1[np.isfinite(data1)] - data2[np.isfinite(data2)]))
    
    del wisoaprt_masked_bin_st



'''
# endregion
# -----------------------------------------------------------------------------


