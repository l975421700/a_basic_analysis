

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
    ]


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
    plot_labels_no_unit,
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
# region import data


composite_sources_isotopes = {}
composite_temp2_isotopes = {}
composite_sam_isotopes_sources_temp2 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes.pkl', 'rb') as f:
        composite_sources_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_temp2_isotopes.pkl', 'rb') as f:
        composite_temp2_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sam_isotopes_sources_temp2.pkl', 'rb') as f:
        composite_sam_isotopes_sources_temp2[expid[i]] = pickle.load(f)


corr_sources_isotopes = {}
corr_temp2_isotopes = {}
corr_sam_isotopes_sources_temp2 = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sources_isotopes.pkl', 'rb') as f:
        corr_sources_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_temp2_isotopes.pkl', 'rb') as f:
        corr_temp2_isotopes[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.corr_sam_isotopes_sources_temp2.pkl', 'rb') as f:
        corr_sam_isotopes_sources_temp2[expid[i]] = pickle.load(f)


wisoaprt_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)


lon = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lon
lat = corr_sources_isotopes[expid[i]]['sst']['d_ln']['mon']['r'].lat

ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

quantile_interval  = np.arange(10, 50 + 1e-4, 10, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot composite_sources_isotopes


iqtl = '10%'

for ivar in ['lat', 'lon', 'sst', 'rh2m', 'wind10', 'distance']:
    # ivar = 'sst'
    print('#-------------------------------- ' + ivar)
    
    for iisotopes in ['wisoaprt', 'dO18', 'dD', 'd_ln', 'd_excess']:
        # iisotopes = 'wisoaprt'
        print('#---------------- ' + iisotopes)
        
        for ialltime in ['mon', 'mon_no_mm']:
            # ['daily', 'mon', 'ann', 'mon_no_mm']
            # ialltime = 'mon'
            print('#-------- ' + ialltime)
            
            if (iisotopes == 'wisoaprt'):
                cm_min       = -100
                cm_max       = 100
                cm_interval1 = 20
                cm_interval2 = 40
            elif (iisotopes == 'dO18'):
                cm_min       = -50
                cm_max       = 50
                cm_interval1 = 10
                cm_interval2 = 20
            elif (iisotopes == 'dD'):
                cm_min       = -100
                cm_max       = 100
                cm_interval1 = 20
                cm_interval2 = 40
            elif (iisotopes == 'd_ln'):
                cm_min       = -50
                cm_max       = 50
                cm_interval1 = 10
                cm_interval2 = 20
            elif (iisotopes == 'd_excess'):
                cm_min       = -50
                cm_max       = 50
                cm_interval1 = 10
                cm_interval2 = 20
            
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=cm_min, cm_max=cm_max,
                cm_interval1=cm_interval1, cm_interval2=cm_interval2,
                cmap='PiYG', reversed=True)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.1_sources_isotopes/8.1.2.1.0 ' + expid[i] + ' ' + ialltime + ' ' + ivar + ' composite of ' + iisotopes + ' ' + str(quantiles[iqtl]) + '.png'
            
            cbar_label = 'Differences in ' + plot_labels_no_unit[iisotopes] +  ' between\ntop and bottom ' + iqtl + ' of ' + plot_labels_no_unit[ivar]
            
            composite_values = composite_sources_isotopes[expid[i]][ivar][iisotopes][ialltime][iqtl]
            correlation_values = corr_sources_isotopes[expid[i]][ivar][iisotopes][ialltime]['r']
            
            if (iisotopes == 'wisoaprt'):
                composite_values = (composite_values / (wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1).values * seconds_per_d) * 100)
            
            fig, ax = hemisphere_plot(northextent=-60, figsize=np.array([5.8, 7.5]) / 2.54,)
            
            cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, ax)
            
            plt1 = plot_t63_contourf(
                lon, lat, composite_values, ax,
                pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
            
            ax.add_feature(
                cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            cbar = fig.colorbar(
                plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
                orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
                pad=0.02, fraction=0.18,
                )
            
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.set_xlabel(cbar_label, fontsize=8)
            fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------




