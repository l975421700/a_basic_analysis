

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_m_502_5.0',
    'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_603_5.3',
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
# sys.path.append('/work/ollie/qigao001')

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
from statsmodels.stats import multitest
import pycircstat as circ
import math

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

dO18_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
    dO18_alltime[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d_xs

d_excess_alltime = {}
d_excess_alltime[expid[i]] = {}

for ialltime in dO18_alltime[expid[i]].keys():
    print(ialltime)
    
    d_excess_alltime[expid[i]][ialltime] = \
        dD_alltime[expid[i]][ialltime] - 8 * dO18_alltime[expid[i]][ialltime]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'wb') as f:
    pickle.dump(d_excess_alltime[expid[i]], f)



'''
#-------------------------------- check

d_excess_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    d_excess_alltime[expid[i]] = pickle.load(f)

ialltime = 'sea'

itime = -1
ilat = 40
ilon = 90

aa = dD_alltime[expid[i]][ialltime][itime, ilat, ilon].values
bb = dO18_alltime[expid[i]][ialltime][itime, ilat, ilon].values
cc = d_excess_alltime[expid[i]][ialltime][itime, ilat, ilon].values

print(aa - 8 * bb)
print(cc)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get d-excess, logarithmic definition

d_ln_alltime = {}
d_ln_alltime[expid[i]] = {}

for ialltime in dO18_alltime[expid[i]].keys():
    print(ialltime)
    # ialltime = 'sm'
    
    ln_dD = 1000 * np.log(1 + dD_alltime[expid[i]][ialltime] / 1000)
    ln_d18O = 1000 * np.log(1 + dO18_alltime[expid[i]][ialltime] / 1000)
    
    d_ln_alltime[expid[i]][ialltime] = \
        ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'wb') as f:
    pickle.dump(d_ln_alltime[expid[i]], f)




'''
#-------------------------------- check

d_ln_alltime = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    d_ln_alltime[expid[i]] = pickle.load(f)

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    for ilat in np.arange(1, 96, 30):
        for ilon in np.arange(1, 192, 60):
            # i = 0; ilat = 40; ilon = 90
            
            for ialltime in ['daily', 'mon', 'sea', 'ann', 'mm', 'sm', 'am']:
                # ialltime = 'ann'
                if (ialltime != 'am'):
                    dO18 = dO18_alltime[expid[i]][ialltime][-1, ilat, ilon]
                    dD = dD_alltime[expid[i]][ialltime][-1, ilat, ilon]
                    d_ln = d_ln_alltime[expid[i]][ialltime][-1, ilat, ilon].values
                else:
                    dO18 = dO18_alltime[expid[i]][ialltime][ilat, ilon]
                    dD = dD_alltime[expid[i]][ialltime][ilat, ilon]
                    d_ln = d_ln_alltime[expid[i]][ialltime][ilat, ilon]
                
                d_ln_new = (1000 * np.log(1 + dD / 1000) - \
                    8.47 * 1000 * np.log(1 + dO18 / 1000) + \
                        0.0285 * (1000 * np.log(1 + dO18 / 1000)) ** 2).values
                
                # print(np.round(d_ln, 2))
                # print(np.round(d_ln_new, 2))
                if (((d_ln - d_ln_new) / d_ln) > 0.000001):
                    print(d_ln)
                    print(d_ln_new)

'''
# endregion
# -----------------------------------------------------------------------------

