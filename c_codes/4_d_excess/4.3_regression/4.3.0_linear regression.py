

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_600_5.0',
    'pi_601_5.1',
    'pi_602_5.2',
    'pi_603_5.3',
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
# import xesmf as xe
import pandas as pd
from statsmodels.stats import multitest
import pycircstat as circ
import xskillscore as xs
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import linregress

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

#---- import dO18 and dD

dO18_alltime = {}
dD_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)

#---- import d_ln

d_ln_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)

#---- import precipitation sources

source_var = ['latitude', 'SST', 'rh2m', 'wind10']
pre_weighted_var = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    pre_weighted_var[expid[i]] = {}
    
    prefix = exp_odir + expid[i] + '/analysis/echam/' + expid[i]
    
    source_var_files = [
        prefix + '.pre_weighted_lat.pkl',
        prefix + '.pre_weighted_sst.pkl',
        prefix + '.pre_weighted_rh2m.pkl',
        prefix + '.pre_weighted_wind10.pkl',
    ]
    
    for ivar, ifile in zip(source_var, source_var_files):
        print(ivar + ':    ' + ifile)
        with open(ifile, 'rb') as f:
            pre_weighted_var[expid[i]][ivar] = pickle.load(f)

#---- import temp2

temp2_alltime = {}

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.temp2_alltime.pkl', 'rb') as f:
        temp2_alltime[expid[i]] = pickle.load(f)

#---- import site locations
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

'''
d_excess_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
    d_excess_alltime[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

pre_weighted_sst = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sst.pkl', 'rb') as f:
    pre_weighted_sst[expid[i]] = pickle.load(f)

pre_weighted_rh2m = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_rh2m.pkl', 'rb') as f:
    pre_weighted_rh2m[expid[i]] = pickle.load(f)

pre_weighted_wind10 = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_wind10.pkl', 'rb') as f:
    pre_weighted_wind10[expid[i]] = pickle.load(f)

lat_lon_sites = {
    'EDC': {'lat': -75.10, 'lon': 123.35,},
    'DOME F': {'lat': -77.32, 'lon': 39.70,}
}

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract data

source_sink_isotopes = {}

ialltime = 'ann'
source_sink_isotopes[ialltime] = {}

ialltime = 'mon'
source_sink_isotopes[ialltime] = {}

for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    source_sink_isotopes[ialltime][expid[i]] = {}
    
    for isite in ten_sites_loc.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        source_sink_isotopes[ialltime][expid[i]][isite] = {}
        
        isitelat = ten_sites_loc.lat[ten_sites_loc.Site == isite].values[0]
        isitelon = ten_sites_loc.lon[ten_sites_loc.Site == isite].values[0]
        
        source_sink_isotopes[ialltime][expid[i]][isite]['d_ln'] = \
            1000 * d_ln_alltime[expid[i]][ialltime].sel(
                lat=isitelat, lon=isitelon, method='nearest',
            )
        
        source_sink_isotopes[ialltime][expid[i]][isite]['dD'] = \
            dD_alltime[expid[i]][ialltime].sel(
                lat=isitelat, lon=isitelon, method='nearest',
            )
        
        source_sink_isotopes[ialltime][expid[i]][isite]['T_site'] = \
            temp2_alltime[expid[i]][ialltime].sel(
                lat=isitelat, lon=isitelon, method='nearest',
            )
        
        source_sink_isotopes[ialltime][expid[i]][isite]['T_source'] = \
            pre_weighted_var[expid[i]]['SST'][ialltime].sel(
                lat=isitelat, lon=isitelon, method='nearest',
            )


'''
d_excess = d_excess_alltime[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

dD = dD_alltime[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

t_site = temp2_alltime[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

t_src = pre_weighted_sst[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

rh2m_src = pre_weighted_rh2m[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

wind10_src = pre_weighted_wind10[expid[i]][ialltime].sel(
    lat=lat_lon_sites[isite]['lat'], lon=lat_lon_sites[isite]['lon'],
    method='nearest').values

# correlation
pearsonr(d_excess, t_src) # highly correlated
pearsonr(d_excess, t_site) # not correlated

pearsonr(dD, t_src) # highly correlated
pearsonr(dD, t_site) # highly correlated

pearsonr(t_src, rh2m_src,)
pearsonr(t_src, wind10_src,)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region linear regression

ialltime = 'ann'

ialltime = 'mon'

for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    for isite in ['EDC']:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        if (ialltime == 'ann'):
            delta_d_ln = source_sink_isotopes[ialltime][expid[i]][isite]['d_ln'] - np.mean(source_sink_isotopes[ialltime][expid[i]][isite]['d_ln'])
            delta_dD = source_sink_isotopes[ialltime][expid[i]][isite]['dD'] - np.mean(source_sink_isotopes[ialltime][expid[i]][isite]['dD'])
            delta_T_site = source_sink_isotopes[ialltime][expid[i]][isite]['T_site'] - np.mean(source_sink_isotopes[ialltime][expid[i]][isite]['T_site'])
            delta_T_source = source_sink_isotopes[ialltime][expid[i]][isite]['T_source'] - np.mean(source_sink_isotopes[ialltime][expid[i]][isite]['T_source'])
        elif (ialltime == 'mon'):
            delta_d_ln = source_sink_isotopes[ialltime][expid[i]][isite]['d_ln'].groupby('time.month') - source_sink_isotopes[ialltime][expid[i]][isite]['d_ln'].groupby('time.month').mean()
            delta_dD = source_sink_isotopes[ialltime][expid[i]][isite]['dD'].groupby('time.month') - source_sink_isotopes[ialltime][expid[i]][isite]['dD'].groupby('time.month').mean()
            delta_T_site = source_sink_isotopes[ialltime][expid[i]][isite]['T_site'].groupby('time.month') - source_sink_isotopes[ialltime][expid[i]][isite]['T_site'].groupby('time.month').mean()
            delta_T_source = source_sink_isotopes[ialltime][expid[i]][isite]['T_source'].groupby('time.month') - source_sink_isotopes[ialltime][expid[i]][isite]['T_source'].groupby('time.month').mean()
        
        fit_T_site = sm.OLS(
            delta_T_site.values,
            sm.add_constant(np.column_stack((
                delta_dD.values, delta_d_ln.values)))
            ).fit()
        # print(fit_T_site.summary())
        print("Parameters: ", np.round(fit_T_site.params, 2))
        print("R2: ", np.round(fit_T_site.rsquared, 2))
        
        fit_T_source = sm.OLS(
            delta_T_source.values,
            sm.add_constant(np.column_stack((
                delta_dD.values, delta_d_ln.values)))
            ).fit()
        # print(fit_T_source.summary())
        print("Parameters: ", np.round(fit_T_source.params, 2))
        print("R2: ", np.round(fit_T_source.rsquared, 2))
        
        predicted_T_site = \
            fit_T_site.params[1] * delta_dD + \
                fit_T_site.params[2] * delta_d_ln
        
        predicted_T_source = \
            fit_T_source.params[1] * delta_dD + \
                fit_T_source.params[2] * delta_d_ln
        
        np.round(((pearsonr(delta_T_site, predicted_T_site)).statistic) ** 2, 2)
        np.round(((pearsonr(delta_T_source, predicted_T_source)).statistic) ** 2, 2)
        
        sns.scatterplot(
            x = delta_T_site,
            y = predicted_T_site,
        )
        sns.scatterplot(
            x = delta_T_source,
            y = predicted_T_source,
        )
        plt.savefig('figures/test/test.png')
        plt.close()
        
        pearsonr(delta_dD, delta_T_site)





#-------- plot

# y_pre = result2.params[0] * X[:, 0] + result2.params[1] * X[:, 1] + \
#     result2.params[2] * X[:, 2]

'''
model1 = sm.OLS(
    t_site - np.mean(t_site),
    sm.add_constant(np.column_stack((
        dD - np.mean(dD),
        d_excess - np.mean(d_excess)))))
result1 = model1.fit()
print(result1.summary())
print("Parameters: ", result1.params)
print("R2: ", result1.rsquared)

model2 = sm.OLS(
    t_src - np.mean(t_src),
    sm.add_constant(np.column_stack((
        dD - np.mean(dD),
        d_excess - np.mean(d_excess)))))
result2 = model2.fit()
print(result2.summary())
print("Parameters: ", result2.params)
print("R2: ", result2.rsquared)

model3 = sm.OLS(t_site - np.mean(t_site), sm.add_constant(dD - np.mean(dD)))
result3 = model3.fit()
print(result3.summary())
print("Parameters: ", result3.params)
print("R2: ", result3.rsquared)

linearfit = linregress(x = dD - np.mean(dD), y = t_site - np.mean(t_site),)
print(linearfit)

model4 = sm.OLS(
    t_src - np.mean(t_src),
    sm.add_constant(d_excess - np.mean(d_excess)))
result4 = model4.fit()
print(result4.summary())
print("Parameters: ", result4.params)
print("R2: ", result4.rsquared)

linearfit = linregress(
    x = d_excess - np.mean(d_excess), y = t_src - np.mean(t_src),)
print(linearfit)

y = t_src - np.mean(t_src)
X = sm.add_constant(np.column_stack((
    dD - np.mean(dD),
    d_excess - np.mean(d_excess))))

model2 = sm.OLS(y, X)
result2 = model2.fit()
print(result2.summary())
print("Parameters: ", result2.params)
print("R2: ", result2.rsquared)

'''
# endregion
# -----------------------------------------------------------------------------


