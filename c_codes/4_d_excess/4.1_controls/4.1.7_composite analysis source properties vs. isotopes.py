

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
# region import data

#---- import wisoaprt

wisoaprt_alltime = {}

for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_alltime.pkl', 'rb') as f:
        wisoaprt_alltime[expid[i]] = pickle.load(f)

lon = wisoaprt_alltime[expid[0]]['am'].lon
lat = wisoaprt_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

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

#---- import site locations
ten_sites_loc = pd.read_pickle('data_sources/others/ten_sites_loc.pkl')

column_names = ['Control', 'Smooth wind regime', 'Rough wind regime',
                'No supersaturation']

quantile_interval  = np.arange(10, 50 + 1e-4, 10, dtype=np.int64)
quantiles = dict(zip(
    [str(x) + '%' for x in quantile_interval],
    [x/100 for x in quantile_interval],
    ))

with open('scratch/others/pi_m_502_5.0.t63_sites_indices.pkl', 'rb') as f:
    t63_sites_indices = pickle.load(f)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check normality

for ivar in source_var:
    # ivar = 'SST'
    print('#-------- ' + ivar)
    
    for ialltime in ['ann', 'mon', 'daily']:
        # ialltime = 'ann'
        print('#---- ' + ialltime)
        
        if (ialltime == 'ann'):
            data = pre_weighted_var[expid[i]][ivar][ialltime].copy()
        elif (ialltime == 'mon'):
            data = (pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - pre_weighted_var[expid[i]][ivar]['mm']).copy()
        elif (ialltime == 'daily'):
            data = (pre_weighted_var[expid[i]][ivar][ialltime]).copy()
        
        print(check_normality_3d(data.sel(lat=slice(-60, -90))))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get composite values

composite_sources_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    composite_sources_isotopes[expid[i]] = {}
    
    for ialltime in ['ann']: # ['daily', 'mon', 'sea', 'ann']
        # ialltime = 'ann'
        print('#---------------- ' + ialltime)
        
        composite_sources_isotopes[expid[i]][ialltime] = {}
        
        for ivar in source_var:
            # ivar = 'SST'
            print('#-------- ' + ivar)
            
            composite_sources_isotopes[expid[i]][ialltime][ivar] = {}
            
            for iisotopes in ['d_ln', 'dD', 'wisoaprt', ]:
                # iisotopes = 'd_ln'
                print('#---- ' + iisotopes)
                
                if (iisotopes == 'd_ln'):
                    isotopevar = d_ln_alltime[expid[i]][ialltime]
                    
                if (iisotopes == 'dD'):
                    isotopevar = dD_alltime[expid[i]][ialltime]
                    
                if (iisotopes == 'wisoaprt'):
                    isotopevar = wisoaprt_alltime[expid[i]][ialltime].sel(
                        wisotype=1) * seconds_per_d
                
                composite_sources_isotopes[
                    expid[i]][ialltime][ivar][iisotopes] = \
                        np.zeros(isotopevar.shape[1:])
                
                for ilat in range(isotopevar.shape[1]):
                    # ilat = 2
                    for ilon in range(isotopevar.shape[2]):
                        # ilon = 2
                        
                        var1 = pre_weighted_var[expid[i]][ivar][ialltime][
                            :, ilat, ilon].values
                        var2 = isotopevar[:, ilat, ilon].values
                        var2 = var2[np.isfinite(var1)]
                        var1 = var1[np.isfinite(var1)]
                        
                        if (len(var1) < 3):
                            composite_sources_isotopes[
                                expid[i]][ialltime][ivar][iisotopes][
                                    ilat, ilon] = np.nan
                        else:
                            var1_mean = np.mean(var1)
                            var1_std = np.std(var1)
                            
                            var1_pos = (var1 > (var1_mean + var1_std))
                            var1_neg = (var1 < (var1_mean - var1_std))
                            # var1[var1_pos]
                            # var1[var1_neg]
                            
                            var2_posmean = np.mean(var2[var1_pos])
                            var2_negmean = np.mean(var2[var1_neg])
                            
                            composite_sources_isotopes[
                                expid[i]][ialltime][ivar][iisotopes][
                                    ilat, ilon] = var2_posmean - var2_negmean
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes.pkl', 'wb') as f:
        pickle.dump(composite_sources_isotopes[expid[i]], f)




'''
#-------------------------------- check
composite_sources_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes.pkl', 'rb') as f:
        composite_sources_isotopes[expid[i]] = pickle.load(f)

ialltime = 'ann'
ivar = 'SST'
iisotopes = 'd_ln'
ilat = 40
ilon = 96

if (iisotopes == 'd_ln'):
    isotopevar = d_ln_alltime[expid[i]][ialltime]

if (iisotopes == 'dD'):
    isotopevar = dD_alltime[expid[i]][ialltime]

if (iisotopes == 'wisoaprt'):
    isotopevar = wisoaprt_alltime[expid[i]][ialltime].sel(
        wisotype=1) * seconds_per_d

var1 = pre_weighted_var[expid[i]][ivar][ialltime][:, ilat, ilon].values
var2 = isotopevar[:, ilat, ilon].values
var2 = var2[np.isfinite(var1)]
var1 = var1[np.isfinite(var1)]

var1_mean = np.mean(var1)
var1_std = np.std(var1)
var1_pos = (var1 > (var1_mean + var1_std))
var1_neg = (var1 < (var1_mean - var1_std))
var2_posmean = np.mean(var2[var1_pos])
var2_negmean = np.mean(var2[var1_neg])

print(composite_sources_isotopes[
    expid[i]][ialltime][ivar][iisotopes][ilat, ilon])
print(var2_posmean - var2_negmean)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the composite analysis

composite_sources_isotopes = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes.pkl', 'rb') as f:
        composite_sources_isotopes[expid[i]] = pickle.load(f)

for ialltime in ['ann']:
    print('#---------------- ' + ialltime)
    
    for ivar in source_var:
        print('#-------- ' + ivar)
        
        for iisotopes in ['d_ln', 'dD', 'wisoaprt']:
            print('#---- ' + iisotopes)
            
            #---------------- settings
            
            if (iisotopes == 'd_ln'):
                iisotopes_label = '$d_{ln}$ [‰]'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='PiYG', reversed=True)
                
            elif (iisotopes == 'dD'):
                iisotopes_label = '$\delta D$ [‰]'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=5,
                    cmap='PiYG', reversed=True)
                
            elif (iisotopes == 'wisoaprt'):
                iisotopes_label = 'precipitation [%] (as percentage of annual mean precipitation)'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20,
                    cmap='PiYG', reversed=True)
            
            output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.0 pi_600_3 ' + ialltime + ' ' + iisotopes + ' diff at posneg source ' + ivar + '.png'
            
            #---------------- plot
            
            nrow = 1
            ncol = 4
            
            fig, axs = plt.subplots(
                nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
                subplot_kw={'projection': ccrs.SouthPolarStereo()},
                gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)
            
            ipanel=0
            for jcol in range(ncol):
                axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])
                plt.text(
                    0.05, 0.95, panel_labels[ipanel],
                    transform=axs[jcol].transAxes,
                    ha='center', va='center', rotation='horizontal')
                ipanel += 1
                
                plt.text(
                    0.5, 1.08, column_names[jcol],
                    transform=axs[jcol].transAxes,
                    ha='center', va='center', rotation='horizontal')
                
                cplot_ice_cores(ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
                
                if (iisotopes == 'wisoaprt'):
                    plt_data = composite_sources_isotopes[expid[jcol]][ialltime][ivar][iisotopes] / (wisoaprt_alltime[expid[jcol]]['am'].sel(wisotype=1).values * seconds_per_d) * 100
                else:
                    plt_data = composite_sources_isotopes[expid[jcol]][ialltime][ivar][iisotopes]
                
                plt1 = plot_t63_contourf(
                    lon, lat,
                    plt_data,
                    axs[jcol], pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
                
                axs[jcol].add_feature(
                    cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            cbar = fig.colorbar(
                plt1, ax=axs, aspect=50,
                orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
                anchor=(0.5, 0.35), format=remove_trailing_zero_pos,
                )
            cbar.ax.set_xlabel(
                'Differences in ' + iisotopes_label +  ' between positive and negative phase of source ' + ivar)
            
            fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
            fig.savefig(output_png)



'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check individual site


i = 0
ialltime = 'daily'
ivar = 'SST'
iisotopes = 'd_ln'

sourcevar = pre_weighted_var[expid[i]][ivar][ialltime].copy()
isotopevar = (d_ln_alltime[expid[i]][ialltime].copy()).compute()

iqtl = '10%'

for isite in ['EDC']: # t63_sites_indices.keys():
    # isite = 'EDC'
    print('#-------- ' + isite)
    
    ilat = t63_sites_indices[isite]['ilat']
    ilon = t63_sites_indices[isite]['ilon']
    
    var1 = sourcevar[:, ilat, ilon].values
    var2 = isotopevar[:, ilat, ilon].values
    
    var2 = var2[np.isfinite(var1)]
    var1 = var1[np.isfinite(var1)]
    
    lower_qtl = np.quantile(var1, quantiles[iqtl])
    upper_qtl = np.quantile(var1, 1-quantiles[iqtl])
    
    var1_pos = (var1 > upper_qtl)
    var1_neg = (var1 < lower_qtl)
    
    var2_posmean = np.mean(var2[var1_pos])
    var2_negmean = np.mean(var2[var1_neg])
    
    var1_mean = np.mean(var1)
    var2_mean = np.mean(var2)
    var2_std = np.std(var2)
    
    # #-------- plot var1 histogram
    
    # sns.histplot(var1)
    # plt.vlines(lower_qtl, ymin=0, ymax=400, colors='red')
    # plt.vlines(upper_qtl, ymin=0, ymax=400, colors='black')
    # plt.savefig('figures/test/test.png')
    # plt.close()
    
    # #-------- plot var2 histogram
    # ax = sns.histplot(var2, binwidth=1, alpha=0.5)
    # sns.histplot(var2[var1_neg], binwidth=1, alpha=0.5, color='red')
    # sns.histplot(var2[var1_pos], binwidth=1, alpha=0.5, color='black')
    # ax.set_xlim(-50, 50)
    # plt.savefig('figures/test/test1.png')
    # plt.close()
    
    
    #-------------------------------- plot var1 histogram
    
    output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.0_quantiles_histgram/8.1.2.0.0 ' + expid[i] + ' histogram of daily source ' + ivar + ' at ' + isite + '.png'
    
    xmax = np.nanmax(var1)
    xmin = np.nanmin(var1)
    xlim_min = xmin - 0.5
    xlim_max = xmax + 0.5
    
    xtickmin = abs(np.ceil(xlim_min / 5)) * 5
    xtickmax = np.floor(xlim_max / 5) * 5
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(var1, binwidth=1,)
    
    ax.axvline(var1_mean, c = 'black', linewidth=0.5,)
    ax.axvline(lower_qtl, c = 'red', linewidth=0.5,)
    ax.axvline(upper_qtl, c = 'red', linewidth=0.5,)
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Daily source SST [$°C$]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 5))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.3, right=0.95, bottom=0.25, top=0.95)
    fig.savefig(output_png)
    
    
    
    
    #-------------------------------- plot var2 histogram
    
    output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.0_quantiles_histgram/8.1.2.0.0 ' + expid[i] + ' histogram of daily ' + iisotopes + ' at ' + isite + '.png'
    
    xmax = var2_mean + var2_std * 2
    xmin = var2_mean - var2_std * 2
    xlim_min = xmin - 0.5
    xlim_max = xmax + 0.5
    
    xtickmin = np.ceil(xlim_min / 20) * 20
    xtickmax = np.floor(xlim_max / 20) * 20
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([4.4, 4]) / 2.54)
    
    sns.histplot(var2, binwidth=2)
    sns.histplot(var2[var1_neg], binwidth=2, alpha=0.5, color='red')
    sns.histplot(var2[var1_pos], binwidth=2, alpha=0.5, color='black')
    
    ax.axvline(var2_negmean, c = 'red', linewidth=0.5,)
    ax.axvline(var2_posmean, c = 'black', linewidth=0.5,)
    
    plt_text = plt.text(0.05, 0.9, isite, transform=ax.transAxes, color='gray',)
    
    ax.set_xlabel('Daily $d_{ln}$ [‰]', labelpad=0.5)
    ax.xaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_xticks(np.arange(xtickmin, xtickmax + 1e-4, 20))
    ax.set_xlim(xlim_min, xlim_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.grid(True, which='both',
            linewidth=0.4, color='lightgray', alpha=0.5, linestyle=':')
    fig.subplots_adjust(left=0.3, right=0.95, bottom=0.25, top=0.95)
    fig.savefig(output_png)







'''
composite_sources_isotopes_top10 = {}

for i in np.arange(0, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes_top10.pkl', 'rb') as f:
        composite_sources_isotopes_top10[expid[i]] = pickle.load(f)

# print(var2_posmean - var2_negmean)
# print(composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl][ilat, ilon])

# (var1 > upper_qtl).sum()
# (var1 < lower_qtl).sum()
# var1[var1_pos]
# var1[var1_neg]





            pearsonr(
                pre_weighted_var[expid[i]][ivar][ialltime].sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ),
                isotopevar.sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ),
            )


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get composite values top vs bottom 10%

composite_sources_isotopes_top10 = {}

for i in [3]: # np.arange(0, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    composite_sources_isotopes_top10[expid[i]] = {}
    
    for ialltime in ['daily', 'mon', 'sea', 'ann']:
        # ialltime = 'ann'
        print('#---------------- ' + ialltime)
        
        composite_sources_isotopes_top10[expid[i]][ialltime] = {}
        
        for ivar in source_var:
            # ivar = 'SST'
            print('#-------- ' + ivar)
            
            composite_sources_isotopes_top10[expid[i]][ialltime][ivar] = {}
            
            for iisotopes in ['d_ln', 'dD', 'wisoaprt', ]:
                # iisotopes = 'd_ln'
                print('#---- ' + iisotopes)
                
                composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes] = {}
                
                if (ialltime == 'ann'):
                    # ialltime = 'ann'
                    sourcevar = pre_weighted_var[expid[i]][ivar]['ann'].copy()
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = (d_ln_alltime[expid[i]]['ann'].copy()).compute()
                    elif (iisotopes == 'dD'):
                        isotopevar = dD_alltime[expid[i]]['ann'].copy()
                    elif (iisotopes == 'wisoaprt'):
                        isotopevar = (wisoaprt_alltime[expid[i]]['ann'].sel(wisotype=1).copy() * seconds_per_d).compute()
                elif (ialltime == 'daily'):
                    # ialltime = 'daily'
                    sourcevar = pre_weighted_var[expid[i]][ivar]['daily'].copy()
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = (d_ln_alltime[expid[i]]['daily'].copy()).compute()
                    elif (iisotopes == 'dD'):
                        isotopevar = dD_alltime[expid[i]]['daily'].copy()
                    elif (iisotopes == 'wisoaprt'):
                        isotopevar = (wisoaprt_alltime[expid[i]]['daily'].sel(wisotype=1).copy() * seconds_per_d).compute()
                elif (ialltime == 'mon'):
                    # ialltime = 'mon'
                    sourcevar = (pre_weighted_var[expid[i]][ivar]['mon'].groupby('time.month') - pre_weighted_var[expid[i]][ivar]['mm']).compute()
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = ((d_ln_alltime[expid[i]]['mon'].groupby('time.month') - d_ln_alltime[expid[i]]['mm'])).compute()
                    elif (iisotopes == 'dD'):
                        isotopevar = (dD_alltime[expid[i]]['mon'].groupby('time.month') - dD_alltime[expid[i]]['mm']).compute()
                    elif (iisotopes == 'wisoaprt'):
                        isotopevar = ((wisoaprt_alltime[expid[i]]['mon'].sel(wisotype=1).groupby('time.month') - wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1)) * seconds_per_d).compute()
                elif (ialltime == 'sea'):
                    # ialltime = 'sea'
                    sourcevar = (pre_weighted_var[expid[i]][ivar]['sea'].groupby('time.season') - pre_weighted_var[expid[i]][ivar]['sm']).compute()
                    
                    if (iisotopes == 'd_ln'):
                        isotopevar = ((d_ln_alltime[expid[i]]['sea'].groupby('time.season') - d_ln_alltime[expid[i]]['sm'])).compute()
                    elif (iisotopes == 'dD'):
                        isotopevar = (dD_alltime[expid[i]]['sea'].groupby('time.season') - dD_alltime[expid[i]]['sm']).compute()
                    elif (iisotopes == 'wisoaprt'):
                        isotopevar = ((wisoaprt_alltime[expid[i]]['sea'].sel(wisotype=1).groupby('time.season') - wisoaprt_alltime[expid[i]]['sm'].sel(wisotype=1)) * seconds_per_d).compute()
                
                for iqtl in quantiles.keys():
                    # iqtl = '10%'
                    print('#-- ' + iqtl + ': ' + str(quantiles[iqtl]))
                    
                    composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl] = np.zeros(d_ln_alltime[expid[i]]['am'].shape)
                    
                    for ilat in range(isotopevar.shape[1]):
                        # ilat = 2
                        for ilon in range(isotopevar.shape[2]):
                            # ilon = 2
                            
                            var1 = sourcevar[:, ilat, ilon].values
                            var2 = isotopevar[:, ilat, ilon].values
                            
                            var2 = var2[np.isfinite(var1)]
                            var1 = var1[np.isfinite(var1)]
                            
                            if (len(var1) < 3):
                                composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl][ilat, ilon] = np.nan
                            else:
                                lower_qtl = np.quantile(var1, quantiles[iqtl])
                                upper_qtl = np.quantile(var1, 1-quantiles[iqtl])
                                
                                var1_pos = (var1 > upper_qtl)
                                var1_neg = (var1 < lower_qtl)
                                # var1[var1_pos]
                                # var1[var1_neg]
                                
                                var2_posmean = np.mean(var2[var1_pos])
                                var2_negmean = np.mean(var2[var1_neg])
                                
                                composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl][ilat, ilon] = var2_posmean - var2_negmean
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes_top10.pkl', 'wb') as f:
        pickle.dump(composite_sources_isotopes_top10[expid[i]], f)




'''
#-------------------------------- check
composite_sources_isotopes_top10 = {}

for i in np.arange(2, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes_top10.pkl', 'rb') as f:
        composite_sources_isotopes_top10[expid[i]] = pickle.load(f)

for i in np.arange(2, 4, 1):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    for ivar in source_var:
        # ivar = 'SST'
        print('#---------------- ' + ivar)
        
        for iqtl in quantiles.keys():
            # iqtl = '10%'
            print('#-------- ' + iqtl + ': ' + str(quantiles[iqtl]))
            
            for ialltime in ['daily', 'mon', 'sea', 'ann']:
                # ialltime = 'ann'
                print('#---- ' + ialltime)
                
                for iisotopes in ['d_ln', 'dD', 'wisoaprt', ]:
                    # iisotopes = 'd_ln'
                    print('#-- ' + iisotopes)
                    
                    if ((ialltime == 'ann') | (ialltime == 'daily')):
                        # ialltime = 'ann'
                        sourcevar = pre_weighted_var[expid[i]][ivar][ialltime].copy()
                    
                        if (iisotopes == 'd_ln'):
                            isotopevar = (d_ln_alltime[expid[i]][ialltime].copy()).compute()
                        elif (iisotopes == 'dD'):
                            isotopevar = dD_alltime[expid[i]][ialltime].copy()
                        elif (iisotopes == 'wisoaprt'):
                            isotopevar = (wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).copy() * seconds_per_d).compute()
                    elif (ialltime == 'mon'):
                        # ialltime = 'mon'
                        sourcevar = (pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.month') - pre_weighted_var[expid[i]][ivar]['mm']).compute()
                        
                        if (iisotopes == 'd_ln'):
                            isotopevar = ((d_ln_alltime[expid[i]][ialltime].groupby('time.month') - d_ln_alltime[expid[i]]['mm'])).compute()
                        elif (iisotopes == 'dD'):
                            isotopevar = (dD_alltime[expid[i]][ialltime].groupby('time.month') - dD_alltime[expid[i]]['mm']).compute()
                        elif (iisotopes == 'wisoaprt'):
                            isotopevar = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).groupby('time.month') - wisoaprt_alltime[expid[i]]['mm'].sel(wisotype=1)) * seconds_per_d).compute()
                    elif (ialltime == 'sea'):
                        # ialltime = 'sea'
                        sourcevar = (pre_weighted_var[expid[i]][ivar][ialltime].groupby('time.season') - pre_weighted_var[expid[i]][ivar]['sm']).compute()
                        
                        if (iisotopes == 'd_ln'):
                            isotopevar = ((d_ln_alltime[expid[i]][ialltime].groupby('time.season') - d_ln_alltime[expid[i]]['sm'])).compute()
                        elif (iisotopes == 'dD'):
                            isotopevar = (dD_alltime[expid[i]][ialltime].groupby('time.season') - dD_alltime[expid[i]]['sm']).compute()
                        elif (iisotopes == 'wisoaprt'):
                            isotopevar = ((wisoaprt_alltime[expid[i]][ialltime].sel(wisotype=1).groupby('time.season') - wisoaprt_alltime[expid[i]]['sm'].sel(wisotype=1)) * seconds_per_d).compute()
                    
                    for ilat in np.arange(1, 96, 30):
                        for ilon in np.arange(1, 192, 60):
                            
                            var1 = sourcevar[:, ilat, ilon].values
                            var2 = isotopevar[:, ilat, ilon].values
                            
                            var2 = var2[np.isfinite(var1)]
                            var1 = var1[np.isfinite(var1)]
                            
                            if (len(var1) < 3):
                                result = np.nan
                            else:
                                lower_qtl = np.quantile(var1, quantiles[iqtl])
                                upper_qtl = np.quantile(var1, 1-quantiles[iqtl])
                                
                                var1_pos = (var1 > upper_qtl)
                                var1_neg = (var1 < lower_qtl)
                                
                                var2_posmean = np.mean(var2[var1_pos])
                                var2_negmean = np.mean(var2[var1_neg])
                                
                                result = var2_posmean - var2_negmean
                            
                            # print(str(np.round(composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl][ilat, ilon], 2)))
                            # print(str(np.round(result, 2)))
                            
                            data1 = composite_sources_isotopes_top10[expid[i]][ialltime][ivar][iisotopes][iqtl][ilat, ilon]
                            data2 = result
                            
                            if (data1 != data2):
                                print(data1)
                                print(data2)





'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot the composite analysis

composite_sources_isotopes_top10 = {}

for i in range(len(expid)):
    # i = 0
    print('#-------------------------------- ' + str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.composite_sources_isotopes_top10.pkl', 'rb') as f:
        composite_sources_isotopes_top10[expid[i]] = pickle.load(f)

for ialltime in ['daily', 'mon', 'ann']: # 'sea',
    # ialltime = 'ann'
    print('#---------------- ' + ialltime)
    
    for ivar in ['latitude']: # source_var:
        # ivar = 'SST'
        print('#-------- ' + ivar)
        
        for iisotopes in ['d_ln']: # ['d_ln', 'dD', 'wisoaprt', ]:
            # iisotopes = 'd_ln'
            print('#---- ' + iisotopes)
            
            #-------------------------------- settings
            
            if (iisotopes == 'd_ln'):
                iisotopes_label = '$d_{ln}$ [‰]'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=2, cm_interval2=8,
                    cmap='PiYG', reversed=True)
                
            elif (iisotopes == 'dD'):
                iisotopes_label = '$\delta D$ [‰]'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=5, cm_interval2=20,
                    cmap='PiYG', reversed=True)
                
            elif (iisotopes == 'wisoaprt'):
                iisotopes_label = 'precipitation [%] (as percentage of annual mean precipitation)'
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20,
                    cmap='PiYG', reversed=True)
            
            for iqtl in ['10%',]: # quantiles.keys():
                # iqtl = '10%'
                print('#-- ' + iqtl + ': ' + str(quantiles[iqtl]))
                
                #-------------------------------- plot
                
                output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.1 pi_600_3 ' + ialltime + ' ' + iisotopes + ' diff at source ' + ivar + ' of percentile ' + str(quantiles[iqtl]) + '.png'
                
                nrow = 1
                ncol = 4
                
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
                    subplot_kw={'projection': ccrs.SouthPolarStereo()},
                    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)
                
                ipanel=0
                for jcol in range(ncol):
                    axs[jcol] = hemisphere_plot(
                        northextent=-60, ax_org = axs[jcol])
                    plt.text(
                        0.05, 0.95, panel_labels[ipanel],
                        transform=axs[jcol].transAxes,
                        ha='center', va='center', rotation='horizontal')
                    ipanel += 1
                    
                    plt.text(
                        0.5, 1.08, column_names[jcol],
                        transform=axs[jcol].transAxes,
                        ha='center', va='center', rotation='horizontal')
                    
                    cplot_ice_cores(
                        ten_sites_loc.lon, ten_sites_loc.lat, axs[jcol])
                    
                    if (iisotopes == 'wisoaprt'):
                        plt_data = composite_sources_isotopes_top10[expid[jcol]][ialltime][ivar][iisotopes][iqtl] / (wisoaprt_alltime[expid[jcol]]['am'].sel(wisotype=1).values * seconds_per_d) * 100
                    else:
                        plt_data = composite_sources_isotopes_top10[expid[jcol]][ialltime][ivar][iisotopes][iqtl]
                    
                    plt1 = plot_t63_contourf(
                        lon, lat, plt_data, axs[jcol],
                        pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
                    
                    axs[jcol].add_feature(
                        cfeature.OCEAN, color='white',
                        zorder=2, edgecolor=None,lw=0)
                
                cbar = fig.colorbar(
                    plt1, ax=axs, aspect=50,
                    orientation="horizontal", shrink=0.8, ticks=pltticks,
                    extend='both',
                    anchor=(0.5, 0.35), format=remove_trailing_zero_pos,)
                cbar.ax.set_xlabel(
                    'Differences in ' + iisotopes_label +  ' between top and bottom ' + iqtl + ' of source ' + ivar)
                
                fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.2, top = 0.98)
                fig.savefig(output_png)
                
                plt.close()




'''
'''
# endregion
# -----------------------------------------------------------------------------



