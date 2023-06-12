

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


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check normality

for ivar in source_var:
    # ivar = 'SST'
    print('#-------- ' + ivar)
    
    for ialltime in ['ann']:
        # ialltime = 'ann'
        print('#---- ' + ialltime)
        
        print(check_normality_3d(
            pre_weighted_var[expid[i]][ivar][ialltime].sel(
                lat=slice(-60, -90))))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check individual site

ialltime = 'ann'

for i in range(len(expid)):
    # i = 0
    print(str(i) + ': ' + expid[i])
    
    for isite in ten_sites_loc.Site:
        # isite = 'EDC'
        print('#-------- ' + isite)
        
        isitelat = ten_sites_loc.lat[ten_sites_loc.Site == isite].values[0]
        isitelon = ten_sites_loc.lon[ten_sites_loc.Site == isite].values[0]
        
        # print(str(isitelon) + ' & ' + str(isitelat))
        
        for ivar in source_var:
            # ivar = 'latitude'
            print('#-------- ' + ivar)
            
            for iisotopes in ['d_ln', 'dD', 'wisoaprt', ]:
                # iisotopes = 'd_ln'
                print('#---- ' + iisotopes)
                
                if (iisotopes == 'd_ln'):
                    isotopevar = d_ln_alltime[expid[i]][ialltime] * 1000
                    
                if (iisotopes == 'dD'):
                    isotopevar = dD_alltime[expid[i]][ialltime]
                    
                if (iisotopes == 'wisoaprt'):
                    isotopevar = wisoaprt_alltime[expid[i]][ialltime].sel(
                        wisotype=1) * seconds_per_d
                
            sns.scatterplot(
                x = pre_weighted_var[expid[i]][ivar][ialltime].sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ),
                y = isotopevar.sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ) * 1000,
            )
            plt.savefig('figures/test/test.png')
            plt.close()
            pearsonr(
                pre_weighted_var[expid[i]][ivar][ialltime].sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ),
                isotopevar.sel(
                    lat=isitelat, lon=isitelon, method='nearest',
                ) * 1000,
            )


'''
'''
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
                    isotopevar = d_ln_alltime[expid[i]][ialltime] * 1000
                    
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
    isotopevar = d_ln_alltime[expid[i]][ialltime] * 1000

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

ialltime = 'ann'

#---------------- settings

#-------- source properties
ivar = 'latitude'
ivar = 'SST'
ivar = 'rh2m'
ivar = 'wind10'


#-------- isotopes
#---- d_ln

iisotopes = 'd_ln'
iisotopes_label = '$d_{ln}$ [‰]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2,
    cmap='PiYG', reversed=True)

#---- dD

iisotopes = 'dD'
iisotopes_label = '$\delta D$ [‰]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=5,
    cmap='PiYG', reversed=True)

#---- wisoaprt

iisotopes = 'wisoaprt'
iisotopes_label = 'precipitation [%] (as percentage of annual mean precipitation)'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20,
    cmap='PiYG', reversed=True)

#---------------- plot

output_png = 'figures/8_d-excess/8.1_controls/8.1.2_composite_analysis/8.1.2.0 pi_600_3 ' + ialltime + ' ' + iisotopes + ' diff at posneg source ' + ivar + '.png'

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
        plt_data = composite_sources_isotopes[expid[i]][ialltime][ivar][iisotopes] / (wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1).values * seconds_per_d) * 100
    else:
        plt_data = composite_sources_isotopes[expid[i]][ialltime][ivar][iisotopes]
    
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



