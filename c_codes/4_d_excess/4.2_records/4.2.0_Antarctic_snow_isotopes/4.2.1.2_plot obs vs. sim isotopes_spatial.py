

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
    find_ilat_ilon_general,
    find_multi_gridvalue_at_site,
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
    plot_labels,
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


dO18_alltime = {}
dD_alltime = {}
d_ln_alltime = {}
d_excess_alltime = {}


for i in range(len(expid)):
    print(str(i) + ': ' + expid[i])
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dO18_alltime.pkl', 'rb') as f:
        dO18_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
        dD_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
        d_ln_alltime[expid[i]] = pickle.load(f)
    
    with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_excess_alltime.pkl', 'rb') as f:
        d_excess_alltime[expid[i]] = pickle.load(f)

lon = d_ln_alltime[expid[0]]['am'].lon
lat = d_ln_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


Antarctic_snow_isotopes = pd.read_csv(
    'data_sources/ice_core_records/Antarctic_snow_isotopic_composition/Antarctic_snow_isotopic_composition_DB.tab',
    sep='\t', header=0, skiprows=97,)

Antarctic_snow_isotopes = Antarctic_snow_isotopes.rename(columns={
    'Latitude': 'lat',
    'Longitude': 'lon',
    'δD [‰ SMOW] (Calculated average/mean values)': 'dD',
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)': 'dO18',
    'd xs [‰] (Calculated average/mean values)': 'd_excess',
})

Antarctic_snow_isotopes = Antarctic_snow_isotopes[[
    'lat', 'lon', 'dD', 'dO18', 'd_excess',
]]

ln_dD = 1000 * np.log(1 + Antarctic_snow_isotopes['dD'] / 1000)
ln_d18O = 1000 * np.log(1 + Antarctic_snow_isotopes['dO18'] / 1000)

Antarctic_snow_isotopes['d_ln'] = ln_dD - 8.47 * ln_d18O + 0.0285 * (ln_d18O ** 2)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean values

for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    for iisotopes in ['dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        # ['dO18', 'dD', 'd_ln', 'd_excess',]
        print('#-------- ' + iisotopes)
        
        if (iisotopes == 'dO18'):
            isotopevar = dO18_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-60, cm_max=-20, cm_interval1=5, cm_interval2=5,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'dD'):
            isotopevar = dD_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-450, cm_max=-100, cm_interval1=25, cm_interval2=50,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'd_ln'):
            isotopevar = d_ln_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10,
                cmap='viridis', reversed=False)
            
        elif (iisotopes == 'd_excess'):
            isotopevar = d_excess_alltime[expid[i]]['am']
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=20, cm_interval1=2, cm_interval2=4,
                cmap='viridis', reversed=False)
        
        output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.1_sim_vs_obs_spatial/8.0.3.1.0 ' + expid[i] + ' observed vs. simulated ' + iisotopes + '_AIS.png'
        
        fig, ax = hemisphere_plot(northextent=-60)
        
        xdata = Antarctic_snow_isotopes['lon']
        ydata = Antarctic_snow_isotopes['lat']
        cdata = Antarctic_snow_isotopes[iisotopes]
        subset = (np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(cdata))
        xdata = xdata[subset]
        ydata = ydata[subset]
        cdata = cdata[subset]
        
        plt_scatter = ax.scatter(
            xdata, ydata, s=8, c=cdata,
            edgecolors='k', linewidths=0.1, zorder=3,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        plt1 = plot_t63_contourf(
            lon, lat, isotopevar, ax,
            pltlevel, 'both', pltnorm, pltcmp, ccrs.PlateCarree(),)
        ax.add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        cbar = fig.colorbar(
            plt_scatter, ax=ax, aspect=30,
            orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
            pad=0.02, fraction=0.2,)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel(plot_labels[iisotopes], linespacing=1.5)
        fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual standard deviation


for i in range(len(expid)):
    # i = 0
    print('#---------------- ' + str(i) + ': ' + expid[i])
    
    for iisotopes in ['dO18', 'dD', 'd_ln', 'd_excess',]:
        # iisotopes = 'd_ln'
        # ['dO18', 'dD', 'd_ln', 'd_excess',]
        print('#-------- ' + iisotopes)
        
        if (iisotopes == 'dO18'):
            isotopevar = dO18_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=4, cm_interval1=0.5, cm_interval2=1,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'dD'):
            isotopevar = dD_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=40, cm_interval1=5, cm_interval2=10,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'd_ln'):
            isotopevar = d_ln_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=4, cm_interval1=0.5, cm_interval2=1,
                cmap='viridis', reversed=True)
            
        elif (iisotopes == 'd_excess'):
            isotopevar = d_excess_alltime[expid[i]]['ann'].std(dim='time', ddof=1)
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=4, cm_interval1=0.5, cm_interval2=1,
                cmap='viridis', reversed=True)
        
        output_png = 'figures/8_d-excess/8.0_records/8.0.3_isotopes/8.0.3.1_sim_vs_obs_spatial/8.0.3.1.1 ' + expid[i] + ' simulated ' + iisotopes + ' std_AIS.png'
        
        fig, ax = hemisphere_plot(northextent=-60)
        
        plt1 = plot_t63_contourf(
            lon, lat, isotopevar, ax,
            pltlevel, 'max', pltnorm, pltcmp, ccrs.PlateCarree(),)
        ax.add_feature(
            cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
        
        cbar = fig.colorbar(
            plt1, ax=ax, aspect=30,
            orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
            pad=0.02, fraction=0.2,)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel(
            'Standard deviation of ' + plot_labels[iisotopes],
            linespacing=1.5)
        fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------




