

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
import matplotlib.patches as mpatches

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
    change_snsbar_width,
    plt_mesh_pars,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data

aprt_geo7_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_alltime.pkl', 'rb') as f:
    aprt_geo7_alltime[expid[i]] = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS_alltime.pkl', 'rb') as f:
    aprt_frc_AIS_alltime = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc.pkl', 'rb') as f:
    aprt_frc = pickle.load(f)

lon = aprt_geo7_alltime[expid[i]]['am'].lon
lat = aprt_geo7_alltime[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)


major_ice_core_site = pd.read_csv('data_sources/others/major_ice_core_site.csv')
major_ice_core_site = major_ice_core_site.loc[
    major_ice_core_site['age (kyr)'] > 120, ]

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am aprt fraction

regions = list(aprt_frc.keys())

cm_mins = [0, 0, 0, 0, 0, 0, 0]
cm_maxs = [5, 10, 35, 55, 45, 20, 60]
cm_interval1s = [0.5, 1, 5, 5, 5, 2, 5]
cm_interval2s = [1, 2, 5, 10, 5, 4, 10]

for count, iregion in enumerate(regions):
    print(str(count) + ': ' + iregion)
    # count = 5; iregion = 'SH seaice'
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=cm_mins[count],
        cm_max=cm_maxs[count],
        cm_interval1=cm_interval1s[count],
        cm_interval2=cm_interval2s[count],
        cmap='Blues',
        reversed=False)
    
    output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am ' + iregion + '.png'
    
    fig, ax = hemisphere_plot(northextent=-60)
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
    plt_cmp = ax.pcolormesh(
        lon, lat,
        aprt_frc[iregion]['am'],
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

    cbar = fig.colorbar(
        plt_cmp, ax=ax, aspect=30,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
        pad=0.02, fraction=0.2,
        )
    cbar.ax.set_xlabel('Fraction of annual mean precipitation from\n' + iregion + ' [%]', linespacing=1.5, fontsize=8)
    fig.savefig(output_png)


#-------- other Ocean

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Open ocean.png'

pltlevel = np.arange(75, 100.01, 2.5)
pltticks = np.arange(75, 100.01, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-60)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Atlantic Ocean']['am'] + aprt_frc['Indian Ocean']['am'] + \
        aprt_frc['Pacific Ocean']['am'] + aprt_frc['Southern Ocean']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nOpen ocean [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)


'''

stats.describe(
    aprt_frc['NHseaice']['am'] + aprt_frc['SHseaice']['am'] + aprt_frc['NHocean']['am'] + aprt_frc['SHocean']['am'] + aprt_frc['Antarctica']['am'] + aprt_frc['NHland']['am'] + aprt_frc['SHland']['am'],
    axis=None, nan_policy='omit')

#-------- precipitation from geo regions


np.max(aprt_frc['NHseaice']['am'].sel(lat=slice(-50, -90))) # 3e-5
np.max(aprt_frc['NHocean']['am'].sel(lat=slice(-50, -90))) # 0.88
np.max(aprt_frc['SHocean']['am'].sel(lat=slice(-50, -90))) # 98.9
np.max(aprt_frc['SHseaice']['am'].sel(lat=slice(-50, -90))) # 42.5
np.max(aprt_frc['Antarctica']['am'].sel(lat=slice(-50, -90))) # 19.4
np.max(aprt_frc['NHland']['am'].sel(lat=slice(-50, -90))) # 0.25
np.max(aprt_frc['SHland']['am'].sel(lat=slice(-50, -90))) # 16.2

#-------- precipitation from Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Antarctica.png'

pltlevel = np.arange(0, 10.01, 1)
pltticks = np.arange(0, 10.01, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Antarctica']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nAntarctica [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)

#-------- SH sea ice

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am SHseaice.png'

pltlevel = np.arange(0, 40.01, 5)
pltticks = np.arange(0, 40.01, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['SHseaice']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nSH sea ice covered area [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)

#-------- other Land

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Otherland.png'

pltlevel = np.arange(0, 10.01, 1)
pltticks = np.arange(0, 10.01, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Otherland']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nland excl. Antarctica [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)


#-------- other Ocean

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Otherocean.png'

pltlevel = np.arange(60, 100.01, 4)
pltticks = np.arange(60, 100.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Greens', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Otherocean']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nOpen ocean [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region histogram: annual circle of aprt frac over AIS

imask = 'AIS'
# imask = 'EAIS'

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.1_aprt_ann_circle/6.1.4.0.1 ' + expid[i] + ' ann circle aprt frc over ' + imask + '.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 10.5]) / 2.54)
lgd_handles = []
colors = [
    'salmon', 'darkviolet',
    'royalblue', 'deepskyblue', 'lightblue',
    'darkorange', 'bisque']
regions = list(aprt_frc_AIS_alltime[imask]['mm'].keys())[-1::-1]

for count, iregion in enumerate(regions):
    print(str(count) + ': ' + iregion)
    
    sns.barplot(
        x = month, y = aprt_frc_AIS_alltime[imask]['mm'][iregion].frc_AIS,
        color=colors[count],
    )
    lgd_handles += [mpatches.Patch(color=colors[count], label=iregion)]

change_snsbar_width(ax, .7)

plt.legend(
    handles=lgd_handles,
    labels=[
        'Southern Ocean: $27.9±1.97$',
        'SH seaice:           $6.1±0.38$',
        'Pacific Ocean:    $28.0±1.63$',
        'Indian Ocean:     $23.2±1.85$',
        'Atlantic Ocean: $9.8±0.79$',
        'Land excl. AIS: $4.4±0.3$',
        'AIS:                   $0.6±0.06$',
        ],
    loc=(-0.1, -0.48), handlelength=0.5, handleheight = 0.5,
    frameon = False, ncol=2, handletextpad = 0.5,
    labelspacing = 0.5, columnspacing = 0.5,
    )

ax.set_xlabel('Fraction of precipitation over ' + imask + ' from each region [$\%$]')
ax.set_ylabel(None)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 100+1e-4, 10))

ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.32, top=0.98)
fig.savefig(output_png)


#-------------------------------- ann+std

imask = 'AIS'

iwisotype = 'AIS'
ann_values = aprt_frc_AIS_alltime[imask]['ann'][iwisotype].frc_AIS.values
am_values  = aprt_frc_AIS_alltime[imask]['am'][iwisotype].frc_AIS.values
print(iwisotype + ': ' + \
    str(np.round(am_values[0], 1)) + '±' + \
        str(np.round(ann_values.std(ddof=1), 2)))

for itype in range(6):
    iwisotype = list(aprt_frc_AIS_alltime[imask]['ann'].keys())[itype+1]
    iwisotype1 = list(aprt_frc_AIS_alltime[imask]['ann'].keys())[itype]
    
    # print(iwisotype + ' vs. ' + iwisotype1)
    
    ann_values = (
        aprt_frc_AIS_alltime[imask]['ann'][iwisotype].frc_AIS - \
        aprt_frc_AIS_alltime[imask]['ann'][iwisotype1].frc_AIS).values
    am_values = (
        aprt_frc_AIS_alltime[imask]['am'][iwisotype].frc_AIS - \
        aprt_frc_AIS_alltime[imask]['am'][iwisotype1].frc_AIS).values
    
    print(iwisotype + ': ' + \
        str(np.round(am_values[0], 1)) + '±' + \
            str(np.round(ann_values.std(ddof=1), 2)))



#-------------------------------- mm

imask = 'AIS'
iwisotype = 'AIS'

mm_values = aprt_frc_AIS_alltime[imask]['mm'][iwisotype].frc_AIS.values
print(iwisotype + ': ' + \
    str(np.round(np.min(mm_values), 1)) + '-' + \
        str(np.round(np.max(mm_values), 1)))

for itype in range(6):
    iwisotype = list(aprt_frc_AIS_alltime[imask]['ann'].keys())[itype+1]
    iwisotype1 = list(aprt_frc_AIS_alltime[imask]['ann'].keys())[itype]
    
    print(iwisotype + ' vs. ' + iwisotype1)
    
    mm_values = aprt_frc_AIS_alltime[imask]['mm'][iwisotype].frc_AIS.values - \
        aprt_frc_AIS_alltime[imask]['mm'][iwisotype1].frc_AIS.values
    
    print(iwisotype + ': ' + \
        str(np.round(np.min(mm_values), 1)) + '-' + \
            str(np.round(np.max(mm_values), 1)))

aprt_frc_AIS_alltime[imask]['mm']['SH seaice'].frc_AIS.values - \
        aprt_frc_AIS_alltime[imask]['mm']['Pacific Ocean'].frc_AIS.values

'''
#-------------------------------- ann+std

imask = 'AIS'

#---- Antarctica: 1.0% ± 0.07%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['Land excl. Antarctica'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- other land: 4.2% ± 0.28%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['Land excl. Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['Land excl. Antarctica'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['SH sea ice'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- SH sea ice: 11.5% ± 0.72%

ann_values = (
    aprt_frc_AIS_alltime[imask]['ann']['SH sea ice'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['ann']['Open ocean'].frc_AIS).values
am_values = (
    aprt_frc_AIS_alltime[imask]['am']['SH sea ice'].frc_AIS - \
    aprt_frc_AIS_alltime[imask]['am']['Open ocean'].frc_AIS).values
# ann_values.mean()
am_values
ann_values.std()


#---- SH sea ice: 83.3% ± 0.80%

ann_values = aprt_frc_AIS_alltime[imask]['ann']['Open ocean'].frc_AIS.values
am_values = aprt_frc_AIS_alltime[imask]['am']['Open ocean'].frc_AIS.values
# ann_values.mean()
am_values
ann_values.std()



# mm
imask = 'AIS'
sea_ice_contribution = aprt_frc_AIS_alltime[imask]['mm']['SH sea ice'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['Open ocean'].frc_AIS
np.max(sea_ice_contribution)
np.min(sea_ice_contribution)

Antarctic_contribution = aprt_frc_AIS_alltime[imask]['mm']['Antarctica'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['Land excl. Antarctica'].frc_AIS

sea_ice_contribution[6] + Antarctic_contribution[6]

land_contribution = aprt_frc_AIS_alltime[imask]['mm']['Land excl. Antarctica'].frc_AIS - aprt_frc_AIS_alltime[imask]['mm']['SH sea ice'].frc_AIS
land_contribution[6]

https://www.python-graph-gallery.com/stacked-and-percent-stacked-barplot

min(aprt_frc_AIS['AIS']['Open ocean'].frc_AIS)
'''
# endregion
# -----------------------------------------------------------------------------

