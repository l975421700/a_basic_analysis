
import xarray as xr
import numpy as np
import pickle

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    'nudged_701_5.0',
    ]
i = 0

dD_q_sfc_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_q_sfc_alltime.pkl', 'rb') as f:
    dD_q_sfc_alltime[expid[i]] = pickle.load(f)

dD_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.dD_alltime.pkl', 'rb') as f:
    dD_alltime[expid[i]] = pickle.load(f)

dD_q_sfc_alltime[expid[i]]['am'].to_netcdf('scratch/test/test0.nc')
dD_alltime[expid[i]]['am'].to_netcdf('scratch/test/test1.nc')



def regional_plot(
    extent=None,
    figsize=None,
    central_longitude = 0,
    xmajortick_int = 20, ymajortick_int = 10,
    xminortick_int = 10, yminortick_int = 10,
    lw=0.25, country_boundaries=True, border_color = 'black',
    grid_color = 'gray',
    set_figure_margin = False, figure_margin=None,
    ticks_and_labels = True,
    ax_org=None,
    ):
    '''
    ----Input
    ----output
    '''
    
    import numpy as np
    import cartopy.feature as cfeature
    import cartopy as ctp
    import matplotlib.pyplot as plt
    from a_basic_analysis.b_module.mapplot import ticks_labels
    
    ticklabel=ticks_labels(extent[0], extent[1], extent[2], extent[3],
                           xmajortick_int, ymajortick_int)
    xminorticks = np.arange(extent[0], extent[1] + 1e-4, xminortick_int)
    yminorticks = np.arange(extent[2], extent[3] + 1e-4, yminortick_int)
    transform = ctp.crs.PlateCarree(central_longitude=central_longitude)
    
    if (figsize is None):
        figsize = np.array([8.8, 8.8]) / 2.54
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': transform},)
    else:
        ax = ax_org
    
    ax.set_extent(extent, crs = ctp.crs.PlateCarree())
    
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0], crs = ctp.crs.PlateCarree())
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2)
    
    if country_boundaries:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if (central_longitude == 0):
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            xlocs = xminorticks, ylocs=yminorticks,
            )
    else:
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            )
    
    if set_figure_margin & (not(figure_margin is None)) & (ax_org is None):
        fig.subplots_adjust(
            left=figure_margin['left'], right=figure_margin['right'],
            bottom=figure_margin['bottom'], top=figure_margin['top'])
    else:
        fig.tight_layout()
    
    if (ax_org is None):
        return fig, ax
    else:
        return ax



import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=8)
import numpy as np
import cartopy as ctp
import matplotlib.ticker as mticker

# plot southern atlantic ocean, 90deg
extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 5.2]) / 2.54)
fig.savefig('figures/test/trial1.png')


# plot southern indian ocean, 120deg
extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 5.2]) / 2.54)
fig.savefig('figures/test/trial2.png')


# plot southern pacific ocean, 150deg
extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 5.2]) / 2.54,
    central_longitude = 180)
fig.savefig('figures/test/trial3.png')






'''
extent=[140, 290, -90, -38]
figsize = np.array([8.8*11/9, 5.2]) / 2.54
central_longitude = 180
xmajortick_int = 20
ymajortick_int = 10
xminortick_int = 10
yminortick_int = 10
lw=0.25
country_boundaries=True
border_color = 'black'
gridlines=True
grid_color = 'gray'
set_figure_margin = False
figure_margin=None
ticks_and_labels = True
ax_org=None

'''


