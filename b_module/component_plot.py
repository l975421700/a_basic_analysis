

# -----------------------------------------------------------------------------
# region Function to plot ice core sites

def cplot_ice_cores(
    lon, lat, ax,
    s=3, c='none', lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    ):
    '''
    #-------- Input
    lon/lat: 1D np.ndarray
    ax: ax to plot
    s: 3
    #-------- Output
    '''
    
    import cartopy.crs as ccrs
    
    ax.scatter(
        x = lon, y = lat,
        s=s, c=c, lw=lw, marker=marker, edgecolors = edgecolors, zorder=zorder,
        transform=ccrs.PlateCarree(),
        )


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to change bar width of sns.barplot

def change_snsbar_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

'''
https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to plot wind vectors

import cartopy.crs as ccrs
def cplot_wind_vectors(
    x, y, u, v, ax,
    iarrow=3, color='blue', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),
    ):
    '''
    #-------- Input
    x, y: location, 1D
    u, v: wind conponents, 2D
    ax
    
    iarrow: intervals. plot every iarrow
    
    #-------- Output
    
    
    '''
    
    plt_quiver = ax.quiver(
        x[::iarrow], y[::iarrow],
        u[::iarrow, ::iarrow], v[::iarrow, ::iarrow],
        color=color, units=units, scale=scale,
        width=width, headwidth=headwidth, headlength=headlength, alpha=alpha,
        transform=transform,)
    
    return(plt_quiver)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to plot lon-plev mesh, from [0, 360] to [-180, 180]

def cplot_lon180(
    lon,
    y,
    ax, pltnorm, pltcmp,
    plt_data,
    ):
    '''
    #-------- Input
    lon: 1d, from 0-360
    y: 1d, y axis
    plt_data: 2d, with lon from 0-360
    pltnorm, pltcmp
    
    #-------- Output
    
    '''
    
    import numpy as np
    import xarray as xr
    
    lon_180 = np.concatenate([lon[int(len(lon) / 2):] - 360,
                              lon[:int(len(lon) / 2)], ])
    
    plt_data_180 = xr.concat([
        plt_data.sel(lon=slice(180, 360)),
        plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')
    
    plt_mesh = ax.pcolormesh(
        lon_180, y, plt_data_180, norm=pltnorm, cmap=pltcmp,)
    
    return(plt_mesh)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to plot lon-plev quiver, from [0, 360] to [-180, 180]


def cplot_lon180_quiver(
    lon, y, ax,
    plt_data,
    iarrow = 5,
    color='magenta', units='height', scale=600, zorder=2,
    width=0.002, headwidth=3, headlength=5, alpha=1
    ):
    '''
    #-------- Input
    lon: 1d, from 0-360
    y: 1d, y axis
    plt_data: 2d, with lon from 0-360
    pltnorm, pltcmp
    
    #-------- Output
    
    '''
    
    import numpy as np
    import xarray as xr
    
    lon_180 = np.concatenate([
        lon[int(len(lon) / 2):] - 360, lon[:int(len(lon) / 2)], ])
    
    plt_data_180 = xr.concat([
        plt_data.sel(lon=slice(180, 360)),
        plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')
    
    plt_quiver = ax.quiver(
        lon_180[::iarrow], y,
        plt_data_180[:, ::iarrow],
        np.zeros(plt_data_180.shape)[:, ::iarrow],
        color=color, units=units, scale=scale, zorder=zorder,
        width=width, headwidth=headwidth, headlength=headlength, alpha=alpha,)
    
    return(plt_quiver)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to plot lon-plev contour, from [0, 360] to [-180, 180]

import numpy as np
from a_basic_analysis.b_module.mapplot import (
    remove_trailing_zero,
)

def cplot_lon180_ctr(
    lon, y, ax,
    plt_data,
    q_intervals = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 10]),
    colors='b', linewidths=0.3, clip_on=True,
    inline_spacing=10, fontsize=6
    ):
    '''
    #-------- Input
    
    #-------- Output
    
    '''
    
    import xarray as xr
    
    lon_180 = np.concatenate([
        lon[int(len(lon) / 2):] - 360, lon[:int(len(lon) / 2)], ])
    
    plt_data_180 = xr.concat([
        plt_data.sel(lon=slice(180, 360)),
        plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')
    
    plt_ctr = ax.contour(
        lon_180, y, plt_data_180,
        colors=colors, levels=q_intervals, linewidths=linewidths,
        clip_on=clip_on)
    
    ax_clabel = ax.clabel(
        plt_ctr, inline=1, colors=colors, fmt=remove_trailing_zero,
        levels=q_intervals, inline_spacing=inline_spacing, fontsize=fontsize,)
    
    return([plt_ctr, ax_clabel])


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Function to derive parameters for mesh plot


def plt_mesh_pars(
    cm_min, cm_max, cm_interval1, cm_interval2, cmap,
    clip=True, reversed=True,
    ):
    '''
    #-------- Input
    
    #-------- Output
    
    '''
    import numpy as np
    from matplotlib.colors import BoundaryNorm
    from matplotlib import cm
    
    pltlevel = np.arange(cm_min, cm_max + 1e-4, cm_interval1, dtype=np.float64)
    pltticks = np.arange(cm_min, cm_max + 1e-4, cm_interval2, dtype=np.float64)
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=clip)
    pltcmp = cm.get_cmap(cmap, len(pltlevel)-1)
    
    if(reversed):
        pltcmp = pltcmp.reversed()
    
    return([pltlevel, pltticks, pltnorm, pltcmp])

'''
# pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
# pltticks = np.arange(-55, -20 + 1e-4, 5)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-55, cm_max=-20, cm_interval1=2.5, cm_interval2=5, cmap='PuOr',
)

'''
# endregion
# -----------------------------------------------------------------------------

