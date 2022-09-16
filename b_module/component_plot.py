

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


def cplot_wind_vectors(
    x, y, u, v, ax,
    iarrow=3, color='blue', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    ):
    '''
    #-------- Input
    x, y: location, 1D
    u, v: wind conponents, 2D
    ax
    
    iarrow: intervals. plot every iarrow
    
    #-------- Output
    
    
    '''
    
    import cartopy.crs as ccrs
    
    plt_quiver = ax.quiver(
        x[::iarrow], y[::iarrow],
        u[::iarrow, ::iarrow], v[::iarrow, ::iarrow],
        color=color, units=units, scale=scale,
        width=width, headwidth=headwidth, headlength=headlength, alpha=alpha,
        transform=ccrs.PlateCarree(),)
    
    return(plt_quiver)


# endregion
# -----------------------------------------------------------------------------

