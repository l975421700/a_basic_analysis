

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

