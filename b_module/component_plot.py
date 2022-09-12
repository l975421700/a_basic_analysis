

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

