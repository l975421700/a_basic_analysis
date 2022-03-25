

# =============================================================================
# region Function to generate ticks and labels

def ticks_labels(
    xmin, xmax, ymin, ymax, xspacing, yspacing
    ):
    '''
    # input ----
    xmin, xmax, ymin, ymax: range of labels
    xspacing: spacing of x ticks
    yspacing: spacing of y ticks
    
    # output ----
    xticks_pos, xticks_label, yticks_pos, yticks_label
    '''
    
    import numpy as np
    
    # get the x ticks
    xticks_pos = np.arange(xmin, xmax + xspacing, xspacing)
    if not isinstance(xspacing, int):
        xticks_pos = np.around(xticks_pos, 1)
    
    # Associate with '° W', '°', and '° E'
    xticks_label = [''] * len(xticks_pos)
    for i in np.arange(len(xticks_pos)):
        if (abs(xticks_pos[i]) == 180) | (xticks_pos[i] == 0):
            xticks_label[i] = str(abs(xticks_pos[i])) + '°'
        elif xticks_pos[i] < 0:
            xticks_label[i] = str(abs(xticks_pos[i])) + '° W'
        elif xticks_pos[i] > 0:
            xticks_label[i] = str(xticks_pos[i]) + '° E'
    
    # get the y ticks
    yticks_pos = np.arange(ymin, ymax + yspacing, yspacing)
    if not isinstance(yspacing, int):
        yticks_pos = np.around(yticks_pos, 1)
    
    # Associate with '° N', '°', and '° S'
    yticks_label = [''] * len(yticks_pos)
    for i in np.arange(len(yticks_pos)):
        if yticks_pos[i] < 0:
            yticks_label[i] = str(abs(yticks_pos[i])) + '° S'
        if yticks_pos[i] == 0:
            yticks_label[i] = str(yticks_pos[i]) + '°'
        if yticks_pos[i] > 0:
            yticks_label[i] = str(yticks_pos[i]) + '° N'
    
    return xticks_pos, xticks_label, yticks_pos, yticks_label

'''
xmin = -180; xmax = 180; ymin = -90; ymax = 90; xspacing = 60; yspacing = 30
xticks_pos = np.arange(xmin, xmax + xspacing, xspacing)
ddd = ticks_labels(xmin, xmax, ymin, ymax, xspacing, yspacing)
'''
# endregion
# =============================================================================


# =============================================================================
# region Function to plot scale bar

def scale_bar(
    ax, bars = 2, length = None, location = (0.1, 0.05),
    barheight = 5, linewidth = 3, col = 'black',
    middle_label=True, fontcolor='black', vline = 1400):
    '''
    ax: the axes to draw the scalebar on.
    bars: the number of subdivisions
    length: in [km].
    location: left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    barheight: in [km]
    linewidth: the thickness of the scalebar.
    col: the color of the scale bar
    middle_label: whether to plot the middle label
    '''
    
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle
    
    # Get the limits of the axis in lat lon
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    
    # Make tmc aligned to the left of the map, vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx = False)
    
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
    
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = 'white'
    
    for i in range(0, bars):
        # Generate the x coordinate for the left of ith bar
        barleft_x = sbx + i * length * 1000 / bars
        
        # plot the chunk
        ax.add_patch(
            Rectangle((barleft_x, sby),
                      length * 1000 / bars, barheight * 1000, ec = 'black',
                      color = barcol, lw = linewidth, transform = tmc))
        
        # ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        
        # alternate the colour
        if barcol == 'white':
            barcol = col
        else:
            barcol = 'white'
        # Plot the scalebar label for that chunk
        if i == 0 or middle_label:
            ax.text(barleft_x, sby + barheight * 1200,
                    str(round(i * length / bars)), transform=tmc,
                    horizontalalignment='center', verticalalignment='bottom',
                    color=fontcolor)
        
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000 * 1.1
    # Plot the last scalebar label
    ax.text(bar_xt, sby + barheight * vline, str(round(length)) + ' km',
            transform=tmc, horizontalalignment='center',
            verticalalignment='bottom', color=fontcolor)

'''
# https://github.com/SciTools/cartopy/issues/490
# bars = 2; length = 1000; location=(0.1, 0.05); linewidth=3; col='black'

bars=scalebar_elements['bars'],
length=scalebar_elements['length'],
location=scalebar_elements['location'],
barheight=scalebar_elements['barheight'],
linewidth=scalebar_elements['linewidth'],
col=scalebar_elements['col'],
middle_label=scalebar_elements['middle_label']
'''
# endregion
# =============================================================================


# =============================================================================
# region Function to plot advance framework

def framework_plot1(
    which_area,
    ax_org=None,
    output_png=None,
    dpi=600,
    figsize=None,
    country_boundaries=True, border_color = 'black',
    gridlines=True, grid_color = 'gray',
    ticks_and_labels = True,
    lw=0.25, labelsize=10, extent=None, ticklabel = None,
    plot_scalebar = True,
    scalebar_elements={'bars': 2, 'length': 200, 'location': (0.02, 0.015),
                       'barheight': 20, 'linewidth': 0.15, 'col': 'black',
                       'middle_label': False},
    set_figure_margin = True,
    figure_margin=None,
    ):
    '''
    ----Input
    which_area: indicate which area to plot, {
        'global', 'self_defined', '12km_out', }
    
    figsize: figure size
    
    ----output
    
    
    '''
    
    import numpy as np
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import cartopy as ctp
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman', size=10)
    
    transform = ctp.crs.PlateCarree()
    
    if which_area == '12km_out':
        extent = [-35, 0, 10, 45]
        ticklabel = ticks_labels(-30, 0, 10, 40, 10, 10)
        plot_scalebar = False
        if (figsize is None):
            figsize = np.array([8.8, 8]) / 2.54
        if (figure_margin is None) & (set_figure_margin):
            figure_margin = {
                'left': 0.12, 'right': 0.96, 'bottom': 0.05, 'top': 0.995
            }
    
    if which_area == 'global':
        extent = [-180, 180, -90, 90]
        ticklabel = ticks_labels(-180, 180, -90, 90, 60, 30)
        plot_scalebar = False
        lw = 0.1
        if (figure_margin is None) & (set_figure_margin):
            figure_margin = {
                'left': 0.07, 'right': 0.97, 'bottom': 0.05, 'top': 0.995
            }
        if (figsize is None):
            figsize = np.array([8.8*2, 8.8]) / 2.54
    
    if which_area == 'self_defined':
        extent = extent
        ticklabel = ticklabel
        plot_scalebar = False
        if (figsize is None):
            figsize = np.array([8.8, 8.8]) / 2.54
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': transform}, dpi=dpi)
    else:
        ax = ax_org
    
    ax.set_extent(extent, crs = transform)
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0])
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2, labelsize=labelsize)
    
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
    
    if gridlines:
        gl = ax.gridlines(crs=transform, linewidth=lw, zorder=2,
                          color=grid_color, alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel[0])
        gl.ylocator = mticker.FixedLocator(ticklabel[2])
    
    if plot_scalebar:
        scale_bar(ax, bars=scalebar_elements['bars'],
                  length=scalebar_elements['length'],
                  location=scalebar_elements['location'],
                  barheight=scalebar_elements['barheight'],
                  linewidth=scalebar_elements['linewidth'],
                  col=scalebar_elements['col'],
                  middle_label=scalebar_elements['middle_label'])
    
    if set_figure_margin & (not(figure_margin is None)) & (ax_org is None):
        fig.subplots_adjust(
            left=figure_margin['left'], right=figure_margin['right'],
            bottom=figure_margin['bottom'], top=figure_margin['top'])
    else:
        fig.tight_layout()
    
    if (not(output_png is None)):
        fig.savefig(output_png, dpi=dpi)
    elif (ax_org is None):
        return fig, ax
    else:
        return ax


'''
# https://www.naturalearthdata.com/downloads/

fig, ax = framework_plot1("global")
fig.savefig('figures/0_test/trial.png')

import os
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'bas_palaeoclim_qino/others/bg_cartopy'
# ax.background_img(name='natural_earth', resolution='high')
# fig.savefig('figures/00_test/natural_earth.png', dpi=1200)

'''

# endregion
# =============================================================================


# =============================================================================
# region Function to plot confidence ellipse
# https://matplotlib.org/3.3.2/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    import numpy as np
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# endregion
# =============================================================================


# =============================================================================
# region Function to plot polar scale bar

def polar_scale_bar(
    ax, bars = None, length = None, location = None,
    barheight = None, linewidth = None, col = 'black', transform = None,
    middle_label=None, fontcolor='black', vline = 1400):
    '''
    ax: the axes to draw the scalebar on.
    bars: the number of subdivisions
    length: in [km].
    location: left side of the scalebar in axis coordinates.
    barheight: in [km]
    linewidth: the thickness of the scalebar.
    col: the color of the scale bar
    middle_label: whether to plot the middle label
    '''
    
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle
    
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent()
    
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
    
    # Plot the scalebar chunks
    barcol = 'white'
    
    for i in range(0, bars):
        # Generate the x coordinate for the left of ith bar
        barleft_x = sbx + i * length * 1000 / bars
        
        # plot the chunk
        ax.add_patch(
            Rectangle((barleft_x, sby),
                      length * 1000 / bars, barheight * 1000, ec = 'black',
                      color = barcol, lw = linewidth,
                      transform=transform, clip_on=False))
        
        # alternate the colour
        if barcol == 'white':
            barcol = col
        else:
            barcol = 'white'
        # Plot the scalebar label for that chunk
        if i == 0 or middle_label:
            ax.text(barleft_x, sby + barheight * 1200,
                    str(round(i * length / bars)), transform=transform,
                    horizontalalignment='center', verticalalignment='bottom',
                    color=fontcolor)
        
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000 * 1.1
    # Plot the last scalebar label
    ax.text(bar_xt, sby + barheight * vline, str(round(length)) + ' km',
            transform=transform, horizontalalignment='center',
            verticalalignment='bottom', color=fontcolor)

'''
# https://github.com/SciTools/cartopy/issues/490
# bars = 2; length = 1000; location=(0.1, 0.05); linewidth=3; col='black'

bars=scalebar_elements['bars'],
length=scalebar_elements['length'],
location=scalebar_elements['location'],
barheight=scalebar_elements['barheight'],
linewidth=scalebar_elements['linewidth'],
col=scalebar_elements['col'],
middle_label=scalebar_elements['middle_label']
'''
# endregion
# =============================================================================


# =============================================================================
# region functions to plot two hemisphere


def hemisphere_plot(
    ax_org = None,
    northextent=None, southextent=None, figsize=None, dpi=600,
    fm_left=0.13, fm_right=0.88, fm_bottom=0.08, fm_top=0.96,
    add_atlas=True, atlas_color='black', lw=0.25,
    add_grid=True, grid_color='gray', add_grid_labels = False,
    output_png=None,
    plot_scalebar=False, sb_bars=2, sb_length=1000, sb_location=(-0.13, 0),
    sb_barheight=100, sb_linewidth=0.15, sb_middle_label=False,
    ):
    '''
    ----Input
    northextent: plot SH, north extent in degree south, e.g. -60, -30, or 0;
    southextent: plot NH, south extent in degree north, e.g. 60, 30, 0;
    figsize: figure size, e.g. np.array([8.8, 9.3]) / 2.54;
    
    ----output
    
    
    ----function dependence
    ticks_labels; polar_scale_bar;
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib as mpl
    import matplotlib.path as mpath
    mpl.rc('font', family='Times New Roman', size=10)
    mpl.rcParams['figure.dpi'] = 600
    import warnings
    warnings.filterwarnings('ignore')
    from cartopy.mpl.ticker import LongitudeFormatter
    
    if (figsize is None):
        figsize = np.array([8.8, 9.3]) / 2.54
    
    if not (northextent is None):
        # northextent = -60
        projections = ccrs.SouthPolarStereo()
        ticklabel = ticks_labels(-180, 180, -90, northextent,
                                 30, int((northextent+90)/3))
        extent = (-180, 180, -90, northextent)
    elif not (southextent is None):
        # southextent = 60
        projections = ccrs.NorthPolarStereo()
        ticklabel = ticks_labels(-180, 180, southextent, 90,
                                 30, int((90-southextent)/3))
        extent = (-180, 180, southextent, 90)
    
    transform = ccrs.PlateCarree()
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': projections},
            dpi=dpi)
    else:
        ax = ax_org
    
    ax.set_extent(extent, crs=transform)
    
    if add_atlas:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=atlas_color,
            facecolor='none', lw=lw)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=atlas_color, facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        ax.add_feature(borders, zorder=2)
    
    if add_grid:
        gl = ax.gridlines(
            crs=transform, linewidth=lw/2, zorder=2,
            draw_labels=add_grid_labels,
            color=grid_color, alpha=0.5, linestyle='--',
            xlocs=ticklabel[0], ylocs=ticklabel[2], rotate_labels=False,
            xformatter=LongitudeFormatter(degree_symbol='° '),
        )
        gl.ylabel_style = {'size': 0, 'color': None, 'alpha': 0}
    
    # set circular axes boundaries
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    plt.setp(ax.spines.values(), linewidth=lw*0.8)
    
    if (ax_org is None):
        fig.subplots_adjust(
            left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
    
    if plot_scalebar:
        polar_scale_bar(
            ax, bars=sb_bars,
            length=sb_length,
            location=sb_location,
            barheight=sb_barheight,
            linewidth=sb_linewidth,
            middle_label=sb_middle_label,
            transform=projections)
    
    if not (output_png is None):
        fig.savefig(output_png)
    elif (ax_org is None):
        return fig, ax
    else:
        return ax

'''
hemisphere_plot(
    northextent=-60, output_png='figures/0_test/trial.png',
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98,
    )
hemisphere_plot(
    northextent=-30, sb_length=2000, sb_barheight=200,
    output_png='figures/0_test/trial',)
hemisphere_plot(
    northextent=0, sb_length=3000, sb_barheight=300,
    output_png='figures/0_test/trial02',)

hemisphere_plot(southextent=60, output_png='figures/0_test/trial03',)
hemisphere_plot(
    southextent=30, sb_length=2000, sb_barheight=200,
    output_png='figures/0_test/trial04',)
hemisphere_plot(
    southextent=0, sb_length=3000, sb_barheight=300,
    output_png='figures/0_test/trial05',)


# northextent=None, southextent=None, figsize=None, dpi=600,
# fm_left=0.12, fm_right=0.88, fm_bottom=0.08, fm_top=0.96,
# add_atlas=True, atlas_color='black', lw=0.25,
# add_grid=True, grid_color='gray', add_grid_labels = True, output_png=None,
# plot_scalebar=True, sb_bars=2, sb_length=1000, sb_location=(-0.13, 0),
# sb_barheight=100, sb_linewidth=0.15, sb_middle_label=False,

'''
# endregion
# =============================================================================


# =============================================================================
# region functions to create a diverging color map

def rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds'):
    '''
    ----Input
    pltlevel: levels used for the color bar
    
    ----output
    cmp_cmap: a colormap from blue to white to red, cmp_cmap.reversed()
    '''
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    
    cmp_top = cm.get_cmap(right_c, int(np.floor(len(pltlevel) / 2)) - 1)
    cmp_bottom = cm.get_cmap(left_c, int(np.floor(len(pltlevel) / 2)) - 1)
    
    cmp_colors = np.vstack(
        (cmp_top(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2)) - 1)),
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         cmp_bottom(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2)) - 1))))
    cmp_cmap = ListedColormap(cmp_colors)
    
    return(cmp_cmap)

# endregion
# =============================================================================


# =============================================================================
# region functions to plot high and low pressure systems


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    '''
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    '''
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center',
                transform=transform)
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top',
                transform=transform)


# endregion
# =============================================================================


# =============================================================================
# region quickly plot variables

def quick_var_plot(
    var=None, varname=None, xlabel=' \n ', whicharea='global',
    lon=None, lat=None, pltlevel=None, pltticks=None, northextent=-60,
    figsize=None, cmap=None, colors=None, extend=None,
    outputfile='figures/0_test/trial.png',#
    fm_left=None, fm_right=None, fm_bottom=None, fm_top=None,
    ):
    '''
    ----Input
    var: variable values.
    varname: variables to plot. e.g. 'pre', 'e'
    whicharea: 'global', 'SH'
    
    '''
    
    import numpy as np
    from matplotlib import cm
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    
    if(lon is None):
        lon = var.lon
    if(lat is None):
        lat = var.lat
    
    if(whicharea == 'global'):
        if(figsize is None):
            figsize=np.array([8.8*2, 11]) / 2.54
        if(varname == 'pre'):
            if(pltlevel is None):
                pltlevel = np.arange(0, 4000.01, 2)
            if(pltticks is None):
                pltticks = np.arange(0, 4000.01, 500)
            if(extend is None):
                extend = 'max'
            if(colors is None):
                colors = 'Blues'
        if(varname == 'evp'):
            if(pltlevel is None):
                pltlevel = np.arange(0, 3500.01, 2)
            if(pltticks is None):
                pltticks = np.arange(0, 3500.01, 500)
            if(extend is None):
                extend = 'both'
            if(colors is None):
                colors = 'Blues'
        if(cmap is None):
            cmap = cm.get_cmap(colors, len(pltlevel))
        
        if(fm_left is None):
            fm_left = 0.06
        if(fm_right is None):
            fm_right = 0.97
        if(fm_bottom is None):
            fm_bottom = 0.08
        if(fm_top is None):
            fm_top = 0.995
        
        fig, ax = framework_plot1(
            whicharea, figsize=figsize,)
    
    
    elif(whicharea == 'SH'):
        if(figsize is None):
            figsize = np.array([8.8, 9.8]) / 2.54
        if(varname == 'pre'):
            if(pltlevel is None):
                pltlevel = np.concatenate(
                    (np.arange(0, 100, 1), np.arange(100, 1600.01, 15)))
            if(pltticks is None):
                pltticks = np.concatenate(
                    (np.arange(0, 100, 20), np.arange(100, 1600.01, 300)))
        if(varname == 'evp'):
            if(pltlevel is None):
                pltlevel = np.concatenate(
                    (np.arange(-30, 0, 0.3), np.arange(0, 800.01, 8)))
            if(pltticks is None):
                pltticks = np.concatenate(
                    (np.arange(-30, 0, 5), np.arange(0, 800.01, 200)))
            if(extend is None):
                extend = 'both'
        if(cmap is None):
            cmap = rb_colormap(pltlevel).reversed()
        
        # fm_left=0.06, fm_right=0.94, fm_bottom=0.06, fm_top=0.99,
        if(fm_left is None):
            fm_left = 0.06
        if(fm_right is None):
            fm_right = 0.94
        if(fm_bottom is None):
            fm_bottom = 0.06
        if(fm_top is None):
            fm_top = 0.99
        
        fig, ax = hemisphere_plot(
            northextent=northextent, figsize=figsize,
            add_grid_labels=False, plot_scalebar=False, grid_color='black',
            fm_left=fm_left, fm_right=fm_right, fm_bottom=fm_bottom,
            fm_top=fm_top,)
    
    plt_cmp = ax.pcolormesh(
        lon, lat, var,
        norm=BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False),
        cmap=cmap, rasterized=True, transform=ccrs.PlateCarree(),)
    
    if(whicharea == 'global'):
        cbar = fig.colorbar(
            plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
            fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
            ticks=pltticks, extend=extend)
        fig.subplots_adjust(
            left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
    elif(whicharea == 'SH'):
        cbar = fig.colorbar(
            plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
            fraction=0.13, shrink=1, aspect=40, anchor=(0.5, 1),
            panchor=(0.5, 0), ticks=pltticks, extend=extend)
    
    cbar.ax.set_xlabel(xlabel, linespacing=1.5)
    fig.savefig(outputfile,)


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region generate mesh components for FESOM2 grid plot

def mesh2plot(
    meshdir='/work/ollie/qigao001/startdump/fesom2/mesh/core2/',
    box=[-180, 180, -90, 90],
    ):
    '''
    # input ----
    
    # output ----
    '''
    
    import pyfesom2 as pf
    import numpy as np
    
    mesh = pf.load_mesh(meshdir)
    
    box_mesh = [box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1]
    left, right, down, up = box_mesh
    
    selection = ((mesh.x2 >= left) & (mesh.x2 <= right) & (mesh.y2 >= down)
                 & (mesh.y2 <= up))
    elem_selection = selection[mesh.elem]
    no_nan_triangles = np.all(elem_selection, axis=1)
    elem_no_nan = mesh.elem[no_nan_triangles]
    d = mesh.x2[elem_no_nan].max(axis=1) - mesh.x2[elem_no_nan].min(axis=1)
    no_cyclic_elem2 = np.argwhere(d < 100).ravel()
    elem2plot = elem_no_nan[no_cyclic_elem2]
    
    tri2plot = {'x2': mesh.x2, 'y2': mesh.y2, 'elem2plot': elem2plot}
    
    return(tri2plot)

# endregion
# =============================================================================
