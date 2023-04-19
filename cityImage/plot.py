import pandas as pd, numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from matplotlib.lines import Line2D  
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

import mapclassify, pylab, colorsys
pd.set_option("display.precision", 3)

from .utilities import scaling_columnDF
from .colors import rand_cmap, random_colors_list, kindlmann, lighten_color

"""
Plotting functions

"""

## Plotting
    
class Plot():
    """
    Class for creating a basic matplotlib plot with a title and customizable settings.

    Parameters
    ----------
    figsize : tuple
        Tuple of width and height for the figure.
    black_background : bool
        Whether to set the background color as black.
    title : str
        Title for the plot.

    Attributes
    ----------
    fig : matplotlib Figure
        Figure object for the plot.
    ax : matplotlib Axes
        Axes object for the plot.
    title : str
        Title for the plot.
    text_color : str
        Color for text elements, white if black_background is True and black otherwise.
    font_size_primary : int
        Font size for the title.
    font_size_secondary : int
        Font size for secondary elements.
    """
    
    def __init__(self, figsize, black_background, fontsize, title = None):
        
        self.font_size_primary = fontsize
        self.font_size_secondary = fontsize*0.90
        self.fig, self.ax = plt.subplots(1, figsize=figsize)
        self.title = title
        self.ax.axis("equal")
        self.black_background = black_background
        self.text_color = "white" if black_background else "black"
        
        rect = self.fig.patch 
        rect.set_facecolor("black" if black_background else "white")      
        if title is not None:
            self.fig.suptitle(title, color = self.text_color, fontsize = self.font_size_primary, fontfamily = 'Times New Roman')
               
class MultiPlot():
    """
    A class for creating multi-plot figures.
    
    Parameters:
    figsize (tuple): The size of the figure in inches.
    nrows (int): The number of rows of subplots in the figure.
    ncols (int): The number of columns of subplots in the figure.
    black_background (bool): Whether to use a black background for the figure.
    title (str, optional): Title of the figure. Default is None.
    """
    def __init__(self, figsize, nrows, ncols, black_background, fontsize, title = None):
        """
        Initializes the MultiPlot class.
        """
        
        self.ncols = ncols
        self.fig, self.grid = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        self.black_background = black_background
        self.text_color = "white" if black_background else "black"
        rect = self.fig.patch 
        rect.set_facecolor("black" if black_background else "white")   
        self.font_size_primary = fontsize
        self.font_size_secondary = fontsize*0.90
        
        if title is not None:
            self.fig.suptitle(title, color = self.text_color, fontsize = self.font_size_primary, fontfamily = 'Times New Roman', 
                         ha = 'center', va = 'center') 
              
def plot_gdf(gdf, column = None, title = None, black_background = True, figsize = (15,15), scheme = None, bins = None, 
            classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, geometry_size = 1.0, 
            geometry_size_column = None, fontsize = 15, geometry_size_factor = None, cbar = False, cbar_ticks = 5, 
            cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, 
            base_map_gdf = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4, base_map_geometry_size = 1.1,  
            base_map_zorder = 0):

    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in column, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    title: string 
        title of the plot
    black_background: boolean 
        black background or white
    fig_size: float
        size of the figure's side extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the ">" and "<" as labels of the lowest and highest ticks' the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        point size value, when plotting a Point GeoDataFrame
    geometry_size_factor: float 
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the marker size 
        accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a Point GeoDataFrame
    base_map_gdf: GeoDataFrame
        a desired additional layer to use as a base map        
    base_map_color: string
        color applied to all geometries of the base map
    base_map_alpha: float
        base map's alpha value
    base_map_geometry_size: float
        base map's marker size when the base map is a Point GeoDataFrame
    base_map_zorder: int   
        zorder of the layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
        
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """   
    
    plot = Plot(figsize = figsize, black_background = black_background, title = title, fontsize = fontsize)
    fig, ax = plot.fig, plot.ax
    
    ax.set_aspect("equal")
    set_axes_frame(axes_frame, ax, black_background, plot.text_color)
 
    zorder = 0
    if cbar:
        legend = False
    if cbar & (norm is None):
        min_value = gdf[column].min()
        max_value = gdf[column].max()
        norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    if (not base_map_gdf.empty):
        plot_baseMap(gdf = base_map_gdf, ax = ax, color = base_map_color, geometry_size = base_map_geometry_size, alpha = base_map_alpha, 
            zorder = base_map_zorder )
        if base_map_zorder == 0:
            zorder = 1
   
    if geometry_size_column is None:
        geometry_size_column = column
    
    plotOn_ax(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, 
                geometry_size = geometry_size, geometry_size_column = geometry_size_column, geometry_size_factor = geometry_size_factor, 
                zorder = zorder, legend = legend)

    if cbar:
        generate_colorbar(plot, cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, shrinkage = cbar_shrinkage)
    elif legend: 
        generate_legend_ax(ax, plot) 

    return plot.fig      
 
def plot_grid_gdfs_column(gdfs = [], column = None, ncols = 1, nrows = 1, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                    classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                    cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, 
                    fontsize = 15, geometry_size = None, geometry_size_columns = [], geometry_size_factor = None):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in the provided columns, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    title: string 
        title of the plot
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the ">" and "<" as labels of the lowest and highest ticks' the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        point size value, when plotting a Point GeoDataFrame
    geometry_size_factor: float 
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the marker 
        size accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a Point GeoDataFrame
    """   
   
    if (len(gdfs)+1 != ncols*nrows) & (len(gdfs) != ncols*nrows):
        raise ValueError("Please provide an appropriate combination of nrows and ncols")
    
    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background, fontsize = fontsize)
    if (cbar) & (norm is None):
        min_value = min([gdf[column].min() for gdf in gdfs])
        max_value = max([gdf[column].max() for gdf in gdfs])
        norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    for n, ax in enumerate(multiPlot.grid.flat):
        if n > len(gdfs)-1: 
            ax.set_visible(False)    
            continue    
        gdf = gdfs[n]
          
        parameters = {'ax' : ax, 'n' : n, 'multiPlot' : multiPlot, 'column' : column , 'gdf' : gdf, 'titles' : titles, 
                      'scheme' : scheme, 'bins' : bins, 'classes' : classes, 'norm' : norm, 'cmap' : cmap, 
                      'color' : color, 'alpha' : alpha, 'legend' : legend, 'axes_frame' : axes_frame,
                      'geometry_size' : geometry_size, 'geometry_size_columns' : geometry_size_columns, 'geometry_size_factor' : geometry_size_factor}        
        subplot(**parameters)    
            
    if (cbar) & (not legend):  
        generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)

    return multiPlot.fig                      

def plot_grid_gdf_columns(gdf, columns = [], ncols = 1, nrows = 1, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                          classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                          cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, 
                          fontsize = 15, geometry_size = None, geometry_size_columns = [], geometry_size_factor = None):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in the provided columns, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    title: string 
        title of the plot
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        number of classes for categorising the data when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    cbar: boolean
        if True, show colorbar, otherwise don't; when True it doesn't show legend
    cbar_ticks: int
        number of ticks along the colorbar
    cbar_max_symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the ">" and "<" as labels of the lowest and highest ticks' the colorbar
    axes_frame: boolean
        if True, it shows the axes' frame
    geometry_size: float
        point size value, when plotting a Point GeoDataFrame
    geometry_size_factor: float 
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the marker 
        size accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a Point GeoDataFrame
    """   
       
    if (len(columns)+1 != ncols*nrows) & (len(columns) != ncols*nrows):
        raise ValueError("Please provide an appropriate combination of nrows and ncols")
    
    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background, fontsize = fontsize)
    if (cbar) & (norm is None):
        min_value = min([gdf[column].min() for column in columns])
        max_value = max([gdf[column].max() for column in columns])
        norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    for n, ax in enumerate(multiPlot.grid.flat):
        if n > len(columns)-1: 
            ax.set_visible(False)    
            continue
        column = columns[n]      
            
        parameters = {'ax' : ax, 'n' : n, 'multiPlot' : multiPlot, 'column' : column , 'gdf' : gdf, 'titles' : titles, 
                      'scheme' : scheme, 'bins' : bins, 'classes' : classes, 'norm' : norm, 'cmap' : cmap, 
                      'color' : color, 'alpha' : alpha, 'legend' : legend, 'axes_frame' : axes_frame,
                      'geometry_size' : geometry_size, 'geometry_size_columns' : geometry_size_columns, 'geometry_size_factor' : geometry_size_factor}        
        subplot(**parameters)    
            
    if (cbar) & (not legend):  
        generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)
    return multiPlot.fig

def plotOn_ax(ax, gdf, column = None, scheme = None, bins = None, classes = 7, norm = None, cmap = None, color = 'red', alpha = 1.0, 
                legend = False, geometry_size = 1.0, geometry_size_column = None, geometry_size_factor = None, zorder = 0):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in column, using a given scheme, on the provided Axes.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        the Axes on which plotting
    gdf: GeoDataFrame
        GeoDataFrame to be plotted 
    column: string
        Column on which the plot is based
    scheme: string
        classification method, choose amongst: https://pysal.org/mapclassify/api.html
    bins: list
        bins defined by the user
    classes: int
        classes for visualising when scheme is not "None"
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    cmap: string, matplotlib.colors.LinearSegmentedColormap
        see matplotlib colormaps for a list of possible values or pass a colormap
    color: string
        categorical color applied to all geometries when not using a column to color them
    alpha: float
        alpha value of the plotted layer
    legend: boolean
        if True, show legend, otherwise don't
    geometry_size: float
        point size value, when plotting a Point GeoDataFrame
    geometry_size_factor: float 
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the marker size 
        accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a Point GeoDataFrame
    geometry_size: float
        line width, when plotting a LineString GeoDataFrame
    geometry_size_factor: float
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the line width 
        accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a LineString GeoDataFrame
    zorder: int   
        zorder of this layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
    """  
    
    gdf = gdf.copy()

    categorical = not (norm is not None) | (scheme is not None)
    if not categorical:
        color = None
        if cmap is None:
            cmap = kindlmann()
        if norm is not None:
            scheme = None
        if (gdf[column].dtype != 'O' ):
            gdf = gdf.reindex(gdf[column].abs().sort_values(ascending = True).index)
        else:
            gdf[column] = gdf[column].astype(float)
            
    elif (column is not None) & (cmap is None):
        cmap = rand_cmap(len(gdf[column].unique())) 
        if len(gdf[column].unique()) == 1:
            legend = False
            cmap = None
            color = 'red'
        
    c_k = dict(k=classes) if bins is None else dict(bins=bins, k=len(bins))
    scheme = 'User_Defined' if bins is not None else scheme
    
    parameters = {'ax' : ax, 'column' : column, 'classification_kwds' : c_k, 'scheme' : scheme, 'norm' : norm,
                   'cmap' : cmap, 'categorical' : categorical, 'color' : color, 'alpha' : alpha, 'legend' : legend, 
                   'zorder': zorder}  
                   
    geometry_type = gdf.iloc[0].geometry.geom_type
    if geometry_type == 'Point':    
        if (geometry_size_factor is not None): 
            gdf[column+'_sc'] = scaling_columnDF(gdf[column])
            geometry_size = np.where(gdf[column+'_sc'] >= 0.20, gdf[column+'_sc']*geometry_size_factor, 0.40) # marker size
        parameters['markersize'] = geometry_size
    
    elif geometry_type == 'LineString':
        if geometry_size_factor is not None:
            geometry_size = [(abs(value)*geometry_size_factor) if (abs(value)*geometry_size_factor) > 1.1 else 1.1 for value in
                             gdf[column]]
        sub_parameters = {'linewidth': geometry_size, 'capstyle': 'round', 'joinstyle':'round'}
        parameters.update(sub_parameters)
    else:
        parameters['edgecolor'] = 'none' 
    
    gdf.plot(**parameters) 
 
 
def subplot(ax, n, multiPlot, gdf, column, scheme, bins, classes, axes_frame, norm, cmap, color, alpha, legend, geometry_size, geometry_size_columns,
                    geometry_size_factor, titles):
    
    ax.set_aspect("equal")
    set_axes_frame(axes_frame, ax, multiPlot.black_background, multiPlot.text_color)
    
    if titles:          
        ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size_primary, color = multiPlot.text_color, 
                     pad = 15)    
    
    geometry_size_column = column
    if geometry_size_columns:
        geometry_size_column = geometry_size_columns[n]
   
    legend_ax = False
    legend_fig = False
    
    if legend:
        legend_ax = (n == 1 and scheme == 'User_Defined') or (scheme != 'User_Defined')
        legend_fig = (n == 1 and scheme == 'User_Defined')

    plotOn_ax(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, 
                alpha = alpha, legend = legend_ax, geometry_size = geometry_size, geometry_size_column = geometry_size_column, 
                geometry_size_factor = geometry_size_factor)
                
    if legend_fig:
        generate_legend_fig(ax, multiPlot)
    elif legend_ax:
        generate_legend_ax(ax, multiPlot)

def plot_baseMap(gdf = None, ax = None, color = None, geometry_size = None, alpha = 0.5, zorder = 0):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in the provided columns, using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing the map data.
    ax : str
        The column on which the plot is based.
    geometry_size : str
        size of the elements (LineString or Point), default value is None
    black_background : bool
        Specifies whether to use a black or white background.
    alpha : float
        transparency of the map elements, default value is 0.5
    zorder : str
        the order of the map elements with respect to other elements in the axis, default value is 0
    
    """
    if gdf.iloc[0].geometry.geom_type == 'LineString':
        gdf.plot(ax = ax, color = color, linewidth = geometry_size, alpha = alpha,zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Point':
        gdf.plot(ax = ax, color = color, markersize = geometry_size, alpha = alpha, zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Polygon':
        gdf.plot(ax = ax, color = color, alpha = alpha, zorder = zorder)
    
def generate_legend_fig(ax, plot):
    """ 
    It generates the legend for an entire figure.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        the Axes on which plotting
    nrows: int
        number of rows in the figure
    text_color: string
        the text color
    font_size: int
        the legend's labels text size
    """ 
    leg = ax.get_legend()    
    fig_leg = plot.fig.legend(handles = leg.legendHandles, labels = [t.get_text() for t in leg.texts], loc=5, 
                              borderaxespad= 0)
    ax.get_legend().remove()
    plt.setp(fig_leg.texts, family='Times New Roman', fontsize = plot.font_size_secondary, color = plot.text_color, 
             va = 'center')

    fig_leg.get_frame().set_linewidth(0.0) # remove legend border
    fig_leg.set_zorder(102)
    fig_leg.get_frame().set_facecolor('none')

    for handle in fig_leg.legendHandles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(15)
        else: 
            break
           
def generate_legend_ax(ax, plot):
    """ 
    It generate the legend for a figure.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        the Axes on which plotting
    text_color: string
        the text color
    font_size: int
        the legend's labels text size
    """
    leg = ax.get_legend()  
    plt.setp(leg.texts, family='Times New Roman', fontsize = plot.font_size_secondary, color = plot.text_color, va = 'center')
    
    bbox_legend = leg.get_window_extent()
    bbox_axes = ax.get_window_extent()
    legend_height = bbox_legend.y1 - bbox_legend.y0
    axes_height = bbox_axes.y1 - bbox_axes.y0
    vertical_position = 0.5+(legend_height/2/axes_height)
    
    leg.set_bbox_to_anchor((1.0, vertical_position))
    
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    
    for handle in leg.legendHandles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(12)
        else:
            break
    leg.get_frame().set_facecolor('none')
    
def generate_colorbar(plot = None, cmap = None, norm = None, ticks = 5, symbol = False, min_max = False, shrinkage = 0.95):
    """ 
    It plots a colorbar, given some settings.
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The figure container for the current plot
    pos: list of float
        the axes positions
    sm: matplotlib.cm.ScalarMappable
        a mixin class to map scalar data to RGBA
    norm: array
        a class that specifies a desired data normalisation into a [min, max] interval
    text_color: string
        the text color
    font_size: int
        the colorbar's labels text size
    ticks: int
        the number of ticks along the colorbar
    symbol: boolean
        if True, it shows the ">" next to the highest tick's label in the colorbar (useful when normalising)
    cbar_min_max: boolean
        if True, it only shows the ">" and "<" as labels of the lowest and highest ticks' the colorbar
    """
    
    if isinstance(plot, Plot):
        ax = plot.ax
    else:
        ax = plot.grid
        
    cb = plot.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, shrink = shrinkage)
    tick_locator = ticker.MaxNLocator(nbins=ticks)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.outline.set_visible(False)

    ticks = list(cb.get_ticks())
    for t in ticks: 
        if (t == ticks[-1]) & (t != norm.vmax) :
            ticks[-1] = norm.vmax

    if min_max:
        ticks = [norm.vmin, norm.vmax]
    
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([round(t,1) for t in ticks])
    if symbol:
        cb.ax.set_yticklabels([round(t,1) if t < norm.vmax else "> "+str(round(t,1)) for t in cb.ax.get_yticks()])

    plt.setp(plt.getp(cb.ax, "yticklabels"), color = plot.text_color, fontfamily = 'Times New Roman', fontsize= plot.font_size_secondary)
                
def set_axes_frame(axes_frame = False, ax = None, black_background = False, text_color = 'black'):
    """ 
    It draws the axis frame.
    
    Parameters
    ----------
    ax: matplotlib.axes
        the Axes on which plotting
    black_background: boolean
        it indicates if the background color is black
    text_color: string
        the text color
    """
    if not axes_frame:
        ax.set_axis_off()
        return
      
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis= 'both', which= 'both', length=0)
    
    for spine in ax.spines:
        ax.spines[spine].set_color(text_color)
    if black_background: 
        ax.set_facecolor('black')


        