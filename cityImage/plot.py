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

class Plot():
    """
    Class for creating a basic matplotlib plot with a title and customizable settings.

    Parameters
    ----------
    figsize: tuple
        Tuple of width and height for the figure.
    black_background: bool
        Whether to set the background color as black.
    title: str
        Title for the plot.

    Attributes
    ----------
    fig: matplotlib.figure.Figure
        Figure object for the plot.
    ax: matplotlib.axes object
        Axes object for the plot.
    title: str
        Title for the plot.
    text_color: str
        Color for text elements, white if black_background is True and black otherwise.
    font_size_primary: int
        Font size for the title.
    font_size_secondary: int
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
    
    Attributes:
    ----------
    figsize: tuple
        The size of the figure (width, height) in inches.
    nrows: int
        The number of rows in the grid layout.
    ncols: int
        The number of columns in the grid layout.
    black_background : bool
        Specifies whether the plot has a black background (True) or white background (False).
    fontsize: int
        The font size to be used in the plot.
    title: str or None, optional
        The title of the figure. If None, no title is displayed.    
    """
    def __init__(self, figsize, nrows, ncols, black_background, fontsize, title = None):
        """
        Initializes the MultiPlot class.
        
        Parameters
        ----------
        figsize: tuple
            The size of the figure (width, height) in inches.
        nrows: int
            The number of rows in the grid layout.
        ncols: int
            The number of columns in the grid layout.
        black_background : bool
            Specifies whether the plot has a black background (True) or white background (False).
        fontsize: int
            The font size to be used in the plot.
        title: str or None
            The title of the figure. If None, no title is displayed.
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
            classes = None, norm = None, cmap = None, color = None, alpha = None, geometry_size = 1.0, 
            geometry_size_column = None, geometry_size_factor = None, legend = False, fontsize = 15, cbar = False, cbar_ticks = 5, 
            cbar_max_symbol = False, cbar_min_max = False, cbar_shrink = 0.75, axes_frame = False, 
            base_map_gdf = pd.DataFrame({"a": []}), base_map_color = None, base_map_alpha = 0.4, base_map_geometry_size = 1.1,  
            base_map_zorder = 0):

    """
    It plots the geometries of a single GeoDataFrame, coloring on the bases of the values contained in column, using a given scheme.
    When only column is provided (no scheme), a categorical map is depicted.
    When no column is provided, a plain map is shown.
    
    The other parameters regulate colorbar, legend, base map. 
    Use this function for plotting in relation to maximum one column, one GeoDataFrame.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted.
    column: str
        Column on which the plot is based.
    title: str
        Title of the plot.
    black_background: bool
        Specifies whether the plot has a black background (True) or white background (False).
    fig_size: float
        Size of the figure's side extent.
    scheme: str
        Classification method. Choose amongst the options listed at https://pysal.org/mapclassify/api.html.
    bins: list
        Bins defined by the user.
    classes: int
        Number of classes for categorizing the data when scheme is not "None".
    norm: array
        A class that specifies a desired data normalization into a [min, max] interval.
    cmap: str or matplotlib.colors.LinearSegmentedColormap
        Color map for the plot. See matplotlib colormaps for a list of possible values or pass a colormap.
    color: str
        Categorical color applied to all geometries when not using a column to color them.
    alpha: float
        Alpha value of the plotted layer.
    geometry_size: float
        Point size value when plotting a Point GeoDataFrame or Width value when plotting LineString GeoDataFrame.
    geometry_size_columns: List of str
        The column name in the GeoDataFrame to be used for scaling the geometry size.
    geometry_size_factor: float
        Rescaling factor for the column provided, if any. The column is rescaled from 0 to 1, and the
        geometry_size_factor is used to rescale the marker size accordingly
        (e.g., rescaled variable's value [0-1] * factor) when plotting a Point GeoDataFrame.
    legend: bool
        When True, show the legend.
    fontsize: int
        Font size.
    cbar: bool
        If True, show the colorbar; otherwise, don't. When True, the legend is not shown.
    cbar_ticks: int
        Number of ticks along the colorbar.
    cbar_max_symbol: bool
        If True, show the ">" next to the highest tick's label in the colorbar (useful when normalizing).
    cbar_min_max: bool
        If True, only show the ">" and "<" as labels of the lowest and highest ticks' the colorbar.
    cbar_shrink:
        Fraction by which to multiply the size of the colorbar.    
    axes_frame: bool
        If True, show the axes' frame.
    base_map_gdf: GeoDataFrame
        Additional layer to use as a base map.
    base_map_color: str
        Color applied to all geometries of the base map.
    base_map_alpha: float
        Base map's alpha value.
    base_map_geometry_size: float
        Base map's marker size when the base map is a Point GeoDataFrame.
    base_map_zorder: int
        Z-order of the layer. If 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
        
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
        generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, cbar_ticks = cbar_ticks, cbar_max_symbol = cbar_max_symbol, cbar_min_max = cbar_min_max, 
                    cbar_shrink = cbar_shrink)
    elif legend: 
        generate_legend_ax(ax, plot) 

    return plot.fig      
 
def plot_grid_gdfs_column(gdfs = [], column = None, ncols = 1, nrows = 1, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                classes = None, norm = None, cmap = None, color = None, alpha = None, geometry_size = None, geometry_size_columns = [], geometry_size_factor = None,
                legend = False, fontsize = 15, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrink = 0.75, axes_frame = False):
    """
    It plots the geometries of different GeoDataFrames, coloring on the bases of the values contained in the provided column, using a given scheme.
    When only column is provided (no scheme), a categorical map is depicted.
    When no column is provided, a plain map is shown.
    
    The other parameters regulate colorbar, legend, etc (no basemap here). 
    Use this function for plotting in relation to maximum one column, for multiple GeoDataFrames.
  
    Parameters
    ----------
    gdfs: list of GeoDataFrame
        The list of GeoDataFrames to be plotted.
    column: str
        Column on which the plot is based.
    ncols: int
        The number of desired columns for organising the subplots.
    nrows: int
        The number of desired rows for organising the subplots.
    titles: list of str
        The list of titles, one per axes (and column, when provided).
    black_background: boolean 
        Black background or white.
    fig_size: float
        Size figure extent.
    scheme: str
        Classification method. Choose amongst the options listed at https://pysal.org/mapclassify/api.html.
    bins: list
        Bins defined by the user.
    classes: int
        Number of classes for categorizing the data when scheme is not "None".
    norm: array
        A class that specifies a desired data normalization into a [min, max] interval.
    cmap: str or matplotlib.colors.LinearSegmentedColormap
        Color map for the plot. See matplotlib colormaps for a list of possible values or pass a colormap.
    color: str
        Categorical color applied to all geometries when not using a column to color them.
    alpha: float
        Alpha value of the plotted layer.
    geometry_size: float
        Point size value when plotting a Point GeoDataFrame or Width value when plotting LineString GeoDataFrame.
    geometry_size_columns: List of str
        The column name(s) in the GeoDataFrames to be used for scaling the geometry size.
    geometry_size_factor: float
        Rescaling factor for the column provided, if any. The column is rescaled from 0 to 1, and the
        geometry_size_factor is used to rescale the marker size accordingly
        (e.g., rescaled variable's value [0-1] * factor) when plotting a Point GeoDataFrame.
    legend: bool
        When True, show the legend.
    fontsize: int
        Font size.
    cbar: bool
        If True, show the colorbar; otherwise, don't. When True, the legend is not shown.
    cbar_ticks: int
        Number of ticks along the colorbar.
    cbar_max_symbol: bool
        If True, show the ">" next to the highest tick's label in the colorbar (useful when normalizing).
    cbar_min_max: bool
        If True, only show the ">" and "<" as labels of the lowest and highest ticks' the colorbar.
    cbar_shrink:
        Fraction by which to multiply the size of the colorbar. 
    axes_frame: bool
        If True, show the axes' frame.
        
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
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
          
        parameters = {'ax': ax, 'n': n, 'multiPlot': multiPlot, 'column': column , 'gdf': gdf, 'titles': titles, 
                      'scheme': scheme, 'bins': bins, 'classes': classes, 'norm': norm, 'cmap': cmap, 
                      'color': color, 'alpha': alpha, 'legend': legend, 'axes_frame': axes_frame,
                      'geometry_size': geometry_size, 'geometry_size_columns': geometry_size_columns, 'geometry_size_factor': geometry_size_factor}        
        subplot(**parameters)    
            
    if (cbar) & (not legend):  
        generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, cbar_ticks = cbar_ticks, cbar_max_symbol = cbar_max_symbol, cbar_min_max = cbar_min_max, 
                    cbar_shrink = cbar_shrink)

    return multiPlot.fig                      

def plot_grid_gdf_columns(gdf, columns = [], ncols = 1, nrows = 1, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                          classes = None, norm = None, cmap = None, color = None, alpha = None, 
                          geometry_size = None, geometry_size_columns = [], geometry_size_factor = None, legend = False, fontsize = 15,
                          cbar = False, cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrink = 0.75, axes_frame = False):
    """
    It plots the geometries of a GeoDataFrame, coloring on the bases of the values contained in two or more provided columns, using a given scheme.
    When no columns are provided, the function raises an error.
    
    The other parameters regulate colorbar, legend, etc (no basemap here). 
    Use this function for plotting in relation to more than one column, for a single GeoDataFrame.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to be plotted.
    columns: list of str
        The list of column on which the plot is based.
    ncols: int
        The number of desired columns for organising the subplots.
    nrows: int
        The number of desired rows for organising the subplots.
    titles: list of str
        Title of the plot.
    black_background: boolean 
        Black background or white.
    fig_size: float
        Size figure extent.
    scheme: str
        Classification method. Choose amongst the options listed at https://pysal.org/mapclassify/api.html.
    bins: list
        Bins defined by the user.
    classes: int
        Number of classes for categorizing the data when scheme is not "None".
    norm: array
        A class that specifies a desired data normalization into a [min, max] interval.
    cmap: str or matplotlib.colors.LinearSegmentedColormap
        Color map for the plot. See matplotlib colormaps for a list of possible values or pass a colormap.
    color: str
        Categorical color applied to all geometries when not using a column to color them.
    alpha: float
        Alpha value of the plotted layer.
    geometry_size: float
        Point size value when plotting a Point GeoDataFrame or Width value when plotting LineString GeoDataFrame.
    geometry_size_columns: List of str
        The column name(s) in the GeoDataFrame to be used for scaling the geometry size.
    geometry_size_factor: float
        Rescaling factor for the column provided, if any. The column is rescaled from 0 to 1, and the
        geometry_size_factor is used to rescale the marker size accordingly
        (e.g., rescaled variable's value [0-1] * factor) when plotting a Point GeoDataFrame.
    legend: bool
        When True, show the legend.
    fontsize: int
        Font size.
    cbar: bool
        If True, show the colorbar; otherwise, don't. When True, the legend is not shown.
    cbar_ticks: int
        Number of ticks along the colorbar.
    cbar_max_symbol: bool
        If True, show the ">" next to the highest tick's label in the colorbar (useful when normalizing).
    cbar_min_max: bool
        If True, only show the ">" and "<" as labels of the lowest and highest ticks' the colorbar.
    cbar_shrink:
        Fraction by which to multiply the size of the colorbar. 
    axes_frame: bool
        If True, show the axes' frame.
    
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """   
    if len(columns) == 0:
        raise ValueError("Provide a list of columns to plot the geometries on. For a plain plot, use plot_gdf")
    
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
            
        parameters = {'ax': ax, 'n': n, 'multiPlot': multiPlot, 'column': column , 'gdf': gdf, 'titles': titles, 
                      'scheme': scheme, 'bins': bins, 'classes': classes, 'norm': norm, 'cmap': cmap, 
                      'color': color, 'alpha': alpha, 'legend': legend, 'axes_frame': axes_frame,
                      'geometry_size': geometry_size, 'geometry_size_columns': geometry_size_columns, 'geometry_size_factor': geometry_size_factor}        
        subplot(**parameters)    
            
    if (cbar) & (not legend):  
        generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, cbar_ticks = cbar_ticks, cbar_max_symbol = cbar_max_symbol, cbar_min_max = cbar_min_max, 
                    cbar_shrink = cbar_shrink)
    return multiPlot.fig

def plotOn_ax(ax, gdf, column = None, scheme = None, bins = None, classes = 7, norm = None, cmap = None, color = 'red', alpha = 1.0, 
                geometry_size = 1.0, geometry_size_column = None, geometry_size_factor = None, legend = False, zorder = 0):
    """

    
    Parameters
    ----------
    ax: matplotlib.axes object
        The axes object on which to create the subplot.
    gdf: GeoDataFrame
        GeoDataFrame to be plotted.
    column: str
        Column on which the plot is based.
    scheme: str
        Classification method. Choose amongst the options listed at https://pysal.org/mapclassify/api.html.
    bins: list
        Bins defined by the user.
    classes: int
        Number of classes for categorizing the data when scheme is not "None".
    norm: array
        A class that specifies a desired data normalization into a [min, max] interval.
    cmap: str or matplotlib.colors.LinearSegmentedColormap
        Color map for the plot. See matplotlib colormaps for a list of possible values or pass a colormap.
    color: str
        Categorical color applied to all geometries when not using a column to color them.
    alpha: float
        Alpha value of the plotted layer.
    geometry_size: float
        Point size value when plotting a Point GeoDataFrame or Width value when plotting LineString GeoDataFrame.
    geometry_size_columns: str
        The column name in the GeoDataFrame to be used for scaling the geometry size.
    geometry_size_factor: float
        Rescaling factor for the column provided, if any. The column is rescaled from 0 to 1, and the
        geometry_size_factor is used to rescale the marker size accordingly
        (e.g., rescaled variable's value [0-1] * factor) when plotting a Point GeoDataFrame.
    legend: bool
        When True, show the legend.
    zorder: int   
        Zorder of this layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
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
    
    parameters = {'ax': ax, 'column': column, 'classification_kwds': c_k, 'scheme': scheme, 'norm': norm,
                   'cmap': cmap, 'categorical': categorical, 'color': color, 'alpha': alpha, 'legend': legend, 
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
 
 
def subplot(ax, n, multiPlot, gdf, column, titles, scheme, bins, classes, norm, cmap, color, alpha, geometry_size, geometry_size_columns,
                    geometry_size_factor, legend, axes_frame):
    """
    Create a subplot with a map plot on the given axes.

    Parameters
    ----------
    ax: matplotlib.axes object
        The axes object on which to create the subplot.
    n: int
        The index of the subplot.
    multiPlot: MultiPlot object
        The MultiPlot object controlling the plot settings.
    gdf: GeoDataFrame
        The GeoDataFrame containing the data to plot.
    column: str
        The column name in the GeoDataFrame to be used for plotting.
    titles: str or sequence
        The title(s) of the subplot(s).
    scheme: str
        The classification scheme to use for mapping the data.
    bins: int or sequence or pandas.IntervalIndex
        The number of bins to use for the classification or the bin intervals.
    classes: int or sequence
        The number of classes to use for the classification or the class intervals.
    geometry_size: float
        Marker size value when plotting a Point GeoDataFrame or line width value when plotting LineString GeoDataFrame.
    geometry_size_columns: List of str
        The column name(s) in the GeoDataFrame to be used for scaling the geometry size.
    geometry_size_factor: float
        The factor by which to scale the geometry size.
    norm: Normalize or str
        The normalization scheme to use for mapping values to colors.
    cmap: str or Colormap
        The colormap to use for mapping values to colors.
    color: str
        Categorical color applied to all geometries when not using a column to color them.
    alpha: float
        Alpha value of the plotted layer.
    legend: bool
        If True, show the legend; otherwise, don't.
    axes_frame: bool
        Flag indicating whether to draw axes frame or not.
    """  
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
    It plots the geometries of a GeoDataFrame, coloring on the basis of the values contained in the provided columns, using a given scheme.
    If only column is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        The GeoDataFrame containing the map data.
    ax: matplotlib.axes object
        The axes object on which to create the subplot.
    color: str
        The color to use for mapping
    geometry_size: float
        Point size value when plotting a Point GeoDataFrame or Width value when plotting LineString GeoDataFrame.
    alpha: float
        Alpha value of the plotted layer.
    zorder: str
        The order of the map elements with respect to other elements in the axis, default value is 0.
    
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
        The axes object on which to create the subplot.
    plot: Plot, MultiPlot Object
        The Plot object.
    
    """ 
    leg = ax.get_legend()    
    fig_leg = plot.fig.legend(handles = leg.legend_handles, labels = [t.get_text() for t in leg.texts], loc=5, 
                              borderaxespad= 0)
    ax.get_legend().remove()
    plt.setp(fig_leg.texts, family='Times New Roman', fontsize = plot.font_size_secondary, color = plot.text_color, 
             va = 'center')

    fig_leg.get_frame().set_linewidth(0.0) # remove legend border
    fig_leg.set_zorder(102)
    fig_leg.get_frame().set_facecolor('none')

    for handle in fig_leg.legend_handles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(15)
        else: 
            break
           
def generate_legend_ax(ax, plot):
    """ 
    It generate the legend for an axes.
    
    Parameters
    ----------
    ax: matplotlib.axes object
        The axes object on which to create the subplot.
    plot: Plot, MultiPlot Object
        The Plot object.
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
    
    for handle in leg.legend_handles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(12)
        else:
            break
    leg.get_frame().set_facecolor('none')
    
def generate_colorbar(plot = None, cmap = None, norm = None, cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrink = 0.95):
    """ 
    It plots a colorbar, given some settings.
    
    Parameters
    ----------
    plot: Plot, MultiPlot Object
        The Plot object.
    cmap: str or matplotlib.colors.LinearSegmentedColormap
        Color map for the plot. See matplotlib colormaps for a list of possible values or pass a colormap.
    norm: array
        A class that specifies a desired data normalisation into a [min, max] interval.
    cbar_ticks: int
        Number of ticks along the colorbar.
    cbar_max_symbol: bool
        If True, show the ">" next to the highest tick's label in the colorbar (useful when normalizing).
    cbar_min_max: bool
        If True, only show the ">" and "<" as labels of the lowest and highest ticks' the colorbar.
    cbar_shrink:
        Fraction by which to multiply the size of the colorbar. 
    """
    
    if isinstance(plot, Plot):
        ax = plot.ax
    else:
        ax = plot.grid
        
    cb = plot.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, shrink = cbar_shrink)
    tick_locator = ticker.MaxNLocator(nbins=cbar_ticks)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.outline.set_visible(False)

    ticks = list(cb.get_ticks())
    for t in ticks: 
        if (t == ticks[-1]) & (t != norm.vmax) :
            ticks[-1] = norm.vmax

    if cbar_min_max:
        ticks = [norm.vmin, norm.vmax]
    
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([round(t,1) for t in ticks])
    if cbar_max_symbol:
        cb.ax.set_yticklabels([round(t,1) if t < norm.vmax else "> "+str(round(t,1)) for t in cb.ax.get_yticks()])

    plt.setp(plt.getp(cb.ax, "yticklabels"), color = plot.text_color, fontfamily = 'Times New Roman', fontsize= plot.font_size_secondary)
                
def set_axes_frame(axes_frame = False, ax = None, black_background = False, text_color = 'black'):
    """ 
    It draws the axis frame.
    
    Parameters
    ----------
    axes_frame: bool
        Flag indicating whether to draw axes frame or not.
    ax: matplotlib.axes object
        The axes object on which to create the subplot.
    black_background: boolean
        It indicates whether the background color is black.
    text_color: str
        The text color.
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


        