import pandas as pd, numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cols
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from matplotlib.lines import Line2D  
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import mapclassify, pylab, colorsys
pd.set_option("display.precision", 3)

from .utilities import scaling_columnDF

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
    
    def __init__(self, figsize, black_background, title):
        
        self.fig, self.ax = plt.subplots(1, figsize=figsize)
        self.title = title
        self.fig.suptitle(title, color = "white" if black_background else "black", fontsize=20, fontfamily = 'Times New Roman')
        self.fig.subplots_adjust(top=0.92)
        self.ax.axis("equal")
        
        rect = self.fig.patch 
        if black_background: 
            self.text_color = "white"
            rect.set_facecolor("black")
        else: 
            self.text_color = "black"
            rect.set_facecolor("white")
        
        # background black or white - basic settings
        rect = self.fig.patch 
        if black_background: 
            self.text_color = "white"
            rect.set_facecolor("black")
        else: 
            self.text_color = "black"
            rect.set_facecolor("white")
        
        self.font_size_primary = figsize[0]+figsize[0]*0.3
        self.font_size_secondary = figsize[0]
        self.fig.suptitle(title, color = self.text_color, fontsize= self.font_size_primary, fontfamily = 'Times New Roman')
        self.fig.subplots_adjust(top=0.92)

                
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
    def __init__(self, figsize, nrows, ncols, black_background, title = None):
        """
        Initializes the MultiPlot class.
        """
    
        self.fig, self.grid = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

        rect = self.fig.patch 
        if black_background: 
            self.text_color = "white"
            rect.set_facecolor("black")
        else: 
            self.text_color = "black"
            rect.set_facecolor("white")
        
        self.font_size_primary = figsize[0]+figsize[0]*0.3
        self.font_size_secondary = figsize[0]
        
        if title is not None:
            fig.suptitle(title, color = self.text_color, fontsize = self.font_size_secondary, fontfamily = 'Times New Roman', 
                         ha = 'center', va = 'center') 
            fig.subplots_adjust(top=0.92)
         
def _single_plot(ax, gdf, column = None, scheme = None, bins = None, classes = 7, norm = None, cmap = None, color = 'red', alpha = 1.0, 
                legend = False, geometry_size = 1.0,  geometry_size_column = None, geometry_size_factor = None, zorder = 0):
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
    if (norm is not None) | (scheme is not None):
        categorical = False
    else:
        categorical = True
    if (column is not None): 
        if (gdf[column].dtype != 'O' ):
            gdf = gdf.reindex(gdf[column].abs().sort_values(ascending = True).index)
        elif not categorical:
            gdf[column] = gdf[column].astype(float)
    
    # categorical map
    if (column is not None) & (scheme is None) & (norm is None) & (cmap is None): 
        cmap = rand_cmap(len(gdf[column].unique()))         
        
    if (norm is not None) | (scheme is not None):    
        color = None
        if cmap is None:
            cmap = kindlmann()
        if norm is not None:
            scheme = None
            legend = False
        
    if bins is None: 
        c_k = dict()
        if classes is not None:
            c_k = {"k" : classes}
    else: 
        c_k = {'bins':bins, "k" : len(bins)}
        scheme = 'User_Defined'
        
    geometry_type = gdf.iloc[0].geometry.geom_type
    if geometry_type == 'Point':    
        if (geometry_size_factor is not None): 
            scaling_columnDF(gdf, column)
            gdf['geometry_size'] = np.where(gdf[column+'_sc'] >= 0.20, gdf[column+'_sc']*geometry_size_factor, 0.40) # marker size
            geometry_size = gdf['geometry_size']
          
        gdf.plot(ax = ax, column = column, markersize = geometry_size, categorical = categorical, color = color, scheme = scheme, cmap = cmap, 
                norm = norm, alpha = alpha, legend = legend, classification_kwds = c_k, zorder = zorder) 
        
    elif geometry_type == 'LineString':
        if geometry_size_factor is not None:
            geometry_size = [(abs(value)*geometry_size_factor) if (abs(value)*geometry_size_factor) > 1.1 else 1.1 for value in gdf[column]]
       
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, linewidth = geometry_size, scheme = scheme, alpha = alpha, 
            cmap = cmap, norm = norm, legend = legend, classification_kwds = c_k, capstyle = 'round', joinstyle = 'round', zorder = zorder) 
                
    else:
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, scheme = scheme, edgecolor = 'none', alpha = alpha, cmap = cmap,
            norm = norm, legend = legend, classification_kwds = c_k, zorder = zorder) 
        
 
def plot_gdf(gdf, column = None, title = None, black_background = True, figsize = (15,15), scheme = None, bins = None, classes = None, norm = None,
            cmap = None, color = None, alpha = None, legend = False, geometry_size = 1.0, geometry_size_column = None, 
            geometry_size_factor = None, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75,
            axes_frame = False, base_map_gdf = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4, base_map_geometry_size = 1.1,  
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
    geometry_size: float
        line width, when plotting a LineString GeoDataFrame
    geometry_size_factor: float
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the line width 
        accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a LineString GeoDataFrame
    base_map_gdf: GeoDataFrame
        a desired additional layer to use as a base map        
    base_map_color: string
        color applied to all geometries of the base map
    base_map_alpha: float
        base map's alpha value
    base_map_geometry_size: float
        base map's marker size when the base map is a Point GeoDataFrame
    base_map_geometry_size: float
        base map's line width when the base map is a LineString GeoDataFrame
    base_map_zorder: int   
        zorder of the layer; e.g. if 0, plots first, thus main GeoDataFrame on top; if 1, plots last, thus on top.
        
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """   
    
    # fig,ax set up
    plot = Plot(figsize = figsize, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.ax
    
    ax.set_aspect("equal")
    _set_axes_frame(axes_frame, ax, black_background, plot.text_color)
 
    zorder = 0
    if (not base_map_gdf.empty):
        _plot_baseMap(gdf = base_map_gdf, ax = ax, color = base_map_color, geometry_size = base_map_geometry_size, alpha = base_map_alpha, 
            zorder = base_map_zorder )
        if base_map_zorder == 0:
            zorder = 1
   
    if geometry_size_column is None:
        geometry_size_column = column
    _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, 
                geometry_size = geometry_size, geometry_size_column = geometry_size_column, geometry_size_factor = geometry_size_factor, 
                zorder = zorder, legend = legend)

    if legend: 
        _generate_legend_ax(ax, plot.font_size_secondary, black_background) 
    elif cbar:
        if norm is None:
            min_value = gdf[column].min()
            max_value = gdf[column].max()
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
            
        _generate_colorbar(plot, cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, shrinkage = cbar_shrinkage)
    
    return fig    
                      
def plot_gdfs(list_gdfs = [], column = None, ncols = 2, main_title = None, titles = [], black_background = True, figsize = (15,30), scheme = None, 
                bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, 
                geometry_size = None, geometry_size_columns = [], geometry_size_factor = None): 
                     
    """
    It plots the geometries of a list of GeoDataFrame, containing the same type of geometry. Coloring is based on a provided column (that needs to 
    be a column in each passed GeoDataFrame), using a given scheme.
    If only "column" is provided, a categorical map is depicted.
    If no column is provided, a plain map is shown.
    
    Parameters
    ----------
    list_gdfs: list of GeoDataFrames
        GeoDataFrames to be plotted
    column: string
        Column on which the plot is based
    main_title: string 
        main title of the plot
    titles: list of string
        list of titles to be assigned to each quadrant (axes) of the grid
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
    geometry_size: float
        line width, when plotting a LineString GeoDataFrame
    geometry_size_factor: float
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the line width 
        accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a LineString GeoDataFrame   
        
    
    Returns
    -------
    fig: matplotlib.figure.Figure object
        the resulting figure
    """              
                     
    if ncols == 2:
        nrows, ncols = int(len(list_gdfs)/2), 2
        if (len(list_gdfs)%2 != 0): 
            nrows = nrows+1
    else:
        nrows, ncols = int(len(list_gdfs)/3), 3
        if (len(list_gdfs)%3 != 0): 
            nrows = nrows+1

    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background, 
                          title = main_title)
    
    fig, grid = multiPlot.fig, multiPlot.grid   
    legend_fig = False
    legend_ax = False
    
    if nrows > 1: 
        grid = [item for sublist in grid for item in sublist]
    if cbar:
        legend = False
        if (norm is None):
            min_value = min([gdf[column].min() for gdf in list_gdfs])
            max_value = max([gdf[column].max() for gdf in list_gdfs])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
            
    for n, ax in enumerate(grid):
        ax.set_aspect("equal")
        _set_axes_frame(axes_frame, ax, black_background, multiPlot.text_color)    

        if n > len(list_gdfs)-1: 
            continue # when odd nr of gdfs    
        
        gdf = list_gdfs[n]
        if len(titles) > 0:
            ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size_primary, color = multiPlot.text_color,  
            pad = 15)
            
        legend_ax = (legend) & ((n == ncols*nrows/2) & (scheme == 'User_Defined') or (scheme != 'User_Defined'))
        legend_fig = (n == ncols*nrows/2) & (legend) & (scheme == 'User_Defined')
        
        geometry_size_column = column
        if geometry_size_columns != []:
            geometry_size_column = geometry_size_columns[n]
        _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, 
                    alpha = alpha, legend = legend_ax, geometry_size = geometry_size, geometry_size_column = geometry_size_column, 
                    geometry_size_factor = geometry_size_factor)
                    
        if legend_fig:
            _generate_legend_fig(ax, nrows, multiPlot.text_color, (multiPlot.font_size_primary), black_background)
        elif legend_ax:
            _generate_legend_ax(ax, (multiPlot.font_size_secondary), black_background)
    
    if cbar:
        _generate_colorbar(multiPlot, cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)
            
    return fig
   
def plot_gdf_grid(gdf = None, columns = [], ncols = 2, titles = [], black_background = True, figsize = (15,15), scheme = None, bins = None, 
                classes = None, norm = None, cmap = None, color = None, alpha = None, legend = False, cbar = False, 
                cbar_ticks = 5, cbar_max_symbol = False, cbar_min_max = False, cbar_shrinkage = 0.75, axes_frame = False, geometry_size = None, 
                geometry_size_columns = [], geometry_size_factor = None): 
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
    geometry_size: float
        line width, when plotting a LineString GeoDataFrame
    geometry_size_factor: float
        when provided, it rescales the column provided, if any, from 0 to 1 and it uses the geometry_size_factor to rescale the line 
        width accordingly 
        (e.g. rescaled variable's value [0-1] * factor), when plotting a LineString GeoDataFrame
    """   
    
    if ncols == 2:
        nrows, ncols = int(len(columns)/2), 2
        if (len(columns)%2 != 0): 
            nrows = nrows+1
    else:
        nrows, ncols = int(len(columns)/3), 3
        if (len(columns)%3 != 0): 
            nrows = nrows+1
     
    nrows = (len(columns) + ncols - 1) // ncols 
    multiPlot = MultiPlot(figsize = figsize, nrows = nrows, ncols = ncols, black_background = black_background)
    fig, grid = multiPlot.fig, multiPlot.grid   
    legend_fig = False
    
    if cbar:
        legend = False
        if norm is None:
            min_value = min([gdf[column].min() for column in columns])
            max_value = max([gdf[column].max() for column in columns])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
    
    for n, ax in enumerate(grid.flat):
        
        ax.set_aspect("equal")
        _set_axes_frame(axes_frame, ax, black_background, multiPlot.text_color)
        
        if n > len(columns)-1: 
            continue # when odd nr of columns
        
        column = columns[n]
        if titles:          
            ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size_primary, color = multiPlot.text_color, 
            pad = 15)
        
        legend_ax = (legend) & ((n == ncols*nrows/2) & (scheme == 'User_Defined') or (scheme != 'User_Defined'))
        legend_fig = (n == ncols*nrows/2) & (legend) & (scheme == 'User_Defined')
                 
        geometry_size_column = column
        if geometry_size_columns != []:
            geometry_size_column = geometry_size_columns[n]
       
        _single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, 
                    alpha = alpha, legend = legend_ax, geometry_size = geometry_size, geometry_size_column = geometry_size_column, 
                    geometry_size_factor = geometry_size_factor)
                            
        if legend_fig:
            _generate_legend_fig(ax, nrows, multiPlot.text_color, multiPlot.font_size_secondary, black_background)
        elif legend_ax:
            _generate_legend_ax(ax, (multiPlot.font_size_secondary), black_background)

    if cbar:   
        _generate_colorbar(plot = multiPlot, cmap = cmap, norm = norm, ticks = cbar_ticks, symbol = cbar_max_symbol, min_max = cbar_min_max, 
                    shrinkage = cbar_shrinkage)

    return fig

def _plot_baseMap(gdf = None, ax = None, color = None, geometry_size = None, alpha = 0.5, zorder = 0):
    
    if gdf.iloc[0].geometry.geom_type == 'LineString':
        gdf.plot(ax = ax, color = color, linewidth = geometry_size, alpha = alpha,zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Point':
        gdf.plot(ax = ax, color = color, markersize = geometry_size, alpha = alpha, zorder = zorder)
    if gdf.iloc[0].geometry.geom_type == 'Polygon':
        gdf.plot(ax = ax, color = color, alpha = alpha, zorder = zorder)
    

def plot_multiplex_network(multiplex_graph, multiplex_edges):
    """
    Plots a multiplex network graph with 3D visualization.
    
    Parameters:
    - multiplex_graph (networkx.MultiGraph): A multiplex network graph object.
    - multiplex_edges (GeoDataFrame): A GeoDataFrame containing the edges of the graph.
    
    Returns:
    - matplotlib.figure.Figure: A 3D figure object containing the plotted multiplex network.
    """
    
    # Extract node coordinates and attributes
    node_xs = [float(node["x"]) for node in multiplex_graph.nodes.values()]
    node_ys = [float(node["y"]) for node in multiplex_graph.nodes.values()]
    node_zs = np.array([float(node["z"])*2000 for node in multiplex_graph.nodes.values()])
    node_sizes = []
    node_colors = []

    # Determine size and color for each node based on attributes
    for i, d in multiplex_graph.nodes(data=True):
        if d["station"]:
            node_sizes.append(9)
            node_colors.append("#ec1a30")
        elif d["z"] == 1:
            node_sizes.append(0.0)
            node_colors.append("#ffffcc")
        elif d["z"] == 0:
            node_sizes.append(8)
            node_colors.append("#ff8566")

    lines = []
    line_widths = []
    default_line_width = 0.4
    
    # Extract edge coordinates and attributes
    for u, v, data in multiplex_graph.edges(data=True):
        xs, ys = data["geometry"].xy
        zs = [multiplex_graph.node[u]["z"]*2000 for i in range(len(xs))]
        if data["layer"] == "intra_layer": 
            zs = [0, 2000]
        
        lines.append([list(a) for a in zip(xs, ys, zs)])
        if data["layer"] == "intra_layer": 
            line_widths.append(0.2)
        elif data["pedestrian"] == 1: 
            line_widths.append(0.1)
        else: 
            line_widths.append(default_line_width)

    # Set up figure and 3D axis
    fig_height = 40
    west, south, east, north = multiplex_edges.total_bounds
    bbox_aspect_ratio = (north - south) / (east - west)*1.5
    fig_width = fig_height + 90 / bbox_aspect_ratio/1.5
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection="3d")
    ax.add_collection3d(Line3DCollection(lines, linewidths=line_widths, alpha=1, color="#ffffff", zorder=1))
    ax.scatter(node_xs, node_ys, node_zs, s=node_sizes, c=node_colors, zorder=2)
    ax.update(xlim=(west, east), ylim=(south, north), zlim=(0,2500), aspect='equal', axis='off', margins=0)
    ax.tick_params(which="both", direction="in")
    fig.canvas.draw()
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    return fig

def _generate_legend_fig(ax, nrows, ncols, text_color, font_size, black_background):
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
    plt.setp(leg.texts, family='Times New Roman', fontsize = font_size, color = text_color, va = 'center')
    
    if ncols == 2:
        if nrows%2 == 0: 
            leg.set_bbox_to_anchor((2.15, 1.00, 0.33, 0.33))    
        else: 
            leg.set_bbox_to_anchor((1.15, 0.5, 0.33, 0.33))
    
    elif ncols == 3:
        if nrows%2 == 0: 
            leg.set_bbox_to_anchor((2.25, 1.15, 0.33, 0.33))    
        else:     
            leg.set_bbox_to_anchor((1.25, 0.65, 0.33, 0.33))
        
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    leg.get_frame().set_facecolor('none')
    
    for handle in leg.legendHandles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(15)
        else: 
            break
def _generate_legend_ax(ax, font_size, black_background):
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
    if black_background:
        text_color = 'black'
    else: 
        text_color = 'white'
    
    plt.setp(leg.texts, family='Times New Roman', fontsize = font_size, color = text_color, va = 'center')
    leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    
    for handle in leg.legendHandles:
        if not isinstance(handle, Line2D):
            handle._legmarker.set_markersize(12)
        else:
            break
    leg.get_frame().set_facecolor('none')
    
def _generate_colorbar(plot = None, cmap = None, norm = None, ticks = 5, symbol = False, min_max = False, shrinkage = 0.95):
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
             
def normalize(n, range1, range2):
    """ 
    It generate the legend for a figure.
    
    Parameters
    ----------
    ax:
    
    nrows:
    
    
    Returns
    -------
    cmap:  matplotlib.colors.Colormap
        the color map
    """  
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]           

def random_colors_list(nlabels, vmin = 0.8, vmax = 1.0, hsv = False):
    """ 
    It generates a list of random HSV colors, given the number of classes, 
    min and max values in the HSV spectrum.
    
    Parameters
    ----------

       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        the color map
    """
    
    randRGBcolors = []
    randHSVcolors = [(np.random.uniform(low=0.0, high=0.95),
                      np.random.uniform(low=0.4, high=0.95),
                      np.random.uniform(low= vmin, high= vmax)) for i in range(nlabels)]
   
    # Convert HSV list to RGB
    if hsv:
        randRGBcolors = []
        for HSVcolor in randHSVcolors: 
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
    return  randHSVcolors  
            
# Generate random colormap
def rand_cmap(nlabels, type_color ='soft'):
    """ 
    It generates a categorical random color map, given the number of classes
    
    Parameters
    ----------
    nlabels: int
        the number of categories to be coloured 
    type_color: string {"soft", "bright"} 
        it defines whether using bright or soft pastel colors, by limiting the RGB spectrum
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        the color map
    """   
    if type_color not in ('bright', 'soft'):
        type_color = 'bright'
    
    # Generate color map for bright colors, based on hsv
    if type_color == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=0.8),
                          np.random.uniform(low=0.2, high=0.8),
                          np.random.uniform(low=0.9, high=1.0)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))


        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type_color == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def kindlmann():
    """ 
    It returns a Kindlmann color map. See https://ieeexplore.ieee.org/document/1183788
       
    Returns
    -------
    cmap: matplotlib.colors.LinearSegmentedColormap
        the color map
    """   

    kindlmann_list = [(0.00, 0.00, 0.00,1), (0.248, 0.0271, 0.569, 1), (0.0311, 0.258, 0.646,1),
            (0.019, 0.415, 0.415,1), (0.025, 0.538, 0.269,1), (0.0315, 0.658, 0.103,1),
            (0.331, 0.761, 0.036,1),(0.768, 0.809, 0.039,1), (0.989, 0.862, 0.772,1),
            (1.0, 1.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list('kindlmann', kindlmann_list)
    return cmap
    
def _set_axes_frame(axes_frame = False, ax = None, black_background = False, text_color = 'black'):
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
      
def cmap_from_colors(list_colors):
    """ 
    It generates a colormap given a list of colors.
    
    Parameters
    ----------
    list_colors: list of string
        the list of colours
       
    Returns
    -------
    cmap:  matplotlib.colors.LinearSegmentedColormap
        the color map
    """   
    cmap = LinearSegmentedColormap.from_list('custom_cmap', list_colors)
    return cmap
    
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

        