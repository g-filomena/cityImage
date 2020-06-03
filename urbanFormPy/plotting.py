import matplotlib as mp, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cols
import matplotlib.patches as mpatches

from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, DivergingNorm

import pylab
import colorsys

pd.set_option("precision", 10)


"""
Plotting functions

"""

## Plotting
    
class Plot():
    
    def __init__(self, fig_size, black_background, title):
    
        fig, ax = plt.subplots(1, figsize=(fig_size, fig_size))
        ax.set_axis_off()

        # background black or white - basic settings
        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        font_size = fig_size+5 # font-size   
        fig.suptitle(title, color = text_color, fontsize=font_size)
        
        plt.axis("equal")
        self.fig = fig
        self.ax = ax
        
class MultiPlotGrid():
    
    def __init__(self, fig_size, nrows, ncols, black_background, title, cbar):
        
        figsize = (fig_size, fig_size*nrows)
        if (nrows == 1) & (ncols == 2): figsize = (fig_size, fig_size/2)
            
        if cbar == False: cbar_mode = None
        else: cbar_mode = "edge"
        fig = plt.figure(figsize=figsize, frameon = True, edgecolor = 'green')
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows,ncols), axes_pad= (0.50, 1.00), share_all=True, cbar_location="right", 
                         cbar_mode=cbar_mode, cbar_size="7%", direction = 'row', cbar_pad= 0.75)
        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        
        font_size = fig_size+5 # font-size   
        if title is not None: fig.suptitle(title, color = text_color, fontsize=40, fontfamily = 'Times New Roman')    
        self.fig, self.grid = fig, grid
        self.font_size, self.text_color = font_size, text_color


    
def plot_points(gdf, column = None, classes = 7, ms = 0.9, ms_col = None, scheme = None, bins = None, color = None,
                cmap = None, title = "Plot", legend = False, color_bar = False, black_background = True,
                fig_size = 15, gdf_base_map = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4,
                base_map_lw = 1.1, base_map_ms = 2.0):
    """
    It creates a plot from a Point GeoDataFrame. 
    It plots the distribution over value and geographical space of variable "column" using "scheme". 
    If only "column" is provided, a categorical map is depicted.
    Otherwise, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string
        Column on which the plot is based
    classes: int
        classes for visualising when scheme is not "None"
    ms: float
        markersize value 
    ms_col: str 
        Column name in the GeoDataFrame's column where markersize values are stored
    scheme: dictionary of str {"Equal_Interval", "Fisher_Jenks"..}
        check: https://pysal.readthedocs.io/en/v1.11.0/library/esda/mapclassify.html
    bins: list
        bins defined by the user
    cmap: string,
        see matplotlib colormaps for a list of possible values
    title: str 
        title of the graph
    legend: boolean
        if True, show legend, otherwise don't
    color_bar: boolean
        if True, show color_bar, otherwise don't (only when legend is False)
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    gdf_base_map: LineString GeoDataFrame
        If provided, it is used as background/base map for visualisation purposes
    
    """
    
    # fig,ax set up
    plot = Plot(fig_size = fig_size, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.ax

    # background (e.g. street network)
    if not gdf_base_map.empty: 
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
    
    if column != None: gdf.sort_values(by = column,  ascending = True, inplace = True) 
    # markers size from column is provided
    if (ms_col != None): ms = gdf[ms_col]
    
    # plain plot:
    if (column == None) & (scheme == None):
        if black_background: 
            if color == None: color = 'white'
            gdf.plot(ax = ax, markersize = ms, color = color)
        else: 
            if color == None: color = 'blue'
            gdf.plot(ax = ax, markersize = ms, color = color)
    
    # categorical map
    elif (column != None) & (scheme == None):
        if cmap == None: cmap =  rand_cmap(len(gdf[column].unique()))
        gdf.plot(ax = ax, column = column, categorical = True, cmap = cmap, k = classes, markersize = ms, legend = legend, alpha = 1)    
    
    # user defined bins
    elif scheme == "User_Defined":
        gdf.plot(ax = ax, column = column, cmap = cmap, markersize = ms, scheme = scheme, legend = legend, classification_kwds={'bins':bins}, alpha = 1)
    # Lynch's bins - only for variables from 0 to 1
    elif scheme == "Lynch_Breaks":  
        bins = [0.125, 0.25, 0.5, 0.75, 1.00]
        gdf.plot(ax = ax, column = column, cmap = cmap, markersize = ms, scheme = scheme, legend = legend, classification_kwds={'bins':bins}, alpha = 1)
    # other schemes
    elif scheme != None: gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, markersize = ms, scheme = scheme,legend = legend, alpha = 1)
    if legend: _generate_legend(ax, black_background)
    
    plt.show() 
                
def plot_lines(gdf, column = None, classes = 7, lw = 1.1, scheme = None, bins = None, color = None, cmap = "Greys_r", 
               title = "Plot", legend = False, color_bar = False, black_background = True,                 
               fig_size = 15, gdf_base_map = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4,
               base_map_lw = 1.1, base_map_ms = 2.0):
    
    """
    It creates a plot from a lineString GeoDataFrame. 
    When column and scheme are not "None" it plots the distribution over value and geographical space of variable "column using scheme
    "scheme". If only "column" is provided, a categorical map is depicted.
    
    It plots the distribution over value and geographical space of variable "column" using "scheme". 
    If only "column" is provided, a categorical map is depicted.
    Otherwise, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string
        Column on which the plot is based
    classes: int
        classes for visualising when scheme is not "None"
    lw: float
        line width
    scheme: dictionary of str {"Equal_Interval", "Fisher_Jenks"..}
        check: https://pysal.readthedocs.io/en/v1.11.0/library/esda/mapclassify.html
    bins: list
        bins defined by the user
    cmap: string,
        see matplotlib colormaps for a list of possible values
    title: str 
        title of the graph
    legend: boolean
        if True, show legend, otherwise don't
    color_bar: boolean
        if True, show color_bar, otherwise don't (only when legend is False)
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent

    """ 
        
    # fig,ax set up
    plot = Plot(fig_size = fig_size, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.ax
    
    if column != None: gdf.sort_values(by = column, ascending = True, inplace = True)  
    
    # background (e.g. street network)
    if not gdf_base_map.empty: 
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
    
    # plain plot:
    if (column == None) & (scheme == None):
        if black_background: gdf.plot(ax = ax, linewidth = lw, color = color)
        else: gdf.plot(ax = ax, linewidth = lw, color = color)
    
    # categorigal plot
    elif (column != None) & (scheme == None):
        # boolean map
        if (cmap == None) & (classes == 2): 
            colors = ["white", "red"]
            gdf.plot(ax = ax, categorical = True, column = column, color = colors, linewidth = lw, legend = legend) 
        # categorical map
        else: 
            if cmap == None: cmap =  rand_cmap(len(gdf[column].unique()))
            gdf.plot(ax = ax, categorical = True, column = column, cmap = cmap, linewidth = lw, legend = legend) 
    # user defined bins
    elif scheme == "User_Defined":
        gdf.plot(ax = ax, column = column, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend, classification_kwds={'bins':bins})
    # Lynch's bins - only for variables from 0 to 1
    elif scheme == "Lynch_Breaks":  
        bins = [0.125, 0.25, 0.5, 0.75, 1.00]
        gdf.plot(ax = ax, column = column, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend, classification_kwds={'bins':bins})
    # other schemes
    elif scheme != None: 
        gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend)
    if legend: _generate_legend(ax, black_background)
                
    plt.show()
    
       
def plot_polygons(gdf, column = None, classes = 7, scheme = None, bins = None, color = None, cmap = None, alpha = 1.0, title =  "Plot", legend = False, 
                color_bar = False, black_background = True,  fig_size = 15, gdf_base_map = pd.DataFrame({"a" : []}),
                base_map_color = None, base_map_alpha = 0.4, base_map_lw = 1.1, base_map_ms = 2.0):
    """
    It creates a plot from a Polygon GeoDataFrame. 
    When column and scheme are not "None" it plots the distribution over value and geographical space of variable "column using scheme
    "scheme". If only "column" is provided, a categorical map is depicted.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string
        Column on which the plot is based
    classes: int
        classes for visualising when scheme is not "None"
    scheme: dictionary of str {"Equal_Interval", "Fisher_Jenks"..}
        check: https://pysal.readthedocs.io/en/v1.11.0/library/esda/mapclassify.html
    bins: list
        bins defined by the user
    cmap: string,
        see matplotlib colormaps for a list of possible values
    title: str 
        title of the graph
    legend: boolean
        if True, show legend, otherwise don't
    color_bar: boolean
        if True, show color_bar, otherwise don't (only when legend is False)
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent
    """
    
    # fig,ax set up
    plot = Plot(fig_size = fig_size, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.ax
       
    # plain plot
    if (column == None) & (scheme == None): gdf.plot(ax = ax, color = "orange",alpha = alpha, edgecolor="none")
    # categorigal plot
    elif (column != None) & (scheme == None): 
        if cmap == None: cmap =  rand_cmap(len(gdf[column].unique()))
        gdf.plot(ax = ax, column = column, cmap = cmap, alpha = alpha, categorical = True, legend = legend, edgecolor="none")  
    # user defined bins
    elif scheme == "User_Defined":
        gdf.plot(ax = ax, column = column, cmap = cmap, alpha = alpha, scheme = scheme, legend = legend, classification_kwds={'bins':bins}, edgecolor="none")
    # Lynch's bins - only for variables from 0 to 1
    elif scheme == "Lynch_Breaks":  
        bins = [0.125, 0.25, 0.5, 0.75, 1.00]
        gdf.plot(ax = ax, column = column, cmap = cmap, alpha = alpha, scheme = scheme, legend = legend, classification_kwds={'bins':bins}, edgecolor="none")
    # other schemes
    elif scheme != None: gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, alpha = alpha,  scheme = scheme, legend = legend, edgecolor="none")

    # background (e.g. street network)

    if not gdf_base_map.empty: 
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = "black", linewidth = base_map_lw, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = "black", markersize = base_map_ms, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha, edgecolor="none")
            else: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha, edgecolor="none")
    
    if legend: _generate_legend(ax, black_background)
    if (color_bar) & (not legend): _generate_color_bar(cmap, gdf[column], ax, text_color, font_size)

    plt.show()    

def plot_barriers(barriers_gdf, lw = 1.1, title = "Plot", legend = False, black_background = True,                 
               fig_size = 15, gdf_base_map = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4,
               base_map_lw = 1.1, base_map_ms = 2.0):
    
    """
    It creates a plot from a lineString GeoDataFrame. 
    When column and scheme are not "None" it plots the distribution over value and geographical space of variable "column using scheme
    "scheme". If only "column" is provided, a categorical map is depicted.
    
    It plots the distribution over value and geographical space of variable "column" using "scheme". 
    If only "column" is provided, a categorical map is depicted.
    Otherwise, a plain map is shown.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    
    lw: float
        line width
    title: str 
        title of the graph
    legend: boolean
        if True, show legend, otherwise don't
    black_background: boolean 
        black background or white
    fig_size: float
        size figure extent

    """ 
    barriers_gdf = barriers_gdf.copy()    
    
    # fig,ax set up
    plot = Plot(fig_size = fig_size, black_background = black_background, title = title)
    fig, ax = plot.fig, plot.ax
    
    # background (e.g. street network)
    if not gdf_base_map.empty: 
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            if black_background: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
            else: gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha)
    
    barriers_gdf['barrier_type'] = barriers_gdf['type']
    barriers_gdf.sort_values(by = 'barrier_type', ascending = False, inplace = True)  
    
    colors = ['green', 'brown', 'grey', 'blue']
    colormap = LinearSegmentedColormap.from_list('new_map', colors, N=4)
    barriers_gdf.plot(ax = ax, categorical = True, column = 'barrier_type', cmap = colormap, linewidth = lw, legend = legend, 
                     label =  'barrier_type') 
    if legend: _generate_legend(ax, black_background)       
    plt.show()  
    
def multi_plot_polygons(list_gdfs, list_sub_titles, main_title, column = None, classes = 7, scheme = None, bins = None, alpha = 1.0, 
                        cmap = "Greens_r", legend = False, color_bar = False, black_background = True):
    """
    It creates a series of subplots from a list of polygons GeoDataFrames
    When column and scheme are not "None" it plots the distribution over value and geographical space of variable "column using scheme
    "scheme". If only "column" is provided, a categorical map is depicted.
    
    Parameters
    ----------
    list_gdfs: list of GeoDataFrames
    list_subtitles: list of str
        subplots'titles
    columns: string
        Column on which the plot is based
    classes: int
        classes for visualising when scheme is not "None"
    scheme: dictionary of str {"Equal_Interval", "Fisher_Jenks"..}
        check: https://pysal.readthedocs.io/en/v1.11.0/library/esda/mapclassify.html
    bins: list
        bins defined by the user
    cmap: string,
        see matplotlib colormaps for a list of possible values
    legend: boolean
        if True, show legend, otherwise don't
    color_bar: boolean
        if True, show color_bar, otherwise don't (only when legend is False)
    black_background: boolean 
        black background or white
    """
    
    # inferring number of columns/rows 
    if len(list_gdf) == 1: nrows, ncols = 1, 1
    elif len(list_gdf) == 2: nrows, ncols = 1, 2
    else: 
        ncols = 3
        nrows = int(len(list_gdf)/ncols)
        if (len(list_gdf)%ncols != 0): nrows = int(nrows)+1;

    # fig,ax set up
    multiPlot = MultiPlot(fig_size = fig_size, nrows = nrows, ncols = ncols, black_background = black_background, title = title)
    fig, axes = multiPlot.fig, multiPlot.ax
        
    if nrows > 1: axes = [item for sublist in axes for item in sublist]
    for n, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_axis_off()
        try: gdf = list_gdf[n]
        except: continue
        
        # subtitles
        ax.set_title(list_sub_titles[n], color = text_color, fontsize = font_size-2)
        # plain plot
        if (column is None) & (scheme is None): gdf.plot(ax = ax, color = "orange", alpha = alpha)  # plain map
        # categorigal plot
        elif (column is not None) & (scheme is None): 
            if cmap is None: cmap = rand_cmap(len(gdf[column].unique()))
            gdf.plot(ax = ax, column = column, cmap = cmap, categorical = True, legend = legend, alpha = alpha)       
        # user defined bins
        elif scheme == "User_Defined":
            gdf.plot(ax = ax, column = column, cmap = cmap, scheme = scheme, legend = legend, alpha = alpha, classification_kwds={'bins':bins})
        # Lynch's bins - only for variables from 0 to 1
        elif scheme == "Lynch_Breaks":  
            bins = [0.125, 0.25, 0.5, 0.75, 1.00]
            gdf.plot(ax = ax, column = column, cmap = cmap, scheme = scheme, legend = legend, alpha = alpha, classification_kwds={'bins':bins})    
        # all other schemes
        elif scheme is not None: gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, alpha = alpha, scheme = scheme, legend = legend)

    plt.subplots_adjust(top = 0.88, hspace= 0.025)
    plt.show()  
    
def plot_lines_grid(gdf, columns = None, title = None, titles = None, fig_size = 15, black_background = True,
                    axis_frame = True, color = 'Black', lw = None, cmap = None, scheme = None, bins = None, norm = None,                   
                    classes = 7, legend = False, color_bar = False):
                          
    nrows =  nrows = int(len(columns)/2)
    multiPlot = MultiPlotGrid(fig_size = fig_size,nrows = nrows, ncols = 2, black_background = black_background,
                              title = title, cbar = color_bar)    
    
    fig, grid = multiPlot.fig, multiPlot.grid   

    for n, ax in enumerate(grid):
        if columns[n] is not None: gdf.sort_values(by = columns[n], ascending = True, inplace = True) 
        if (lw is None) & (scheme is None) & (norm is None): lw = 1.00
        elif lw is None: lw = [value*0.10 if value*0.10 > 1.1 else 1.1 for value in gdf[columns[n]]]

        ax.set_aspect("equal")
#         
        if axis_frame:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis= 'both', which= 'both', length=0)
            for spine in ax.spines: ax.spines[spine].set_color(multiPlot.text_color)
            if black_background: x.set_facecolor('black')
        else: ax.set_axis_off()
            
        ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = 30, color = multiPlot.text_color)
        
        # single color map
        if (columns[n] is None) & (scheme is None): gdf.plot(ax = ax, color = color, linewidth = lw)
        
        # boolean map
        elif (columns[n] is not None) & (scheme is None) & (norm is None): 
            if (cmap is None) & (classes == 2): 
                colors = ["white", "red"]
                gdf.plot(ax = ax, categorical = True, column = columns[n], color = colors, linewidth = lw, legend = legend,
                         capstyle = 'round', joinstyle = 'round')
            else:
                if cmap is None: cmap = rand_cmap(len(gdf[column].unique()))
                gdf.plot(ax = ax, categorical = True, column = columns[n], cmap=cmap, linewidth = lw, legend = legend,
                        capstyle = 'round', joinstyle = 'round')
            
        # user defined bins
        elif scheme == "User_Defined":
            gdf.plot(ax = ax, column = columns[n], cmap = cmap, linewidth = lw, scheme = scheme, legend = legend, 
                     classification_kwds={'bins':bins}, capstyle = 'round', joinstyle = 'round')
        
        # Lynch's bins - only for variables from 0 to 1 
        elif scheme == "Lynch_Breaks":  
            bins = [0.125, 0.25, 0.5, 0.75, 1.00]
            gdf.plot(ax = ax, column = columns[n], cmap = cmap, linewidth = lw, scheme = scheme, legend = legend,
                     classification_kwds={'bins':bins}, capstyle = 'round', joinstyle = 'round')
        
        # all other schemes        
        elif scheme is not None: 
            gdf.plot(ax = ax, column = columns[n], k = classes, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend)
        elif norm is not None:
            gdf.plot(ax = ax, column = columns[n], cmap = cmap, norm = norm, linewidth = lw, legend = False, capstyle = 'round',
                    joinstyle = 'round')
        
        if color_bar:
            if (n == nrows): _generate_color_bar(cmap, ax, multiPlot.text_color, multiPlot.font_size, norm = norm, series = gdf[columns[n]])
            elif (n == nrows-1): pass
            else: 
                try: ax.cax.remove()
                except: pass

        if legend:
            leg = ax.get_legend()
            leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
            
    return fig
    
 
    
def plot_multiplex(M, multiplex_edges):
    node_Xs = [float(node["x"]) for node in M.nodes.values()]
    node_Ys = [float(node["y"]) for node in M.nodes.values()]
    node_Zs = np.array([float(node["z"])*2000 for node in M.nodes.values()])
    node_size = []
    size = 1
    node_color = []

    for i, d in M.nodes(data=True):
        if d["station"] == True:
            node_size.append(9)
            node_color.append("#ec1a30")
        elif d["z"] == 1:
            node_size.append(0.0)
            node_color.append("#ffffcc")
        elif d["z"] == 0:
            node_size.append(8)
            node_color.append("#ff8566")

    lines = []
    line_width = []
    lwidth = 0.4
    
    # edges
    for u, v, data in M.edges(data=True):
        xs, ys = data["geometry"].xy
        zs = [M.node[u]["z"]*2000 for i in range(len(xs))]
        if data["layer"] == "intra_layer": zs = [0, 2000]
        
        lines.append([list(a) for a in zip(xs, ys, zs)])
        if data["layer"] == "intra_layer": line_width.append(0.2)
        elif data["pedestrian"] == 1: line_width.append(0.1)
        else: line_width.append(lwidth)

    fig_height = 40
    lc = Line3DCollection(lines, linewidths=line_width, alpha=1, color="#ffffff", zorder=1)

    west, south, east, north = multiplex_edges.total_bounds
    bbox_aspect_ratio = (north - south) / (east - west)*1.5
    fig_width = fig_height +90 / bb_aspect_ratio/1.5
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection="3d")
    ax.add_collection3d(lc)
    ax.scatter(node_Xs, node_Ys, node_Zs, s=node_size, c=node_color, zorder=2)
    ax.set_ylim(south, north)
    ax.set_xlim(west, east)
    ax.set_zlim(0, 2500)
    ax.axis("off")
    ax.margins(0)
    ax.tick_params(which="both", direction="in")
    fig.canvas.draw()
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    return(fig)
    
def _generate_legend(ax, black_background):

    leg = ax.get_legend()  
    leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    for text in leg.get_texts(): text.set_color("white")
    if not black_background:
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_alpha(1)
            
def _generate_color_bar(cmap, ax, text_color, font_size, norm = None, series = None):
    
    if font_size is None: font_size = 20
    if (norm is None) & (series is not None): norm = plt.Normalize(vmin = series.min(), vmax = series.max()) 
    elif (norm is None) & (series is None): return
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = ax.cax.colorbar(sm) 
    cbar.ax.set_yticklabels([t if t < norm.vmax else "> "+str(t) for t in cbar.ax.get_yticks()])
    plt.setp(plt.getp(ax.cax.axes, "yticklabels"), size = 2, color = text_color, fontfamily = 'Times New Roman', fontsize=(font_size))
    
def normalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]           


def random_colors_list_hsv(nlabels, vmin = 0.8, vmax = 1.0):
    randHSVcolors = [(np.random.uniform(low=0.0, high=0.95),
                      np.random.uniform(low=0.4, high=0.95),
                      np.random.uniform(low= vmin, high= vmax)) for i in range(nlabels)]

    return  randHSVcolors


def random_colors_list_rgb(nlabels, vmin = 0.8, vmax = 1.0):
    randHSVcolors = [(np.random.uniform(low=0.0, high=0.95),
                      np.random.uniform(low=0.4, high=0.95),
                       np.random.uniform(low= vmin, high= vmax)) for i in range(nlabels)]

    # Convert HSV list to RGB
    randRGBcolors = []
    for HSVcolor in randHSVcolors: randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
    return  randRGBcolors
    
            
# Generate random colormap
def rand_cmap(nlabels, type_color ='soft'):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :return: colormap for matplotlib
    """

    import numpy as np

    if type_color not in ('bright', 'soft'): type_color = 'bright'
    
    # Generate color map for bright colors, based on hsv
    if type_color == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=0.8),
                          np.random.uniform(low=0.2, high=0.8),
                          np.random.uniform(low=0.7, high=0.9)) for i in range(nlabels)]

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

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap
    
