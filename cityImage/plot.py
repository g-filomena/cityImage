import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cols
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, DivergingNorm
import mapclassify

import pylab
import colorsys

pd.set_option("precision", 10)

from .utilities import scaling_columnDF
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
        font_size = fig_size*2+5 # font-size
        fig.suptitle(title, color = text_color, fontsize=font_size, fontfamily = 'Times New Roman')
        
        plt.axis("equal")
        self.fig, self.ax = fig, ax
        self.font_size, self.text_color = font_size, text_color
        
class MultiPlotGrid():
    
    def __init__(self, fig_size, nrows, ncols, black_background):
        
        figsize = (fig_size, fig_size*nrows)
        if (nrows == 1) & (ncols == 2): 
            figsize = (fig_size, fig_size/2)
            
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows,ncols), axes_pad= (0.50, 1.00))
        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        
        font_size = fig_size+5 # font-size   
        self.fig, self.grid = fig, grid
        self.font_size, self.text_color = font_size, text_color
        
class MultiPlot():
    
    
    def __init__(self, fig_size, nrows, ncols, black_background, title = None):
    
        figsize = (fig_size, fig_size/2*nrows)          
        fig, grid = plt.subplots(nrows = nrows, ncols = ncols, figsize=figsize)

        rect = fig.patch 
        if black_background: 
            text_color = "white"
            rect.set_facecolor("black")
        else: 
            text_color = "black"
            rect.set_facecolor("white")
        
        font_size = fig_size*2+5
        if title is not None:
            fig.suptitle(title, color = text_color, fontsize = font_size, fontfamily = 'Times New Roman', 
                         ha = 'center', va = 'center') 
            fig.subplots_adjust(top=0.92)
        
        plt.axis("equal")    
        self.fig, self.grid = fig, grid
        self.font_size, self.text_color = font_size, text_color

def single_plot(ax, gdf, column = None, scheme = None, bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, 
                legend = False, axis_frame = False, ms = None, ms_factor = None, lw = None, lw_factor = None,  zorder = 0):
    
    gdf = gdf.copy()
    categorical = True
    if alpha is None:
        alpha = 1
    if column is not None: 
        gdf.sort_values(by = column,  ascending = True, inplace = True) 
    
    # single-colour map
    if (column is None) & (scheme is None) & (color is None):
        color = 'red'
    # categorical map
    elif (column is not None) & (scheme is None) & (norm is None) & (cmap is None): 
        cmap = rand_cmap(len(gdf[column].unique()))         
    # Lynch's bins - only for variables from 0 to 1 
    elif scheme == "Lynch_Breaks":  
        scaling_columnDF(gdf, column)
        column = column+"_sc"
        bins = [0.125, 0.25, 0.5, 0.75, 1.00]
        scheme = 'User_Defined'
        categorical = False
    elif norm is not None:
        legend = False
        categorical = False
        scheme = None
    elif (scheme is not None) & (classes is None) & (bins is None):
        classes = 7   
    if (scheme is not None) & (cmap is None) :
        cmap = kindlmann()
    if (scheme is not None) | (norm is not None):
        categorical = False
        color = None
    
    if (column is not None) & (not categorical):
        if (gdf[column].dtype == 'O'):
            gdf[column] = gdf[column].astype(float)
    
    if bins is None: 
        c_k = {None}
        if classes is not None:
            c_k = {"k" : classes}
    else: 
        c_k = {'bins':bins, "k" : len(bins)}
        scheme = 'User_Defined'
    
    if gdf.iloc[0].geometry.geom_type == 'Point':
        if (ms_factor is not None): 
            # rescale
            scaling_columnDF(gdf, column)
            gdf['ms'] = np.where(gdf[column+'_sc'] >= 0.20, gdf[column+'_sc']*ms_factor, 0.40) # marker size
            ms = gdf['ms']
        elif ms is None:
            ms = 1.0
        else: ms = ms

        gdf.plot(ax = ax, column = column, markersize = ms, categorical = categorical, color = color, scheme = scheme, cmap = cmap, norm = norm, alpha = alpha,
            legend = legend, classification_kwds = c_k, zorder = zorder) 
        
    if gdf.iloc[0].geometry.geom_type == 'LineString':
        if (lw is None) & (lw_factor is None): 
            lw = 1.00
        elif lw_factor is not None:
            lw = [value*lw_factor if value*lw_factor> 1.1 else 1.1 for value in gdf[column]]
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, linewidth = lw, scheme = scheme, alpha = alpha, cmap = cmap, norm = norm,
            legend = legend, classification_kwds = c_k, capstyle = 'round', joinstyle = 'round', zorder = zorder) 
                
    if gdf.iloc[0].geometry.geom_type == 'Polygon': 
        gdf.plot(ax = ax, column = column, categorical = categorical, color = color, scheme = scheme, edgecolor = 'none', alpha = alpha, cmap = cmap,
            norm = norm, legend = legend, classification_kwds = c_k, zorder = zorder) 
        
 
def plot_gdf(gdf, column = None, title = None, black_background = True, fig_size = 15, scheme = None, bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, 
                legend = False, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, only_min_max = False, axis_frame = False, ms = None, ms_factor = None, lw = None, lw_factor = None, gdf_base_map = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4,
                base_map_lw = 1.1, base_map_ms = 2.0, base_map_zorder = 0):
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
    
    ax.set_aspect("equal")
    if axis_frame: 
        set_axis_frame(ax, black_background, plot.text_color)
    else: ax.set_axis_off()     
    zorder = 0
    # base map (e.g. street network)
    if (not gdf_base_map.empty):
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha, zorder = base_map_zorder)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha, zorder = base_map_zorder)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha, zorder = base_map_zorder)
        if base_map_zorder == 0:
            zorder = 1
   
    single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, 
                axis_frame = axis_frame, ms = ms, ms_factor = ms_factor, lw = lw, lw_factor = lw_factor, zorder = zorder, legend = legend)

    if legend: 
        _generate_legend_ax(ax, plot.font_size-5, black_background) 
        
    if (cbar) & (not legend):
        if norm is None:
            min_value = gdf[column].min()
            max_value = gdf[column].max()
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
            
        generate_row_colorbar(cmap, fig, ax, ncols = 1, text_color = plot.text_color, font_size = plot.font_size, norm = norm, 
                             ticks = cbar_ticks,symbol = cbar_max_symbol, only_min_max = only_min_max)
    
    plt.show()    
                
def plot_barriers(barriers_gdf, lw = 1.1, title = "Plot", legend = False, axis_frame = False, black_background = True,                 
               fig_size = 15, gdf_base_map = pd.DataFrame({"a" : []}), base_map_color = None, base_map_alpha = 0.4,
               base_map_lw = 1.1, base_map_ms = 2.0, base_map_zorder = 0):
    
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
    
    ax.set_aspect("equal")
    if axis_frame: 
        set_axis_frame(ax, black_background, plot.text_color)
    else: ax.set_axis_off()     
    
    zorder = 0
    # background (e.g. street network)
    if (not gdf_base_map.empty):
        if gdf_base_map.iloc[0].geometry.geom_type == 'LineString':
            gdf_base_map.plot(ax = ax, color = base_map_color, linewidth = base_map_lw, alpha = base_map_alpha,zorder = base_map_zorder)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Point':
            gdf_base_map.plot(ax = ax, color = base_map_color, markersize = base_map_ms, alpha = base_map_alpha, zorder = base_map_zorder)
        if gdf_base_map.iloc[0].geometry.geom_type == 'Polygon':
            gdf_base_map.plot(ax = ax, color = base_map_color, alpha = base_map_alpha, zorder = base_map_zorder)
        if base_map_zorder == 0:
            zorder = 1
    
    barriers_gdf['barrier_type'] = barriers_gdf['type']
    barriers_gdf.sort_values(by = 'barrier_type', ascending = False, inplace = True)  
    
    colors = ['green', 'brown', 'grey', 'blue']
    colormap = LinearSegmentedColormap.from_list('new_map', colors, N=4)
    barriers_gdf.plot(ax = ax, categorical = True, column = 'barrier_type', cmap = colormap, linewidth = lw, legend = legend, 
                     label =  'barrier_type', zorder = zorder )             
                     
    if legend: 
        _generate_legend_ax(ax, plot.font_size-10, black_background)
    
    plt.show()  
    
def plot_gdfs(list_gdfs = None, column = None, main_title = None, titles = None, black_background = True, fig_size = 15, scheme = None, bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, 
                legend = False, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, only_min_max = False, axis_frame = False, ms = None, ms_factor = None, lw = None, lw_factor = None): 
                     
    """
    It creates of subplots from a list of polygons GeoDataFrames
    When column and scheme are not "None" it plots the distribution over value and geographical space of variable "column" using scheme
    "scheme". If only "column" is provided, a categorical map is generated.
    
    Parameters
    ----------
    list_gdfs: list of GeoDataFrames

    columns: string
        Column on which the plot is based
        list_titles: list of str
        subplots'titles    
    
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
                     
    nrows, ncols = int(len(list_gdfs)/2), 2
    if (len(list_gdfs)%2 != 0): 
        nrows = nrows+1
     
    multiPlot = MultiPlot(fig_size = fig_size, nrows = nrows, ncols = ncols, black_background = black_background, title = main_title)
    fig, grid = multiPlot.fig, multiPlot.grid   
    legend_fig = False
    
    if nrows > 1: 
        grid = [item for sublist in grid for item in sublist]
    for n, ax in enumerate(grid):
                
        ax.set_aspect("equal")
        if axis_frame: 
            set_axis_frame(ax, black_background, multiPlot.text_color)
        else: ax.set_axis_off()      

        if n > len(list_gdfs)-1: 
            continue # when odd nr of gdfs    
        
        gdf = list_gdfs[n]
        if titles is not None:
            ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size, color = multiPlot.text_color,  pad = 15)
            
        if (n == ncols*nrows/2) & legend & ((scheme == 'User_Defined') | (scheme == 'Lynch_Breaks')):
            legend_ax = True
            legend_fig = True
        elif legend & ((scheme != 'User_Defined') & (scheme != 'Lynch_Breaks')):
            legend_ax = True
        else: 
            legend_ax = False
            legend_fig = False
        
        single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, legend = legend_ax, 
                    axis_frame = axis_frame, ms = ms, ms_factor = ms_factor, lw = lw, lw_factor = lw_factor)
                    
        if legend_fig:
            _generate_legend_fig(ax, nrows, multiPlot.text_color, (multiPlot.font_size-5), black_background)
        elif legend_ax:
            _generate_legend_ax(ax, (multiPlot.font_size-15), black_background)
    
    if (cbar) & (not legend):
        if norm is None:
            min_value = min([gdf[column].min() for gdf in list_gdfs])
            max_value = max([gdf[column].max() for gdf in list_gdfs])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
        generate_grid_colorbar(cmap, fig, grid, nrows, ncols, multiPlot.text_color,(multiPlot.font_size-5), norm = norm, ticks = cbar_ticks, 
                              symbol = cbar_max_symbol, only_min_max = only_min_max )
            
    return fig
   
def plot_gdf_grid(gdf = None, columns = None, titles = None, black_background = True, fig_size = 15, scheme = None, bins = None, classes = None, norm = None, cmap = None, color = None, alpha = None, 
                legend = False, cbar = False, cbar_ticks = 5, cbar_max_symbol = False, only_min_max = False, axis_frame = False, ms = None, ms_factor = None, lw = None, lw_factor = None): 
       
    nrows, ncols = int(len(columns)/2), 2
    if (len(columns)%2 != 0): 
        nrows = nrows+1
     
    multiPlot = MultiPlotGrid(fig_size = fig_size, nrows = nrows, ncols = ncols, black_background = black_background)
    fig, grid = multiPlot.fig, multiPlot.grid   
    legend_fig = False
    
    for n, ax in enumerate(grid):
        
        ax.set_aspect("equal")
        if axis_frame: 
            set_axis_frame(ax, black_background, multiPlot.text_color)
        else: ax.set_axis_off()
        
        if n > len(columns)-1: 
            continue # when odd nr of columns
        
        column = columns[n]
        if titles is not None:          
            ax.set_title(titles[n], loc='center', fontfamily = 'Times New Roman', fontsize = multiPlot.font_size, color = multiPlot.text_color,  pad = 15)
        
        if (n == ncols*nrows/2) & legend & ((scheme == 'User_Defined') | (scheme == 'Lynch_Breaks')):
            legend_ax = True
            legend_fig = True
        elif legend & ((scheme != 'User_Defined') & (scheme != 'Lynch_Breaks')):
            legend_ax = True
        else: 
            legend_ax = False
            legend_fig = False
        
        single_plot(ax, gdf, column = column, scheme = scheme, bins = bins, classes = classes, norm = norm, cmap = cmap, color = color, alpha = alpha, legend = legend_ax,
                    axis_frame = axis_frame, ms = ms, ms_factor = ms_factor, lw = lw, lw_factor = lw_factor)
                            
        if legend_fig:
            _generate_legend_fig(ax, nrows, multiPlot.text_color, multiPlot.font_size-5, black_background)
        elif legend_ax:
            _generate_legend_ax(ax, (multiPlot.font_size-5), black_background)

    if (cbar) & (not legend):
        if norm is None:
            min_value = min([gdf[column].min() for column in columns])
            max_value = max([gdf[column].max() for column in columns])
            norm = plt.Normalize(vmin = min_value, vmax = max_value)
        generate_grid_colorbar(cmap, fig, grid, nrows, ncols, multiPlot.text_color,multiPlot.font_size-5, norm = norm, ticks = cbar_ticks, 
                              symbol = cbar_max_symbol, only_min_max = only_min_max)

            
    return fig
    
def plot_multiplex(M, multiplex_edges):
    node_Xs = [float(node["x"]) for node in M.nodes.values()]
    node_Ys = [float(node["y"]) for node in M.nodes.values()]
    node_Zs = np.array([float(node["z"])*2000 for node in M.nodes.values()])
    node_size = []
    size = 1
    node_color = []

    for i, d in M.nodes(data=True):
        if d["station"]:
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
        if data["layer"] == "intra_layer": 
            zs = [0, 2000]
        
        lines.append([list(a) for a in zip(xs, ys, zs)])
        if data["layer"] == "intra_layer": 
            line_width.append(0.2)
        elif data["pedestrian"] == 1: 
            line_width.append(0.1)
        else: line_width.append(lwidth)

    fig_height = 40
    lc = Line3DCollection(lines, linewidths=line_width, alpha=1, color="#ffffff", zorder=1)

    west, south, east, north = multiplex_edges.total_bounds
    bbox_aspect_ratio = (north - south) / (east - west)*1.5
    fig_width = fig_height +90 / bbox_aspect_ratio/1.5
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
    
def _generate_legend_fig(ax, nrows, text_color, font_size, black_background):
    
    leg = ax.get_legend() 
    plt.setp(leg.texts, family='Times New Roman', fontsize = font_size, color = text_color, va = 'center')
    if nrows%2 == 0: 
        leg.set_bbox_to_anchor((2.15, 1.00, 0.33, 0.33))    
    else: leg.set_bbox_to_anchor((1.15, 0.5, 0.33, 0.33))
    
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    leg.get_frame().set_facecolor('none')
    
    for handle in leg.legendHandles:
        handle._legmarker.set_markersize(15)

def _generate_legend_ax(ax, font_size, black_background):

    leg = ax.get_legend()  
    if black_background:
        text_color = 'black'
    else: text_color = 'white'
    
    plt.setp(leg.texts, family='Times New Roman', fontsize = font_size, color = text_color, va = 'center')
    leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
    leg.get_frame().set_linewidth(0.0) # remove legend border
    leg.set_zorder(102)
    
    for handle in leg.legendHandles:
        handle._legmarker.set_markersize(12)
    if not black_background:
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_alpha(0.90)  
    else:
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.90)  
 
def generate_grid_colorbar(cmap, fig, grid, nrows, ncols, text_color, font_size, norm = None, ticks = 5, symbol = False, only_min_max = False):
    
    if font_size is None: 
        font_size = 20
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    vr_p = 1/30.30
    hr_p = 0.5/30.30
    ax = grid[0]

    if ncols == 2:
        width = ax.get_position().x1*ncols-hr_p-ax.get_position().x0
    elif ncols > 2:
        width = ax.get_position().x1*(ncols-1)-hr_p*ncols
   
    if nrows == 1: 
        pos = [ax.get_position().x0+width, ax.get_position().y0, 0.027, ax.get_position().height]
    elif nrows%2 == 0:
        y0 = (ax.get_position().y0-(ax.get_position().height*(nrows-1))-vr_p)+(nrows/2-0.5)*ax.get_position().height
        pos = [ax.get_position().x0+width, y0, 0.027, ax.get_position().height]
    else:
        ax = grid[nrows-1]
        pos = [ax.get_position().x0+width, ax.get_position().y0, 0.027, ax.get_position().height]

    _set_colorbar(fig, pos, sm, ticks, norm, symbol, text_color, font_size, only_min_max)    
    
def generate_row_colorbar(cmap, fig, ax, ncols, text_color, font_size, norm = None, ticks = 5, symbol = False, only_min_max = False):
    
    if font_size is None: 
        font_size = 20
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    vr_p = 1/30.30
    hr_p = 0.5/30.30
    
    width = ax.get_position().x1
    if ncols == 2:
        width = ax.get_position().x1*ncols-hr_p-ax.get_position().x0
    elif ncols > 2:
        width = ax.get_position().x1*(ncols-1)-hr_p*ncols
    pos = [ax.get_position().x0+width, ax.get_position().y0, 0.05, ax.get_position().height]
    
    _set_colorbar(fig, pos, sm, ticks, norm, symbol, text_color, font_size, only_min_max)    
    
    
def _set_colorbar(fig, pos, sm, ticks, norm, symbol, text_color, font_size, only_min_max = False):
    cax = fig.add_axes(pos, frameon = False)
    cax.tick_params(size=0)
    cb = plt.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    tick_locator = ticker.MaxNLocator(nbins=ticks)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.outline.set_visible(False)
    
    ticks = list(cax.get_yticks())
    for t in ticks: 
        if (t == ticks[-1]) & (t != norm.vmax) :
            ticks[-1] = norm.vmax

    if only_min_max:
        ticks = [norm.vmin, norm.vmax]
    cb.set_ticks(ticks)
    
    if symbol:
        cax.set_yticklabels([round(t,1) if t < norm.vmax else "> "+str(round(t,1)) for t in cax.get_yticks()])
    else: cax.set_yticklabels([round(t,1) for t in cax.get_yticks()])
    
    plt.setp(plt.getp(cax.axes, "yticklabels"), size = 0, color = text_color, fontfamily = 'Times New Roman', fontsize=font_size)
             
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
    for HSVcolor in randHSVcolors: 
        randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
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

    kindlmann_list = [(0.00, 0.00, 0.00,1), (0.248, 0.0271, 0.569, 1), (0.0311, 0.258, 0.646,1),
            (0.019, 0.415, 0.415,1), (0.025, 0.538, 0.269,1), (0.0315, 0.658, 0.103,1),
            (0.331, 0.761, 0.036,1),(0.768, 0.809, 0.039,1), (0.989, 0.862, 0.772,1),
            (1.0, 1.0, 1.0)]
    return LinearSegmentedColormap.from_list('kindlmann', kindlmann_list)
    
def set_axis_frame(ax, black_background, text_color):
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis= 'both', which= 'both', length=0)
    for spine in ax.spines: 
        ax.spines[spine].set_color(text_color)
    if black_background: 
        ax.set_facecolor('black')
    
def cmap_two_colors(from_rgb,to_rgb):
    
    from_rgb = cols.to_rgb(from_rgb)
    to_rgb = cols.to_rgb(to_rgb) 
        
    # from color r,g,b
    r1,g1,b1 = from_rgb
    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1), (1, r2, r2)),
           'green': ((0, g1, g1), (1, g2, g2)),
           'blue': ((0, b1, b1), (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap
    
def cmap_three_colors(col1, col2, col3):

    list_colors = [col1, col2, col3]

    return LinearSegmentedColormap.from_list('red_to green', list_colors)