import matplotlib as mp, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cols
import matplotlib.patches as mpatches

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pylab




pd.set_option('precision', 10)


"""
Series of utilies for plotting LineString, Points or Polygons geodataframes, and other operations.

"""

## Plotting
	
    
def plot_points(gdf, column, classes = 7, ms = 0.9, ms_col = None, title = 'Plot', scheme = 'fisher_jenks', cmap = 'Greys_r', 
                legend = False, bb = True, line_back = pd.DataFrame({'a' : []}), f = 10):
        
    """
    It creates a plot from a points GDF. 
    When column and scheme are not 'None' it plots the distribution over value and geographical space of variable 'column using scheme
    'scheme'. If only 'column' is provided, a categorical map is depicted.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string, column on which the plot is based
    classes: = 7, classes for visualising when scheme is not None and different from 'LynchBreaks'
    ms: float, markersize value (fixed)
    ms_col: String, column name in the GDF'S column where markersize values are stored
    title: string, title of the graph
    scheme: {'fisher_jenks', 'quantiles', 'equal_interval', 'LynchBreaks}
    cmap: String, see matplotlib colormaps for a list of possible values
    legend: boolean
    bb: boolean, black background or white
    f: float, size figure extent
    
    """
    # background black or white - basic settings
    if column != None: gdf.sort_values(by = column, ascending = True, inplace = True)    
    fig, ax = plt.subplots(1, figsize=(f, f))
    if scheme == "LynchBreaks": legend = True
    if bb == True: tcolor = 'white'
    else: tcolor = 'black'
    rect = fig.patch    
    if bb == True: rect.set_facecolor('black')
    else: rect.set_facecolor('white')
    fs = f+5 # font-size   
    fig.suptitle(title, color = tcolor, fontsize=fs)
    plt.axis('equal')
    ax.set_axis_off()
    
    if line_back.empty == False: line_back.plot(ax = ax, color = 'white', linewidth = 1.1, alpha = 0.3)
    
    # markers size from column
    if (ms_col != None): ms = gdf[ms_col]
    if scheme != None: gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, s = ms, scheme = scheme, legend = legend, alpha = 1)
    else: gdf.plot(ax = ax, linewidth = lw, color = 'black')
        
    if legend == True:
        leg = ax.get_legend()  
        leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
        leg.get_frame().set_linewidth(0.0) # remove legend border
        leg.set_zorder(102)
        if bb == True:
            for text in leg.get_texts(): text.set_color("white")   

    plt.show() 
    
    
def plot_lines(gdf, column = None, classes = 7, lw = 1.1, title = 'Plot', scheme = None, bins = None,  colors = None, cmap = 'Greys_r', 
               legend = False, cb = False, bb = True, f=10):
    """
    It creates a plot from a lineStrings GDF.
    When column and scheme are not 'None' it plots the distribution over value and geographical space of variable 'column using scheme
    'scheme'. If only 'column' is provided, a categorical map is depicted.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string, column on which the plot is based
    classes: = 7, classes for visualising when scheme is not None and different from 'LynchBreaks'
    lw: float, line width (fixed)
    ms_col: String, column name in the GDF'S column where markersize values are stored
    title: string, title of the graph
    scheme: {'fisher_jenks', 'quantiles', 'equal_interval', 'LynchBreaks}
    cmap: String, see matplotlib colormaps for a list of possible values
    legend: boolean
    cb: boolean, show color bar
    bb: boolean, black background or white
    f: float, size figure extent
    
    """
    
    # background black or white - basic settings    
    if column != None: gdf.sort_values(by = column, ascending = True, inplace = True)    
    fig, ax = plt.subplots(1, figsize=(f, f))
    if scheme == "LynchBreaks": legend = True
    if bb == True: tcolor = 'white'
    else: tcolor = 'black'
    rect = fig.patch  
    rect.set_zorder(0)
    if bb == True: rect.set_facecolor('black')
    else: rect.set_facecolor('white')
        
    fs = f+5 # font-size    
    fig.suptitle(title, color = tcolor, fontsize=fs)
    plt.axis('equal')
    ax.set_axis_off()

    # categorigal variable
    if (column != None) & (scheme == None):
        if cmap == None: # boolean map
            colors = ['white', 'red']
            gdf.plot(ax = ax, categorical = True, column = column, color = colors, linewidth = lw, legend = legend) 
        else: gdf.plot(ax = ax, column = column, cmap = cmap, linewidth = lw, legend = legend) # cateogircal map
    

    elif scheme == "pedestrians":
        bins = bins
        gdf['cat'] = 0
        text = []

        for en, b in enumerate(bins):
            if en == 0: 
                gdf['cat'][(gdf[column] >= 0) & (gdf[column] < bins[en])] = en
                text.append("0 - "+str(bins[en]))
                continue
            gdf['cat'][(gdf[column] > bins[en-1]) & (gdf[column] < b)] = en
            text.append(str(bins[en-1])+" - "+str(b))
            if en == len(bins)-1:
                gdf['cat'][(gdf[column] >= b)] = en+1
                text.append(">"+str(b))

        categories = [i for i in range(0,len(bins)+1,1)]
        if colors == None: colors = ['#000000', '#800026', '#e31a1c','#fc4e2a','#fd8d3c','#feb24c','#fed976','#ffffcc']
        colordict = dict(zip(categories, colors))
        gdf["Color"] = gdf['cat'].apply(lambda x: colordict[x])
        gdf.plot(ax = ax, categorical = True, color=gdf.Color,  linewidth = lw, legend=True)

        legend_dict = dict(zip(text, colors))
        patchList = []
        for key in legend_dict:
            data_key = mpatches.Patch(color=legend_dict[key], label=key)
            patchList.append(data_key)
        leg = plt.legend(handles=patchList, loc = 3)
        for text in leg.get_texts():
            plt.setp(text, color = 'w')  

    elif scheme != None:
        gdf.plot(ax = ax, column = column, k = classes, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend)
    
    else: gdf.plot(ax = ax, linewidth = lw, color = 'black') # plain map
        
    if legend == True:
        leg = ax.get_legend()  
        leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
        leg.get_frame().set_linewidth(0.0) # remove legend border
        leg.set_zorder(102)
        if bb == True:
            for text in leg.get_texts(): text.set_color("white")
                
    plt.show()

    
        
def plot_polygons(gdf, column = None, classes = 7, title = 'Plot', scheme = None, cmap = 'Greens_r', legend = False, 
                  cb = False, bb = True,  f = 10):
    """
    It creates a plot from a polygons GDF.
    When column and scheme are not 'None' it plots the distribution over value and geographical space of variable 'column using scheme
    'scheme'. If only 'column' is provided, a categorical map is depicted.
    
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string, column on which the plot is based
    classes: = 7, classes for visualising when scheme is not None and different from 'LynchBreaks'
    title: string, title of the graph
    scheme: {'fisher_jenks', 'quantiles', 'equal_interval', 'LynchBreaks}
    cmap: String, see matplotlib colormaps for a list of possible values
    legend: boolean
    cb: boolean, show color bar
    bb: boolean, black background or white
    f: float, size figure extent
    
    """
    
    # background black or white - basic settings   
    fig, ax = plt.subplots(1, figsize=(f, f))
    if scheme == "LynchBreaks": legend = True
    if bb == True: tcolor = 'white'
    else: tcolor = 'black'
    rect = fig.patch  
    rect.set_zorder(0)
    if bb == True: rect.set_facecolor('black')
    else: rect.set_facecolor('white')
    fs = f+5 # font-size    
    fig.suptitle(title, color = tcolor, fontsize=fs)
    plt.axis('equal')
    ax.set_axis_off()
    
    # categorigal variable
    if (column != None) & (scheme == None): gdf.plot(ax = ax, column = column, cmap = cmap, categorical = True, legend = legend)       
      
    # Other schemes
    elif scheme != None:
        if cb == True: legend = False # if colorbar is activated, don't show legend
        gdf.plot(ax = ax, column = column, k = classes, cmap = cmap,  scheme = scheme, legend = legend)
        
    else: gdf.plot(ax = ax, color = 'orange')  # plain map
    
    if legend == True:
        leg = ax.get_legend()  
        leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
        leg.get_frame().set_linewidth(0.0) # remove legend border
        leg.set_zorder(102)
        if bb == True:
            for text in leg.get_texts(): text.set_color("white")
                
    if cb == True:
        # create color bar
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = gdf[column].min(), vmax = gdf[column].max()))
        fig = ax.get_figure()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, ticks=[gdf[column].min(), gdf[column].max()]) 
        cbar.ax.set_yticklabels(['min', 'max'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=tcolor, fontsize=(fs-5))   
       
    plt.show()    
    
    
def multi_plot_polygons(list_gdf, list_sub_titles, main_title, column = None, classes = 7, scheme = None, 
                        cmap = 'Greens_r', legend = False, cb = False, bb = True):
    """
    It creates a plot from a polygons GDF.
    When column and scheme are not 'None' it plots the distribution over value and geographical space of variable 'column using scheme
    'scheme'. If only 'column' is provided, a categorical map is depicted.
    
    
    Parameters
    ----------
    gdf: GeoDataFrame
    column: string, column on which the plot is based
    classes: = 7, classes for visualising when scheme is not None and different from 'LynchBreaks'
    title: string, title of the graph
    scheme: {'fisher_jenks', 'quantiles', 'equal_interval', 'LynchBreaks}
    cmap: String, see matplotlib colormaps for a list of possible values
    legend: boolean
    cb: boolean, show color bar
    bb: boolean, black background or white
    f: float, size figure extent
    
    """

    if len(list_gdf) == 1: nrows, ncols = 1, 1
    elif len(list_gdf) == 2: nrows, ncols = 1, 2
    else: 
        ncols = 3
        nrows = int(len(list_gdf)/ncols)
        if (len(list_gdf)%ncols != 0): nrows = int(nrows)+1;
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (15, 5*nrows))
    plt.axis('equal')
 
    if scheme == "LynchBreaks": legend = True
        
    # background settings (black vs white)
    if bb == True: tcolor = 'white'
    else: tcolor = 'black'
    rect = fig.patch  
    rect.set_zorder(0)
    if bb == True: rect.set_facecolor('black')
    else: rect.set_facecolor('white')
    fig.suptitle(main_title, color = tcolor, fontsize= 20)
    #     fs = f+5 # font-size    
    
    if nrows > 1: axes = [item for sublist in axes for item in sublist]
    for n, ax in enumerate(axes):
        ax.set_axis_off()
        try: gdf = list_gdf[n]
        except: continue
            
        ax.set_title(list_sub_titles[n], color = tcolor, fontsize= 18)
        # categorigal variable
        if (column != None) & (scheme == None): gdf.plot(ax = ax, column = column, cmap = cmap, categorical = True, legend = legend)       

        # Other schemes
        elif scheme != None:
            if cb == True: legend = False # if colorbar is activated, don't show legend
            gdf.plot(ax = ax, column = column, k = classes, cmap = cmap,  scheme = scheme, legend = legend)

        else: gdf.plot(ax = ax, color = 'orange')  # plain map

#     if legend == True:
#         leg = ax[n].get_legend()  
#         leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
#         leg.get_frame().set_linewidth(0.0) # remove legend border
#         leg.set_zorder(102)
#         if bb == True:
#             for text in leg.get_texts(): text.set_color("white") 
    
    plt.subplots_adjust(top = 0.88, hspace= 0.025)
    plt.show()  
    
def plot_lines_aside(gdf, gdf_c = None, classes = 7, lw = 0.9, columnA = None, columnB = None, title = 'Plot', 
                     scheme = 'fisher_jenks', bins = None, colors = None, cmap = 'Greys_r', legend = False, bb = True, f = 15):
    
    # background black or white - basic settings 
    
    if columnA != None: gdf.sort_values(by = columnA, ascending = True, inplace = True)    
    
    if bb == True: fcolor = 'black'
    else: fcolor = 'white'
    print(fcolor)
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(15, 8), facecolor = fcolor)
    if bb == True: tcolor = 'white'
    else: tcolor = 'black'
    rect = fig.patch    
    if bb == True: rect.set_facecolor('black')
    else: rect.set_facecolor('white')  
    fig.suptitle(title, color = tcolor)
    
    plt.axis('equal')
    ax2.set_axis_off()
    ax1.set_axis_off()
    col = [columnA, columnB]
    
    for n, i in enumerate([ax1, ax2]):
        
        if (n == 1) & (columnB != None): gdf.sort_values(by = columnB, ascending = True, inplace = True)  
        i.set_aspect('equal')
        
        if (col[n] != None) & (scheme == None): 
            colors = ['white', 'red']
            gdf.plot(ax = i, categorical = True, column = col[n], color = colors, linewidth = lw, legend = legend) 
#             gdf.plot(ax = i, linewidth = lw, color = 'black') # categorical map
              
        elif scheme == "pedestrians":
                       
            gdf['cat'] = 0
            text = []
            
            for en, b in enumerate(bins):
                if en == 0: 
                    gdf['cat'][(gdf[col[n]] >= 0) & (gdf[col[n]] < bins[en])] = en
                    text.append("0 - "+str(bins[en]))
                    continue
                gdf['cat'][(gdf[col[n]] > bins[en-1]) & (gdf[col[n]] < b)] = en
                text.append(str(bins[en-1])+" - "+str(b))
                if en == len(bins)-1:
                    gdf['cat'][(gdf[col[n]] >= b)] = en+1
                    text.append(">"+str(b))
            
            categories = [i for i in range(0,len(bins)+1,1)]
            if colors == None: colors = ['#000000', '#800026', '#e31a1c','#fc4e2a','#fd8d3c','#feb24c','#fed976','#ffffcc']
            colordict = dict(zip(categories, colors))
            gdf["Color"] = gdf['cat'].apply(lambda x: colordict[x])
            gdf.plot(ax = i, categorical = True, color=gdf.Color,  linewidth = lw, legend=True)
           
            legend_dict = dict(zip(text, colors))
            patchList = []
            for key in legend_dict:
                data_key = mpatches.Patch(color=legend_dict[key], label=key)
                patchList.append(data_key)
            leg = plt.legend(handles=patchList, loc = 3)
            for text in leg.get_texts():
                plt.setp(text, color = 'w')
   
 
        # other schemes
        elif scheme != None: 

            gdf.plot(ax = i, column = col[n], k = classes, cmap = cmap, linewidth = lw, scheme = scheme, legend = legend)
            leg = i.get_legend()
            leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
            
            
        
        else: gdf.plot(ax = i, linewidth = lw, color = 'black')  # plain map     
    
#         if legend == True:
#             i.legend(loc=1)
#             leg = i.get_legend()
#             
#             leg.set_zorder(102)
#             leg.get_frame().set_linewidth(0.0) # remove legend border
#             if bb == True:
#                 for text in leg.get_texts(): text.set_color("white")

    plt.show()
    
    
def plot_multiplex(M, multiplex_edges):
    node_Xs = [float(node['x']) for node in M.nodes.values()]
    node_Ys = [float(node['y']) for node in M.nodes.values()]
    node_Zs = np.array([float(node['z'])*2000 for node in M.node.values()])
    node_size = []
    size = 1
    node_color = []

    for i, d in M.nodes(data=True):
        if d['station'] == True:
            node_size.append(9)
            node_color.append('#ec1a30')
        elif d['z'] == 1:
            node_size.append(0.0)
            node_color.append('#ffffcc')
        elif d['z'] == 0:
            node_size.append(8)
            node_color.append('#ff8566')

    lines = []
    line_width = []
    lwidth = 0.4
    
    # edges
    for u, v, data in M.edges(data=True):
        
        xs, ys = data['geometry'].xy
        zs = [M.node[u]['z']*2000 for i in range(len(xs))]
        if data['layer'] == 'intra_layer': zs = [0, 2000]
        
        lines.append([list(a) for a in zip(xs, ys, zs)])
        if data['layer'] == 'intra_layer': line_width.append(0.2)
        elif data['pedestrian'] == 1: line_width.append(0.1)
        else: line_width.append(lwidth)

    fig_height = 40
    lc = Line3DCollection(lines, linewidths=line_width, alpha=1, color="#ffffff", zorder=1)

    west, south, east, north = multiplex_edges.total_bounds
    bbox_aspect_ratio = (north - south) / (east - west)*1.5
    fig_width = fig_height +90 / bbox_aspect_ratio/1.5
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    ax.add_collection3d(lc)
    ax.scatter(node_Xs, node_Ys, node_Zs, s=node_size, c=node_color, zorder=2)
    ax.set_ylim(south, north)
    ax.set_xlim(west, east)
    ax.set_zlim(0, 2500)
    ax.axis('off')
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    fig.canvas.draw()
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    return(fig)
            
        


            
            
            
            


    
