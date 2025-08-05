import networkx as nx
import pandas as pd
import numpy as np

from shapely.geometry import Point, LineString
from shapely.ops import unary_union
pd.set_option("display.precision", 3)

from .graph import graph_fromGDF
from .graph_load import obtain_nodes_gdf, join_nodes_edges_by_coordinates
from .utilities import split_line_at_MultiPoint 

def fix_network_topology(nodes_gdf, edges_gdf, edgeID_column = 'edgeID'):
    """
    Fix the network topology by splitting intersecting edges and adding fixed edges to the network.
    This function considers as segments to be fixed only segments that are actually fully intersecting, thus sharing coordinates, excluding their 
    from and to vertices coordinates, but withouth actually generating, in the given GeoDataFrame, the right number of features.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.
    
    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    edges_gdf.copy()
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # spatial index
    sindex = edges_gdf.sindex 

    def find_intersections(ix, line_geometry, coords):
        """
        Find intersection points between a line and other intersecting edges.

        Parameters
        ----------
        ix : int
            The index of the line to check for intersections.
        line_geometry : LineString
            The LineString geometry of the line to check for intersections.
        coords : list
            The list of coordinates of the line's geometry.

        Returns
        -------
        list
            A list of actual intersection points between the line and other intersecting edges.
        """
        
        possible_matches_index = list(sindex.intersection(line_geometry.buffer(5).bounds))
        possible_matches = edges_gdf.iloc[possible_matches_index].copy()
        # lines intersecting the given line
        tmp = possible_matches[possible_matches.intersects(line_geometry)]
        tmp = tmp.drop(ix, axis = 0)
        union = tmp.union_all()
        
        # find actual intersections
        actual_intersections = []
        intersections = line_geometry.intersection(union)
        if intersections is None:
            return actual_intersections      
        if intersections.geom_type == 'LineString': 
            # probably overlapping (to resolve)
            return actual_intersections     
        
        if intersections.geom_type == 'Point': 
            intersections = [intersections]
        else:
            intersections = intersections.geoms
        
        # from and to vertices of the given line
        segment_vertices = [coords[0], coords[-1]]
        # obtaining all the intersecting Points
        intersection_points = [intersection for intersection in intersections if intersection.geom_type == 'Point']
        
        # keeping intersections that are in the coordinate list of the given line, without actually coinciding with the from and to vertices
        for point in intersection_points: 
            if (point.coords[0] not in coords):
                pass
            if (point.coords[0] in segment_vertices): 
                pass 
            else: 
                actual_intersections.append(point)
        return actual_intersections
    
    # verify which street segment needs to be fixed
    edges_gdf['to_fix'] = edges_gdf.apply(lambda row: find_intersections(row.name, row.geometry, row.coords), axis=1)
    # verify which street segment needs to be fixed
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]   
    if len(to_fix) == 0:
        return edges_gdf    
    return _add_fixed_edges(edges_gdf, to_fix, edgeID_column)
    
def fix_fake_self_loops(nodes_gdf, edges_gdf, edgeID_column = 'edgeID'):
    """
    Fix the network topology by removing (fake) self-loops and adding fixed edges.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.

    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    
    edges_gdf = edges_gdf.copy()
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # all the coordinates but the from and to vertices' ones.
    edges_gdf['coords'] = [coords[1:-1] for coords in edges_gdf.coords]
    
    # convert nodes_gdf['x'] and nodes_gdf['y'] to numpy arrays for faster computation
    x = list(nodes_gdf['x'])
    y = list(nodes_gdf['y'])
    # create a set of all coordinates in nodes. This essentially correspond to the from and to nodes of the edges currently in the edges_gdf
    nodes_set = set(zip(x, y))

    to_fix = []
    # loop through the coordinates in edges_gdf.coords and check if they are in the nodes_set. This means that one of the edges coords (not from and to),
    # coincide with some other edge from or to vertex (indicating some sort of loop) 
    for coords in edges_gdf.coords:
        fix_coords = []
        for coord in coords:
            if coord in nodes_set:
                fix_coords.append(coord)
        to_fix.append(fix_coords)

    # assign the results to self_loops['to_fix']
    edges_gdf['to_fix'] = to_fix
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]
    if len(to_fix) == 0:
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix, edgeID_column)    
 
def _add_fixed_edges(edges_gdf, to_fix_gdf, edgeID_column = 'edgeID'):
    """
    Add fixed edges to the edges GeoDataFrame.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    to_fix_gdf: GeoDataFrame
        The GeoDataFrame containing the edges to be fixed.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    dfs = []
    new_geometries = to_fix_gdf.apply(lambda row: split_line_at_MultiPoint(row.geometry, [Point(coord) for coord in row.to_fix]), axis=1)
    new_geometries = pd.DataFrame(new_geometries, columns = ['lines'])
    
    def append_new_geometries(row):
        for n, line in enumerate(row): # assigning the resulting geometries
            ix = row.name
            if n == 0: 
                index = ix
            else: 
                index = max(edges_gdf.index)+1

            # copy attributes
            row = to_fix_gdf.loc[ix].copy()
            # and assign geometry an new edgeID 
            row[edgeID_column] = index 
            row['geometry'] = line 
            dfs.append(row.to_frame().T)

    new_geometries.apply(lambda row: append_new_geometries(row), axis = 1)
    rows = pd.concat(dfs, ignore_index = True)
    rows = rows.explode(column = 'geometry')
    
    # concatenate the dataframes and assign to edges_gdf
    edges_gdf = pd.concat([edges_gdf, rows], ignore_index=True)
    edges_gdf.drop(['u', 'v', 'to_fix', 'fixing', 'coords'], inplace=True, axis=1)
    edges_gdf['length'] = edges_gdf.geometry.length
    edges_gdf[edgeID_column] = edges_gdf.index
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    
    return nodes_gdf, edges_gdf
 
def remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID_column = 'nodeID'):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrame.
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column)
    if not nx.is_connected(Ng):  
        largest_component = max(nx.connected_components(Ng), key=len)
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        to_keep = list(G.nodes())
        nodes_gdf = nodes_gdf[nodes_gdf[nodeID_column].isin(to_keep)]
        edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf[nodeID_column])) & (edges_gdf.v.isin(nodes_gdf[nodeID_column]))]
        
    return nodes_gdf, edges_gdf