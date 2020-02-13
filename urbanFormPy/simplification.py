import pandas as pd, numpy as np, geopandas as gpd
import math
from math import sqrt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points, polygonize_full
import warnings
pd.set_option('precision', 10)
warnings.simplefilter(action='ignore', category=FutureWarning)

import statistics
import ast
from .graph import*
from .utilities import *
from .cleaning_network import *
from .angles import *

def is_perpendicular(line_geometry_A, line_geometry_B):
    
    angle = angle_line_geometries(line_geometry_A, line_geometry_B, degree = True)
    if ((angle >= 70) & (angle <= 110)): 
        if is_parallel(line_geometry_A, line_geometry_B, hard = False): return False 
        else: return True
    else: return False

def is_parallel(line_geometry_A, line_geometry_B, hard = False):
    
    difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
    if (difference_angle <= 20) & (difference_angle >= -20): return True
        
    line_coordsA = list(line_geometry_A.coords)
    line_coordsB = list(line_geometry_B.coords)
    if ((len(line_coordsA) == 2) | (len(line_coordsB) == 2)): return False
       
    if not hard:
        # remove first coordinates (A,B)
        line_geometry_A = LineString([coor for coor in line_coordsA[1:]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): return True
        
        # remove first (A) and last (B)
        line_geometry_B = LineString([coor for coor in line_coordsB[:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): return True
        
        # remove last (A) and first (B)
        line_geometry_A = LineString([coor for coor in line_coordsA[:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): return True
        
        # remove last coordinates (A, B)
        line_geometry_A = LineString([coor for coor in line_coordsA[:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): return True
        
        if ((len(line_coordsA) == 3) | (len(line_coordsB) == 3)): return False
        line_geometry_A = LineString([coor for coor in line_coordsA[1:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): return True
        
    return False
    
def is_continuation(ix_lineA, ix_lineB, edges_gdf):

    nameA = edges_gdf.loc[ix_lineA]['name']
    nameB = edges_gdf.loc[ix_lineB]['name']
    line_geometry_A = edges_gdf.loc[ix_lineA]['geometry']
    line_geometry_B = edges_gdf.loc[ix_lineB]['geometry']
    if is_parallel(line_geometry_A, line_geometry_B, hard = True): return True
    
    if nameA == nameB: 
        if not is_perpendicular(line_geometry_A, line_geometry_B): return True
        else: return False
    if not is_perpendicular(line_geometry_A, line_geometry_B): return True
    else: return False


def simplify_dual_lines_junctions(nodes_gdf, edges_gdf, max_difference_length = 0.40, max_distance_between_lines = 30):

    """
    This function simplifies parallel or semi-parallel lines - which may represent dual carriageway roads.
    In this case, the roads originate and terminate from the same pair of nodes:
    - An uninterrupted (no intersecting roads along) street segment A is examined
    - The lines originating from its vertexes (u, v) are assesed.
    - Lines which are not parallel are disregarded.
    - The parallel lines are kept and their natural continuations are examined, again in relation to segment A.
      This line can originate for example in segment A's "u", traverse a certain amount of intermediate nodes and reach segment A's "v".
    - Thus, road B, if existing, is composed of continuous sub-segments parallel to segment A. The geometry obtained by merging road B continuous segments starts either in
      segmentA's "u" or "v" and terminates in either "v" or "u".
    - If such line is found a center line geometry is obtained.
    
    Interesecting roads are interpolated in the simplified road-center-line resulting geometry.
            
    If the researcher has assigned specific values to edges (e.g. densities of pedestrians, vehicular traffic or similar) please allow the function to combine
    the relative densities values during the cleaning process.
    
    Two parameters depend on street morphology and the user assessment:
    - max_difference_length: indicate here the max difference in length between the two lines (segmentA's geometry and roadB's). 
                             Specify the max percente difference in float. e.g. 40% --> 0.40
    - max_distance_between_lines: float
    
    A new dataframe is returned with the simplified geometries.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
    edges_gdf: LineString GeoDataFrames
    max_difference_length: float
    max_distance_between_lines: float
  
    Returns
    -------
    GeoDataFrames
    """
    
    nodes_gdf.set_index('nodeID', drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
   
    edges_gdf['name'][edges_gdf.name.isnull()] = None
    original_edges_gdf = edges_gdf.copy()
       
    ix_geo = edges_gdf.columns.get_loc("geometry")+1  
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    processed = []
    
    # the original geometries and edges are iterated and examined;
    for row in original_edges_gdf.itertuples():
        if row.Index in processed: continue  
    
        for r in [ix_u, ix_v]:
            found = False
            possible_matches = original_edges_gdf[(original_edges_gdf['u'] == row[r]) | (original_edges_gdf['v'] == row[r])].copy()
            possible_matches.drop(row.Index, axis = 0, inplace = True)
            possible_matches = possible_matches[~possible_matches.index.isin(processed)]
            
            # first check
            for connector in possible_matches.itertuples(): 
                if (not is_continuation(row.Index, connector.Index, edges_gdf)) | (connector[ix_geo].length > row[ix_geo].length):
                    possible_matches.drop(connector.Index, axis = 0, inplace = True)
                    continue

            if len(possible_matches) == 0: continue
            if r == ix_u: 
                direction = 'v'
                to_reach = row[ix_v]    
            else: 
                direction = 'u'
                to_reach = row[ix_u]           
                    
            for connector in possible_matches.itertuples():
                if connector[ix_u] == row[r]: search = connector[ix_v]  
                else: search = connector[ix_u]

                nodes_encountered = [search]
                lines_traversed = [connector[ix_geo]]
                lines = [connector.Index]
                next_line = False # to determine when moving to the next candidate
                last_line = connector.Index

                while (not found) & (not next_line):
                    # look for a new possible set of connectors
                    next_possible_matches = original_edges_gdf[(original_edges_gdf['u'] == search) | (original_edges_gdf['v'] == search)].copy()      
                    next_possible_matches.drop([last_line, row.Index], axis = 0, inplace = True, errors = 'ignore') # remove the previous lines, in case
                    next_possible_matches = next_possible_matches[~next_possible_matches.index.isin(processed)]

                    for other_connector in next_possible_matches.itertuples():
                        if not is_continuation(last_line, other_connector.Index, edges_gdf): next_possible_matches.drop(other_connector.Index, axis = 0, inplace = True)

                    if len(next_possible_matches) == 0: 
                        next_line = True
                        break

                    if len(next_possible_matches) > 1: # if more than one candidate
                        next_possible_matches['angle'] = 0.0
                        for candidate in next_possible_matches.itertuples():
                            angle = angle_line_geometries(edges_gdf.loc[last_line].geometry, candidate[ix_geo], deflection = True, degree = True)
                            next_possible_matches.at[candidate.Index, 'angle'] = angle
                        next_possible_matches.sort_values(by = 'angle', ascending = True, inplace = True)    
                    
                    # take the best candidate's attribute
                    u, v = next_possible_matches.iloc[0]['u'], next_possible_matches.iloc[0]['v']

                    if u == search: 
                        search = next_possible_matches.iloc[0]['v']
                        other = next_possible_matches.iloc[0]['u']
                    else: 
                        search = next_possible_matches.iloc[0]['u']
                        other = next_possible_matches.iloc[0]['v']

                    distA = nodes_gdf.loc[search].geometry.distance(nodes_gdf.loc[to_reach].geometry)
                    distB = nodes_gdf.loc[other].geometry.distance(nodes_gdf.loc[to_reach].geometry)

                    if (search in nodes_encountered) | (distB < distA):           
                        next_line = True
                        continue
                    elif search == to_reach:
                        lines_traversed.append(next_possible_matches.iloc[0].geometry)
                        lines.append(next_possible_matches.iloc[0].name)
                        found = True
                        break
                    else: 
                        nodes_encountered.append(search)
                        lines_traversed.append(next_possible_matches.iloc[0].geometry)
                        lines.append(next_possible_matches.iloc[0].name)
                        last_line = next_possible_matches.iloc[0].name

                if next_line: continue
                else: break

            if not found: continue # no parallel dual lines at this node
            u, v, geo = row[ix_u], row[ix_v], row[ix_geo]    
            merged_line = merge_lines(lines_traversed)
            
            # check whether it makes sense to merge or not
            if (geo.length*(max_difference_length+1) < merged_line.length) | (geo.length > merged_line.length*(max_difference_length+1)): continue
            if (geo.centroid.distance(merged_line.centroid) > max_distance_between_lines): continue
            
            # obtaining center line
            cl = center_line(geo, merged_line)
            processed = processed + lines
            processed.append(row.Index)
            if "pedestrian" in edges_gdf.columns:
                if len(edges_gdf.loc[lines][edges_gdf.pedestrian == 1]) > 0: edges_gdf.at[row.Index, 'pedestrian'] = 1
            if direction == 'u': nodes_encountered.reverse()
            
            # interpolate nodes encountered along the parallel lines
            interpolate(u, v, cl, nodes_encountered, lines, nodes_gdf, edges_gdf, row.Index)
            edges_gdf.drop(lines, axis = 0, inplace = True) 
            break
            
    # correct the coordinates and clean the network
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True)
    
    return(nodes_gdf, edges_gdf)

def simplify_complex_junctions(nodes_gdf, edges_gdf):
    
    """
    This function simplifies complex junctions as trinagular-like junctions formed mainly by secondary links.
    The junction may be as well represented by one node rather than, for example, three nodes. 
            
    If the researcher has assigned specific values to edges (e.g. densities of pedestrians, vehicular traffic or similar) please allow the function to combine
    the relative densities values during the cleaning process.
    
    The function takes a node and check whether the intersecting edges give shape to a triangular-cyclic junction.
    
    A new dataframe with the simplified geometries is returned.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
    edges_gdf: LineString GeoDataFrames
   
    Returns
    -------
    GeoDataFrames
    """
    
    nodes_gdf.set_index('nodeID', drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    edges_gdf['name'][edges_gdf.name.isnull()] = None
    original_edges_gdf = edges_gdf.copy()
    
    ix_geo = edges_gdf.columns.get_loc("geometry")+1  
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    processed = []
    
    for node in nodes_gdf.itertuples():
        tmp =  edges_gdf[(edges_gdf['u'] == node.Index) | (edges_gdf['v'] == node.Index)].copy()
        found = False
        
        # take one of these lines and examine its relationship with the others at the same junction
        for row in tmp.itertuples():
            if row.Index in processed: continue

            for other in tmp.itertuples():
                if (row.Index == other.Index) | (other.Index in processed) : continue
                
                # determining the relationship
                if row[ix_u] == other[ix_u]: # the last one is 'v'
                    v1, v2 = ix_v, ix_v
                    last_vertex, code = -1, 'v'
                    
                elif row[ix_u] == other[ix_v]: # the last one is 'u'
                    v1, v2 = ix_v, ix_u
                    last_vertex, code = -1, 'v'
                
                elif row[ix_v] == other[ix_u]: # the last one is 'u'
                    v1, v2 = ix_u, ix_v
                    last_vertex, code = 0, 'u'
                    
                elif row[ix_v] == other[ix_v]: # the last one is 'u'
                    v1, v2 = ix_u, ix_u
                    last_vertex, code = 0, 'u'
                else: continue
                 
                # look for the connector segment
                possible_matches = edges_gdf[((edges_gdf['u'] == row[v1]) & (edges_gdf['v'] == other[v2])) | ((edges_gdf['u'] == other[v2]) & (edges_gdf['v'] == row[v1]))].copy()
                if len(possible_matches) == 0: continue
                connector = possible_matches.iloc[0]
                
                u, v, u_other, v_other = row[ix_u], row[ix_v], other[ix_u], other[ix_v]
                geo, other_geometry, connector_geometry = row[ix_geo], other[ix_geo], connector.geometry
                if any(i > 100 for i in [geo.length, other_geometry.length, connector_geometry.length]): break # segments are too long
                
                diff_A = abs(geo.length - other_geometry.length)    
                diff_B = abs(geo.length - connector_geometry.length)
                diff_C = abs(other_geometry.length- connector_geometry.length)
                if (diff_B < diff_A) | (diff_C < diff_A): continue 
                if (diff_A > geo.length*0.75) | (diff_A > other_geometry.length*0.75): continue
                if (connector_geometry.length > (geo.length + other_geometry.length)*1.25): continue  
                if (diff_A > geo.length*0.25) | (diff_A > other_geometry.length*0.25): continue
                
                if "pedestrian" in edges_gdf.columns: 
                    if edges_gdf.loc[other.Index]['pedestrian'] == 1: edges_gdf.at[row.Index, 'pedestrian'] = 1
                
                # drop the other line
                edges_gdf.drop(other.Index, axis = 0, inplace = True)
                cl =  center_line(geo, other_geometry)
                intersection = cl.intersection(connector_geometry)
                ix_node = nodes_gdf.index.max()+1
                nodes_gdf.loc[ix_node] = nodes_gdf.loc[row[v1]] # copy attributes
                nodes_gdf.at[ix_node, 'nodeID'] = ix_node
                
                ix_edge = edges_gdf.index.max()+1
                edges_gdf.loc[ix_edge] = edges_gdf.loc[connector.name]
                edges_gdf.at[ix_edge, 'edgeID'] = ix_edge
                edges_gdf.at[row.Index, code] = ix_node

                if intersection.geom_type == 'Point': # check if the center line reaches the connector
                    last = intersection.coords[0]
                    line = split_line_interpolation(intersection, cl)[0]
                    nodes_gdf.at[ix_node, 'geometry'] = intersection
                    
                    if code == 'u': edges_gdf.at[row.Index,'geometry'] = line[1]
                    else: edges_gdf.at[row.Index,'geometry'] = line[0]
                    
                    line = split_line_interpolation(intersection, connector_geometry)[0]
                    edges_gdf.at[connector.name, 'geometry'] = line[0]
                    edges_gdf.at[connector.name, 'v'] = ix_node
                    edges_gdf.at[ix_edge, 'u'] = ix_node
                    edges_gdf.at[ix_edge, 'geometry'] = line[1]

                else: # no intersection, extend lines towards center line
                    last = list(cl.coords)[last_vertex]
                    nodes_gdf.at[ix_node, 'geometry'] = Point(last)
                    edges_gdf.at[row.Index,'geometry'] = cl

                    line_geometry_A = LineString([coor for coor in [connector_geometry.coords[0], last]])
                    line_geometry_B = LineString([coor for coor in [last, connector_geometry.coords[-1]]])
                    edges_gdf.at[connector.name, 'geometry'] = line_geometry_A
                    edges_gdf.at[ix_edge, 'geometry'] = line_geometry_B
                    edges_gdf.at[connector.name, 'v'] = ix_node
                    edges_gdf.at[ix_edge, 'u'] = ix_node
                
                processed = processed + [row.Index, other.Index]
                nodes_gdf.at[ix_node, 'x'] = last[0]
                nodes_gdf.at[ix_node, 'y'] = last[1]
                
                found = True
                break
                                    
            if found: break
                        
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True) 
    return(nodes_gdf, edges_gdf)



def dissolve_roundabouts(nodes_gdf, edges_gdf, max_length_segment = 80, angle_tolerance = 40):

    nodes_gdf.set_index('nodeID', drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    ix_geo = edges_gdf.columns.get_loc("geometry")+1  
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1

    processed_segments = []
    processed_nodes = []
    
    dd_u, dd_v = dict(edges_gdf['u'].value_counts()), dict(edges_gdf['v'].value_counts())
    dd = {k: dd_u.get(k, 0) + dd_v.get(k, 0) for k in set(dd_u) | set(dd_v)}
    
    # editing the ones which only connect two edges
    to_edit = {k: v for k, v in dd.items() if v == 3}
    if len(to_edit) == 0: return(nodes_gdf, edges_gdf)
    to_edit_gdf = nodes_gdf[nodes_gdf.nodeID.isin(to_edit)]
    
    
    for node in to_edit_gdf.itertuples():

        if node in processed_nodes: continue
        tmp =  edges_gdf[(edges_gdf['u'] == node.Index) | (edges_gdf['v'] == node.Index)].copy()
        found = False
        not_a_roundabout = False
               
        # take one of these lines and examine its relationship with the others at the same junction
        for row in tmp.itertuples():
        
            if row[ix_geo].length > max_length_segment: continue #too long for being a roundabout segment
            sequence_nodes = [node.Index]
            sequence_segments = [row.Index]
            if row.Index in processed_segments: continue
            
            if row[ix_u] == node.Index: last_vertex = row[ix_v]
            else: last_vertex = row[ix_u]
            
            sequence_nodes.append(last_vertex)
            segment = row
            distance = 0
            second_candidate = False
            
            while not found:
                if distance >= 400: break # too much traversed distance for a roundabout
                if last_vertex in processed_nodes: # the node has been dissolved already
                    if not second_candidate: break
                    else:
                        distance -= segment[ix_geo].length
                        segment = sc
                        distance += segment[ix_geo].length
                        sequence_segments[-1] = segment[0]
                        last_vertex = sc_last_vertex
                        sequence_nodes[-1] = sc_last_vertex
                        second_candidate = False
                        continue
                        
                possible_connectors = edges_gdf[(edges_gdf['u'] == last_vertex) | (edges_gdf['v'] == last_vertex)].copy()
                for connector in possible_connectors.itertuples():
        
                    if (segment[0] == connector.Index) | (connector.Index in processed_segments): possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                    elif connector[ix_geo].length > max_length_segment: possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                    else: 
                        angle = angle_line_geometries(segment[ix_geo], connector[ix_geo], angular_change = True, degree = True)
                        if angle > angle_tolerance: possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                        else: possible_connectors.at[connector.Index, 'angle'] = angle
                    
                if (len(possible_connectors) == 0) | (last_vertex in processed_nodes):
                    if not second_candidate: break
                    else:
                        distance -= segment[ix_geo].length
                        segment = sc
                        distance += segment[ix_geo].length
                        sequence_segments[-1] = segment[0]
                        last_vertex = sc_last_vertex
                        sequence_nodes[-1] = sc_last_vertex
                        second_candidate = False
                        continue

                else: possible_connectors.sort_values(by = 'angle', ascending = True, inplace = True) 
                
                segment = list(possible_connectors.iloc[0])
                segment.insert(0, possible_connectors.iloc[0].name)
                
                if len(possible_connectors) > 1:
                    sc = list(possible_connectors.iloc[1])
                    sc.insert(0, possible_connectors.iloc[1].name)
                    second_candidate = True
                    if sc[ix_u] == last_vertex: sc_last_vertex = sc[ix_v]
                    else: sc_last_vertex = sc[ix_u]
                
                if segment[ix_u] == last_vertex: last_vertex = segment[ix_v]
                else: last_vertex = segment[ix_u]

                sequence_nodes.append(last_vertex)
                sequence_segments.append(segment[0])                
                distance += segment[ix_geo].length
                if last_vertex == node.Index:
                    lm = linemerge(edges_gdf.loc[i].geometry for i in sequence_segments)
                    roundabout = polygonize_full(lm)[0]
                    centroid = roundabout.centroid
                    distances = [nodes_gdf.loc[i].geometry.distance(centroid) for i in sequence_nodes]
                    shortest, longest, mean = min(distances), max(distances), statistics.mean(distances) 
                    if (shortest < mean * 0.80) | (longest > mean * 1.20): 
                        not_a_roundabout = True
                        break

                    found = True
                    new_index = max(nodes_gdf.index)+1

                    nodes_gdf.loc[new_index] = nodes_gdf.loc[node.Index]
                    nodes_gdf.at[new_index,'nodeID'] = new_index
                    nodes_gdf.at[new_index,'geometry'] = centroid
                    nodes_gdf.at[new_index,'x'] = centroid.coords[0][0]
                    nodes_gdf.at[new_index,'y'] = centroid.coords[0][1]
                    processed_segments = processed_segments + sequence_segments
                    processed_nodes = processed_nodes + sequence_nodes + [new_index]
                    edges_gdf.loc[edges_gdf['u'].isin(sequence_nodes), 'u'] = new_index 
                    edges_gdf.loc[edges_gdf['v'].isin(sequence_nodes), 'v'] = new_index 
                    nodes_gdf.drop(sequence_nodes, axis = 0, inplace = True)
                    edges_gdf.drop(sequence_segments, axis = 0, inplace = True)   
            if not_a_roundabout: break
            if found: break
            
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True)         
     
            
    return nodes_gdf, edges_gdf
                    

            # envelope = building_geometry.envelope
            # coords = mapping(t)["coordinates"][0]
            # d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
            # width = min(d)

def extract_centroids(nodes_gdf, edges_gdf, radius = 10):   
    
    """
    This function simplifies complex junctions as trinagular-like junctions formed mainly by secondary links.
    The junction may be as well represented by one node rather than, for example three nodes. 
               
    The function takes a node and check whether the intersecting edges give shape to a triangular-cyclic junction.
    
    A new dataframe with the simplified geometries is returned.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
    edges_gdf: LineString GeoDataFrames
   
    Returns
    -------
    GeoDataFrames
    """   
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    
    buffered_nodes = nodes_gdf.buffer(radius).unary_union
    if isinstance(buffered_nodes, Polygon): buffered_nodes = [buffered_nodes]
        
    buffered_nodes_geoS = gpd.GeoSeries(list(buffered_nodes))
    buffered_nodes_df =  pd.concat([buffered_nodes_geoS.rename('geometry'), pd.Series(buffered_nodes_geoS.index).rename('code')], axis=1)

    buffered_nodes_gdf = gpd.GeoDataFrame(buffered_nodes_df, geometry = buffered_nodes_df.geometry)
    buffered_nodes_gdf['area']= buffered_nodes_gdf['geometry'].area
    buffered_nodes_gdf['centroid'] = buffered_nodes_gdf.geometry.centroid
    
    clusters_gdf = buffered_nodes_gdf[buffered_nodes_gdf["area"] > (radius*radius*3.14159)]
    clusters_gdf['x'], clusters_gdf['y'] = (clusters_gdf.geometry.centroid.x, clusters_gdf.geometry.centroid.y)
    clusters_gdf.index += nodes_gdf.index.max()+1
    clusters_gdf['code'] = clusters_gdf.index
    
    nodes_gdf['cluster'] = None

    # set cluster column values
    nodes_gdf["cluster"] = nodes_gdf.apply(lambda row: _assign_cluster_nodes(row["geometry"], clusters_gdf), axis = 1)
    geometry = clusters_gdf['centroid']
    data = clusters_gdf.drop(['centroid', 'geometry'], axis=1)
    clusters_gdf = gpd.GeoDataFrame(data, crs=nodes_gdf.crs, geometry=geometry)
    edges_gdf = assign_cluster_edges(nodes_gdf, edges_gdf)
    
    return(nodes_gdf, edges_gdf, clusters_gdf)
 
def _assign_cluster_nodes(node_geometry, cluster_gdf):
        
    ix_geo = clusters_gdf.columns.get_loc("geometry")+1
    ix_code = clusters_gdf.columns.get_loc("code")+1
    
    tmp = clusters_gdf[clusters_gdf["geometry"].intersects(node_geometry.buffer(1))]
    if len(tmp) == 0: return None
    for cluster in tmp.itertuples():
        if node_geometry.within(cluster[ix_geo]): return cluster[ix_code]

def assign_cluster_edges(nodes_gdf, edges_gdf):
    
    edges_gdf.drop(['nodeID_x', 'nodeID_y','clus_uR', 'clus_vR', 'clus_u', 'clus_v'], axis = 1, inplace = True, errors = 'ignore')
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['cluster', 'nodeID']], how = 'left', left_on= "u", right_on = "nodeID")
    edges_gdf = edges_gdf.rename(columns = {'cluster':'clus_u'})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['cluster', 'nodeID']], how = 'left', left_on= "v", right_on = "nodeID")
    edges_gdf = edges_gdf.rename(columns = {'cluster':'clus_v'})  
    edges_gdf.set_index('edgeID', drop = False, append = False, inplace = True)
    edges_gdf.index.name = None
    
    edges_gdf['clus_uR'], edges_gdf['clus_vR'] = None, None
    ix_clus_u, ix_clus_v  = edges_gdf.columns.get_loc("clus_u")+1, edges_gdf.columns.get_loc("clus_v")+1
    ix_clus_uR, ix_clus_vR = edges_gdf.columns.get_loc("clus_uR")+1, edges_gdf.columns.get_loc("clus_vR")+1
   
    # assigning cluster
    tmp = edges_gdf[(edges_gdf['clus_u'].isnull()) | (edges_gdf['clus_v'].isnull())].copy()
    for row in tmp.itertuples():
        if row[ix_clus_u] is None:
            cluster = indirect_cluster(nodes_gdf, edges_gdf, row.Index, 'u')[0]
            edges_gdf.at[row.Index, 'clus_uR'] = cluster
        if row[ix_clus_v] is None:
            cluster = indirect_cluster(nodes_gdf, edges_gdf, row.Index, 'v')[0]
            edges_gdf.at[row.Index, 'clus_vR'] = cluster
    
    edges_gdf.drop(['nodeID_x', 'nodeID_y'], axis = 1, inplace = True, errors = 'ignore')       
    return(edges_gdf)

def indirect_cluster(nodes_gdf, edges_gdf, ix_line, search_dir):
    
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    
    u, v = edges_gdf.loc[ix_line]['u'], edges_gdf.loc[ix_line]['v']
    line = edges_gdf.loc[ix_line].geometry
    name = edges_gdf.loc[ix_line]['name']
    line_coords = list(line.coords)
    
    if search_dir == 'v': 
        coming_from = v
        other_node = u
        possible_matches = edges_gdf[(edges_gdf.u == v) | (edges_gdf.v == v)].copy()
    else: 
        line_coords.reverse()
        coming_from = u
        other_node = v
        possible_matches = edges_gdf[(edges_gdf.u == u) | (edges_gdf.v == u)].copy()
     
    possible_matches.drop(ix_line, axis = 0, inplace = True)
    nodes_encountered = []
    lines_traversed = []
    last_line = ix_line

    found = False
    while not found:
        if len(possible_matches) == 0: return(None, None, None, None, None)
      
        if len(possible_matches) > 1:
            possible_matches['angle'] = 0.0
            for connector in possible_matches.itertuples():
                    angle = angle_line_geometries(edges_gdf.loc[last_line].geometry, connector[ix_geo], deflection = True, degree = True)
                    possible_matches.at[connector.Index, 'angle'] = angle
            possible_matches.sort_values(by = 'angle', ascending = True, inplace = True)    
            
        for connector in possible_matches.itertuples():
            if not is_continuation(last_line, connector.Index, edges_gdf):
                possible_matches.drop(connector.Index, axis = 0, inplace = True)
                continue
            
            else:
                uCP, vCP = connector[ix_u], connector[ix_v]
                
                if uCP == coming_from:
                    cluster = nodes_gdf.loc[vCP].cluster
                    coming_from = vCP
                    distance_to = nodes_gdf.loc[vCP].geometry.distance(nodes_gdf.loc[other_node].geometry)
                    distance_from = nodes_gdf.loc[uCP].geometry.distance(nodes_gdf.loc[other_node].geometry)
                    if (vCP in nodes_encountered) | (distance_to < distance_from):
                        possible_matches = possible_matches[0:0]
                        break
                else: 
                    cluster = nodes_gdf.loc[uCP].cluster
                    coming_from = uCP
                    distance_to = nodes_gdf.loc[uCP].geometry.distance(nodes_gdf.loc[other_node].geometry)
                    distance_from = nodes_gdf.loc[vCP].geometry.distance(nodes_gdf.loc[other_node].geometry)
                    if (uCP in nodes_encountered) | (distance_to < distance_from):
                        possible_matches = possible_matches[0:0]
                        break
                                    
                if cluster is None:
                    lines_traversed.append(connector.Index)
                    last_line = connector.Index

                    if vCP == coming_from:
                        possible_matches = edges_gdf[(edges_gdf.u == vCP) | (edges_gdf.v == vCP) ].copy()
                        nodes_encountered.append(uCP) 
                        line_coords = line_coords + list(connector[ix_geo].coords)
                    else:
                        possible_matches = edges_gdf[(edges_gdf.u == uCP) | (edges_gdf.v == uCP)].copy()
                        nodes_encountered.append(vCP)
                        tmp = list(connector[ix_geo].coords)
                        tmp.reverse()
                        line_coords = line_coords + tmp
                    break
                
                else:
                    found = True
                    lines_traversed.append(connector.Index)
                    
                    if vCP == coming_from:
                        nodes_encountered.append(uCP)
                        last_node = vCP
                        line_coords = line_coords + list(connector[ix_geo].coords)
                    else: 
                        nodes_encountered.append(vCP)
                        last_node = uCP
                        tmp = list(connector[ix_geo].coords)
                        tmp.reverse()
                        line_coords = line_coords + tmp
                    break
    
    merged_line = LineString([coor for coor in line_coords])   
    return(cluster, merged_line, lines_traversed, nodes_encountered, last_node)
        
def center_line_cluster(line_geometry_A, line_geometry_B, nodes_gdf, clusters_gdf, cluster_from, cluster_to, one_cluster = False): 
    
    if one_cluster: coord_from = (nodes_gdf.loc[cluster_from]['x'], nodes_gdf.loc[cluster_from]['y'])
    else: coord_from = (clusters_gdf.loc[cluster_from]['x'], clusters_gdf.loc[cluster_from]['y'])
    
    coord_to =  (clusters_gdf.loc[cluster_to]['x'], clusters_gdf.loc[cluster_to]['y'])
    line_coordsA = list(line_geometry_A.coords)
    line_coordsB = list(line_geometry_B.coords)
    
    # no need to reverse lines, as they should arrive already in the same order      
    # different number of vertexes, connect the line
    while len(line_coordsA) > len(line_coordsB):
        index = int(len(line_coordsA)/2)
        del line_coordsA[index]
    while len(line_coordsB) > len(line_coordsA):
        index = int(len(line_coordsB)/2)
        del line_coordsB[index]      
    
    new_line = line_coordsA
    for n, i in enumerate(line_coordsA):
        link = LineString([coor for coor in [line_coordsA[n], line_coordsB[n]]])
        np = link.centroid.coords[0]           
        new_line[n] = np
        
    new_line[0] = coord_from
    new_line[-1] = coord_to
    center_line = LineString([coor for coor in new_line])           
        
    return center_line

def center_line_cluster_four(list_lines, nodes_gdf, clusters_gdf, cluster_from, cluster_to): #rev
    
    coord_from = (clusters_gdf.loc[cluster_from]['x'], clusters_gdf.loc[cluster_from]['y'])
    coord_to = (clusters_gdf.loc[cluster_to]['x'], clusters_gdf.loc[cluster_to]['y'])
    
    list_coords = []
    for i in list_lines: list_coords.append(list(i.coords))
    
    # no need to reverse lines, as they should arrive already in the same order      
    # different number of vertexes, connect the line
    
    for line in list_coords:
        for other_line in list_coords:
            if line == other_line: continue               
            while len(line) > len(other_line):
                index = int(len(line)/2)
                del line[index]    

    new_line = list_coords[0]
    for n, i in enumerate(list_coords[0]):       
        pairs = [coor[n] for coor in list_coords]
        pairs = list(set(pairs))
        
        maxDistance = 0
        for point in pairs:
            for other_point in pairs:
                if point == other_point: continue
                distance = Point(point).distance(Point(other_point))
                if distance > maxDistance:
                    furthest = [point, other_point]
                    maxDistance = distance
                    
        if len(pairs) == 2:
            link = LineString([coor for coor in [pairs[0], pairs[1]]])
        
        elif len(pairs) == 3:
            second = [x for x in pairs if x not in furthest]
            link = LineString([coor for coor in [furthest[0], second[0], furthest[1]]])
        else:
            # find second
            maxDistance = 0.0
            last = None
            
            for point in pairs:
                if point in furthest: continue
                distance = Point(point).distance(Point(furthest[0]))
                if distance > maxDistance:
                    maxDistance = distance
                    last = point   
                    
            second = [x for x in pairs if x not in furthest]
            second.remove(last)
            link = LineString([coor for coor in [furthest[0], second[0], last, furthest[1]]])
        
        np = link.centroid.coords[0]      
        new_line[n] = np
        
    new_line[0] = coord_from
    new_line[-1] = coord_to
    center_line = LineString([coor for coor in new_line])           
        
    return center_line

def split_line_interpolation(point, line_geometry):
    
    line_coords = list(line_geometry.coords)
    starting_point = Point(line_coords[0])
    np = nearest_points(point, line_geometry)[1]
    distance_start = line_geometry.project(np)
    
    new_line_A = []
    new_line_B = []

    if len(line_coords) == 2:
        new_line_A = [line_coords[0],  np.coords[0]]
        new_line_B = [np.coords[0], line_coords[-1]]
        line_geometry_A = LineString([coor for coor in new_line_A])
        line_geometry_B = LineString([coor for coor in new_line_B])

    else:
        new_line_A.append(line_coords[0])
        new_line_B.append(np.coords[0])

        for n, i in enumerate(line_coords):
            if (n == 0) | (n == len(line_coords)-1): continue
            if line_geometry.project(Point(i)) < distance_start: new_line_A.append(i)
            else: new_line_B.append(i)

        new_line_A.append(np.coords[0])
        new_line_B.append(line_coords[-1])
        line_geometry_A = LineString([coor for coor in new_line_A])
        line_geometry_B = LineString([coor for coor in new_line_B])
    
    return((line_geometry_A, line_geometry_B), np)

def interpolate(first_node, last_node, center_line, list_nodes, list_lines, nodes_gdf, edges_gdf, ix_line):
   
    line_geometry = center_line    
    new_index = ix_line
    
    for counter, node in enumerate(list_nodes):
        point = nodes_gdf.loc[node].geometry
        lines, np = split_line_interpolation(point, line_geometry)
              
        # adjusting node coordinates
        nodes_gdf.at[node, 'x'] = np.coords[0][0]
        nodes_gdf.at[node, 'y'] = np.coords[0][1]
        nodes_gdf.at[node, 'geometry'] = np
        
        #first part of the segment
        if counter == 0: edges_gdf.at[new_index, 'u'] = first_node
        edges_gdf.at[new_index, 'geometry'] = lines[0]
        edges_gdf.at[new_index, 'v'] = node
        
        # second part of the segment
        new_index = max(edges_gdf.index)+1
        edges_gdf.loc[new_index] = edges_gdf.loc[ix_line]
        edges_gdf.at[new_index, 'geometry'] = lines[1]
        edges_gdf.at[new_index, 'u'] = node
        edges_gdf.at[new_index, 'v'] = last_node
        edges_gdf.at[new_index, 'edgeID'] = new_index
        line_geometry = lines[1]
                                                                                                         

def interpolate_multi(first_node, last_node, center_line, list_nodes, list_lines, nodes_gdf, edges_gdf, ix_line):

    line_geometry = center_line   
    new_index = ix_line                                                                                        
    distances = {}
    lines_distances = {}
    
    for node in list_nodes:
        distance = nodes_gdf.loc[node]['geometry'].project(Point(center_line.coords[0])) #rev!
        distances[node] = distance
                                                                                                         
    for line in list_lines:
        distance = edges_gdf.loc[line]['geometry'].project(Point(center_line.coords[-1])) #rev!
        lines_distances[line] = distance                                                                                                   

    distances_sorted = sorted(distances.items(), key=lambda kv: kv[1])               
    lines_distances_sorted = sorted(lines_distances.items(), key=lambda kv: kv[1])
                                                                                                         
    for counter, node in enumerate(distances_sorted):
        
        node = distances_sorted[counter][0]
        point = nodes_gdf.loc[node].geometry
        result, np = split_line_interpolation(point, gline)
        
        #first part of the segment, adjusting node coordinates
        nodes_gdf.at[node, 'x'] = np.coords[0][0]
        nodes_gdf.at[node, 'y'] = np.coords[0][1]
        nodes_gdf.at[node, 'geometry'] = np
        
        if counter == 0: edges_gdf.at[new_index, 'u'] = first_node
        edges_gdf.at[new_index, 'geometry'] = result[0]
        edges_gdf.at[new_index, 'v'] = node
          
        # second part of the segment
        new_index = max(edges_gdf.index)+1
        
        edges_gdf.loc[new_index] = edges_gdf.loc[ix_line]
        edges_gdf.at[new_index, 'geometry'] = result[1]
        edges_gdf.at[new_index, 'u'] = node
        edges_gdf.at[new_index, 'v'] = last_node
        edges_gdf.at[new_index, 'edgeID'] = new_index
        gline = result[1]                                       
                                                                         
def merge_two(ix_lineA, ix_lineB, line_geometry_A, line_geometry_B, nodes_gdf, edges_gdf, clusters_gdf, cluster, goal, direction):
    
    # some pre criteria
    if (((line_geometry_A.centroid.distance(line_geometry_B.centroid) > 18) & 
       (edges_gdf.loc[ix_lineA]['name'] != edges_gdf.loc[ix_lineB]['name'])) | 
        ((line_geometry_A.length > line_geometry_B.length*1.50) | (line_geometry_B.length > line_geometry_A.length*1.50))):
        return None

    cl = center_line_cluster(line_geometry_A, line_geometry_B, nodes_gdf, clusters_gdf, cluster, goal)
    
    if direction == 'u':
        line_coords = list(cl.coords)
        line_coords.reverse() 
        cl = LineString([coor for coor in line_coords])
    
    edges_gdf.at[ix_lineA, 'geometry'] = cl
    return 'processed'

    
def merge_two_inter(ix_lineA, ix_lineB, line_geometry_A, line_geometry_B, nodes_gdf, edges_gdf, clusters_gdf, cluster, goal, 
                               starting_node, last_node, list_nodes, list_lines, multi = False):
    # the center line is built in relation to the variable cluster as 'u', or from_node --> to_node
    if (((line_geometry_A.centroid.distance(line_geometry_B.centroid) > 18) & 
       (edges_gdf.loc[ix_lineA]['name'] != edges_gdf.loc[ix_lineB]['name'])) | 
        ((line_geometry_A.length > line_geometry_B.length*1.50) | (line_geometry_B.length > line_geometry_A.length*1.50))):
        return None
    
    cl = center_line_cluster(line_geometry_A, line_geometry_B, nodes_gdf, clusters_gdf, cluster, goal)
    if multi: interpolate_multi(starting_node, last_node, cl, 
                                        list_nodes, list_lines, nodes_gdf, edges_gdf, ix_lineA)
    else: interpolate(starting_node, last_node, cl, list_nodes, list_lines, nodes_gdf, edges_gdf, ix_lineA) 
    return 'processed'

def find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal):
    secondary_lines = []
    max_dist = 0
    
    if len(dict_lines)%2 != 0:                                                        
        for key, value in dict_lines.items():
            for keyC, valueC in dict_lines.items():
                if key == keyC: continue
                distance = (value.centroid).distance(valueC.centroid)
                if distance > max_dist: 
                    max_dist = distance
                    secondary_lines = [key, keyC]

        central = [x for x in list(dict_lines.keys()) if x not in secondary_lines][0]  
        geo_central = dict_lines[central]
    
    else:
        geo_central = center_line_cluster_four(list(dict_lines.values()), nodes_gdf, clusters_gdf, cluster, goal)                         
        central, secondary_lines = None, None
       
    return(central, secondary_lines, geo_central)
  

def simplify_dual_lines(nodes_gdf, edges_gdf, clusters_gdf):
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    ix_u, ix_v  = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    ix_cluster = nodes_gdf.columns.get_loc("cluster")+1
    ix_clus_u, ix_clus_v  = edges_gdf.columns.get_loc("clus_u")+1, edges_gdf.columns.get_loc("clus_v")+1
    ix_clus_uR, ix_clus_vR = edges_gdf.columns.get_loc("clus_uR")+1, edges_gdf.columns.get_loc("clus_vR")+1
    
    ################################ FROM NODES TO CLUSTERED JUNCTIONS
    
    clusters_gdf['keep'] = False
    original_edges_gdf = edges_gdf.copy()  
    processed = []
    to_drop = []
    list_cluster = clusters_gdf.index.values.tolist() 
    
    print('Simplifying dual lines: First part - clusters')

    for cluster in list_cluster:
           
        edges_tmp = original_edges_gdf[((original_edges_gdf.clus_u == cluster) | (original_edges_gdf.clus_v == cluster))].copy()
        edges_tmp = edges_tmp[edges_tmp.clus_u != edges_tmp.clus_v].copy()
        
        if len(edges_tmp) == 1: continue
        for row in edges_tmp.itertuples():
            if row.Index in processed: continue  
            pDL = edges_tmp.copy() 
            
            # disregard unparallel lines 
            for rowC in pDL.itertuples():
                if row.Index == rowC.Index: continue
                elif rowC.Index in processed: pDL.drop(rowC.Index, axis = 0, inplace = True)
                elif ((row[ix_u] == rowC[ix_u]) | (row[ix_u] == rowC[ix_v]) |  (row[ix_v] == rowC[ix_v]) |
                (row[ix_v] == rowC[ix_u])): pDL.drop(rowC.Index, axis = 0, inplace = True)
                elif is_continuation(row.Index, rowC.Index, original_edges_gdf): continue
                else: 
                    pDL.drop(rowC.Index, axis = 0, inplace = True)            

            # does the line considered in the loop reach a cluster? if not straight away, at some point?
            pDL['dir'] = 'v'
            # orientate everything from "u" to "v"
            
            for rowC in pDL.itertuples():
                if rowC[ix_clus_v] == cluster:
                    line_coords = list(rowC[ix_geo].coords)
                    line_coords.reverse() 
                    new_gline = LineString([coor for coor in line_coords])
                    old_u = rowC[ix_u]
                    old_clus_u = rowC[ix_clus_u]
                    old_clus_uR = rowC[ix_clus_uR]

                    pDL.at[rowC.Index,'geometry'] = new_gline
                    pDL.at[rowC.Index,'u']  = rowC[ix_v]
                    pDL.at[rowC.Index,'v'] = old_u
                    pDL.at[rowC.Index,'clus_u'] = rowC[ix_clus_v]
                    pDL.at[rowC.Index,'clus_v'] = old_clus_u
                    pDL.at[rowC.Index,'clus_uR'] = rowC[ix_clus_vR]
                    pDL.at[rowC.Index,'clus_vR'] = old_clus_uR
                    pDL.at[rowC.Index, 'dir'] = 'u' # indicates original dir
     
            if pDL.loc[row.Index]['clus_v'] != None: pDL_goal = pDL.loc[row.Index]['clus_v']
            else: pDL_goal = pDL.loc[row.Index]['clus_vR']
            if (pDL_goal == None) | (pDL_goal == cluster): continue
            for rowC in pDL.itertuples():
                if rowC[ix_clus_v] != None: secondary_goal = rowC[ix_clus_v]
                else: secondary_goal = rowC[ix_clus_vR]
                if (secondary_goal != pDL_goal): pDL.drop(rowC.Index, axis = 0, inplace = True)
            
            done = False
            ######################################################## OPTION 1
            while (not done):
                
                if len(pDL) == 1: break # no parallel streets to row.Index 
                if len(pDL) > 4: 
                    break
                    print("cannot handle this set")
                    
                ######################################################## OPTION 2

                elif len(pDL) == 2:

                    list_nodes = []
                    c_u, c_uC = pDL.iloc[0]['clus_u'], pDL.iloc[1]['clus_u']
                    c_v, c_vC, = pDL.iloc[0]['clus_v'], pDL.iloc[1]['clus_v']
                    u, uC =  pDL.iloc[0]['u'], pDL.iloc[1]['u']
                    v, vC = pDL.iloc[0]['v'], pDL.iloc[1]['v']
                    dr, drC = pDL.iloc[0]['dir'], pDL.iloc[1]['dir']
                    gline, glineC = pDL.iloc[0]['geometry'], pDL.iloc[1]['geometry']
                    ix_line, ix_lineC  = pDL.iloc[0].name, pDL.iloc[1].name
                    lines = [ix_line, ix_lineC]

                    ######################################################## 
                    ## SUB-OPTION 1: they all reach another cluster:

                    if (c_v == c_vC) & (c_v != None):
                        lines_traversed = []
                        goal = c_v
                        
                        if (gline.length > glineC.length*1.50) & (glineC.length > gline.length*1.50):
                            print(ix_line, ix_lineC, 'not COMPLETED: OPTION 2 - SECTION 2')
                            break
                        
                        p = merge_two(ix_line, ix_lineC, gline, glineC, nodes_gdf, edges_gdf, clusters_gdf, cluster, goal, dr)
                        if p is None: 
                            print(ix_line, ix_lineC, 'not COMPLETED: OPTION 2 - SECTION 1')
                            break
                        print(ix_line, ix_lineC, 'OPTION 2 - SECTION 1')
                        to_drop = to_drop + [ix_lineC]

                    ######################################################## 
                    ## SUB-OPTION 2: only one reaches another cluster:

                    elif (c_v != None) | (c_vC != None):
                        if c_v != None: 
                            goal = c_v
                            found, glineC, lines_t, list_nodes, vC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                            lines_t.insert(0, ix_lineC)
                            last_node = vC                          
                        else: 
                            goal = c_vC
                            found, gline, lines_t, list_nodes, v = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                            lines_t.insert(0, ix_line)
                            last_node = v
                        
                        if (gline.length > glineC.length*1.50) & (glineC.length > gline.length*1.50):
                            print(ix_line, ix_lineC, 'not COMPLETED: OPTION 2 - SECTION 2')
                            break
                            
                        lines_traversed = lines_t
              
                        p = merge_two_inter(ix_line, ix_lineC, gline, glineC, nodes_gdf, edges_gdf, clusters_gdf,
                                     cluster, goal, u, last_node, list_nodes, lines_traversed)
                        if p is None: 
                            print(ix_line, ix_lineC, 'not COMPLETED: OPTION 2 - SECTION 2')
                            break
                            
                        print(ix_line, ix_lineC, 'OPTION 2 - SECTION 2')
                        to_drop = to_drop + lines_t + [ix_lineC, ix_line]
                        to_drop = list(filter(lambda a: a != ix_line, to_drop))

                    ####################################################### 
                    # SUB-OPTION 3: none reaches a cluster directly; comparing the first reached cluster
                    else: 
                        goal, gline, lines_t, nodes_en, v = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                        goalC, glineC, lines_tC, nodes_enC, vC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)  
                        common = list(set(lines_t).intersection(lines_tC))
                        if len(common) > 0: break
                        list_nodes = nodes_en + nodes_enC                       
                        lines_traversed = lines_t + lines_tC
                                            
                        # last node does not matter, as it will be reassigned to the relative cluster
                        p = merge_two_inter(ix_line, ix_lineC, gline, glineC, nodes_gdf, edges_gdf, clusters_gdf, 
                                          cluster, goal, u, v, list_nodes, lines_traversed, multi = True)
                        if p is None: 
                            print(ix_line, ix_lineC, 'not COMPLETED: OPTION 2 - SECTION 3')
                            break
                            
                        to_drop = to_drop + lines_tC + lines_t
                        to_drop = list(filter(lambda a: a != ix_line, to_drop))
                        print(ix_line, ix_lineC, 'OPTION 2 - SECTION 3') 
                          
                    clusters = [cluster, goal]    
                    between = (
                            list(original_edges_gdf.index[(original_edges_gdf.u.isin(list_nodes)) & (original_edges_gdf.v.isin(list_nodes))])+
                            list(original_edges_gdf.index[(original_edges_gdf.clus_u.isin(clusters)) & (original_edges_gdf.v.isin(list_nodes))])+
                            list(original_edges_gdf.index[(original_edges_gdf.clus_v.isin(clusters)) & (original_edges_gdf.u.isin(list_nodes))])+ 
                            list(original_edges_gdf.index[(original_edges_gdf.clus_uR.isin(clusters)) & (original_edges_gdf.v.isin(list_nodes))])+
                            list(original_edges_gdf.index[(original_edges_gdf.clus_vR.isin(clusters)) & (original_edges_gdf.u.isin(list_nodes))]))
                    
                    between = list(set(between)-set(lines_traversed)-set(lines))     
                    to_drop = to_drop + between  
                    processed = processed + [ix_line] + to_drop
                    clusters_gdf.at[clusters, 'keep'] =  True
                    if len(original_edges_gdf.loc[processed][original_edges_gdf.pedestrian == 1]) > 0: edges_gdf.at[ix_line, 'pedestrian'] = 1
#                     edges_gdf.drop(to_drop, axis = 0, inplace = True, errors = 'ignore')
                    done = True

                ####################################################### OPTION 3

                elif len(pDL) == 3:
                    list_nodes = []
                    c_u, c_uC, c_uCC = pDL.iloc[0]['clus_u'], pDL.iloc[1]['clus_u'], pDL.iloc[2]['clus_u']
                    c_v, c_vC, c_vCC = pDL.iloc[0]['clus_v'], pDL.iloc[1]['clus_v'], pDL.iloc[2]['clus_v']
                    u, uC, uCC =  pDL.iloc[0]['u'], pDL.iloc[1]['u'], pDL.iloc[2]['u']
                    v, vC, vCC = pDL.iloc[0]['v'], pDL.iloc[1]['v'], pDL.iloc[2]['v']
                    if (uC == uCC) | (uC == vCC) | (vC == uCC) | (vC == vCC): break
                    dr, drC, drCC = pDL.iloc[0]['dir'], pDL.iloc[1]['dir'], pDL.iloc[2]['dir']
                    gline, glineC, glineCC = pDL.iloc[0]['geometry'], pDL.iloc[1]['geometry'], pDL.iloc[2]['geometry']
                    ix_line, ix_lineC, ix_lineCC  = pDL.iloc[0].name, pDL.iloc[1].name, pDL.iloc[2].name            
                    lines = [ix_line, ix_lineC, ix_lineCC]
                    ######################################################## 
                    ## SUB-OPTION 1: they all reach another cluster (the same)
                    
                    if ((c_v == c_vC) & (c_v == c_vCC) & (c_v != None)):
                        goal = c_v

                        # checking length
                        if (gline.length > glineC.length*1.50) & (gline.length > glineCC.length*1.50):
                            pDL.drop(ix_line, axis = 0, inplace = True)
                        elif (glineC.length > gline.length*1.50) & (glineC.length > glineCC.length*1.50):
                            pDL.drop(ix_lineC, axis = 0, inplace = True)
                        elif (glineCC.length > gline.length*1.50) & (glineCC.length > glineC.length*1.50):
                            pDL.drop(ix_lineC, axis = 0, inplace = True)
                        else:  
                            dict_lines = {ix_line: gline, ix_lineC: glineC, ix_lineCC: glineCC}
                            
                            print(ix_line, ix_lineC, ix_lineCC, 'OPTION 3 - SECTION 1')
                            ix_line, secondary = find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal)[0:2]
                            to_drop = to_drop + secondary
                            done = True
                            # no need to change geometry here

                    ########################################################  
                    ## SUB-OPTION 2: two reach another cluster:   
                    
                    elif (((c_v == c_vC) & (c_v != None))| ((c_v == c_vCC) & (c_v != None)) | ((c_vC == c_vCC) & (c_vC != None))):

                        if (c_v == c_vC) & (c_v != None):
                            goal, glineCC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                            ix = ix_line

                        elif (c_v == c_vCC) & (c_v != None):
                            goal, glineC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                            ix = ix_line

                        elif (c_vC == c_vCC) & (c_vC != None):
                            goal, gline, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                            ix = ix_lineC
                          
                        if (gline.length > glineC.length*1.50) & (gline.length > glineCC.length*1.50):
                            pDL.drop(ix_line, axis = 0, inplace = True)
                        elif (glineC.length > gline.length*1.50) & (glineC.length > glineCC.length*1.50):
                            pDL.drop(ix_lineC, axis = 0, inplace = True)
                        elif (glineCC.length > gline.length*1.50) & (gline.length > glineC.length*1.50):
                            pDL.drop(ix_lineCC, axis = 0, inplace = True)
                        else:
                            dict_lines = {ix_line: gline, ix_lineC: glineC, ix_lineCC: glineCC}
                            ix_central, secondary_lines, cl = find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal)
                            
                            to_drop = to_drop + secondary_lines + lines_t + [ix_central]
                            to_drop = list(filter(lambda a: a != ix, to_drop))

                            lines_traversed = lines_t                                                                 
                            interpolate(u, last_node, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix)                           
                            done = True
                            print(ix_line, ix_lineC, ix_lineCC, 'OPTION 3 - SECTION 2')
                            ix_line = ix
                            
                    ########################################################  
                    ## SUB-OPTION 3: only one reaches a cluster:

                    elif (c_v != None)| (c_vC != None) | (c_vCC != None):

                        only_two = False # a line connects from the existing main lines to the cluster

                        if (c_v != None):
                            if (uC == u) | (uCC == u): only_two = True
                            else:
                                goalC, glineC, lines_tC, nodes_enC, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                                goalCC, glineCC, lines_tCC, nodes_enCC, last_nodeCC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                                lines_t, nodes_en = [], []
                                goal = c_v
                        elif (c_vC != None):
                            if (u == uC) | (uCC == uC): only_two = True
                            else:
                                goal, gline, lines_t, nodes_en, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                                goalCC, glineCC, lines_tCC, nodes_enCC, last_nodeCC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                                lines_tC, nodes_enC = [], []
                                goal = c_vC
                        else:
                            if (u == uCC) | (uC == uCC): only_two = True
                            else:
                                goal, gline, lines_t, nodes_en, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                                goalC, glineC, lines_tC, nodes_enC, last_nodeC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC) 
                                lines_tCC, nodes_enCC = [], []
                                goal = c_vCC

                        if only_two:          
                            
                            if c_v != None:
                                goal = c_v
                                if uC == u:
                                    found, glineC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                                else: #uCC = u 
                                    found, glineC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                                
                                if (gline.length > glineC.length*1.50) | (glineC.length > gline.length*1.50): break
                                cl = center_line_cluster(gline, glineC, nodes_gdf, clusters_gdf, cluster, goal)
                                to_drop = to_drop + lines_t +[ix_lineC, ix_lineCC]

                            elif c_vC != None:
                                goal = c_vC
                                if u == uC: # use CC
                                    found, gline,lines_t,list_nodes,last_node=indirect_cluster(nodes_gdf, original_edges_gdf,ix_lineCC,drCC)
                                else: #uCC = uC # use --
                                    found, gline,lines_t,list_nodes,last_node = indirect_cluster(nodes_gdf, original_edges_gdf,ix_line,dr)
                          
                                if (gline.length > glineC.length*1.50) | (glineC.length > gline.length*1.50): break
                                cl = center_line_cluster(glineC, glineC, nodes_gdf, clusters_gdf, cluster, goal)
                                ix_line = ix_lineC
                                to_drop = to_drop + lines_t +[ix_line, ix_lineCC]

                            elif c_vCC != None:
                                goal = c_vCC
                                if u == uCC: #use C
                                    found, glineC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf,ix_lineC,drC)
                                    last_node = v

                                else: # uC = uCC #use --
                                    found, glineC, lines_t, list_nodes, last_node = indirect_cluster(nodes_gdf, original_edges_gdf,ix_line,dr)
                                    last_node = vC  
                                
                                if (gline.length > glineC.length*1.50) | (glineC.length > gline.length*1.50): break
                                cl = center_line_cluster(glineCC, glineC, nodes_gdf, clusters_gdf, cluster, goal)
                                ix_line = ix_lineCC
                                to_drop = to_drop + lines_t +[ix_line, ix_lineC]
                            
                            to_drop = list(filter(lambda a: a != ix_line, to_drop))
                            edges_gdf.at[ix_line, 'counts'] = edges_gdf.loc[ix_line].counts + to_add
                            lines_traversed = lines_t
                                          
                            interpolate(u, last_node, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix_line)             
                            processed = processed + [ix_line] + to_drop
                            done = True
                            print(ix_line, ix_lineC, ix_lineCC, 'OPTION 3 - SECTION 3 - SUBSECTION2')
                        
                        else:
                            if (gline.length > glineC.length*1.50) & (gline.length > glineCC.length*1.50):
                                pDL.drop(ix_line, axis = 0, inplace = True)
                            elif (glineC.length > gline.length*1.50) & (glineC.length > glineCC.length*1.50):
                                pDL.drop(ix_lineC, axis = 0, inplace = True)
                            elif (glineCC.length > gline.length*1.50) & (glineCC.length > glineC.length*1.50):
                                pDL.drop(ix_lineCC, axis = 0, inplace = True)
                            else:
                                # exclude the 2 lines furter away                          
                                list_nodes = nodes_en + nodes_enC + nodes_enCC
                                dict_lines = {ix_line: gline, ix_lineC: glineC, ix_lineCC: glineCC}
                                
                                print(ix_line, ix_lineC, ix_lineCC, 'OPTION 3 - SECTION 3')
                                ix_central, secondary_lines, cl = find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal)
                                ix_line = ix_central
                                to_drop = to_drop + secondary_lines + lines_t + lines_tC + lines_tCC
                                lines_traversed = lines_t + lines_tC + lines_tCC
 
                                to_drop = list(filter(lambda a: a != ix_line, to_drop))
                                interpolate_multi(u, last_node, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix_line)
                                done = True
                                
                                
                    ########################################################  
                    ## SUB-OPTION 4: none reaches a cluster:
                    else: 
                        goal, gline, lines_t, nodes_en, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                        goalC, glineC, lines_tC, nodes_enC, last_nodeC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                        goalCC, glineCC, lines_tCC, nodes_enCC, last_nodeCC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                        
                        if (gline.length > glineC.length*1.50) & (gline.length > glineCC.length*1.50):
                            pDL.drop(ix_line, axis = 0, inplace = True)
                        elif (glineC.length > gline.length*1.50) & (glineC.length > glineCC.length*1.50):
                            pDL.drop(ix_lineC, axis = 0, inplace = True)
                        elif (glineCC.length > gline.length*1.50) & (glineCC.length > glineC.length*1.50):
                            pDL.drop(ix_lineCC, axis = 0, inplace = True)
                        
                        else:
                            # exclude the 2 lines furter away  
                            list_nodes = nodes_en + nodes_enC + nodes_enCC   
                            print(ix_line, ix_lineC, ix_lineCC, 'OPTION 3 - SECTION 4')
                            dict_lines = {ix_line: gline, ix_lineC: glineC, ix_lineCC: glineCC}
                            ix_central, secondary_lines, cl = find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal)
                            ix_line = ix_central
                            to_drop = to_drop + secondary_lines + lines_t + lines_tC + lines_tCC
                            lines_traversed = lines_t + lines_tC + lines_tCC
                                                                
                            to_drop = list(filter(lambda a: a != ix_line, to_drop))  
                            interpolate_multi(u, last_node, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix_line)
                            done = True
                    
                    if not done: pass
                    else:
                        clusters = [cluster, goal]
                        between = (
                        list(original_edges_gdf.index[(original_edges_gdf.u.isin(list_nodes)) & (original_edges_gdf.v.isin(list_nodes))])+
                        list(original_edges_gdf.index[(original_edges_gdf.clus_u.isin(clusters)) & (original_edges_gdf.v.isin(list_nodes))])+
                        list(original_edges_gdf.index[(original_edges_gdf.clus_v.isin(clusters)) & (original_edges_gdf.u.isin(list_nodes))])+ 
                        list(original_edges_gdf.index[(original_edges_gdf.clus_uR.isin(clusters))& (original_edges_gdf.v.isin(list_nodes))])+
                        list(original_edges_gdf.index[(original_edges_gdf.clus_vR.isin(clusters)) & (original_edges_gdf.u.isin(list_nodes))]))
                        
                        between = list(set(between)-set(lines_traversed)-set(lines))     
                        to_drop = to_drop + between
                        to_drop = list(filter(lambda a: a != ix_line, to_drop))
                        
                        processed = processed + [ix_line] + to_drop
                        clusters_gdf.at[clusters, 'keep'] = True
                        if len(original_edges_gdf.loc[processed][original_edges_gdf.pedestrian == 1]) > 0:
                            edges_gdf.at[ix_line, 'pedestrian'] = 1
#                         edges_gdf.drop(to_drop, axis = 0, inplace = True, errors = 'ignore')
                
                ######################################################## OPTION 1
                elif len(pDL) >3:
                    
#                     c_u = [pDL.iloc[i]['clus_u'] for i in range(0, len(pDL)+1)]
#                     c_v = [pDL.iloc[i]['clus_v'] for i in range(0, len(pDL)+1)]
#                     u =  [pDL.iloc[i]['u'] for i in range(0, len(pDL)+1)]   
#                     v =  [pDL.iloc[i]['v'] for i in range(0, len(pDL)+1)] 
#                     dr = [pDL.iloc[i]['dr'] for i in range(0, len(pDL)+1)] 
#                     gline = [pDL.iloc[i]['geometry'] for i in range(0, len(pDL)+1)]       
#                     ix_lines = [pDL.iloc[i].name for i in range(0, len(pDL)+1)]      
#                     nodes_en = [[] for i in range(0, len(pDL)+1)]
#                     lines_t = [[] for i in range(0, len(pDL)+1)]    
                    c_u,c_uC,c_uCC,c_uCCC = pDL.iloc[0]['clus_u'],pDL.iloc[1]['clus_u'],pDL.iloc[2]['clus_u'],pDL.iloc[3]['clus_u']
                    c_v,c_vC,c_vCC,c_vCCC = pDL.iloc[0]['clus_v'],pDL.iloc[1]['clus_v'],pDL.iloc[2]['clus_v'], pDL.iloc[3]['clus_v']
                    u, uC,uCC, uCCC =  pDL.iloc[0]['u'], pDL.iloc[1]['u'], pDL.iloc[2]['u'],pDL.iloc[3]['u']
                    v, vC, vCC, vCCC = pDL.iloc[0]['v'], pDL.iloc[1]['v'], pDL.iloc[2]['v'],pDL.iloc[3]['v']
                    dr, drC, drCC, drCCC = pDL.iloc[0]['dir'],pDL.iloc[1]['dir'],pDL.iloc[2]['dir'],pDL.iloc[3]['dir']
                    gline,glineC,glineCC, glineCCC=pDL.iloc[0]['geometry'],pDL.iloc[1]['geometry'],pDL.iloc[2]['geometry'],pDL.iloc[3]['geometry']
                    ix_line, ix_lineC, ix_lineCC, ix_lineCCC=pDL.iloc[0].name, pDL.iloc[1].name,pDL.iloc[2].name, pDL.iloc[3].name
                    nodes_en, nodes_enC, nodes_enCC, nodes_enCCC = [],[],[],[]
                    lines_t, lines_tC,lines_tCC, lines_tCCC = [],[],[],[]
                    last_node, goal = None, None
                    
                    if c_v == None:
                        goal, gline, lines_t, nodes_en, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                        lines_t.insert(0, ix_line)
                    if c_vC == None:
                        goal, glineC, lines_tC, nodes_enC, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                        lines_t.insert(0, ix_lineC)
                    if c_vCC == None:
                        goal, glineCC, lines_tCC, nodes_enCC, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCC, drCC)
                        lines_tC.insert(0, ix_lineCC)
                    if (c_vCCC == None):
                        goal, glineCCC, lines_tCCC, nodes_enCCC, last_node = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineCCC, drCCC)  
                        lines_tCC.insert(0, ix_lineCCC)
                   
                    for i in [c_v, c_vC,c_vCC,c_vCCC]: 
                        goal = i
                        if goal != None: break
                        
                        
                    if last_node == None: last_node = v
                    dict_lines = {ix_line: gline, ix_lineC: glineC, ix_lineCC: glineCC, ix_lineCCC: glineCCC}
                    cl = find_central(dict_lines, nodes_gdf, clusters_gdf, cluster, goal)[2]

                    list_nodes = nodes_en + nodes_enC + nodes_enCC+nodes_enCCC 
                    lines_traversed = lines_t + lines_tC + lines_tCC + lines_tCCC

                    to_drop = to_drop + lines_traversed +[ix_lineC, ix_lineCC, ix_lineCCC]
                    to_drop = list(filter(lambda a: a != ix_line, to_drop))
                    
                    print(ix_line, ix_lineC, ix_lineCC, ix_lineCCC, 'OPTION 4')     
                    if len(list_nodes) == 0: 
                        edges_gdf.at[ix_line, 'counts']  = main_count
                        edges_gdf.at[ix_line, 'geometry']  = cl
                    else:
                        interpolate_multi(u, last_node, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix_line) 
                    done = True
                    
                    clusters = [cluster, goal]
                    between = (
                    list(original_edges_gdf.index[(original_edges_gdf.u.isin(list_nodes))&(original_edges_gdf.v.isin(list_nodes))])+
                    list(original_edges_gdf.index[(original_edges_gdf.clus_u.isin(clusters))&(original_edges_gdf.v.isin(list_nodes))])+
                    list(original_edges_gdf.index[(original_edges_gdf.clus_v.isin(clusters))&(original_edges_gdf.u.isin(list_nodes))])+ 
                    list(original_edges_gdf.index[(original_edges_gdf.clus_uR.isin(clusters))&(original_edges_gdf.v.isin(list_nodes))])+
                    list(original_edges_gdf.index[(original_edges_gdf.clus_vR.isin(clusters))&(original_edges_gdf.u.isin(list_nodes))]))
                    
                    between = list(set(between)-set(lines_traversed)-set(lines))                                 
                    to_drop = to_drop + between
                    processed = processed + [ix_line] + to_drop
                    clusters_gdf.at[clusters, 'keep'] = True

                    if len(original_edges_gdf.loc[processed][original_edges_gdf.pedestrian == 1]) > 0:
                        edges_gdf.at[ix_line, 'pedestrian'] = 1
    
    edges_gdf.drop(to_drop, axis = 0, inplace = True, errors = 'ignore')
    edges_gdf['edgeID'] = edges_gdf.index.values.astype(int)
    nodes_gdf['nodeID'] = nodes_gdf.index.values.astype(int)
    nodes_gdf, edges_gdf = reassign_edges(nodes_gdf, edges_gdf, clusters_gdf)   
    return(nodes_gdf, edges_gdf, clusters_gdf)    


def simplify_dual_linesNodes(nodes_gdf, edges_gdf, clusters_gdf):
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    processed = []
    print('Simplifying dual lines: Second part - nodes')
    edges_gdf = assign_cluster_edges(nodes_gdf, edges_gdf)

    original_edges_gdf = edges_gdf.copy()
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    ix_u, ix_v  = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    ix_cluster = nodes_gdf.columns.get_loc("cluster")+1
    ix_clus_u, ix_clus_v  = edges_gdf.columns.get_loc("clus_u")+1, edges_gdf.columns.get_loc("clus_v")+1
    ix_clus_uR, ix_clus_vR = edges_gdf.columns.get_loc("clus_uR")+1, edges_gdf.columns.get_loc("clus_vR")+1
    
    clusters_gdf['keep'] = False
    to_drop = []
    
    for n in nodes_gdf.itertuples():
        tmp = original_edges_gdf[((original_edges_gdf.u == n[0]) | (original_edges_gdf.v == n[0]))].copy()
        for row in tmp.itertuples():
            if row.Index in processed: continue 
            if row[ix_u] == n[0]:
                goal = row[ix_clus_v]
                if goal is None: goal = row[ix_clus_vR]
            elif row[ix_v] == n[0]:
                goal = row[ix_clus_u]
                if goal is None: goal = row[ix_clus_uR]
            if goal is None: continue
                
            pDL = tmp[(tmp.clus_u == goal) | (tmp.clus_uR == goal) 
                         | (tmp.clus_v == goal) | (tmp.clus_vR == goal)].copy()
                
            # orientate everything from "u" to "v"
            pDL['dir'] = 'v'
            for g in pDL.itertuples():
                if g[ix_v] == n[0]:
                    line_coords = list(g[ix_geo].coords)
                    line_coords.reverse() 
                    new_gline = LineString([coor for coor in line_coords])
                    old_u, old_clus_u, old_clus_uR = g[ix_u], g[ix_clus_u], g[ix_clus_uR]
                    pDL.at[g[0],'geometry'] = new_gline
                    pDL.at[g[0],'u'] = g[ix_v]
                    pDL.at[g[0],'v'] = old_u
                    pDL.at[g[0],'clus_u'] = g[ix_clus_v]
                    pDL.at[g[0],'clus_v'] = old_clus_u
                    pDL.at[g[0],'clus_uR'] = g[ix_clus_vR]
                    pDL.at[g[0],'clus_vR'] = old_clus_uR
                    pDL.at[g[0], 'dir'] = 'u' # indicates original dir
                
            pDL = pDL[(pDL.clus_v == goal) | (pDL.clus_vR == goal)].copy()
            pDL = pDL[~pDL.index.isin(processed)]

            ######################################################## OPTION 1
            
            if len(pDL) == 1: continue # no parallel streets to row.Index 
                            
                ######################################################## OPTION 2
            list_nodes = []
                
            if len(pDL) == 2:                   
                c_v, c_vC, = pDL.iloc[0]['clus_v'], pDL.iloc[1]['clus_v']
                u, uC =  pDL.iloc[0]['u'], pDL.iloc[1]['u']
                v, vC = pDL.iloc[0]['v'], pDL.iloc[1]['v']
                gline, glineC = pDL.iloc[0]['geometry'], pDL.iloc[1]['geometry']
                dr, drC = pDL.iloc[0]['dir'], pDL.iloc[1]['dir']
                ix_line, ix_lineC  = pDL.iloc[0].name, pDL.iloc[1].name
                name, nameC = pDL.iloc[0]['name'], pDL.iloc[1]['name']
                if is_continuation(ix_line, ix_lineC, original_edges_gdf): pass
                else: continue

                ######################################################## 
                ## SUB-OPTION 1: they all reach another cluster:
                    
                if (c_v == c_vC) & (c_v != None):
                        
                    goal = c_v
                    if (gline.length > glineC.length *1.50) | (glineC.length > gline.length *1.50): continue
                    print(ix_line, ix_lineC, 'sub1 NODE', 'node ', n[0], 'cluster ', goal)
                    cl = center_line(gline, glineC)
                    if dr == 'u':
                        line_coords = list(cl.coords)
                        line_coords.reverse() 
                        cl = LineString([coor for coor in line_coords])
                    
                    edges_gdf.at[ix_line, 'geometry'] = cl
                    to_drop = to_drop + [ix_lineC]

                ######################################################## 
                ## SUB-OPTION 2: only one reaches another cluster:

                elif (c_v != None) | (c_vC != None):
                    if c_v != None: 
                        goal = c_v
                        found, glineC, lines_t, list_nodes, vC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)
                        lines_t.insert(0, ix_lineC)
                        ix = ix_line
                        last_node = vC
                    else: 
                        goal = c_vC
                        found, gline, lines_t, list_nodes, v = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                        lines_t.insert(0, ix_line)
                        ix = ix_lineC
                        last_node = v

                    if (gline.length > glineC.length *1.50) | (glineC.length > gline.length *1.50): continue  
                    print(ix_line, ix_lineC, 'sub2 NODE', 'node ', n[0], 'cluster ', goal)    
                    cl = center_line_cluster(gline, glineC, nodes_gdf, clusters_gdf, u, goal, one_cluster = True)
                    interpolate(u, last_node, cl, list_nodes, lines_t, nodes_gdf, edges_gdf, ix)                  
                    to_drop = to_drop + lines_t + [ix_lineC] + [ix_line]
                    to_drop = list(filter(lambda a: a != ix, to_drop))
                    ix_line = ix

                ####################################################### 
                # SUB-OPTION 3: none reaches a cluster directly; comparing the first reached cluster
                else:  
                    print(ix_line, ix_lineC, 'sub3 NODE', 'node ', n[0], 'cluster ', goal)    
                    goal, gline, lines_t, nodes_en, v = indirect_cluster(nodes_gdf, original_edges_gdf, ix_line, dr)
                    goalC, glineC, lines_tC, nodes_enC, vC = indirect_cluster(nodes_gdf, original_edges_gdf, ix_lineC, drC)    
                    if (gline.length > glineC.length *1.50) | (glineC.length > gline.length *1.50): continue
                                        
                    lines_t.insert(0, ix_line)
                    lines_t.insert(0, ix_lineC)
                    # the center line is built in relation to the variable cluster as 'u', or from_node --> to_node
                    cl =  center_line_cluster(gline, glineC, nodes_gdf, clusters_gdf, u, goal, one_cluster = True)
                    # last node does not matter, as it will be reassigned to the relative cluster
                    list_nodes = nodes_en + nodes_enC
                    lines_traversed = lines_t + lines_tC
                    interpolate_multi(u, v, cl, list_nodes, lines_traversed, nodes_gdf, edges_gdf, ix_line)
                    to_drop = to_drop+ lines_t + lines_tC + [ix_lineC]
                    to_drop = list(filter(lambda a: a != ix_line, to_drop))

                processed = processed + [ix_line] + to_drop
                
                if len(original_edges_gdf.loc[processed][original_edges_gdf.pedestrian == 1]) > 0: edges_gdf.at[ix_line, 'pedestrian'] = 1
#                 edges_gdf.drop(to_drop, axis = 0, inplace = True, errors = 'ignore')
                clusters_gdf.at[goal, 'keep'] = True
                continue
    
    edges_gdf.drop(to_drop, axis = 0, inplace = True, errors = 'ignore')
    nodes_gdf, edges_gdf = reassign_edges(nodes_gdf, edges_gdf, clusters_gdf)            
    edges_gdf['edgeID'] = edges_gdf.index.values.astype(int)
    nodes_gdf['nodeID'] = nodes_gdf.index.values.astype(int)
    nodes_gdf.drop(['cluster'], axis = 1, inplace = True)
    return(nodes_gdf, edges_gdf)

def reassign_edges(nodes_gdf, edges_gdf, clusters_gdf):
    
    print("Assigning centroids coordinates")
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    edges_gdf = edges_gdf.rename(columns = {'u':'old_u'})
    edges_gdf = edges_gdf.rename(columns = {'v':'old_v'})
    
    edges_gdf['u'], edges_gdf['v'] = 0, 0
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1 
    ix_old_u, ix_old_ = edges_gdf.columns.get_loc("old_u")+1, edges_gdf.columns.get_loc("old_v")+1 
    ix_geo = edges_gdf.columns.get_loc("geometry")+1 
    
    ix_cluster = nodes_gdf.columns.get_loc("cluster")+1 
    ix_x, ix_y = clusters_gdf.columns.get_loc("x")+1, clusters_gdf.columns.get_loc("y")+1
    ix_centroid = clusters_gdf.columns.get_loc("geometry")+1
    ix_check = clusters_gdf.columns.get_loc("keep")+1
    
    for row in edges_gdf.itertuples():
            
        line_coords = list(row[ix_geo].coords)
        u = nodes_gdf.loc[row[ix_old_u]]["cluster"]
        v = nodes_gdf.loc[row[ix_old_v]]["cluster"]
        old_u = row[ix_old_u]
        old_v = row[ix_old_v]
        
        if ((u != None) & (v != None)):  # change starting and ending node in the list of coordinates for the line
                if ( clusters_gdf.loc[u].keep) & (not clusters_gdf.loc[v].keep): 
                    u = old_u
                    v = old_v
                elif not clusters_gdf.loc[v].keep:
                    v = old_v
                    line_coords[0] = (clusters_gdf.loc[u]['x'], clusters_gdf.loc[u]['y'])
                elif not clusters_gdf.loc[u].keep: 
                    u = old_u    
                    line_coords[-1] = (clusters_gdf.loc[v]['x'], clusters_gdf.loc[v]['y'])
                else:
                    line_coords[0] = (clusters_gdf.loc[u]['x'], clusters_gdf.loc[u]['y'])
                    line_coords[-1] = (clusters_gdf.loc[v]['x'], clusters_gdf.loc[v]['y'])

        elif ((u is None) & (v is None)):  # maintain old_u and old_v
                u = old_u
                v = old_v
        elif ((u is None) & (v != None)) : # maintain old_u
                u = old_u
                if not clusters_gdf.loc[v].keep: v = old_v
                else: line_coords[-1] = (clusters_gdf.loc[v]['x'], clusters_gdf.loc[v]['y'])

        else: #(( u =! None) & (v == None) !: # maintain old_v
                v = old_v
                if not clusters_gdf.loc[u].keep: u = old_u
                else: line_coords[0] = (clusters_gdf.loc[u]['x'], clusters_gdf.loc[u]['y'])

        gline = (LineString([coor for coor in line_coords]))
        if u == v: 
            edges_gdf.drop(row.Index, axis = 0, inplace = True)
            continue
            
        edges_gdf.at[row.Index,"u"] = u
        edges_gdf.at[row.Index,"v"] = v
        edges_gdf.at[row.Index,"geometry"] = gline

    edges_gdf.drop(['old_u', 'old_v'], axis = 1, inplace=True)
    edges_gdf['u'] = edges_gdf['u'].astype(int)
    edges_gdf['v'] = edges_gdf['v'].astype(int)
    nodes_gdf['x'] = nodes_gdf['x'].astype(float)
    nodes_gdf['y'] = nodes_gdf['y'].astype(float)
       
    for row in clusters_gdf.itertuples():
        if not row[ix_check]: continue
        oldIDs = list(nodes_gdf[nodes_gdf.cluster == row.Index]['oldIDs'])
        oldIDs = [item for sublist in oldIDs for item in sublist]
               
        nodes_gdf.at[row.Index, 'x'] = row[ix_x]
        nodes_gdf.at[row.Index, 'y'] = row[ix_y]
        nodes_gdf.at[row.Index, 'geometry'] = row[ix_centroid]
        nodes_gdf.at[row.Index, 'nodeID'] = row.Index
        nodes_gdf.at[row.Index, 'oldIDs'] = oldIDs
        nodes_gdf.at[row.Index, 'cluster'] = None
        
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf.drop(['clus_u','clus_v', 'clus_uR', 'clus_vR'], axis = 1, inplace = True)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True)
    print("Done") 
    return(nodes_gdf, edges_gdf)
          
def simplify_pipeline(nodes_gdf, edges_gdf, radius = 15):
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, remove_disconnected_islands = True, same_uv_edges = True, 
        self_loops = False)
    nodes_gdf, edges_gdf = simplify_dual_lines_junctions(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = simplify_complex_junctions(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf, clusters, buffers = extract_centroids(nodes_gdf, edges_gdf, radius = radius)
    nodes_gdf, edges_gdf, clusters = simplify_dual_lines(nodes_gdf, edges_gdf, clusters)
    nodes_gdf, edges_gdf = simplify_dual_lines_junctions(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = simplify_complex_junctions(nodes_gdf, edges_gdf)
    
    return nodes_gdf, edges_gdf
    
          

  
         
    
    
    
    
    
    
