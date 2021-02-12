import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import linemerge, polygonize_full
import statistics

pd.set_option('precision', 10)
pd.options.mode.chained_assignment = None

from .graph import nodes_degree
from .utilities import center_line, merge_lines, split_line_at_interpolation
from .clean import clean_network, correct_edges
from .angles import angle_line_geometries, is_continuation

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
               
    Two parameters depend on street morphology and the user assessment:
    - max_difference_length: indicate here the max difference in length between the two lines (segmentA's geometry and roadB's). 
                             Specify the max percente difference in float. e.g. 40% --> 0.40
    - max_distance_between_lines: float
    
    A new dataframe is returned with the simplified geometries.
    
    Parameters
    ----------
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        the street junctions GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        the street segment GeoDataFrame
    max_difference_length: float
        indicate the max difference in length between the two lines (segmentA's geometry and roadB's). Specify the max percente difference in float. e.g. 40% --> 0.40
    max_distance_between_lines: float
        the distance separating lines that could be considered as dual lines
  
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the simplified GeoDataFrames
    """

    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    nodes_gdf, edges_gdf = _check_indexes(nodes_gdf, edges_gdf)
    original_edges_gdf = edges_gdf.copy()
       
    ix_geo = edges_gdf.columns.get_loc("geometry")+1  
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_name = edges_gdf.columns.get_loc("name")+1
    processed = []
    if max_difference_length > 1.0:
        max_difference_length = 1.0
        
    # the original geometries and edges are iterated and examined;
    for row in original_edges_gdf.itertuples():
        if row.Index in processed: 
            continue  
        current_index = row.Index
        for look in [ix_u, ix_v]:
            found = False
            possible_matches = original_edges_gdf[(original_edges_gdf['u'] == row[look]) | (original_edges_gdf['v'] == row[look])].copy()
            possible_matches.drop(current_index, axis = 0, inplace = True)
            possible_matches = possible_matches[~possible_matches.index.isin(processed)]
            
            possible_matches = possible_matches[possible_matches.geometry.length <  row[ix_geo].length]
            possible_matches['continuation'] = possible_matches.apply(lambda c: is_continuation(current_index, c.name, edges_gdf), axis = 1)
            possible_mathces = possible_matches[possible_matches.continuation]
            
            if len(possible_matches) == 0: 
                continue
            if look == ix_u: 
                direction = 'v'
                to_reach = row[ix_v]    
            else: 
                direction = 'u'
                to_reach = row[ix_u]           
                    
            for connector in possible_matches.itertuples():
                
                if connector[ix_u] == row[look]: 
                    search = connector[ix_v]  
                else: search = connector[ix_u]

                nodes_traversed = [search]
                lines_traversed = [connector[ix_geo]]
                lines = [connector.Index]
                next_line = False # to determine when moving to the next candidate
                last_line = connector.Index

                while (not found) & (not next_line):
                    # look for a new possible set of connectors
                    next_possible_matches = original_edges_gdf[(original_edges_gdf['u'] == search) | (original_edges_gdf['v'] == search)].copy()      
                    next_possible_matches.drop([last_line, current_index], axis = 0, inplace = True, errors = 'ignore') # remove the previous lines, in case
                    next_possible_matches = next_possible_matches[~next_possible_matches.index.isin(processed)]

                    for other_connector in next_possible_matches.itertuples():
                        if not is_continuation(last_line, other_connector.Index, edges_gdf): 
                            next_possible_matches.drop(other_connector.Index, axis = 0, inplace = True)

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

                    if (search in nodes_traversed) | (distB < distA):           
                        next_line = True
                        continue
                    elif search == to_reach:
                        lines_traversed.append(next_possible_matches.iloc[0].geometry)
                        lines.append(next_possible_matches.iloc[0].name)
                        found = True
                        break
                    else: 
                        nodes_traversed.append(search)
                        lines_traversed.append(next_possible_matches.iloc[0].geometry)
                        lines.append(next_possible_matches.iloc[0].name)
                        last_line = next_possible_matches.iloc[0].name

                if next_line: 
                    continue
                else: 
                    break

            if not found: 
                continue # no parallel dual lines at this node
            u, v, geo = row[ix_u], row[ix_v], row[ix_geo]    
            merged_line = merge_lines(lines_traversed)
            
            # check whether it makes sense to merge or not
            if (geo.length*(max_difference_length+1) < merged_line.length) | (geo.length > merged_line.length*(max_difference_length+1)): 
                continue
            if (geo.centroid.distance(merged_line.centroid) > max_distance_between_lines):
                continue
            
            # obtaining center line
            cl = center_line(geo, merged_line)
            processed = processed + lines
            processed.append(current_index)
            if ("pedestrian" in edges_gdf.columns) & (len(edges_gdf.loc[lines][edges_gdf.pedestrian == 1]) > 0):
                edges_gdf.at[current_index, 'pedestrian'] = 1
            if direction == 'u': 
                nodes_traversed.reverse()
                
            # interpolate nodes encountered along the parallel lines
            interpolate_on_center_line(current_index, cl, nodes_gdf, edges_gdf, u, v, nodes_traversed)
            edges_gdf.drop(lines, axis = 0, inplace = True) 
            break
            
    # correct the coordinates and clean the network
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True)
    
    return(nodes_gdf, edges_gdf)

def simplify_complex_junctions(nodes_gdf, edges_gdf):
    
    """
    This function simplifies complex junctions as triangular-like junctions formed mainly by secondary links.
    The junction may be as well represented by one node rather than three nodes. 
                
    The function takes a node and check whether the intersecting edges give shape to a triangular-cyclic junction.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        the street junctions GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        the street segment GeoDataFrame
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the simplified GeoDataFrames
    """
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    nodes_gdf, edges_gdf = _check_indexes(nodes_gdf, edges_gdf)
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
            if row.Index in processed: 
                continue
            current_index = row.Index
            for other in tmp.itertuples():
                if (current_index == other.Index) | (other.Index in processed): 
                    continue
                
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
                if len(possible_matches) == 0: 
                    continue
                connector = possible_matches.iloc[0]
                
                u, v, u_other, v_other = row[ix_u], row[ix_v], other[ix_u], other[ix_v]
                geo, other_geometry, connector_geometry = row[ix_geo], other[ix_geo], connector.geometry
                if any(i > 100 for i in [geo.length, other_geometry.length, connector_geometry.length]): 
                    break # segments are too long
                
                diff_A = abs(geo.length - other_geometry.length)    
                diff_B = abs(geo.length - connector_geometry.length)
                diff_C = abs(other_geometry.length- connector_geometry.length)
                if (diff_B < diff_A) | (diff_C < diff_A): 
                    continue 
                if (diff_A > geo.length*0.75) | (diff_A > other_geometry.length*0.75):
                    continue
                if (connector_geometry.length > (geo.length + other_geometry.length)*1.25): 
                    continue  
                if (diff_A > geo.length*0.25) | (diff_A > other_geometry.length*0.25): 
                    continue
                
                if "pedestrian" in edges_gdf.columns: 
                    if edges_gdf.loc[other.Index]['pedestrian'] == 1: 
                        edges_gdf.at[current_index, 'pedestrian'] = 1
                
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
                edges_gdf.at[current_index, code] = ix_node

                if intersection.geom_type == 'Point': # check if the center line reaches the connector
                    last = intersection.coords[0]
                    line = split_line_at_interpolation(intersection, cl)[0]
                    nodes_gdf.at[ix_node, 'geometry'] = intersection
                    
                    if code == 'u': 
                        edges_gdf.at[current_index,'geometry'] = line[1]
                    else: 
                        edges_gdf.at[current_index,'geometry'] = line[0]
                    
                    line = split_line_at_interpolation(intersection, connector_geometry)[0]
                    edges_gdf.at[connector.name, 'geometry'] = line[0]
                    edges_gdf.at[connector.name, 'v'] = ix_node
                    edges_gdf.at[ix_edge, 'u'] = ix_node
                    edges_gdf.at[ix_edge, 'geometry'] = line[1]

                else: # no intersection, extend lines towards center line
                    last = list(cl.coords)[last_vertex]
                    nodes_gdf.at[ix_node, 'geometry'] = Point(last)
                    edges_gdf.at[current_index,'geometry'] = cl

                    line_geometry_A = LineString([coor for coor in [connector_geometry.coords[0], last]])
                    line_geometry_B = LineString([coor for coor in [last, connector_geometry.coords[-1]]])
                    edges_gdf.at[connector.name, 'geometry'] = line_geometry_A
                    edges_gdf.at[ix_edge, 'geometry'] = line_geometry_B
                    edges_gdf.at[connector.name, 'v'] = ix_node
                    edges_gdf.at[ix_edge, 'u'] = ix_node
                
                processed = processed + [current_index, other.Index]
                nodes_gdf.at[ix_node, 'x'] = last[0]
                nodes_gdf.at[ix_node, 'y'] = last[1]
                
                found = True
                break
                                    
            if found: 
                break
                        
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True) 
    return(nodes_gdf, edges_gdf)


def dissolve_roundabouts(nodes_gdf, edges_gdf, max_length_segment = 80, angle_tolerance = 40):
    """
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        the street junctions GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        the street segment GeoDataFrame
    max_length_segment: float
        if a segment in the possible roundbout-like junction is longer than this threshold, the junction examined is not considered a roundabout
    angle_tolerance: float
        if two segments in the possible roundbout-like junction form an angle whose magnitude is higher than this threshold, the junction examined is not
        considered a roundabout
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the simplified GeoDataFrames
    """


    nodes_gdf.index, edges_gdf.index = nodes_gdf.nodeID, edges_gdf.edgeID
    nodes_gdf.index.name, edges_gdf.index.name = None, None
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    
    ix_geo = edges_gdf.columns.get_loc("geometry")+1  
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1

    processed_segments = []
    processed_nodes = []
       
    # editing the ones which only connect three edges
    to_edit = {k: v for k, v in nodes_degree(edges_gdf).items() if v == 3}
    if len(to_edit) == 0: 
        return(nodes_gdf, edges_gdf)
    
    to_edit_gdf = nodes_gdf[nodes_gdf.nodeID.isin(list(to_edit.keys()))].copy()
        
    for node in to_edit_gdf.itertuples():

        if node in processed_nodes: 
            continue
        tmp =  edges_gdf[(edges_gdf['u'] == node.Index) | (edges_gdf['v'] == node.Index)].copy()
        found = False
        not_a_roundabout = False
        sc, sc_last_vertex = None, None
        
        # take one of these lines and examine its relationship with the others at the same junction
        for row in tmp.itertuples():
            if row[ix_geo].length > max_length_segment: 
                continue #too long for being a roundabout segment
            sequence_nodes = [node.Index]
            sequence_segments = [row.Index]
            if row.Index in processed_segments: 
                continue
            
            if row[ix_u] == node.Index: 
                last_vertex = row[ix_v]
            else: last_vertex = row[ix_u]
            
            sequence_nodes.append(last_vertex)
            segment = row
            distance = 0
            second_candidate = False

            
            while not found:
                if distance >= 400: 
                    break # too much traversed distance for a roundabout
                if last_vertex in processed_nodes: # the node has been dissolved already
                    if not second_candidate: 
                        break
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
        
                    if (segment[0] == connector.Index) | (connector.Index in processed_segments): 
                        possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                    elif connector[ix_geo].length > max_length_segment: 
                        possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                    else: 
                        angle = angle_line_geometries(segment[ix_geo], connector[ix_geo], angular_change = True, degree = True)
                        if angle > angle_tolerance: 
                            possible_connectors.drop(connector.Index, axis = 0, inplace = True)
                        else: 
                            possible_connectors.at[connector.Index, 'angle'] = angle
                    
                if (len(possible_connectors) == 0) | (last_vertex in processed_nodes):
                    if not second_candidate: 
                        break
                    else:
                        distance -= segment[ix_geo].length
                        segment = sc
                        distance += segment[ix_geo].length
                        sequence_segments[-1] = segment[0]
                        last_vertex = sc_last_vertex
                        sequence_nodes[-1] = sc_last_vertex
                        second_candidate = False
                        continue

                else: 
                    possible_connectors.sort_values(by = 'angle', ascending = True, inplace = True) 
                
                segment = list(possible_connectors.iloc[0])
                segment.insert(0, possible_connectors.iloc[0].name)
                
                if len(possible_connectors) > 1:
                    sc = list(possible_connectors.iloc[1])
                    sc.insert(0, possible_connectors.iloc[1].name)
                    second_candidate = True
                    if sc[ix_u] == last_vertex:
                        sc_last_vertex = sc[ix_v]
                    else: 
                        sc_last_vertex = sc[ix_u]
                
                if segment[ix_u] == last_vertex:
                    last_vertex = segment[ix_v]
                else: 
                    last_vertex = segment[ix_u]
                sequence_nodes.append(last_vertex)
                sequence_segments.append(segment[0])                
                distance += segment[ix_geo].length
                
                if last_vertex == node.Index:
                    lm = linemerge(edges_gdf.loc[i].geometry for i in sequence_segments)
                    roundabout = polygonize_full(lm)[0]
                    if len(roundabout) == 0:
                        not_a_roundabout = True
                        break
                    
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
            
            if not_a_roundabout:
                break
            if found: 
                break
            
    edges_gdf = correct_edges(nodes_gdf, edges_gdf)
    nodes_gdf, edges_gdf = clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_disconnected_islands = False, same_uv_edges = True, self_loops = True)         
            
    return nodes_gdf, edges_gdf
                    
def _check_indexes(nodes_gdf, edges_gdf): 

    nodes_gdf.index, edges_gdf.index = nodes_gdf.nodeID, edges_gdf.edgeID
    nodes_gdf.index.name, edges_gdf.index.name = None, None
    edges_gdf = edges_gdf.where(pd.notnull(edges_gdf), None)
          

def interpolate_on_center_line(ix_line, center_line, nodes_gdf, edges_gdf, first_node, last_node, nodes_traversed, 
                                clusters_gdf = None, clusters_traversed = []):
       
    line_geometry = center_line   
    new_index = ix_line                                                                                        
    distances = {}
    
    if len(clusters_traversed)> 0:
        nodes_traversed = nodes_traversed + clusters_traversed
    for node in nodes_traversed:
        if node in clusters_traversed: 
            node_geometry = clusters_gdf.loc[node]['geometry']
        else: 
            node_geometry = nodes_gdf.loc[node]['geometry']
        np = nearest_points(node_geometry, center_line)[1]
        distance = center_line.project(np)
        distances[node] = distance                                                                                               

    distances_sorted = sorted(distances.items(), key=lambda kv: kv[1])               
                                                                                                    
    for counter, node in enumerate(distances_sorted):
        
        node = distances_sorted[counter][0]
        if node in clusters_traversed: 
            point = clusters_gdf.loc[node].geometry
        else: 
            point = nodes_gdf.loc[node].geometry
        result, np = split_line_at_interpolation(point, line_geometry)
            
        if node in clusters_traversed:
            clusters_gdf.at[node, 'x'] = np.coords[0][0]
            clusters_gdf.at[node, 'y'] = np.coords[0][1]
            clusters_gdf.at[node, 'geometry'] = np
            if counter == 0: 
                edges_gdf.at[new_index, 'u'] = first_node
            continue
            
        nodes_gdf.at[node, 'x'] = np.coords[0][0]
        nodes_gdf.at[node, 'y'] = np.coords[0][1]
        nodes_gdf.at[node, 'geometry'] = np 
        
        #first part of the segment, adjusting node coordinates        
        tmp = edges_gdf[(edges_gdf.u == node) | (edges_gdf.v == node)].copy()
        tmp.drop(ix_line, axis = 0, inplace = True, errors = 'ignore')
                
        if counter == 0: 
            edges_gdf.at[new_index, 'u'] = first_node
        
        edges_gdf.at[new_index, 'geometry'] = result[0]
        edges_gdf.at[new_index, 'v'] = node
        edges_gdf.at[new_index, 'new_geo'] = True
          
        # second part of the segment
        new_index = max(edges_gdf.index)+1
        
        edges_gdf.loc[new_index] = edges_gdf.loc[ix_line]
        edges_gdf.at[new_index, 'geometry'] = result[1]
        edges_gdf.at[new_index, 'u'] = node
        edges_gdf.at[new_index, 'v'] = last_node
        edges_gdf.at[new_index, 'edgeID'] = new_index
        edges_gdf.at[new_index, 'new_geo'] = True
        line_geometry = result[1]      
         
    
    
    
    
    
    
