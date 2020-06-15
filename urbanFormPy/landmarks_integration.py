import pandas as pd
import numpy as np 
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import cascaded_union
pd.set_option("precision", 10)

from .utilities import *
from .angles import*

def assign_local_landmarks_to_nodes(nodes_gdf, buildings_gdf, radius = 50):
    """
    The function assigns a set of adjacent buildings (within a certain radius) to each node in a nodes GeoDataFrame.
    It moreover stores the local, scaled landmarkness scores of the adjacent buildings.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network   
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    radius: float
        The radius which regulates the search of adjacent buildings
        
    Returns
    -------
    GeoDataFrame
    """
    
    sindex = buildings_gdf.sindex
    # list of local landmarks at each junction and their scores
    nodes_gdf[["loc_land", "loc_scor"]] = nodes_gdf.apply(lambda row: _find_local_landmarks(row["geometry"], buildings_gdf, sindex, radius), axis = 1, result_type="expand")
    return nodes_gdf
    
def _find_local_landmarks(node_geometry, buildings_gdf, buildings_gdf_sindex, radius):
    """
    The function finds the set of adjacent buildings (within a certain radius) for a given node.
    It moreover stores the local, scaled scores of such adjacent buildings.
     
    Parameters
    ----------
    node_geometry: Point
        The geometry of the node considered
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    buildings_gdf_sindex: Rtree spatial index
        The spatial index on the GeoDataFrame of the buildings of a city
    radius: float
        The radius which regulates the search of adjacent buildings
        
    Returns
    -------
    tuple
    """
    
    list_local, list_scores = [], []
    b = node_geometry.buffer(radius)    
    possible_matches_index = list(buildings_gdf_sindex.intersection(b.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(b)]
    
    if len(precise_matches) == 0: 
        pass
    else:
        precise_matches = precise_matches.sort_values(by = "lScore_sc", ascending = False).reset_index()
        precise_matches = precise_matches.round({"lScore_sc":3})
        list_local = precise_matches["buildingID"].tolist()
        list_scores = precise_matches["lScore_sc"].tolist()
    
    return list_local, list_scores

def assign_anchors_to_nodes(nodes_gdf, buildings_gdf, radius = 2000, threshold = 0.3):
    """
    The function assigns a set of anchoring or orienting landmark (within a certan radius) to each node in a nodes GeoDataFrame.
    Amongst the buildings GeoDataFrame, only the one with a global landmark score higher than a certain threshold are kept.
    The global landmarks around a node, within the radius, may work as orienting landmarks towards the node, if the landmark is visible in other locations across the city.
    The function moreover stores the distances of each of the global landmarks from the considered node.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    radius: float
        It determintes the area within which the point of references for a node are searched
    threshold: float
        It regulates the selection of global_landmarks buildings (lanmdarks), by filtering out buildings whose global landmark score is lower than the argument.
        
    Returns
    -------
    GeoDataFrame
    """

    global_landmarks = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    global_landmarks = global_landmarks.round({"gScore_sc":3})
    sindex = global_landmarks.sindex

    nodes_gdf[["anchors", "distances"]] = nodes_gdf.apply(lambda row: _find_anchors(row["geometry"], global_landmarks, sindex, radius), axis = 1, result_type="expand")

    return nodes_gdf
    
def _find_anchors(node_geometry, global_landmarks, global_landmarks_sindex, radius):
    """
    The function finds the set of anchoring or orienting landmark (within a certan radius) to a given node.
    The function moreover computes the distances of each of the global landmarks from the given node.
     
    Parameters
    ----------
    node_geometry: Point
        The geometry of the node considered
    global_landmarks: Polygon GeoDataFrame
        The GeoDataFrame of the global_landmarks buildings of a city
    global_landmarks_sindex: Rtree spatial index
        The spatial index on the GeoDataFrame of the global landmarks of a city
    radius: float
        It determintes the area within which the point of references for a node are searched
    
    Returns
    -------
    tuple
    """

    # anchors around the node (intended as a destination)
    list_anchors, list_distances = [], []
    b = node_geometry.buffer(radius) 
    possible_matches_index = list(global_landmarks_sindex.intersection(b.bounds))
    possible_matches = global_landmarks.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(b)]
    if len(precise_matches) == 0: 
        pass
    else:  
        precise_matches["dist"] = precise_matches.apply(lambda row: node_geometry.distance(row.geometry), axis=1)
        precise_matches = precise_matches.round({"dist":3})
        precise_matches.sort_values(by = "gScore_sc", ascending = False, inplace = True)
        anchors = precise_matches.iloc[0:5] # first seven rows of dataframe
        list_anchors = anchors["buildingID"].tolist()
        list_distances = anchors["dist"].tolist()
    
    return list_anchors, list_distances
        
def assign_3d_visible_landmarks_to_nodes(nodes_gdf, buildings_gdf, sight_lines, threshold = 0.3):
    """
    The function assigns to each node in a nodes GeoDataFrame the set of visibile buildings, on the basis of pre-computed 3d sight_lines.
    Only global landmarks, namely buildings with global landmarkness higher than a certain threshold, are considered as buildings.
    It moreover stores the global, scaled scores of the visible buildings.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    sight_lines: LineString GeoDataFrame
        The GeoDataFrame of 3d sight lines from nodes to buildings. 
        The nodeID and buildingID fields are expected to be in this GeoDataFrame, referring respectively to obeserver and target of the line
    threshold: float
        It regulates the selection of global_landmarks buildings (lanmdarks), by filtering out buildings whose global landmark score is lower than the argument
   
    Returns
    -------
    GeoDataFrame
    """
    
    global_landmarks = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    global_landmarks = global_landmarks.round({"gScore_sc":3})
    index_global_landmarks = buildings_gdf["buildingID"].values.astype(int) 
    sight_lines_to_global_landmarks = sight_lines[sight_lines["buildingID"].isin(index_global_landmarks)]

    nodes_gdf[["dist_land","dist_scor"]] = nodes_gdf.apply(lambda row: _find_visible_landmarks(row["nodeID"], global_landmarks, 
                                                        sight_lines_to_global_landmarks), axis = 1, result_type="expand")
    return nodes_gdf
    
def _find_visible_landmarks(nodeID, global_landmarks, sight_lines_to_global_landmarks):
    """
    The function finds the set of visibile buildings from a certain node, on the basis of pre-computed 3d sight_lines.
    Only global landmarks, namely buildings with global landmarkness higher than threshold, are considered as buildings.
    It moreover stores the global, scaled scores of the visible buildings.
     
    Parameters
    ----------
    nodesID: int
        The nodeID of the node considered
    global_landmarks: Polygon GeoDataFrame
        The GeoDataFrame of buildings considered global landmarks
    sight_lines_to_global_landmarks: LineString GeoDataFrame
        The GeoDataFrame of 3d sight lines from nodes to buildings (only landmarks).
        The nodeID and buildingID fields are expected to be in this GeoDataFrame, referring respectively to obeserver and target of the line
        
    Returns
    -------
    tuple
    """

    # per each node, the sight lines to the global_landmarks landmarks are extracted.  
    global_landmarks_list, scores_list = [], []
    sight_node = sight_lines_to_global_landmarks[sight_lines_to_global_landmarks["nodeID"] == nodeID] 
    ix_global_landmarks_node = list(sight_node["buildingID"].values.astype(int))
    global_landmarks_from_node = global_landmarks[global_landmarks["buildingID"].isin(ix_global_landmarks_node)] 
    global_landmarks_from_node.sort_values(by = "gScore_sc", ascending = False, inplace = True)
    if len(global_landmarks_from_node) == 0: pass
    else:
        global_landmarks_list = global_landmarks_from_node["buildingID"].tolist()
        scores_list = global_landmarks_from_node["gScore_sc"].tolist()
        while len(str(global_landmarks_list)) > 254:
            del global_landmarks_list[-1]
            del scores_list[-1]             

    return global_landmarks_list, scores_list

def nodes_openess_polygons(nodes_gdf, buildings_gdf, distance_along = 50, max_distance_from_node = 600):
    """
    The function creates a Polygons GeoDataFrame in which each polygon represent the 2d visibility (openess) polygon of each node in a nodes GeoDataFrame.
    A node's visibility polygon is built based on visibiliy lines, vector built on the basis of the parameter "distance_along" (origin = node geometry, 
    position dependeing on "max_distance_from_node" and "distance_along"). Essentially, from a node several vectors are built in every direction around it. Each of their position points, is separated by
    "distance_along" meters; interesections with possible obstructing buildings are identified and a polygon is then finalised. 
    The paramter "max_distance_from_node" 
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city, with landmark scores
    distance_along: float
        The parameter defines the distance between each visibility line (vector) destination point in the space around the node (origin). 
        The lower the number, more the lines, more precise the polygon, slower the function
    max_distance_from_node: float
        It regulates the maximum expansion of the polygon from the node
        
    Returns
    -------
    GeoDataFrame
    """
    
    # visibility_polygons = pd.DataFrame(columns = ["nodeID", "geometry"])
    sindex = buildings_gdf.sindex
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["openess_polygon"] = nodes_gdf.apply(lambda row: _openess_polygon(row["geometry"], sindex, distance_along, max_distance_from_node), axis = 1)
    
    visibility_polygons_gdf = gpd.GeoDataFrame(nodes_gdf["nodeID"], crs = nodes_gdf.crs, geometry = nodes_gdf["openess_polygon"])
    return visibility_polygons_gdf
    
def _openess_polygon(node_geometry, buildings_gdf_sindex, distance_along, max_distance_from_node):
    """
    This function creates a Polygon which represents the 2d visibility polygon, or the openess, of a given node.
    A node's visibility polygon is built based on visibiliy lines, here computed on the basis of the parameter "distance_along". Essentially, from the node,
    several lines are built around the node considered every x meters; interesections with possible obstructing buildings are identified and a polygon is then finalised. 
    The paramter "max_distance_from_node" regulates the maximum expansion of the polygon from the node.
    
    Parameters
    ----------
    nodes_geometry: Point
        The geometry of the node considered 
    buildings_gdf_sindex: Rtree spatial index
        The spatial index on the GeoDataFrame of the buildings of a city
    distance_along: float
        The parameter defines the distance between each visibility line (vector) destination point in the space around the node (origin). 
        The lower the number, more the lines, more precise the polygon, slower the function
    max_distance_from_node: float
    
    Returns
    -------
    Polygon
    """

    origin = (float("{0:.10f}".format(node_geometry.coords[0][0])), float("{0:.10f}".format(node_geometry.coords[0][1])))
    possible_obstacles_index = list(sindex.intersection(node_geometry.buffer(max_distance_from_node).bounds))
    possible_obstacles = buildings_gdf.iloc[possible_obstacles_index]
    
    start = 0.0
    i = start
    list_lines = []
    
    while(i <= 360):
        coords = get_coord_angle(origin, distance = max_distance_from_node, angle = i)
        line = LineString([node_geometry, Point(coords)])
        obstacles = possible_obstacles[possible_obstacles.crosses(line)]

        if len(obstacles) == 0: 
            lineNew = line
        else:
            ob = cascaded_union(obstacles.geometry)
            t = line.intersection(ob)
            try: 
                intersection = t[0].coords[0]
            except: 
                intersection = t.coords[0]

            lineNew = LineString([node_geometry, Point(intersection)])

        list_lines.append(lineNew)
        i = i+distance_along

    list_points = [node_geometry]
    for i in list_lines: 
        list_points.append(Point(i.coords[1]))
    list_points.append(node_geometry)
    poly = Polygon([[p.x, p.y] for p in list_points])
    
    return poly
    

def compute_2d_visibility(nodes_gdf, buildings_gdf, max_distance_node_to_building = 300):
    """
    The function determines for each node in a nodes GeoDataFrame the relative visible buildings in a buildings GeoDataFrame,
    on the basis of 2d lines of visibility. It returns a dictionary where keys represent nodes' IDs and items are lists of visible buildings' IDs.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city
    max_distance_node_to_building: float
        It regulates the search space from the node
        
    Returns
    -------
    dictionary
    """
    
    nodes_gdf = nodes_gdf.copy()
    buildings_gdf = buildings_gdf.copy()
    
    nodes_gdf.set_index("nodeID", drop = False, inplace = True, append = False)
    del nodes_gdf.index.name
    buildings_gdf.set_index("buildingID", drop = False, inplace = True, append = False)
    del buildings_gdf.index.name
    
    sindex_n = nodes_gdf.sindex
    sindex_b = buildings_gdf.sindex
    ix_geo_b = buildings_gdf.columns.get_loc("geometry")+1
    ix_geo_n = nodes_gdf.columns.get_loc("geometry")+1
    d = {el:[] for el in nodes_gdf.nodeID}
    interval = max_distance_node_to_building
    
    for row_b in buildings_gdf.itertuples(): 
        exteriors = row_b[ix_geo_b].exterior
        coords = list(exteriors.coords)
        no_holes = Polygon(coords)

        possible_obstacles_index = list(sindex_b.intersection(no_holes.buffer(interval).bounds))
        possible_obstacles = buildings_gdf.iloc[possible_obstacles_index]
        obstacles = possible_obstacles[possible_obstacles.intersects(no_holes.buffer(interval))]
        obstacles.drop(row_b.Index, axis = 0, inplace = True, errors = "ignore")

        possible_nodes_index = list(sindex_n.intersection(no_holes.buffer(interval).bounds))
        possible_nodes = nodes_gdf.iloc[possible_nodes_index]
        nodes_around = possible_nodes[possible_nodes.intersects(no_holes.buffer(interval))]
        if len(nodes_around) == 0: 
            continue

        new_ring = coords.copy()
        distance_so_far = 0

        for n, i in enumerate(coords):
            if (n == 0) | (n == len(coords)-1) : 
                continue
            distance = Point(i).distance(Point(coords[n-1]))
            if distance < interval: 
                distance_so_far = distance_so_far + distance
                continue

            vertexes_to_add = int(distance/interval)
            index = new_ring.index(i)

            for v in range(0, vertexes_to_add):
                distance_along = distance_so_far + interval + (interval*v)
                next_vertex = exteriors.interpolate(distance_along)
                new_index = index+v
                new_ring.insert(new_index, next_vertex)

            distance_so_far = distance_so_far + distance

        new_ring = new_ring[:-1]
        no_obstacles = False
        if (len(obstacles) > 0): union = obstacles.unary_union
        else: no_obstacles = True
        
        for row_n in nodes_around.itertuples():
           
            for coord in new_ring:
                v = LineString([row_n[ix_geo_n], Point(coord)])
                if not no_obstacles: 
                    if v.intersects(union): 
                        continue        
                    
                self_intersection = v.intersection(exteriors)
                if (self_intersection.geom_type == "Point") | (self_intersection.geom_type == "GeometryCollection"): 
                    d[row_n.Index] = d[row_n.Index]+[row_b.Index]
                    break                           
                else: continue              
    return(d)
    
def build_2d_visibility_matrix(nodes_gdf, buildings_gdf, v_dict):
    """
    Given a dictionary where keys represent nodes' IDs and items are lists of relative visible buildings', the function creates a DataFrame where rows are buildings' IDs and columns are nodes' IDs.
    A cell's value is set to 1 when the row's buildingID is in the item list of the dictionary provided, at the key equal to the nodeID of the column considered. In other words, when a building (t) is 
    visible from a node (z), the value of the cell at row t and column z is set to 1, otherwise 0.
         
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The GeoDataFrame of the nodes of a street network
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame of the buildings of a city
    v_dict: dictionary
        The dictionary returned by the function "2d _visibility"
   
    Returns
    -------
    DataFrame
    """
    
    columns = ["buildingID"] + nodes_gdf["nodeID"].astype(str).tolist()
    matrix = pd.DataFrame(columns = columns)
    matrix["buildingID"] = buildings_gdf["buildingID"]
    
    for n, buildings in v_dict.items():
        mask = matrix["buildingID"].isin(buildings)
        matrix.loc[mask, str(n)] = 1
        
    matrix.fillna(value = 0, inplace = True)
    return matrix
    