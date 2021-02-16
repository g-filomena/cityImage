"""Unit tests for the package."""

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd

from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon


import cityImage as ci

# define queries to use throughout tests

place = "Susa, Italy"
download_method = 'OSMplace'
epsg = 3003
OSMPolygon = 'Susa (44287)'
address = 'Susa, via Roma 1'
distance = 1500
location = (45.1383, 7.0509)
nodes_gdf, edges_gdf = None, None
graph = None

## Test load.py
def test_loadOSM():
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg)
    polygon = ci.convex_hull_wgs(edges_gdf)
    nodes_gdf2, edges_gdf2 = ci.get_network_fromOSM(polygon, 'polygon', network_type = "all", epsg = epsg)
    nodes_gdf3, edges_gdf3 = ci.get_network_fromOSM(address, 'distance_from_address', network_type = "all", epsg = epsg, distance = distance)
    
def test_loadSHP_topology():  
    epsg = 2019 
    input_path = 'tests/input/York_street_network.shp'
    dict_columns = {"roadType_field": "type",  "direction_field": "oneway", "speed_field": "maxspeed", "name_field": "name"}    
    nodes_gdf, edges_gdf = ci.get_network_fromSHP(input_path, epsg, dict_columns = dict_columns, other_columns = [])
    # fix topology
    nodes_gdf, edges_gdf = ci.clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = True, same_uv_edges = True, self_loops = False, fix_topology = True)
    
# Test graph.py
def test_graph():

    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg)
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    multi_graph_fromGDF = ci.multiGraph_fromGDF(nodes_gdf, edges_gdf, 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

# Test barriers.py
def test_barriers(): 
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, download_method, network_type = "all", epsg = epsg)
    barriers_gdf = ci.get_barriers(place, download_method, epsg = epsg)

    # assign barriers to street network
    edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    edges_gdf_updated_sindex = edges_gdf_updated.sindex
    edges_gdf_updated = ci.along_water(edges_gdf, barriers_gdf)
    edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    edges_gdf_updated = ci.assign_structuring_barriers(edges_gdf_updated, barriers_gdf)
    
    base_map_dict = {'gdf_base_map': edges_gdf, 'base_map_alpha' : 0.4, 'base_map_lw' : 1.1, 'base_map_zorder' : 0}
    plot = ci.plot_barriers(barriers_gdf, lw = 1.1, legend = True, axis_frame = False, black_background = True, fig_size = 15, **base_map_dict)

# Test angles.py
def test_angles():
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg)
    tmp = edges_gdf[(edges_gdf.u == 0) | (edges_gdf.v == 0)]
    
    line_geometryA = tmp.iloc[0].geometry
    line_geometryB = tmp.iloc[1].geometry
    difference = ci.difference_angle_line_geometries(line_geometryA, line_geometryB)
    
    angle = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = False, angular_change = False)
    angle_deflection = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = True, angular_change = False)
    angle_angular = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = True, angular_change = True)
    
# Test centrality.py
def test_centrality():
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg)
    nodes_gdf, edges_gdf = ci.clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_disconnected_islands = True, same_uv_edges = True, self_loops = True, fix_topology = False)
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    weight = 'length'
    
    services = ox.geometries_from_address(address, tags = {'amenity':True}, dist = distance).to_crs(epsg=epsg)
    services = services[services['geometry'].geom_type == 'Point']
    graph = ci.weight_nodes(nodes_gdf, services, graph, field_name = 'services', radius = 50)
    rc = ci.reach_centrality(graph,  weight = weight, radius = 400, attribute = 'services')
    
    measure = 'betweenness_centrality'
    Bc = ci.centrality(graph, nodes_gdf, measure = 'betweenness_centrality', weight = weight, normalized = False)
    Sc = ci.centrality(graph, nodes_gdf, measure = 'straightness_centrality', weight = weight, normalized = False)
    Cc = ci.centrality(graph, nodes_gdf, measure = 'closeness_centrality', weight = weight, normalized = False)
    Ic = ci.centrality(graph, nodes_gdf, measure = 'information_centrality', weight = weight, normalized = False)
    
    # Appending the attributes to the geodataframe
    dicts = [Bc, Sc, Cc, Ic]
    columns = ['Bc', 'Sc', 'Cc', 'Ic']
    for n, c in enumerate(dicts): 
        nodes_gdf[columns[n]] = nodes_gdf.nodeID.map(c)
    
    cmap = ci.kindlmann()
    base_map_dict = {'gdf_base_map': edges_gdf, 'base_map_alpha' : 0.4, 'base_map_lw' : 1.1, 'base_map_zorder' : 0}
    
    plot = ci.plot_gdf(gdf = nodes_gdf, column = 'Bc', black_background = True, fig_size = 15, scheme = 'Lynch_Breaks', 
        cmap = cmap, legend = True, axis_frame = True, ms = 25, **base_map_dict) 
    
    cmap = ci.cmap_three_colors('yellow', 'orange', 'red')
    plot = ci.plot_gdf_grid(gdf = nodes_gdf, columns = columns, black_background = True, fig_size = 15, scheme = 'Lynch_Breaks', 
        cmap = cmap, legend = True, ms_factor = 30) 
    
    Eb = nx.edge_betweenness_centrality(graph, weight = weight, normalized = False)
    edges_gdf = ci.append_edges_metrics(edges_gdf, graph, [Eb], ['Eb'])
    
    cbar_dict = {'cbar' : True, 'cbar_ticks' : 2, 'cbar_max_symbol' : True, 'only_min_max' : True}
    cmap = ci.cmap_two_colors('red', 'blue')
    plot_edges = ci.plot_gdf(edges_gdf, column = 'Eb', black_background = True, fig_size = 15, scheme = 'Fisher_Jenks', cmap = cmap, norm = None, legend = False, **cbar_dict)

def test_clean_network():
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg)
    nodes_gdf, edges_gdf = ci.simplify_pipeline(nodes_gdf, edges_gdf, radius = 12)

def test_landmarks():
    
    epsg = 3003
    buildings_gdf = ci.get_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg)
    buildings_gdf_address = ci.get_buildings_fromOSM(address, download_method = 'distance_from_address', epsg = epsg, distance = 1000)
    buildings_gdf_point = ci.get_buildings_fromOSM(location, download_method = 'from_point', epsg = epsg, distance = 1000)
    historical = ci.get_historical_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg)
    
    epsg = 25832
    input_path = 'tests/input/Muenster_buildings.shp'
    
    buildings_shp, _ = ci.get_buildings_fromSHP(input_path, epsg = epsg, height_field = 'height', base_field = 'base', land_use_field = 'land_use')
    # buildings_attributes = ci.attach_attributes(buildings_gdf, attributes_gdf, height_field, base_field, land_use_field)
    
    buildings_shp, _ = ci.get_buildings_fromSHP(input_path, epsg = epsg, height_field = 'height', base_field = 'base', land_use_field = 'land_use')
    sight_lines = gpd.read_file('tests/input/Muenster_sight_lines.shp')
    buildings_shp, _ = ci.visibility_score(buildings_shp, sight_lines = sight_lines)
    
    _, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "drive", epsg = epsg)
    buildings_gdf = ci.structural_score(buildings_gdf, buildings_gdf, edges_gdf, max_expansion_distance = 100, distance_along = 50, radius = 100)   
    
    buildings_gdf = ci.cultural_score_from_OSM(buildings_gdf)
    buildings_gdf, _ = ci.visibility_score(buildings_gdf)
    buildings_gdf['land_use'] = buildings_gdf['land_use_raw']
    buildings_gdf = ci.pragmatic_score(buildings_gdf, radius = 200)
    
    g_cW = {'vScore': 0.50, 'sScore' : 0.30, 'cScore': 0.20, 'pScore': 0.10}
    g_iW = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.30, '2dvis':0.30, 'neigh': 0.20 , 'road': 0.20}

    l_cW = {'vScore': 0.25, 'sScore' : 0.35, 'cScore':0.10 , 'pScore': 0.30}
    l_iW = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.40, '2dvis': 0.00, 'neigh': 0.30 , 'road': 0.30}
    
    buildings_gdf = ci.compute_global_scores(buildings_gdf, g_cW, g_iW)
    buildings_gdf = ci.compute_local_scores(buildings_gdf, l_cW, l_iW, radius = 1500)

    cmap = ci.random_colors_list_rgb(nlabels = len(buildings_gdf.land_use.unique()))
    plot_buildings = ci.plot_gdf(buildings, column = 'land_use', black_background = True, fig_size = 15, cmap = cmap, norm = None, legend = True)
    plot_flat = ci.plot_gdf(buildings, black_background = True, fig_size = 15)
    
    
def test_regions():
        
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "drive", epsg = epsg)
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    
    dual_regions = ci.identify_regions(dual_graph, edges_gdf, weight = None)
    primal_regions = ci.identify_regions_primal(graph, nodes_gdf, weight = None)
    
    polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = False, buffer = 30)
    polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = True, buffer = 30)
    edges_updated = ci.districts_to_edges_from_nodes(primal_regions, edges_gdf, 'p_topo')
    nodes_updated = ci.district_to_nodes_from_edges(nodes_gdf, dual_regions, 'p_topo')
    
    nodes_gdf_ped, edges_gdf_ped = ci.get_network_fromOSM(place, 'OSMplace', network_type = "walk", epsg = epsg)
    nodes_gdf_ped = ci.district_to_nodes_from_polygons(nodes_gdf_ped, polygons_gdf, 'p_topo')
    min_size_district = 10
    nodes_gdf_ped = ci.amend_nodes_membership(nodes_gdf_ped, edges_gdf_ped, 'p_topo', min_size_district = min_size_district)
    nodes_gdf_ped = ci.find_gateways(nodes_gdf_ped, edges_gdf_ped, 'p_topo')

