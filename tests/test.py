"""Unit tests for the package."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

import osmnx as ox
import cityImage as ci

# define queries to use throughout tests

place = "Susa, Torino"
download_method = 'OSMPlace'
epsg = 3003
OSMPolygon = 'Susa (44287)'
address = 'Susa, via Roma 1'
location = (45.1383, 7.0509)
nodes_gdf, edges_gdf = None, None
graph = None

amenities = ['arts_centre', 'atm', 'bank', 'bar', 'bbq', 'bicycle_rental', 'bicycle_repair_station', 'biergarten', 
             'boat_rental', 'boat_sharing', 'brothel', 'bureau_de_change', 'cafe', 'car_rental', 'car_sharing', 'car_wash', 'casino', 'childcare',
           'cinema', 'clinic', 'college', 'community_centre', 'courthouse', 'crematorium', 'dentist', 'dive_centre', 'doctors', 
           'driving_school', 'embassy', 'fast_food', 'ferry_terminal', 'fire_station', 'food_court', 'fuel', 'gambling', 'gym', 
           'hospital', 'ice_cream', 'internet_cafe', 'kindergarten', 'kitchen', 'language_school', 'library', 'marketplace', 
           'monastery', 'motorcycle_parking', 'music_school', 'nightclub', 'nursing_home', 'pharmacy', 'place_of_worship', 
           'planetarium', 'police', 'post_office', 'prison', 'pub', 'public_building', 'ranger_station', 'restaurant', 'sauna',
           'school', 'shelter', 'shower', 'social_centre', 'social_facility', 'stripclub', 'studio', 'swingerclub', 'theatre', 
           'toilets', 'townhall', 'university', 'veterinary']


## Test load.py
def test_loadOSM():
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg, distance = None)
    self.nodes_gdf, self.edges_gdf = ci.get_network_fromOSM(OSMPolygon, 'OSMpolygon', network_type = "all", epsg = epsg, distance = None)
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(address, 'distance_from_address', network_type = "all", epsg = epsg, distance = 2000)
    
    
def test_loadSHP():   
    dict_columns = {"roadType_field": "highway",  "direction_field": "oneway", "speed_field": None, "name_field": "name"}    
    # nodes_gdf, edges_gdf = ci.get_network_fromSHP(path, epsg, dict_columns = dict_columns, other_columns = [])

## Test graph.py
def test_graph()
 
    graph_fromGDF = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID)
    self.graph = graph_fromGDF
    assert len(graph_fromGDF) == len(nodes_gdf)
    assert len(graph_fromGDF.edges()) == len(edges_gdf)
    
    # multi_graph_fromGDF = ci.multiGraph_fromGDF(nodes_gdf, edges_gdf, 'nodeID')
    
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    assert len(graph_fromGDF) == len(nodes_dual)
    assert len(graph_fromGDF.edges()) == len(edges_dual)

## Test barriers.py
def test_barriers(): 
    
    barriers = get_barriers(place, download_method, distance = None, epsg)

    # assign barriers to street network
    edges_gdf = ci.along_within_parks(edges_gdf, barriers_gdf)
    edges_gdf_sindex = edges_gdf.sindex
    edges_gdf = ci.barriers_along(ix_line, edges_gdf, barriers_gdf, edges_gdf_sindex, offset = 100)
    edges_gdf = ci.assign_structuring_barriers(edges_gdf, barriers_gdf)

## Test angles.py
def test_angles():

    tmp = edges_gdf[(edges_gdf.u == 0) | (edges_gdf.v == 0)]
    line_geometryA = tmp.iloc[0].geometry
    line_geometryB = tmp.iloc[1].geometry
    assert line_geometryA.geom_type == 'LineString'
    assert line_geometryB.geom_type == 'LineString'
    difference = ci.difference_angle_line_geometries(line_geometryA, line_geometryB)
    
    angle = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = False, angular_change = False)
    angle_deflection = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = True, angular_change = False)
    angle_angular = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, deflection = True, angular_change = True)
    
## Test centrality.py
def test_centrality():

    weight = 'length'
    sc = ci.straightness_centrality(graph, weight = weight, normalized = True)
    
    services = ox.pois.pois_from_address(address, distance = 3000, amenities=amenities).to_crs(epsg=epsg)
    services = services[services['geometry'].geom_type == 'Point']

    nodes_gdf = ci.weight_nodes(nodes_gdf, services, G, field_name = 'services', radius = 50)
    rc = reach_centrality(graph,  weight = weight, radius = 400, attribute = 'services')
    
    Eb = nx.edge_betweenness_centrality(G, weight = weight, normalized = False)
    edges_gdf = ci.append_edges_metrics(edges_gdf, G, dicts, ['Eb'])

def test_clean_network():

    clean_nodes_gdf, clean_edges_gdf = ci.clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_disconnected_islands = True, same_uv_edges = True, 
        self_loops = True, fix_topology = True)

def test_landmarks():

    buildings_gdf_shp,_ = get_buildings_fromSHP(path, epsg, case_study_area = None, height_field = None, base_field = None, distance_from_center = None)
    buildings_gdf = ci.get_buildings_fromOSM(place, download_method = 'OSMPlace', epsg = epsg, distance = None)
    buildings_gdf_address = ci.get_buildings_fromOSM(address, download_method = 'distance_from_address', epsg = epsg, distance = 1000)
    buildings_gdf_point = ci.get_buildings_fromOSM(location, download_method = 'from_point', epsg = epsg, distance = 1000)
    
    _, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg, distance = None)
    obstructions = buildings_gdf.copy()
    buildings_gdf = ci.structural_score(buildings_gdf, obstructions, edges_gdf, max_expansion_distance = 300, distance_along = 50, radius = 150)
    sight_lines = 
    buildings_gdf = ci.visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a' : []}))
    buildings_gdf = cultural_score_from_OSM(buildings_gdf)
    buildings_gdf = ci.pragmatic_score(buildings_gdf, radius = 200)
    
    g_cW = {'vScore': 0.50, 'sScore' : 0.30, 'cScore': 0.20, 'pScore': 0.10}
    g_iW = {'vis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.30, 'a_vis':0.30, 'neigh': 0.20 , 'road': 0.20}

    l_cW = {'vScore': 0.25, 'sScore' : 0.35, 'cScore':0.10 , 'pScore': 0.30}
    l_iW = {'vis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.40, 'a_vis': 0.00, 'neigh': 0.30 , 'road': 0.30}
    
    buildings_gdf =  ci.compute_global_scores(buildings_gdf, g_cW, g_iW)
    buildings_gdf = ci.compute_local_scores(buildings_gdf, l_cW, l_iW, radius = 1500)
    
def test_regions():
    


def reset_index_gdfsIG(nodes_gdf, edges_gdf):
    
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg, distance = None)
    G = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    
    dual_regions = ci.identify_regions(dual_graph, edges_gdf, weight = None):
    primal_regions = ci.dentify_regions_primal(G, nodes_gdf, weight = None)
    
    nodes_dualIG, edges_dualIG = ci.reset_index_dual_gdfsIG(nodesDual_gdf, edgesDual_gdf):
    dual_graphIG = ci.dual_graphIG_fromGDF(nodes_dualIG, edges_dualIG):
    polygonise_partition(edges_gdf, partition_field, method = None, buffer = 30):
    districts_to_edges_from_nodes(edges_gdf, nodes_gdf, district_field):
    district_to_nodes_from_edges(node_geometry, edges_gdf, sindex):
    district_to_nodes_from_polygons(node_geometry, polygons_gdf):
    find_gateways(nodeID, nodes_gdf, edges_gdf):
    check_disconnected_regions(nodes_gdf, edges_gdf, min_size):
    amend_node_membership(nodeID, nodes_gdf, edges_gdf):
