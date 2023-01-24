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
epsg_york = 2019
OSMPolygon = 'Susa (44287)'
address = 'Susa, via Roma 1'
distance = 1500
location = (45.1383, 7.0509)
nodes_gdf, edges_gdf, edges_gdf_II = None, None, None
graph = None 

# Test load.py
def test_loadOSM():
    global edges_gdf_II
    epsg_susa = 3003
    _, edges_gdf_IV = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg_susa)
    polygon = ci.convex_hull_wgs(edges_gdf_IV)
    nodes_gdf_II, edges_gdf_II = ci.get_network_fromOSM(polygon, 'polygon', network_type = "all", epsg = epsg_susa)
    nodes_gdf_III, edges_gdf_III = ci.get_network_fromOSM(address, 'distance_from_address', network_type = "all", 
                                                          epsg = epsg_susa, distance = distance)
    
def test_loadSHP_topology():  
    global nodes_gdf
    global edges_gdf
    global epsg_york
    input_path = 'C:/Users/G.Filomena/Scripts/cityImage/tests/input/York_street_network.shp'
    dict_columns = {"roadType_field": "type",  "direction_field": "oneway", "speed_field": "maxspeed", "name_field": "name"}    
    nodes_gdf, edges_gdf = ci.get_network_fromSHP(input_path, epsg_york, dict_columns = dict_columns, other_columns = [])
    # fix topology
    nodes_gdf, edges_gdf = ci.clean_network(nodes_gdf, edges_gdf, dead_ends = True, remove_islands = True, 
                                            same_uv_edges = True, self_loops = True, fix_topology = False)
    
# Test graph.py
def test_graph():
    global nodes_gdf
    global edges_gdf
    global graph
    global epsg_york
    
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    multi_graph_fromGDF = ci.multiGraph_fromGDF(nodes_gdf, edges_gdf, 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg_york, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

# Test angles.py
def test_angles():
    global edges_gdf

    nodeID = edges_gdf.iloc[0].u
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    line_geometryA = tmp.iloc[0].geometry
    line_geometryB = tmp.iloc[1].geometry
    
    angle = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'vectors')
    angle_deflection = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'deflection')
    angle_angular = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'angular_change')
    
# # Test centrality.py
def test_centrality():
    global graph
    global nodes_gdf
    global edges_gdf
    global epsg_york
    weight = 'length'
    services = ox.geometries_from_address(address, tags = {'amenity':True}, dist = distance).to_crs(epsg=epsg_york)
    services = services[services['geometry'].geom_type == 'Point']
    graph = ci.weight_nodes(nodes_gdf, services, graph, field_name = 'services', radius = 50)

    measure = 'betweenness_centrality'
    Rc = ci.reach_centrality(graph,  weight = weight, radius = 400, attribute = 'services')
    Bc = ci.centrality(graph, nodes_gdf, measure = 'betweenness_centrality', weight = weight, normalized = False)
    Sc = ci.centrality(graph, nodes_gdf, measure = 'straightness_centrality', weight = weight, normalized = False)
    Cc = ci.centrality(graph, nodes_gdf, measure = 'closeness_centrality', weight = weight, normalized = False)
    Ic = ci.centrality(graph, nodes_gdf, measure = 'information_centrality', weight = weight, normalized = False)
    
    # Appending the attributes to the geodataframe
    dicts = [Bc, Sc, Cc, Ic]
    columns = ['Bc', 'Sc', 'Cc', 'Ic']
    for n, c in enumerate(dicts): 
        nodes_gdf[columns[n]] = nodes_gdf.nodeID.map(c)
        
    Eb = nx.edge_betweenness_centrality(graph, weight = weight, normalized = False)
    edges_gdf = ci.append_edges_metrics(edges_gdf, graph, [Eb], ['Eb'])
    
def test_plot():
    global nodes_gdf
    global edges_gdf
    global edges_gdf_II
    
    tmp_nodes = nodes_gdf.copy()
    base_map_dict = {'base_map_gdf': edges_gdf, 'base_map_alpha' : 0.4, 'base_map_geometry_size' : 1.1, 'base_map_zorder' : 0}
    # Lynch's bins - only for variables from 0 to 1    
        

    ci.scaling_columnDF(tmp_nodes, 'Bc')
    scheme_dict = {'column' : "Bc_sc", 'bins' : [0.125, 0.25, 0.5, 0.75, 1.00], 'scheme' : 'User_Defined'}
    cbar_dict = {'cbar' : True, 'cbar_ticks' : 2, 'cbar_max_symbol' : True, 'cbar_min_max' : True}
    cmap = ci.kindlmann()
    plot = ci.plot_gdf(gdf = tmp_nodes, black_background = True, cmap = cmap, legend = True, axes_frame = True, 
                       geometry_size = 25, **base_map_dict, **scheme_dict) 
    
    columns = ['Bc', 'Sc', 'Cc', 'Ic']
    cmap = ci.cmap_from_colors(['yellow', 'orange', 'red'])
    plot = ci.plot_gdf_grid(gdf = tmp_nodes, columns = columns, black_background = True, cmap = cmap, legend = False, 
                           geometry_size_factor = 30, **cbar_dict) 
    
    cmap = ci.cmap_from_colors(['red', 'blue'])
    plot_edges = ci.plot_gdf(edges_gdf, column = 'Eb', black_background = True, scheme = 'Fisher_Jenks', cmap = cmap, 
                             norm = None, legend = False, **cbar_dict)
    plot_multi = ci.plot_gdfs([edges_gdf, edges_gdf_II], column = "length", black_background = True, scheme = 'Fisher_Jenks', 
                              cmap = cmap, **cbar_dict)
    
# # Test barriers.py
# def test_barriers(): 

    # barriers_gdf = ci.get_barriers(place, download_method, epsg = epsg)

    # # assign barriers to street network
    # edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    # edges_gdf_updated_sindex = edges_gdf_updated.sindex
    # edges_gdf_updated = ci.along_water(edges_gdf, barriers_gdf)
    # edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    # edges_gdf_updated = ci.assign_structuring_barriers(edges_gdf_updated, barriers_gdf)
    
    # base_map_dict = {'gdf_base_map': edges_gdf, 'base_map_alpha' : 0.4, 'base_map_lw' : 1.1, 'base_map_zorder' : 0}
    # plot = ci.plot_barriers(barriers_gdf, lw = 1.1, legend = True, axis_frame = False, black_background = True, fig_size = 15, **base_map_dict)
    
# def test_landmarks():
    
    # epsg = 3003
    # buildings_gdf = ci.get_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg)
    # buildings_gdf_address = ci.get_buildings_fromOSM(address, download_method = 'distance_from_address', epsg = epsg, distance = 1000)
    # buildings_gdf_point = ci.get_buildings_fromOSM(location, download_method = 'from_point', epsg = epsg, distance = 1000)
    # historical = ci.get_historical_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg)
    
    # epsg = 25832
    # input_path = 'input/Muenster_buildings.shp'
    
    # # buildings_shp, _ = ci.get_buildings_fromSHP(input_path, epsg = epsg, height_field = 'height', base_field = 'base', land_use_field = 'land_use')
    # # buildings_attributes = ci.attach_attributes(buildings_gdf, attributes_gdf, height_field, base_field, land_use_field)
    
    # buildings_shp, _ = ci.get_buildings_fromSHP(input_path, epsg = epsg, height_field = 'height', base_field = 'base', land_use_field = 'land_use')
    # sight_lines = gpd.read_file('input/Muenster_sight_lines.shp')
    # buildings_shp, _ = ci.visibility_score(buildings_shp, sight_lines = sight_lines)
    
    # _, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "drive", epsg = epsg)
    # buildings_gdf = ci.structural_score(buildings_gdf, buildings_gdf, edges_gdf, max_expansion_distance = 100, distance_along = 50, radius = 100)   
    
    # buildings_gdf = ci.cultural_score_from_OSM(buildings_gdf)
    # buildings_gdf, _ = ci.visibility_score(buildings_gdf)
    # buildings_gdf['land_use'] = buildings_gdf['land_use_raw']
    # buildings_gdf = ci.pragmatic_score(buildings_gdf, radius = 200)
    
    # g_cW = {'vScore': 0.50, 'sScore' : 0.30, 'cScore': 0.20, 'pScore': 0.10}
    # g_iW = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.30, '2dvis':0.30, 'neigh': 0.20 , 'road': 0.20}

    # l_cW = {'vScore': 0.25, 'sScore' : 0.35, 'cScore':0.10 , 'pScore': 0.30}
    # l_iW = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.40, '2dvis': 0.00, 'neigh': 0.30 , 'road': 0.30}
    
    # buildings_gdf = ci.compute_global_scores(buildings_gdf, g_cW, g_iW)
    # buildings_gdf = ci.compute_local_scores(buildings_gdf, l_cW, l_iW, radius = 1500)

    # cmap = ci.random_colors_list_rgb(nlabels = len(buildings_gdf.land_use.unique()))
    # plot_buildings = ci.plot_gdf(buildings_gdf, column = 'land_use', black_background = True, fig_size = 15, legend = True)
    # plot_flat = ci.plot_gdf(buildings_gdf, black_background = True, fig_size = 15)
    
    
# def test_regions():
        
    # nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "drive", epsg = epsg)
    # graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    # nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg, oneway = False, angle = 'degree')
    # dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    
    # dual_regions = ci.identify_regions(dual_graph, edges_gdf, weight = None)
    # primal_regions = ci.identify_regions_primal(graph, nodes_gdf, weight = None)
    
    # polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = False, buffer = 30)
    # polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = True, buffer = 30)
    # edges_updated = ci.districts_to_edges_from_nodes(primal_regions, edges_gdf, 'p_topo')
    # nodes_updated = ci.district_to_nodes_from_edges(nodes_gdf, dual_regions, 'p_topo')
    
    # nodes_gdf_ped, edges_gdf_ped = ci.get_network_fromOSM(place, 'OSMplace', network_type = "walk", epsg = epsg)
    # nodes_gdf_ped = ci.district_to_nodes_from_polygons(nodes_gdf_ped, polygons_gdf, 'p_topo')
    # min_size_district = 10
    # nodes_gdf_ped = ci.amend_nodes_membership(nodes_gdf_ped, edges_gdf_ped, 'p_topo', min_size_district = min_size_district)
    # nodes_gdf_ped = ci.find_gateways(nodes_gdf_ped, edges_gdf_ped, 'p_topo')

