"""Unit tests for the package."""

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd

from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
import cityImage as ci
from matplotlib.colors import LinearSegmentedColormap

# define queries to use throughout tests

place = "Susa, Italy"
download_method = 'OSMplace'
epsg_york = 2019
epsg_susa = 3003
OSMPolygon = 'Susa (44287)'
address = 'Susa, via Roma 1'
distance = 1500
location = (45.1383, 7.0509)

buildings_gdf = None
nodes_gdf, edges_gdf = None, None
nodes_gdf_y, edges_gdf_y = None, None
barriers_gdf = None
graph = None 

def test_loadOSM():
    global nodes_gdf
    global edges_gdf
    global epsg_susa
    
    nodes_gdf, edges_gdf = ci.get_network_fromOSM(place, 'OSMplace', network_type = "all", epsg = epsg_susa)
    polygon = ci.convex_hull_wgs(edges_gdf)
    _, _ = ci.get_network_fromOSM(polygon, 'polygon', network_type = "all", epsg = epsg_susa)
    _, _ = ci.get_network_fromOSM(address, 'distance_from_address', network_type = "all", 
                                                          epsg = epsg_susa, distance = distance)
    
def test_loadSHP_topology():  
    global nodes_gdf_y
    global edges_gdf_y
    global epsg_york
    input_path = 'tests/input/York_street_network.shp'
    dict_columns = {"roadType_field": "type",  "direction_field": "oneway", "speed_field": "maxspeed", "name_field": "name"}    
    nodes_gdf_y, edges_gdf_y = ci.get_network_fromSHP(input_path, epsg_york, dict_columns = dict_columns, other_columns = [])
    # fix topology
    nodes_gdf_y, edges_gdf_y = ci.clean_network(nodes_gdf_y, edges_gdf_y, dead_ends = True, remove_islands = True, 
                                                same_vertexes_edges = True, self_loops = True, fix_topology = False)
    
def test_graph():
    global nodes_gdf
    global edges_gdf
    global graph
    global epsg_susa
    
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    multi_graph_fromGDF = ci.multiGraph_fromGDF(nodes_gdf, edges_gdf, 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg_susa, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

def test_angles():
    global edges_gdf

    nodeID = edges_gdf.iloc[0].u
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    line_geometryA = tmp.iloc[0].geometry
    line_geometryB = tmp.iloc[1].geometry
    
    angle = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'vectors')
    angle_deflection = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'deflection')
    angle_angular = ci.angle_line_geometries(line_geometryA, line_geometryB, degree = True, calculation_type = 'angular_change')
    
def test_centrality():
    global graph
    global nodes_gdf
    global edges_gdf
    global epsg_susa
   
    weight = 'length'
    services = ox.geometries_from_address(address, tags = {'amenity':True}, dist = distance).to_crs(epsg=epsg_susa)
    services = services[services['geometry'].geom_type == 'Point']
    graph = ci.weight_nodes(nodes_gdf, services, graph, field_name = 'services', radius = 50)

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
    
def test_regions():
    global nodes_gdf
    global edges_gdf
    global epsg_susa
    
    graph_susa = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID = 'nodeID')
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, epsg = epsg_susa, oneway = False, angle = 'degree')
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    
    dual_regions = ci.identify_regions(dual_graph, edges_gdf, weight = None)
    primal_regions = ci.identify_regions_primal(graph_susa, nodes_gdf, weight = None)
    
    polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = False, buffer = 30)
    polygons_gdf = ci.polygonise_partitions(dual_regions, 'p_topo', convex_hull = True, buffer = 30)
    edges_updated = ci.districts_to_edges_from_nodes(primal_regions, edges_gdf, 'p_topo')
    nodes_updated = ci.district_to_nodes_from_edges(nodes_gdf, dual_regions, 'p_topo')
    
    nodes_gdf = ci.district_to_nodes_from_polygons(nodes_gdf, polygons_gdf, 'p_topo')
    min_size_district = 10
    nodes_gdf = ci.amend_nodes_membership(nodes_gdf, edges_gdf, 'p_topo', min_size_district = min_size_district)
    nodes_gdf = ci.find_gateways(nodes_gdf, edges_gdf, 'p_topo')
   
def test_barriers(): 
    global edges_gdf
    global barriers_gdf
    global place
    global epsg_susa
    
    barriers_gdf = ci.get_barriers(place, download_method, epsg = epsg_susa, parks_min_area = 200)
    # assign barriers to street network
    edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    edges_gdf_updated = ci.along_water(edges_gdf_updated, barriers_gdf)
    edges_gdf_updated = ci.along_within_parks(edges_gdf_updated, barriers_gdf)
    edges_gdf_updated = ci.assign_structuring_barriers(edges_gdf_updated, barriers_gdf)
 
def test_landmarks():
    
    global epsg_susa
    global address
    global location
    global nodes_gdf
    global edges_gdf
    global buildings_gdf
    
    _ = ci.get_buildings_fromOSM(address, download_method = 'distance_from_address', epsg = epsg_susa, distance = 1000)
    _ = ci.get_buildings_fromOSM(location, download_method = 'from_point', epsg = epsg_susa, distance = 1000)
    
    # weights
    global_indexes_weights = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.30, '2dvis':0.30, 'neigh': 0.20 , 'road': 0.20}
    global_components_weights = {'vScore': 0.50, 'sScore' : 0.30, 'cScore': 0.20, 'pScore': 0.10}   

    local_indexes_weights = {'3dvis': 0.50, 'fac': 0.30, 'height': 0.20, 'area': 0.40, '2dvis': 0.00, 'neigh': 0.30 , 'road': 0.30}
    local_components_weights = {'vScore': 0.25, 'sScore' : 0.35, 'cScore':0.10 , 'pScore': 0.30}
    
    buildings_gdf = ci.get_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg_susa)
    buildings_gdf['height'] = np.random.choice([10, 1, 50], buildings_gdf.shape[0]) 
    
    # testing with only 5 nodes, to avoid time issues
    sight_lines = ci.compute_3d_sight_lines(nodes_gdf.iloc[:5], buildings_gdf, distance_along = 300,
                                                         distance_min_observer_target = 300)
    # historical elements                                                    
    historical = ci.get_historical_buildings_fromOSM(place, download_method = 'OSMplace', epsg = epsg_susa)
    
    # scores    
    buildings_gdf = ci.structural_score(buildings_gdf, buildings_gdf, edges_gdf, advance_vis_expansion_distance = 100, neighbours_radius = 100)   

    buildings_gdf = ci.cultural_score(buildings_gdf, historical_elements_gdf = historical, from_OSM = False)
    buildings_gdf = ci.pragmatic_score(buildings_gdf, research_radius = 200)
    buildings_gdf, _ = ci.visibility_score(buildings_gdf, sight_lines)
    
    buildings_gdf = ci.compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights)
    buildings_gdf = ci.compute_local_scores(buildings_gdf, local_indexes_weights, local_components_weights, rescaling_radius = 1500)
 
 
def test_plot():
    global nodes_gdf
    global edges_gdf
    global barriers_gdf
    global buildings_gdf
    
    tmp_nodes = nodes_gdf.copy()
    base_map_dict = {'base_map_gdf': edges_gdf, 'base_map_alpha' : 0.4, 'base_map_geometry_size' : 1.1, 'base_map_zorder' : 0}
    # Lynch's bins - only for variables from 0 to 1    

    tmp_nodes['Bc_sc'] = ci.scaling_columnDF(tmp_nodes['Bc'])
    scheme_dict = {'column' : "Bc_sc", 'bins' : [0.125, 0.25, 0.5, 0.75, 1.00], 'scheme' : 'User_Defined'}
    cmap = ci.kindlmann()
    plot = ci.plot_gdf(title = 'Example', gdf = tmp_nodes, black_background = True, cmap = cmap, legend = True, axes_frame = True, 
                       geometry_size = 25, **base_map_dict, **scheme_dict, figsize = (10,5)) 
    
   
    # Appending the attributes to the geodataframe
    columns = ['Bc', 'Sc', 'Cc', 'Ic']
    # 2x2 color bar
    cbar_dict = {'cbar' : True, 'cbar_ticks' : 2, 'cbar_max_symbol' : True, 'cbar_min_max' : True, 'cbar_shrinkage' : 0.75}
    plot = ci.plot_grid_gdf_columns(gdf = tmp_nodes, columns = columns, black_background = True, cmap = cmap, legend = False, ncols = 2, 
                         nrows = 2, figsize = (15,5), geometry_size_factor = 30, axes_frame = True, **cbar_dict, titles = columns)
    
    
    #2x2 legend
    scheme_dict = {'bins' : [0.125, 0.25, 0.5, 0.75, 1.00], 'scheme' : 'User_Defined'}
    plot = ci.plot_grid_gdf_columns(gdf = tmp_nodes , columns = columns, black_background = True, cmap = cmap, nrows= 2, ncols = 2, 
                         figsize = (15,7), classes = 5, legend = True, axes_frame = True, **scheme_dict, titles = columns)
    
    # 4x2 white
    plot = ci.plot_grid_gdf_columns(gdf = tmp_nodes , columns = columns+columns, black_background = False,
                         cmap = cmap, nrows= 4, ncols = 2, classes = 6, scheme = 'quantiles', legend = True, figsize = (20, 10), 
                         axes_frame = True, fontsize = 15, titles = columns+columns)
                         
                         
    #3 x 1
    columns = ['Bc', 'Sc', 'Cc']
    plot = ci.plot_grid_gdf_columns(gdf = tmp_nodes, columns = columns, black_background = True, cmap = cmap, ncols = 1, nrows = 3,
                        classes = 5, legend = True, figsize = (9,9), **scheme_dict, axes_frame = True,
                         titles = columns)
    #1x3
    plot = ci.plot_grid_gdf_columns(gdf = tmp_nodes, columns = columns, black_background = True, cmap = cmap, ncols = 3, nrows =1,
                        classes = 5, legend = True, figsize = (12,4), **scheme_dict, axes_frame = True, titles = columns)                     
    
    # edges
    plot_edges = ci.plot_gdf(edges_gdf, column = 'Eb', black_background = True, scheme = 'Fisher_Jenks', cmap = cmap, 
                             norm = None, legend = False, **cbar_dict, figsize = (10,5), title = 'Testing', axes_frame = True)   

    # multi_gdfs
    plot_multi = ci.plot_grid_gdfs_column(gdfs = [edges_gdf, edges_gdf], column = "length", black_background = True, ncols = 2, nrows = 1,                        
                                          figsize = (10,5), scheme = 'Fisher_Jenks', cmap = cmap, **cbar_dict)  
                                                  
    barriers_gdf.sort_values(by = 'barrier_type', ascending = False, inplace = True)  
    colors = ['green', 'red', 'gray', 'blue']
    if 'secondary_road' in list(barriers_gdf['barrier_type'].unique()):
        colors.append('gold')

    base_map_dict = {'base_map_gdf': buildings_gdf, 'base_map_alpha' : 1, 'base_map_zorder' : 0, 'base_map_color' : 'yellow'}
    cmap = LinearSegmentedColormap.from_list('cmap', colors, N=len(colors)) 
    plot = ci.plot_gdf(barriers_gdf, column = 'barrier_type', geometry_size = 1.1, legend = True, axes_frame = True, 
                    black_background = True, cmap = cmap, figsize = (15, 5), title = 'Barriers',
                    **base_map_dict)                        
     
    cmap = ci.rand_cmap(nlabels = len(buildings_gdf.land_use.unique()))
    plot_buildings = ci.plot_gdf(buildings_gdf, column = 'land_use', black_background = True, legend = True, figsize = (25,10))

# def test_landuse():

    epsg = 25832
    input_path = 'tests/input/Muenster_buildings.shp'
    buildings_shp, _ = ci.get_buildings_fromSHP(input_path, epsg = epsg, height_field = 'height', base_field = 'base', land_use_field = 'land_use')
    attributes_gdf = gpd.read_file('tests/input/Muenster_buildings_attributes.shp').to_crs('EPSG:'+str(epsg))  
    
    adult_entertainment = ['brothel','casino', 'swingerclub', 'stripclub', 'nightclub', 'gambling'] 
    agriculture = ['shed', 'silo', 'greenhouse', 'stable', 'agricultural and forestry',  'greenhouse (botany)',  
                   'building in the botanical garden']
    attractions = ['attractions',   'attraction','aquarium', 'monument',  'gatehouse', 'terrace', 'tower', 'attraction and Leisure',
                   'information', 'viewpoint', 'tourist information center', 'recreation and amusement park',  'zoo',
                   'exhibition hall, trade hall', 'boathouse', 'bath house, thermal baths', 'entertainment hall', 'sauna']
    business_services = ['bank', 'service','offices', 'foundation', 'office', 'atm', 'bureau_de_change', 'post_office', 
                  'post_office;atm', 'coworking_space', 'conference_centre',  'trade and services', 'trade and services building',
                          'customs office', 'insurance', 'tax_office', 'post', 'administrative building',  'facility building',
                          'residential building with trade and services', 'data_center', 'tax office']
    commercial = [ 'commercial',  'retail', 'pharmacy', 'commercial;educa', 'shop', 'supermarket', 'books', 'commercial services',
                  'commercial land', 'car_wash', 'internet_cafe', 'driving_school', 'marketplace', 'fuel', 'car_sharing', 
                  'commercial and industry buidling',  'crematorium', 'commercial building', 'commercial and industry building',  
                  'commercial building to traffic facilities (general)', 'funeral parlor', 'gas station', 'car wash',
                  'pumping station','boat_rental', 'boat_sharing',  'bicycle_rental', 'car_rental', 'dive_centre'] 
    culture = ['club_house','gallery', 'arts_centre','cultural facility', 'cultural_centre', 'theatre', 'cinema', 'studio',
                'exhibition_centre', 'music_school', 'theater','castle', 'museum', 'culture']
    eating_drinking = ['bbq', 'restaurant', 'fast_food', 'cafe', 'bar',  'pub', 'accommodation, eating and drinking', 
                       'ice_cream', 'kitchen', 'food_court', 'cafe;restaurant', 'biergarten']
    education_research = ['university', 'research', 'university building', 'education and research', 'research_institute']
    emergency_service = [ 'fire brigade','fire_station','police', 'emergency service', 'resque_station', 'ranger_station',  
                         'security']
    general_education = ['school', 'college', 'kindergarten', 'education', 'education and health', 'childcare',
                 'language_school', 'children home',  'nursery',  'general education school']
    hospitality = [ 'hotel',  'hostel', 'guest_house',  'building for accommodation',  'hotel, motel, pension',  'refuge']
    industrial = ['industrial', 'factory', 'construction', 'manufacturing and production', 'gasometer', 'workshop', 
                  'production building']
    medical_care = ['hospital', 'doctors', 'dentist','clinic','veterinary', 'medical Care', 'nursing_home',  
                    'sanatorium, nursing home', 'retirement home', 'healthcare']
    military_detainment = ['general aviation', 'barracks',  'military', 'penitentiary', 'prison']
    other = ['toilets', 'picnic_site','hut', 'storage_tank', 'canopy','toilet',  'bunker, shelter',  'warehouse',  'converter',
             'garage', 'garages','parking'] 
    public = ['townhall', 'public_building',  'library','civic', 'courthouse', 'public', 'embassy',
              'public infrastructure', 'community_centre', 'court',  'district government', 
              'residential building with public facilities']
    religious = ['church', 'place_of_worship','convent', 'rectory', 'chapel', 'religious building', 'monastery', 'nuns home',
                  'vocational school',  'cathedral']
    residential = [ 'apartments', None, 'NaN', 'residential','flats', 'houses', 'building', 'residential land', 
                   'residential building', 'student dorm', 'building usage mixed with living']
    social = ['social_facility', 'community_centre', 'community buidling', 'dormitory', 'social_centre', 'social serives building', 
             'social services',  'community hall',  'commercial social facility',  'recreational']
    sport = ['stadium', 'sport and entertainment', 'sports or exercise facility', 'gym', 'sports building', 'sports hall', 
             'horse riding school',  'swimming pool',  'sport hall', 'bowling hall',  'indoor swimming pool']
    transport = ['transport', 'road transport', 'station', 'subway_entrance', 'bus_station', 'shipping facility building', 
                 'train_station',  'railway building']
    utilities = ['gas supply', 'electricity supply', 'electricity substation', 'waste treatment building',
                'water supply', 'waste water treatment plant', 'smokestack', 'supply systems', 'waste management', 'water works',
                  'heating plant', 'boiler house',  'telecommunication']      
          
    categories = [adult_entertainment, agriculture, attractions, business_services, commercial, culture, eating_drinking,
                  education_research, emergency_service, general_education, hospitality, industrial, medical_care, military_detainment,
                  other, public, religious, residential, social, sport, transport, utilities]
    strings = ['adult_entertainment', 'agriculture', 'attractions', 'business_services', 'commercial', 'culture', 'eating_drinking',
                'education_research', 'emergency_service', 'general_education', 'hospitality', 'industrial', 'medical_care', 
               'military_detainment', 'other', 'public', 'religious', 'residential', 'social', 'sport', 'transport', 'utilities']
     
    attributes_gdf = ci.classify_land_use(attributes_gdf, new_land_use_field = 'land_use', land_use_field = 'lu_eng', categories= categories, strings = strings)   
    attributes_gdf['land_use'] = attributes_gdf['land_use'] .str.lower()   
    buildings_gdf = ci.land_use_from_other_gdf(buildings_shp, other_gdf =  attributes_gdf, new_land_use_field = 'land_use' , land_use_field = 'land_use')
    _ = ci.polygons_gdf_multiparts_to_singleparts(attributes_gdf)
    
    


