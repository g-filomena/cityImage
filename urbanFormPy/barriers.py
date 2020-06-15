import osmnx as ox, networkx as nx, pandas as pd, numpy as np, geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import cascaded_union, linemerge, polygonize, polygonize_full, unary_union, nearest_points
pd.set_option("precision", 10)

from .graph import *
from .utilities import *

"""
Set of functions to extract barriers within a certain urban space (Major roads, water bodies, parks and railway structures).
The concept of barrier refers to Lynch's "edges" in The Image of the City.
"""

def road_barriers(place, download_method, distance = None, epsg = None, include_primary = False):
    """
    The function downloads major roads from OSM. These can be considered to be barrier to pedestrian movement or, at least, to structure people's cognitive Image of the City.
    if 'include_primary' considers also primary roads, beyond motorway and trunk roads.
        
    Parameters
    ----------
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"polygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    distance: float
        it is used only if download_method == "distance from address"
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    include_primary: boolean
        it is used only if download_method_graphB == "distance from address"
        
    Returns
    -------
    LineString GeoDataFrame
    """
    
    crs = {'init': 'epsg:' + str(epsg)}
    
    if download_method == 'distance_from_address': 
        roads_graph = ox.graph_from_address(place, network_type = 'drive', distance = distance, simplify = False)
    elif download_method == 'OSMplace': roads_graph = ox.graph_from_place(place, network_type = 'drive', simplify = False)
    else: roads_graph = ox.graph_from_polygon(place, network_type = 'drive', simplify = False)
    
    roads = ox.graph_to_gdfs(roads_graph, nodes=False, edges=True, node_geometry= False, fill_edge_geometry=True)
    roads = roads.to_crs(crs)
    # exclude tunnels
    if "tunnel" in roads.columns:
        roads["tunnel"].fillna(0, inplace = True)
        roads = roads[roads["tunnel"] == 0] 
    
    # resolving lists 
    roads["highway"] = [x[0] if type(x) == list else x for x in roads["highway"]]        
    roads.drop(['osmid', 'oneway', 'bridge', 'tunnel'], axis = 1, inplace = True, errors = 'ignore')
    main_types = ['trunk', 'motorway']
    if include_primary == True: main_types = ['trunk', 'motorway', 'primary']
    roads = roads[roads.highway.isin(main_types)]
    roads = roads.unary_union
    roads = linemerge(roads)
    if roads.type != "LineString": features = [i for i in roads]
    else: features = [roads]
    
    features = [i for i in roads]
    df = pd.DataFrame({'geometry': features, 'type': ['road'] * len(features)})
    road_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    road_barriers["from"] = road_barriers.apply(lambda row: row.geometry.coords[0], axis = 1)
    road_barriers["to"] = road_barriers.apply(lambda row: row.geometry.coords[-1], axis = 1)

    dd_u = dict(road_barriers['from'].value_counts())
    dd_v = dict(road_barriers['to'].value_counts())
    dd = {k: dd_u.get(k, 0) + dd_v.get(k, 0) for k in set(dd_u) | set(dd_v)}
    to_ignore = {k: v for k, v in dd.items() if v == 1}
    
    # exclude short sectors which are probably not continuous enough to be considered barriers
    road_barriers = road_barriers[~((road_barriers['from'].isin(to_ignore) & road_barriers['to'].isin(to_ignore)) & (road_barriers.length < 500))]
    road_barriers = road_barriers[~((road_barriers['from'].isin(to_ignore) | road_barriers['to'].isin(to_ignore)) & (road_barriers.length < 200))]
    road_barriers.drop(['from', 'to'], axis = 1, inplace = True)
    
    return road_barriers


def water_barriers(place, download_method, distance = None, epsg = None):
    """
    The function downloads water bodies from OSM. Lakes, rivers and see coastlines can be considered structuring barriers.
        
    Parameters
    ----------
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"polygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    distance: float
        it is used only if download_method == "distance from address"
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
        
    Returns
    -------
    LineString GeoDataFrame
    """
    
    crs = {'init': 'epsg:' + str(epsg)}
    
    # rivers
    try:
        if download_method == 'distance_from_address': 
            rivers_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["waterway" = "river"]' )
        elif download_method == 'OSMplace': rivers_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["waterway" = "river"]' )
        else: rivers_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["waterway" = "river"]') 

        rivers = ox.graph_to_gdfs(rivers_graph, nodes=False, edges=True, node_geometry= False, fill_edge_geometry=True)
        rivers = rivers.to_crs(crs)
        if "tunnel" in rivers.columns:
            rivers["tunnel"].fillna(0, inplace = True)
            rivers = rivers[rivers["tunnel"] == 0] 
        rivers = rivers.unary_union
        
        # adding canals
        try:
            if download_method == 'distance_from_address': 
                canals_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "canal"]')                           
            elif download_method == 'OSMplace': 
                canals_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "canal"]')    
            else: canals_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "canal"]')    
                                   
            canals = ox.graph_to_gdfs(canal_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            canals = canals.to_crs(crs)
            cc = canals.unary_union
            rivers = rivers.union(cc)
            
        except: pass
        
        # removing possible duplicates with different tags
        try:
            if download_method == 'distance_from_address': 
                river_banks_graph =  ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]')                  
            elif download_method == 'OSMplace': 
                river_banks_graph =  ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]') 
            else: river_banks_graph =  ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]')
                                   
            river_banks = ox.graph_to_gdfs(river_banks_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            river_banks = river_banks.to_crs(crs)
            rb = river_banks.unary_union
            rivers = rivers.difference(rb)      
            
        except: pass
    
        rivers = linemerge(rivers)
        if rivers.type != "LineString": features = [i for i in rivers]
        else: features = [rivers]
        df = pd.DataFrame({'geometry': features})
        rivers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
        rivers['length'] = rivers['geometry'].length
            
    except: rivers = None
    
    # lakes #########
    
    try:
        if download_method == 'distance_from_address': 
            lakes_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["natural"="water"]')
        elif download_method == 'OSMplace': 
            lakes_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["natural"="water"]' )
        else: lakes_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["natural"="water"]' )
                               
        lakes = ox.graph_to_gdfs(lakes_graph, nodes=False, edges=True, node_geometry= False, fill_edge_geometry=True)
        lakes = lakes.to_crs(crs)                       
        lakes = lakes.unary_union
        
        # removing possible duplicates with different tags - riverbanks
        try:
            if download_method == 'distance_from_address': 
                river_banks_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "riverbank"]')                           
            elif download_method == 'OSMplace': 
                river_banks_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "riverbank"]')        
            else: river_banks_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway" = "riverbank"]') 
                                   
            river_banks = ox.graph_to_gdfs(river_banks_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            river_banks = river_banks.to_crs(crs)
            rb = river_banks.unary_union
            lakes = lakes.difference(rb)
        except: pass
        
        # removing possible duplicates with different tags - rivers
        try:
            if download_method == 'distance_from_address': 
                river_banks_graph =  ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]')                  
            elif download_method == 'OSMplace': 
                river_banks_graph =  ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]') 
            else: river_banks_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="river"]')                        
                                   
            river_banks = ox.graph_to_gdfs(river_banks_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            river_banks = river_banks.to_crs(crs)
            rb = river_banks.unary_union
            lakes = lakes.difference(rb)  
            
        except: pass
        
        # removing possible duplicates with different tags - steams
        try:
            if download_method == 'distance_from_address': 
                steams_graph =  ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="steam"]')                  
            elif download_method == 'OSMplace': 
                steams_graph =  ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="steam"]') 
            else: steams_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["natural"="water"]["water"="steam"]')                        
                                   
            steams = ox.graph_to_gdfs(steams_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            steams = steams.to_crs(crs)
            st = steams.unary_union
            lakes = lakes.difference(st)  
        
        except: pass
        
        lakes = linemerge(lakes)
        if lakes.type != "LineString": features = [i for i in lakes]
        else: features = [lakes]
        df = pd.DataFrame({'geometry': features})
        lakes = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
        lakes['length'] = lakes['geometry'].length
        
        # final control, possible duplicates with different tags - generic waterways
        try:
            if download_method == 'distance_from_address': 
                waterway_graph =  ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway"]')                  
            elif download_method == 'OSMplace': 
                waterway_graph =  ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway"]') 
            else: waterway_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                                   network_type='none', infrastructure= 'way["waterway"]')     
            waterway = ox.graph_to_gdfs(waterway_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            waterway = waterway.to_crs(crs)
            waterway = waterway.unary_union
            if waterway.type != "LineString": features = [i for i in waterway]
            else: features = [waterway]
            df = pd.DataFrame({'geometry': features})
            waterway = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)      
        except: waterway = None
                
        lakes = lakes[~lakes.intersects(waterway)]
        lakes = lakes[lakes['length'] >=500]
        
    except: lakes = None
    
    # sea
    try:
        if download_method == 'distance_from_address': 
            sea_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                           network_type='none', infrastructure= 'way["natural"="coastline"]')
        elif download_method == 'OSMplace': sea_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["natural"="coastline"]')
        else: sea_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["natural"="coastline"]')
        
        sea = ox.graph_to_gdfs(sea_graph, nodes=False, edges=True, node_geometry= False, fill_edge_geometry=True)
        sea = sea.to_crs(crs)
        sea = linemerge(sea)
        if sea.type != "LineString": features = [i for i in sea]
        else: features = [sea]
        df = pd.DataFrame({'geometry': features})
        sea['length'] = sea['geometry'].length
        sea = sea[['geometry', 'length']]
        
    except: sea = None
    
    water = rivers.append(lakes)
    water = water.append(sea)
    water = water.unary_union
    water = linemerge(water)
    if water.type != "LineString": features = [i for i in water]
    else: features = [water]
    df = pd.DataFrame({'geometry': features, 'type': ['water'] * len(features)})
    water_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return water_barriers
    
def railway_barriers(place, download_method,distance = None, epsg = None, keep_light_rail = False):
    """
    The function downloads overground railway structures from OSM. Such structures can be considered barriers which shape the Image of the City and obstruct sight and movement.
        
    Parameters
    ----------
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"polygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    distance: float
        it is used only if download_method == "distance from address"
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    keep_light_rail: boolean
        considering light rail, like tramway
        
    Returns
    -------
    LineString GeoDataFrame
    """
    
    
    crs = {'init': 'epsg:' + str(epsg)}
    if download_method == 'distance_from_address': 
            railway_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge= False, simplify = False,
                           network_type='none', infrastructure= 'way["railway"~"rail"]')
    elif download_method == 'OSMplace':
        railway_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["railway"~"rail"]')
    else: railway_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge=False, simplify = False,
                               network_type='none', infrastructure= 'way["railway"~"rail"]')                           
                               
    railways = ox.graph_to_gdfs(railway_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    railways = railways.to_crs(crs)
    if "tunnel" in railways.columns:
        railways["tunnel"].fillna(0, inplace = True)
        railways = railways[railways["tunnel"] == 0]     
    r = railways.unary_union
    
    # removing light_rail, in case
    if not keep_light_rail:
        try:
            if download_method == 'distance_from_address': 
                light_graph = ox.graph_from_address(place, distance = distance, retain_all = True, truncate_by_edge=False, simplify = False,
                       network_type='none', infrastructure= 'way["railway"~"light_rail"]' )
            elif download_method == 'OSMplace':
                light_graph = ox.graph_from_place(place, retain_all = True, truncate_by_edge = False, simplify = False,
                       network_type='none', infrastructure= 'way["railway"~"light_rail"]')
            else: light_graph = ox.graph_from_polygon(place, retain_all = True, truncate_by_edge = False, simplify = False,
                       network_type='none', infrastructure= 'way["railway"~"light_rail"]') 

            light_railways = ox.graph_to_gdfs(light_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            light_railways = light_railways.to_crs(crs)
            lr = light_railways.unary_union
            r = r.difference(lr)
        except: pass

    p = polygonize_full(r)
    railways = unary_union(p).buffer(10).boundary # to simpify a bit
    railways = railways
    railways = linemerge(railways)
    if railways.type != "LineString": features = [i for i in railways]
    else: features = [railways]
    df = pd.DataFrame({'geometry': features, 'type': ['railway'] * len(features)})
    railway_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return railway_barriers
    
def park_barriers(place, download_method, distance = None, epsg = None, min_area = 100000):
    """
    The function downloads parks areas with a certain extent and converts them to LineString features. Parks may break continuity in the urban structure, besides being attractive areas for pedestrians.
        
    Parameters
    ----------
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"polygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    distance: float
        it is used only if download_method == "distance from address"
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    min_area: double
        parks with an extension smaller that this parameter are disregarded
      
    Returns
    -------
    LineString GeoDataFrame
    """

    crs = {'init': 'epsg:' + str(epsg)}
    if download_method == 'distance_from_address':
        parks_polygon = ox.footprints_from_address(place, distance = distance, footprint_type="leisure", retain_invalid = True)
    elif download_method == 'OSMplace': parks_polygon = ox.footprints_from_place(place, footprint_type="leisure", retain_invalid = True)
    else: parks_polygon = ox.footprints_from_polygon(place, footprint_type="leisure", retain_invalid = True)
    
    parks_polygon = parks_polygon[parks_polygon.leisure == 'park']
    ix_geo = parks_polygon.columns.get_loc("geometry")+1
    to_drop = []
    
    for row in parks_polygon.itertuples():
        type_geo = None
        try: type_geo = row[ix_geo].geom_type
        except: to_drop.append(row.Index)
        
    parks_polygon.drop(to_drop, axis = 0, inplace = True)
    parks_polygon = parks_polygon.to_crs(crs)
    parks_polygon.area = parks_polygon.geometry.area
    parks_polygon = parks_polygon[parks_polygon.area >= min_area]
 
    pp = parks_polygon['geometry'].unary_union  
    pp = polygonize_full(pp)
    parks = unary_union(pp).buffer(10).boundary # to simpify a bit
    parks = linemerge(parks)
    if parks.type != "LineString": features = [i for i in parks]
    else: features = [parks]
    features = [i for i in parks]

    df = pd.DataFrame({'geometry': features, 'type': ['park'] * len(features)})
    park_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return park_barriers
    
   
def along_water(edges_gdf, barriers_gdf):
    """
    The function assigns to each street segment in a GeoDataFrame the list of barrierIDs corresponding to waterbodies which lay along the street segment. No obstructions between the street segment
    and the barriers are admitted.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        Street segments 
    barriers_gdf: LineString GeoDataFrame
        Barriers 
    Returns
    -------
    LineString GeoDataFrame
    """
    
    sindex = edges_gdf.sindex
    tmp = barriers_gdf[barriers_gdf['type'].isin(['water'])]
    edges_gdf['ac_rivers'] = edges_gdf.apply(lambda row: barriers_along(row['edgeID'], edges_gdf, tmp, sindex, offset = 200), axis = 1)
    edges_gdf['c_rivers'] = edges_gdf.apply(lambda row: _crossing_barriers(row['geometry'], tmp), axis = 1)
    edges_gdf['bridge'] = edges_gdf.apply(lambda row: True if len(row['c_rivers']) > 0 else False, axis = 1)
    # excluding bridges
    edges_gdf['a_rivers'] = edges_gdf.apply(lambda row: list(set(row['ac_rivers'])-set(row['c_rivers'])), axis = 1)
    edges_gdf['a_rivers'] = edges_gdf.apply(lambda row: row['ac_rivers'] if row['bridge'] == False else [], axis = 1)
    edges_gdf.drop(['ac_rivers', 'c_rivers'], axis = 1, inplace = True)
    
    return edges_gdf

def along_within_parks(edges_gdf, barriers_gdf):
    """
    The function assigns to each street segment in a GeoDataFrame the list of barrierIDs corresponding to parks which lay along the street segment.
    Also street segments within parks are considered
    and the barriers are admitted.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        Street segments 
    barriers_gdf: LineString GeoDataFrame
        Barriers 
    Returns
    -------
    LineString GeoDataFrame
    """
    
    sindex = edges_gdf.sindex
    tmp = barriers_gdf[barriers_gdf['type']=='park']
    edges_gdf['a_parks'] = edges_gdf.apply(lambda row: barriers_along(row['edgeID'], edges_gdf, tmp, sindex, offset = 200), axis = 1)
    
    # polygonize parks
    park_polygons = barriers_gdf[barriers_gdf['type']=='park'].copy()
    park_polygons['geometry'] = park_polygons.apply(lambda row: (polygonize_full(row['geometry']))[0][0], axis = 1)
    park_polygons = gpd.GeoDataFrame(park_polygons['barrierID'], geometry = park_polygons['geometry'], crs = edges_gdf.crs)
    
    edges_gdf['w_parks'] = edges_gdf.apply(lambda row: _within_parks(row['geometry'], park_polygons), axis = 1) #within
    edges_gdf['aw_parks'] = edges_gdf.apply(lambda row: list(set(row['a_parks']+row['w_parks'])), axis = 1) #along
    edges_gdf.drop(['a_parks', 'w_parks'], axis = 1, inplace = True)

    return edges_gdf
    
def barriers_along(ix_line, edges_gdf, barriers_gdf, edges_gdf_sindex, offset = 100):
    """
    The function returns list of barrierIDs along the edgeID of a street segment, given a certain offset.
    Touching and intersecting barriers are ignored.
        
    Parameters
    ----------
    ix_line: int
        index street segment
    edges_gdf: LineString GeoDataFrame
        Street segments 
    barriers_gdf: LineString GeoDataFrame
        Barriers 
    edges_gdf_sindex: RTree Sindex
        spatial index on edges_gdf
    offset: int
        offset along the street segment considered
      
    Returns
    -------
    LineString GeoDataFrame
    """
    
    buffer = edges_gdf.loc[ix_line].geometry.buffer(offset)
    barriers_along = []
    intersecting_barriers = barriers_gdf[barriers_gdf.geometry.intersects(buffer)]
    touching_barriers = barriers_gdf[barriers_gdf.geometry.touches(edges_gdf.loc[ix_line].geometry)]
    intersecting_barriers = intersecting_barriers[~intersecting_barriers.barrierID.isin(list(touching_barriers.barrierID))]
    if len(intersecting_barriers) == 0: return []
    
    possible_matches_index = list(edges_gdf_sindex.intersection(buffer.bounds))
    pm = edges_gdf.iloc[possible_matches_index]
    pm.drop(ix_line, axis = 0, inplace = True)
    
    for ix, barrier in intersecting_barriers.iterrows():
        midpoint = edges_gdf.loc[ix_line].geometry.interpolate(0.5, normalized = True)
        line = LineString([midpoint, nearest_points(midpoint, barrier['geometry'])[1]])
        if len(pm[pm.geometry.intersects(line)]) >= 1: continue
        else: barriers_along.append(barrier['barrierID'])
    
    if len(barriers_along) == 0: return []
    return barriers_along
    

    
def _within_parks(line_geometry, park_polygons):
    """
    The function supports the along_within_parks function. Returns a list containing the barrierID of possibly intersecting parks (should be one).
        
    Parameters
    ----------
    line_geometry: LineString 
        street segment geometry
    park_polygons: Polygon GeoDataFrame
        Parks GeoDataFrame
      
    Returns
    -------
    List
    """
    
    within = []
    intersecting_parks = park_polygons[park_polygons.geometry.intersects(line_geometry)]
    touching_parks = park_polygons[park_polygons.geometry.touches(line_geometry)]
    if len(intersecting_parks) == 0: return within
    intersecting_parks = intersecting_parks[~intersecting_parks.barrierID.isin(list(touching_parks.barrierID))]
    within = list(intersecting_parks.barrierID)
    return within
    

def assign_structuring_barriers(edges_gdf, barriers_gdf):
    """
    The function return a GeoDataFrame with an added boolean column field that indicates whether the street segment intersects a separating/structuring barrier.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        Street segments 
    barriers_gdf: LineString GeoDataFrame
        Barriers 
    Returns
    -------
    LineString GeoDataFrame
    """
    
    barriers_gdf = barriers_gdf.copy()
    edges_gdf = edges_gdf.copy()
    tmp = barriers_gdf[barriers_gdf['type'] != 'park'].copy() # parks are disregarded
    
    edges_gdf['c_barr'] = edges_gdf.apply(lambda row: _crossing_barriers(row['geometry'], tmp ), axis = 1)
    edges_gdf['sep_barr'] = edges_gdf.apply(lambda row: True if len(row['c_barr']) > 0 else False, axis = 1)
    edges_gdf.drop('c_barr', axis = 1, inplace = True)
    
    return edges_gdf

def _crossing_barriers(line_geometry, barriers_gdf):
    """
    The function supports the assign_structuring_barriers and along_water functions. It returns a list of intersecting barrierIDs.
    
    Parameters
    ----------
    line_geometry: LineString 
        street segment geometry
    barriers_gdf: LineString GeoDataFrame
        Barriers 
        
    Returns
    -------
    list
    """
    
    adjacent_barriers = []
    intersecting_barriers = barriers_gdf[barriers_gdf.geometry.intersects(line_geometry)]
    touching_barriers = barriers_gdf[barriers_gdf.geometry.touches(line_geometry)]
    if len(intersecting_barriers) == 0: return adjacent_barriers
    intersecting_barriers = intersecting_barriers[~intersecting_barriers.barrierID.isin(list(touching_barriers.barrierID))]
    adjacent_barriers = list(intersecting_barriers.barrierID)
    return adjacent_barriers
    
def get_barriers(place, download_method, distance, epsg): 
 
    rb = road_barriers(place, download_method, distance, epsg, include_primary = True)
    wb = water_barriers(place, download_method, distance, epsg)
    ryb = railway_barriers(place,download_method, distance, epsg)
    pb = park_barriers(place,download_method, distance, epsg, min_area = 100000)
    
    barriers_gdf = rb.append(wb)
    barriers_gdf = ryb.append(barriers_gdf)
    barriers_gdf = pb.append(barriers_gdf)
    barriers_gdf.reset_index(inplace = True, drop = True)
    barriers_gdf['barrierID'] = barriers_gdf.index.astype(int)

    return barriers_gdf
    
