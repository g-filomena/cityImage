import osmnx as ox, networkx as nx, pandas as pd, numpy as np, geopandas as gpd
import functools

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, polygonize, polygonize_full, unary_union
pd.set_option("precision", 10)

from .graph import *
from .utilities import *

def waterway_barriers(place, distance, convex_hull, crs):

    rivers_graph = ox.graph_from_address(place, retain_all = True, truncate_by_edge=True, simplify=True, distance = distance,
                           network_type='none', infrastructure= 'way["waterway"]' )
    rivers = ox.graph_to_gdfs(rivers_graph, nodes=False, edges=True, node_geometry= False, fill_edge_geometry=True)
    rivers = rivers.to_crs(crs)
    rivers.length = rivers.geometry.length
    rivers = rivers[rivers.length> 300]
    rivers.drop(['osmid', 'oneway', 'bridge', 'tunnel'], axis = 1, inplace = True, errors = 'ignore')
    waterways = rivers.unary_union.intersection(convex_hull)
    waterways = linemerge(waterways)
    features = [i for i in waterways]
    df = pd.DataFrame({'geometry': features, 'type': ['waterway'] * len(features)})
    waterway_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)

    return waterway_barriers
    
def railway_barriers(place, distance, convex_hull, crs, keep_light_rail = False):


    railway_graph = ox.graph_from_address(place, retain_all = True, truncate_by_edge=True, simplify=True, distance = distance,
                           network_type='none', infrastructure= 'way["railway"~"rail"]' )
    railways = ox.graph_to_gdfs(railway_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    railways = railways.to_crs(crs)
    if "tunnel" in railways.columns:
        railways["tunnel"].fillna(0, inplace = True)
        railways = railways[railways["tunnel"] == 0]     
    r = railways.unary_union
    
    if not keep_light_rail:
        light_graph = ox.graph_from_address(place, retain_all = True, truncate_by_edge=True, simplify=True, distance = distance,
                   network_type='none', infrastructure= 'way["railway"~"light_rail"]' )

        light_railways = ox.graph_to_gdfs(light_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
        light_railways = light_railways.to_crs(crs)
        lr = light_railways.unary_union
        r = r.symmetric_difference(lr)

    p = polygonize_full(r)
    railways = unary_union(p).buffer(10).boundary # to simpify a bit
    railways = railways.intersection(convex_hull)
    railways = linemerge(railways)
    features = [i for i in railways]
    df = pd.DataFrame({'geometry': features, 'type': ['railway'] * len(features)})
    railway_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return railway_barriers
    