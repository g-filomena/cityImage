import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import osmnx as ox, networkx as nx, matplotlib.cm as cm, pandas as pd, numpy as np, geopandas as gpd
import math
from math import sqrt
import ast
import functools

from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import split
pd.set_option("precision", 10)

from .utilities import *
from .angles import *

"""
This set of functions handles interoperations between GeoDataFrames and graphs. It allows data conversion and the extraction of nodes and edges GeoDataFrames from roads shapefile or OpenStreetMap.

"""
    
## Graph preparation functions ###############
    
def get_network_fromOSM(download_type, place, network_type = "all", epsg = None, distance = 7000, fix_topology = False): 

    """
    The function downloads and creates a simplified OSMNx graph for a selected area. 
    Afterwards, GeoDataFrames for nodes and edges are created, assigning new nodeID and edgeID identifiers.
        
    Parameters
    ----------
    download_type: string, {"OSMpolygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data. of dowload
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    network_type: string,  {"walk", "bike", "drive", "drive_service", "all", "all_private", "none"}
        it indicates type of street or other network to extract - from OSMNx paramaters
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        it is used only if download_type == "distance from address"
    fix_topology: boolean
        if True it breaks lines at intersections with other lines in the streets GeoDataFrame, apart from segments categorised as bridges in OSM
        
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    # using OSMNx to download data from OpenStreetMap     
    if download_type == "OSMpolygon":
        query = ox.osm_polygon_download(place, limit=1, polygon_geojson=1)
        OSMplace = query[0]["display_name"]
        G = ox.graph_from_place(OSMplace, network_type = network_type, simplify = True)
        
    elif download_type == "distance_from_address":
        G = ox.graph_from_address(place, network_type = network_type, distance = distance, simplify = True)
    
    # (download_type == "OSMplace")
    else: G = ox.graph_from_place(place, network_type = network_type, simplify = True)
    
    # fix list of osmid assigned to same edges
    for i, item in enumerate(G.edges()):
        if isinstance(G[item[0]][item[1]][0]["osmid"], (list,)): 
            G[item[0]][item[1]][0]["osmid"] = G[item[0]][item[1]][0]["osmid"][0]
    
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    edges_gdf = edges_gdf[["geometry", "length", "u", "v","highway","key", "oneway", "bridge", "maxspeed","name"]]
    
    if not fix_topology:
        nodes = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=False)
        nodes_gdf = nodes.drop(["highway", "ref"], axis=1, errors = "ignore")
        # getting rid of OSMid and preparing geodataframes
        nodes_gdf.index = nodes_gdf.osmid.astype("int64")
        nodes_gdf, edges_gdf = reset_index_street_network_gdfs(nodes_gdf, edges_gdf)    
        
    else: # when topology needs to be fixed
        edges_gdf = fix_network_topology(edges_gdf)
        if epsg == None: 
            edges_gdf = ox.projection.project_gdf(edges_gdf)
            epsg = edges_gdf.crs['init'][5:] # extract epsg
        nodes_gdf = _obtain_nodes_gdf(edges_gdf, epsg)
        nodes_gdf, edges_gdf =_join_by_coordinates(nodes_gdf, edges_gdf)
        
    # columns to keep (u and v represent "from" and "to" node)
    nodes_gdf = nodes_gdf[["nodeID","x","y","geometry"]]
    edges_gdf = edges_gdf[["edgeID","u","v","key","geometry", "length", "highway","oneway", "name"]]
    edges_gdf["oneway"] *= 1
    
    # resolving lists 
    edges_gdf["highway"] = [x[0] if type(x) == list else x for x in edges_gdf["highway"]]
    edges_gdf["name"] = [x[0] if type(x) == list else x for x in edges_gdf["name"]]
    
    # finalising geodataframes
    if epsg == None: nodes_gdf, edges_gdf = ox.projection.project_gdf(nodes_gdf), ox.projection.project_gdf(edges_gdf)
    else: nodes_gdf, edges_gdf = nodes_gdf.to_crs(epsg = epsg), edges_gdf.to_crs(epsg = epsg)
    
    nodes_gdf["x"], nodes_gdf["y"] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    nodes_gdf['height'] = 2 # this will be used for 3d visibility analysis
    return nodes_gdf, edges_gdf

def get_network_fromSHP(path, epsg, case_study_area = None, radius = 0, dict_columns = {}, fix_topology = False):
    
    """
    The function loads a vector lines shapefile from a specified directory, along with the epsg coordinate code.
    It creates two GeoDataFrame, one for street junctions (nodes) and one for street segments (edges).
    The GeoDataFrames are built assuming a planar undirected graph. 
    The "case_study_area" polygon is optional and when provided is used to select geometries within the area + a buffer of x meters, fixed by the researcher (distance_from_boundary)
     
    Parameters
    ----------
    path: string
        the local path where the .shp file is stored
    epsg: int
        epsg of the area considered 
    case_study_area: Polygon
        the polygon representing the extension of the case-study area, if not provided the 2 following parameters are used to extract the street network
    radius: float
        it is employed, when the case_study_area polygon is not provided, to determine the extension of the area of interest
    dict_columns: dict
        it should be structured as: {"roadType_field": "highway",  "direction_field": "oneway", "speed_field": None, "name_field": "name"}
        Replace the items with the field names in the input data (if the relative attributes are relevant and existing)
    fix_topology: boolean
        if True it breaks lines at intersections with other lines in the streets GeoDataFrame, apart from segments categorised as bridges in OSM
    
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    # try reading street network from directory
    edges_gdf = gpd.read_file(path).to_crs(epsg=epsg)
        
    # using a buffer to clip the area of study
    if case_study_area != None: edges_gdf = edges_gdf[edges_gdf.geometry.within(case_study_area)]
    elif radius > 0:
        cn = edges_gdf.geometry.unary_union.centroid
        buffer = cn.buffer(radius) 
        edges_gdf = edges_gdf[edges_gdf.geometry.within(buffer)]
        
    edges_gdf["from"] = None
    edges_gdf["to"] = None
    edges_gdf["key"] = 0
    
    # creating the dataframes
    new_columns = ["highway", "oneway", "maxspeed","name"]
    if len(dict_columns) > 0:
        for n, (key, value) in enumerate(dict_columns.items()):
            if (value != None): edges_gdf[new_columns[n]] = edges_gdf[value]
    else: new_columns = []
     
    standard_columns = ["geometry", "from", "to", "key"]
    edges_gdf = edges_gdf[standard_columns + new_columns]
    
    # remove z coordinates, if any
    edges_gdf["geometry"] = edges_gdf.apply(lambda row: LineString([coor for coor in [row["geometry"].coords[i][0:2] for i in range(0, len(row["geometry"].coords))]]), axis = 1)
    edges_gdf['edgeID'] = edges_gdf.index.values.astype(int)
    if fix_topology: edges_gdf = fix_network_topology(edges_gdf) # fixing topology
    
    # assigning indexes
    edges_gdf.reset_index(inplace=True, drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype(int) 
    nodes_gdf = _obtain_nodes_gdf(edges_gdf, epsg)
    
    # linking on coordinates
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype(int)
    nodes_gdf, edges_gdf =_join_by_coordinates(nodes_gdf, edges_gdf)
    edges_gdf["length"] = gpd.GeoSeries(edges_gdf["geometry"].length) # computing length
    nodes_gdf['height'] = 2 # this will be used for 3d visibility analysis
    
    return nodes_gdf, edges_gdf
    
def _obtain_nodes_gdf(edges_gdf, epsg):

    """
    It obtains the nodes GeoDataFrame from the unique coordinates pairs in the edges_gdf GeoDataFrame.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    epsg: int
        epsg of the area considered 
    Returns
    -------
    Point GeoDataFrames
    """
    
    edges_gdf["from"] = edges_gdf.apply(lambda row: row.geometry.coords[0], axis = 1)
    edges_gdf["to"] = edges_gdf.apply(lambda row: row.geometry.coords[-1], axis = 1)
    unique_nodes_tmp = list(edges_gdf["to"].unique()) + list(edges_gdf["from"].unique())
    unique_nodes = list(set(unique_nodes_tmp))
    #preparing nodes geodataframe
    nodes_data = pd.DataFrame.from_records(unique_nodes, columns=["x", "y"]).astype("float")
    geometry = [Point(xy) for xy in zip(nodes_data.x, nodes_data.y)]
    crs = {'init': 'epsg:' + str(epsg)}
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=crs, geometry=geometry)
    nodes_gdf.reset_index(drop=True, inplace = True)
    
    return nodes_gdf
    
def _join_by_coordinates(nodes_gdf, edges_gdf):

    """
    The function merge the u-v nodes information, from the nodes GeoDataFrame, with the edges_gdf GeoDataFrame.
    The process exploits coordinates pairs of the edges for finding the relative nodes in the nodes GeoDataFrame.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    if not "nodeID" in nodes_gdf.columns: nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    edges_gdf["from"] = edges_gdf.apply(lambda row: row.geometry.coords[0], axis = 1)
    edges_gdf["to"] = edges_gdf.apply(lambda row: row.geometry.coords[-1], axis = 1)
    nodes_gdf["coordinates"] = list(zip(nodes_gdf.x, nodes_gdf.y))

    edges_tmp = pd.merge(edges_gdf, nodes_gdf[["nodeID","coordinates"]], how="left", left_on="from", right_on="coordinates")
    edges_tmp.drop(edges_tmp[["coordinates"]], axis = 1, inplace = True)
    edges_tmp.rename(columns = {"nodeID":"u"}, inplace = True)
    
    edges_gdf = pd.merge(edges_tmp, nodes_gdf[["nodeID","coordinates"]], how="left", left_on="to", right_on="coordinates")
    edges_gdf = edges_gdf.drop(edges_gdf[["coordinates", "from", "to"]], axis = 1)
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"v"})
    nodes_gdf.drop(["coordinates"], axis = 1, inplace = True)
    
    return nodes_gdf, edges_gdf

def reset_index_street_network_gdfs(nodes_gdf, edges_gdf):
    """
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    edges_gdf = edges_gdf.rename(columns = {"u":"old_u", "v":"old_v"})
    nodes_gdf["old_nodeID"] = nodes_gdf.index.values.astype("int64")
    nodes_gdf = nodes_gdf.reset_index(drop = True)
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", "nodeID"]], how="left", left_on="old_u", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"u"})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", "nodeID"]], how="left", left_on="old_v", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"v"})

    edges_gdf.drop(["old_u", "old_nodeID_x", "old_nodeID_y", "old_v"], axis = 1, inplace = True)
    nodes_gdf.drop(["old_nodeID", "index"], axis = 1, inplace = True, errors = "ignore")
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype(int)
    
    return nodes_gdf, edges_gdf

## Obtaining graphs ###############

def graph_fromGDF(nodes_gdf, edges_gdf, nodeID = "nodeID"):

    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX undirected Graph.
       
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    nodeID: str
        please provide here the column name which indicates the node identifier column (if different from "nodeID")
        
    Returns
    -------
    NetworkX undirected Graph
    """

    nodes_gdf.set_index(nodeID, drop = False, inplace = True, append = False)
    del nodes_gdf.index.name
    G = nx.Graph()   
    G.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    
    for attribute_name in nodes_gdf.columns:
        if type(nodes_gdf.iloc[0][attribute_name]) == list: 
            attribute_values = {k: v for k, v in attributes[attribute_name].items()}        
        # only add this attribute to nodes which have a non-null value for it
        else: attribute_values = {k: v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v (as they're added separately) or null
    for _, row in edges_gdf.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ["u", "v"]) and (isinstance(value, list) or pd.notnull(value)):  attrs[label] = value
        G.add_edge(row["u"], row["v"], **attrs)
    
    return G


def multiGraph_fromGDF(nodes_gdf, edges_gdf, nodeID):

    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX MultiGraph.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    nodeID: str
        please provide here the column name which indicates the node identifier column (if different from "nodeID")
    
    Returns
    -------
    NetworkX MultiGraph
    """
    
    nodes_gdf.set_index(nodeID, drop = False, inplace = True, append = False)
    del nodes_gdf.index.name
    
    Mg = nx.MultiGraph()   
    Mg.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    
    for attribute_name in nodes_gdf.columns:
        # only add this attribute to nodes which have a non-null value for it
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Mg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added separately) or null
    for _, row in edges_gdf.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ["u", "v", "key"]) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        Mg.add_edge(row["u"], row["v"], key=row["key"], **attrs)
      
    return Mg
    
## Building geo-dataframes for dual graph representation ###############

def dual_gdf(nodes_gdf, edges_gdf, epsg):

    """
    It creates two dataframes that are later exploited to generate the dual graph of a street network. The nodes_dual gdf contains edges 
    centroids; the edges_dual gdf, instead, contains links between the street segment centroids. Those dual edges link real street segments 
    that share a junction. The centroids are stored with the original edge edgeID, while the dual edges are associated with several
    attributes computed on the original street segments (distance between centroids, deflection angle).
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    nodeID: str
        please provide here the column name which indicates the node identifier column (if different from "nodeID")
    epsg: int
        epsg of the area considered 

    Returns
    -------
    tuple of GeoDataFrames
    """
    
    if list(edges_gdf.index.values) != list(edges_gdf.edgeID.values): 
        edges_gdf.index = edges_gdf.edgeID
        del edges_gdf.index.name
    
    # computing centroids                                       
    centroids_gdf = edges_gdf.copy()
    centroids_gdf["centroid"] = centroids_gdf["geometry"].centroid
    centroids_gdf["intersecting"] = None
    
    ix_u, ix_v = centroids_gdf.columns.get_loc("u")+1, centroids_gdf.columns.get_loc("v")+1
    ix_edgeID = centroids_gdf.columns.get_loc("edgeID")+1
         
    # find_intersecting segments and storing them in the centroids gdf
    centroids_gdf["intersecting"] = centroids_gdf.apply(lambda row: list(centroids_gdf.loc[(centroids_gdf["u"] == row["u"])|(centroids_gdf["u"] == row["v"])|
                                                    (centroids_gdf["v"] == row["v"])|(centroids_gdf["v"] == row["u"])].index), axis=1)
            
    # creating vertexes representing street segments (centroids)
    centroids_data = centroids_gdf[["edgeID", "intersecting", "length"]]
    if epsg == None: crs = nodes_gdf.crs
    else: crs = {'init': 'epsg:' + str(epsg)}
    nodes_dual = gpd.GeoDataFrame(centroids_data, crs=crs, geometry=centroids_gdf["centroid"])
    nodes_dual["x"], nodes_dual["y"] = [x.coords.xy[0][0] for x in centroids_gdf["centroid"]], [y.coords.xy[1][0] for y in centroids_gdf["centroid"]]
    nodes_dual.index =  nodes_dual.edgeID
    del nodes_dual.index.name
    
    # creating fictious links between centroids
    edges_dual = pd.DataFrame(columns=["u","v", "geometry", "length"])
    ix_length = nodes_dual.columns.get_loc("length")+1
    ix_intersecting = nodes_dual.columns.get_loc("intersecting")+1
    ix_geo = nodes_dual.columns.get_loc("geometry")+1

    # connecting nodes which represent street segments share a linked in the actual street network   
    processed = []
    for row in nodes_dual.itertuples():                                           
        # intersecting segments:  # i is the edgeID                                      
        for intersecting in row[ix_intersecting]:
            if ((row.Index == intersecting) | ((row.Index, intersecting) in processed) | ((intersecting, row.Index) in processed)): continue
            length_intersecting =  nodes_dual.loc[intersecting]["length"]
            distance = (row[ix_length]+length_intersecting)/2
        
            # adding a row with u-v, key fixed as 0, Linestring geometry 
            # from the first centroid to the centroid intersecting segment 
            ls = LineString([row[ix_geo], nodes_dual.loc[intersecting]["geometry"]])
            edges_dual.loc[-1] = [row.Index, intersecting, ls, distance] 
            edges_dual.index = edges_dual.index + 1
            processed.append((row.Index, intersecting))
            
    edges_dual = edges_dual.sort_index(axis=0)
    edges_dual = gpd.GeoDataFrame(edges_dual[["u", "v", "length"]], crs=crs, geometry=edges_dual["geometry"])
    
    # setting angle values in degrees and radians
    edges_dual["deg"] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row["u"]].geometry, edges_gdf.loc[row["v"]].geometry, degree = True, deflection = True), axis = 1)
    edges_dual["rad"] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row["u"]].geometry, edges_gdf.loc[row["v"]].geometry, degree = False, deflection = True), axis = 1)
        
    return nodes_dual, edges_dual

def dual_graph_fromGDF(nodes_dual, edges_dual):

    """
    The function generates a NetworkX graph from dual-nodes and -edges GeoDataFrames.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        the GeoDataFrame of the dual nodes, namely the street segments' centroids
    edges_dual: LineString GeoDataFrame
        the GeoDataFrame of the dual edges, namely the links between street segments' centroids 
        
    Returns
    -------
    NetworkX Graph
    """
   
    nodes_dual.set_index("edgeID", drop = False, inplace = True, append = False)
    del nodes_dual.index.name
    edges_dual.u = edges_dual.u.astype(int)
    edges_dual.v = edges_dual.v.astype(int)
    
    Dg = nx.Graph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
    
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if attribute_name == "intersecting": continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ["u", "v"]) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        Dg.add_edge(row["u"], row["v"], **attrs)

    return Dg

def dual_id_dict(dict_values, G, node_attribute):

    """
    It could be used when one deals with a dual graph and wants to link analyses conducted on this representation to the
    the primal graph. For instance, it takes the dictionary containing the betweennes-centrality values of the
    nodes in the dual graph, and associates these variables to the corresponding edgeID.
    
    Parameters
    ----------
    dict_values: dictionary 
        it should be in the form {nodeID: value} where values is a measure that has been computed on the graph, for example
    G: networkx graph
        the graph that was used to compute or to assign values to nodes or edges
    node_attribute: string
        the attribute of the node to link to the edges GeoDataFrame
    
    Returns
    -------
    dictionary
    """
    
    view = dict_values.items()
    ed_list = list(view)
    ed_dict = {}
    for p in ed_list: ed_dict[G.nodes[p[0]][node_attribute]] = p[1] # attribute and measure
        
    return ed_dict

def fix_network_topology(edges_gdf):

    """
    The function breaks lines at intersections, in the edges_gdf GeoDataFrame. 
    The interesections may not be corrently represented prior to the analysis. Lines are therefore split and the intersection coordinate is 
    added to the sequence of coordinates in the considered edges.
    As a consequence, new geometries may be creadeted and added in the edges_gdf GeoDataFrame.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        the GeoDataFrame of the dual nodes, namely the street segments' centroids
    edges_dual: LineString GeoDataFrame
        the GeoDataFrame of the dual edges, namely the links between street segments' centroids 
        
    Returns
    -------
    LineString GeoDataFrame
    """

    edges_gdf = edges_gdf.sort_index(axis=0)
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    restart = True
    ix_bridge = 0
    
    # taking into account bridges, which are not supposed to be split at intersections 
    if "bridge" in edges_gdf.columns:
        edges_gdf["bridge"].fillna(0, inplace = True)
        ix_bridge = edges_gdf.columns.get_loc("bridge")+1
    last = 0
    
    # starting
    while restart:
        restart = False

        for row in edges_gdf.itertuples():
            if restart: break 
            if row.Index < last: continue # already checked
            if (ix_bridge > 0) & (row[ix_bridge] != 0): continue # bridges are not checked
            pm = pm = edges_gdf[edges_gdf.geometry.intersects(row[ix_geo].buffer(1))] # find possible intersecting other lines..
            if len(pm) == 0: continue
            
            #..iterating through them
            for other_edge in pm.itertuples():
                if other_edge.Index == row.Index: continue
                if not (row[ix_geo].intersects(other_edge[ix_geo]) | row[ix_geo].touches(other_edge[ix_geo])): continue # no intersection
                intersection = row[ix_geo].intersection(other_edge[ix_geo])
                line_geometry = row[ix_geo]
                line_geometry_other = other_edge[ix_geo]
                coordsA = line_geometry.coords
                coordsB = line_geometry_other.coords
                
                if intersection.geom_type == 'Point': # just a single intersection
                    coordsI =  intersection.coords
                    # don't need to fix when:
                    # the line considered is ok. The intersection is on u or v. The other line will be fixed later, if necessary
                    if (coordsA[-1] == coordsI[0]) | (coordsA[0] == coordsI[0]): continue 
                    # fix:
                    else: 
                        new_line_geometries = split_line_at_Point(line_geometry, intersection) # checking if the intersection's coordinates are in the coordinates sequence of the line and split
                        edges_gdf.at[row.Index, 'geometry'] = new_line_geometries[0] # first resulting split geometry
                        index = edges_gdf.index.max()+1
                        # copy attributes
                        edges_gdf.loc[index] = edges_gdf.loc[row.Index]
                        # and assign geometry an new edgeID                 
                        edges_gdf.at[index, 'geometry'] = new_line_geometries[1] # first resulting split geometry
                        edges_gdf.at[index, 'edgeID'] = index
    
                elif intersection.geom_type == 'MultiPoint': # multiPoint intersection
                    new_collection = []
                    for int in intersection:
                        if (int.coords[0] == coordsA[0]) | (int.coords[0] == coordsA[-1]): pass # disregarding the ones which lie on the line's u-v nodes
                        else: new_collection.append(int) # only checking the others
                    if len(new_collection) == 0: continue    
                    geometry_collection = MultiPoint([point.coords[0] for point in new_collection])                         
                    new_line_geometries = split_line_at_MultiPoint(line_geometry, geometry_collection) # including the intersecting geometries in the coordinates sequence of the line and split

                    for n, line in enumerate(new_line_geometries): # assigning the resulting geometries
                        if n == 0: index = row.Index
                        else: index = max(edges_gdf.index)+1
                        # copy attributes
                        edges_gdf.loc[index] = edges_gdf.loc[row.Index]  
                        # and assign geometry an new edgeID 
                        edges_gdf.at[index, 'geometry'] = line
                        edges_gdf.at[index, 'edgeID'] = index           
                
                else: continue  
                last = row.Index-1 # store the last
                restart = True
                break
                    
    return edges_gdf
    
def split_line_at_Point(line_geometry, intersection):   

    """
    The function checks whether a Point's coordinate are part of the sequence of coordinates of a LineString.
    When this has been ascerted or fixed, the LineString line_geometry is split at the Point
               
    The input intersection, must be an actual intersection.
               
    Parameters
    ----------
    line_geometry: LineString
        the LineString which has to be split
    intersection: Point
        the intersecting point
        
    Returns
    -------
    MultiLineString
    """

    lines = split(line_geometry, intersection)
    if len(lines) == 1: 
    # the split function can't work in this case; we have to add the intersection coordinates to the line_geometry coordinates
    # we have first to add the vertex in the LineString coordinate sequence
        new_line_coords = list(line_geometry.coords)
        for n, v in enumerate(new_line_coords):
            if n == 0: continue
            tmp_line = LineString([Point(new_line_coords[n-1]), Point(v)])
            if ((intersection.intersects(tmp_line)) | (tmp_line.distance(intersection) < 1e-8)): # determine if the point is within the line
                new_line_coords.insert(n, intersection.coords[0])
                break
        
        line_geometry = LineString([coor for coor in new_line_coords])
        lines = split(line_geometry, intersection)        
    return lines
    
def split_line_at_MultiPoint(line_geometry, intersection):   

    """
    The function checks whether the coordinates of Point(s) in a Point Collections coordinate are part of the sequence of coordinates of a LineString.
    When this has been ascerted or fixed, the LineString line_geometry is split at each of the intersecting points in the collection.
    
    The input intersection, must be an actual intersection.
               
    Parameters
    ----------
    line_geometry: LineString
        the LineString which has to be split
    intersection: MultiPoint
        the intersecting points
        
    Returns
    -------
    MultiLineString
    """
    for point in intersection:
        # if point.coords[0] in line_geometry.coords: continue
        # else:
        new_line_coords = list(line_geometry.coords)
        for n, v in enumerate(new_line_coords):
            if n == 0: continue
            line = LineString([Point(new_line_coords[n-1]), Point(v)])
            if ((point.intersects(line)) | (line.distance(point) < 1e-8)):
                new_line_coords.insert(n, point.coords[0])
                break
        line_geometry = LineString([coor for coor in new_line_coords])
                     
    lines = split(line_geometry, intersection)   
    return lines
    

 