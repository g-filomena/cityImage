import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option("precision", 10)

from .utilities import *
from .angles import *

"""
This set of functions is designed for extracting the computational Image of The City.
Computational landmarks can be extracted employing the following functions (see notebooks "2_Landmarks.ipynb" for usages and pipeline).

"""
        
def get_buildings_from_SHP(path, epsg, case_study_area = None, obstructions_area = None, height_field = None, base_field = None, distance_from_center = 1000):

    """    
    The function take a sets of buildings, returns two smaller GDFs of buildings: the case-study area, plus a larger area containing other 
    buildings, called "obstructions" (for analyses which include adjacent buildings). If the area for clipping the obstructions is not
    provided a buffer from the case-study is used to build the obstructions GDF.
            
    Parameters
    ----------
    path: string
    
    epsg: int
    
    case_study_area: Polygon
    
    obstructions_area: Polygon
    
    height_field, base_field: str 
        height and base fields name in the original data-source
    
    distance: float
    
    Returns
    -------
    tuple of GeoDataFrames
    """   
    
    # trying reading buildings footprints shapefile from directory
    city_buildings = gpd.read_file(path).to_crs(epsg=epsg)  
    
    # computing area, reassigning columns
    city_buildings["area"] = city_buildings["geometry"].area
    if height_field != None: city_buildings["height"] = city_buildings[height_field]
    if (base_field == None): 
        city_buildings["base"] = 0.0
        if height_field != None: city_buildings["height_r"] = city_buildings["height"]
    else: 
        city_buildings["base"] = city_buildings[base_field]
        if height_field != None: city_buildings["height_r"] = city_buildings["height"]+city_buildings["base"] # relative_height
        
    # dropping small buildings and buildings with null height
    city_buildings = city_buildings[city_buildings["area"] >= 200]
    if height_field != None: city_buildings = city_buildings[city_buildings["height"] >= 1]
    city_buildings = city_buildings[["height", "height_r", "base","geometry", "area"]]
    
    # assigning ID
    city_buildings["buildingID"] = city_buildings.index.values.astype(int)
    
    # check if case_study_area is defined
    if (case_study_area == None): case_study_area = city_buildings.geometry.unary_union.centroid.buffer(distance_from_center)
    # obtaining obstructions
    if  (obstructions_area == None): obstructions_area = case_study_area.buffer(800)
    obstructions_gdf = city_buildings[city_buildings.geometry.within(obstructions_area)]
        
    # clipping buildings in the case-study area
    buildings_gdf = city_buildings[city_buildings.geometry.within(area_to_clip.geometry.loc[0])]

    return buildings_gdf, obstructions_gdf
    
def get_buildings_from_OSM(place, method = "from_address", distance = 1000, epsg = None):

    """    
    The function downloads and cleans buildings footprint geometries and create as many buildings GeoDataFrames as the number of places specified in the list "list_places".
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.
    
    "list_places""  is used to indicate names of case-study cities.
    "distance" regulates the extension of the area within wich the building are extracted from.
    
            
    Parameters
    ----------
    place: str or tuple 
    
    method: str
    
    distance: float
    
    epsg: int
    
    Returns
    -------
    GeoDataFrames
    """   
    
    columns_to_keep = ['amenity', 'building', 'geometry', 'historic','land_use']

    if method == "from_address": buildings_gdf = ox.footprints.footprints_from_address(address = place, distance = distance, footprint_type = 'building', retain_invalid = False)
    elif method == "from_point": buildings_gdf = ox.footprints.footprints_from_point(point = place, distance = distance, footprint_type= 'building', retain_invalid=False)
    
    if epsg == None: buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else: buildings_gdf.to_crs({'init': epsg})

    buildings_gdf['land_use'] = None
    for column in buildings_gdf.columns: 
        if column.startswith('building:use:'): buildings_gdf.loc[pd.notnull(buildings_gdf[column]), 'land_use'] = column[13:]
        if column not in coluns_to_keep: buildings_gdf.drop(column, axis = 1, inplace = True)

    buildings_gdf = buildings_gdf[~buildings_gdf['geometry'].is_empty]
    buildings_gdf['building'].replace('yes', np.nan, inplace = True)
    buildings_gdf['building'][buildings_gdf['building'].isnull()] = buildings_gdf['amenity']
    buildings_gdf['land_use'][buildings_gdf['land_use'].isnull()] = buildings_gdf['building']
    buildings_gdf['land_use'][buildings_gdf['land_use'].isnull()] = 'residential'

    buildings_gdf = buildings_gdf[['geometry', 'cult', 'land_use']]
    buildings_gdf['area'] = buildings_gdf.geometry.area
    buildings_gdf = buildings_gdf['area' >= 200] 
    
    # reset index
    buildings_gdf = buildings_gdf.reset_index(drop = True)
    buildings_gdf['buildingID'] = buildings_gdf.index.values.astype('int')  
    
    return buildings_gdf
        
def structural_score(buildings_gdf, obstructions_gdf, street_gdf, max_expansion_distance = 300, distance_along = 50, radius = 150):

    """
    The function computes the structural properties of each building properties.
    
    neighbours
    "radius" research radius for other adjacent buildings.
    
    2d advance visibility
    "max_expansion_distance" indicates up to which distance from the building boundaries the visibility polygon can expand.
    "distance_along" defines the interval between each line's destination, namely the search angle.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    
    street_gdf: LineString GeoDataFrame
    
    obstructions_gdf: Polygon GeoDataFrame
    
    radius: float
    
    max_expansion_distance: float
    
    distance_along: float
   
    Returns
    -------
    GeoDataFrame
    """
    # spatial index
    sindex = obstructions_gdf.sindex
    street_network = street_gdf.geometry.unary_union
    buildings_gdf = buildings_gdf.copy()
    
    # distance from road
    buildings_gdf["road"] =  buildings_gdf.apply(lambda row: row["geometry"].distance(street_network), axis = 1)
    # 2d advance visibility
    buildings_gdf["2dvis"] = buildings_gdf.apply(lambda row: _advance_visibility(row["geometry"], obstructions_gdf, sindex, max_expansion_distance = max_expansion_distance,
                                distance_along = distance_along), axis = 1)
    # neighbours
    buildings_gdf["neigh"] = buildings_gdf.apply(lambda row: _number_neighbours(row["geometry"], obstructions_gdf, sindex, radius = radius), axis = 1)
    
    return buildings_gdf
    
def _number_neighbours(building_geometry, obstructions_gdf, obstructions_sindex, radius):

    """
    The function computes a buildings" number of neighbours.
    "radius" research radius for other adjacent buildings.
     
    Parameters
    ----------
    building_geometry: Polygon
    
    obstructions_gdf: Polygon GeoDataFrame
    
    obstructions_sindex: Rtree Spatial Index
    
    radius: float
   
    Returns
    -------
    int
    """
        
    buffer = building_geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)

def _advance_visibility(building_geometry, obstructions_gdf, obstructions_sindex, max_expansion_distance = 600, distance_along = 20):

    """
    It creates a 2d polygon of visibility around a building. The extent of this polygon is assigned as a 2d advance
    visibility measure. The polygon is built constructing lines around the centroid, breaking them at obstructions and connecting 
    the new formed geometries to get the final polygon.
    "max_expansion_distance" indicates up to which distance from the building boundaries the visibility polygon can expand.
    "distance_along" defines the interval between each line's destination, namely the search angle.
     
    Parameters
    ----------
    building_geometry: Polygon
    obstructions_gdf: Polygon GeoDataFrame
    obstructions_sindex: Rtree Spatial Index
    max_expansion_distance: float
    distance_along: float

    Returns
    -------
    float
    """
      
    # creating buffer
    origin = building_geometry.centroid
    exteriors = list(building_geometry.exterior.coords)
    no_holes = Polygon(exteriors)
    max_expansion_distance = max_expansion_distance + origin.distance(building_geometry.envelope.exterior)
    
    # identifying obstructions in an area of x (max_expansion_distance) mt around the building
    possible_obstacles_index = list(obstructions_sindex.intersection(origin.buffer(max_expansion_distance).bounds))
    possible_obstacles = obstructions_gdf.iloc[possible_obstacles_index]
    possible_obstacles = obstructions_gdf[obstructions_gdf.geometry != row[ix_geo]]
    possible_obstacles = obstructions_gdf[~obstructions_gdf.geometry.within(no_holes)]

    start = 0.0
    i = start
    list_lines = [] # list of lines
    
    # creating lines all around the building till a defined distance
    while(i <= 360):
        coords = get_coord_angle([origin.x, origin.y], max_expansion_distance = max_expansion_distance, angle = i)
        line = LineString([origin, Point(coords)])
        
        # finding actual obstacles to this line
        obstacles = possible_obstacles[possible_obstacles.crosses(line)]
        ob = cascaded_union(obstacles.geometry)
        
        """
        if there are obstacles: indentify where the line from the origin is interrupted, create the geometry and
        append it to the list of lines
        """
        
        if len(obstacles > 0):
            t = line.intersection(ob)
            # taking the coordinates
            try: intersection = t[0].coords[0]
            except: intersection = t.coords[0]
            lineNew = LineString([origin, Point(intersection)])
        
        # the line is not interrupted, keeping the original one
        else: lineNew = line 

        list_lines.append(lineNew)
        # increase the angle
        i = i+distance_along
   
    # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
    list_points = [Point(origin)]
    for i in list_lines: list_points.append(Point(i.coords[1]))
    list_points.append(Point(origin))
    poly = Polygon([[p.x, p.y] for p in list_points])
    
    # subtracting th area of the building and computing the area of the polygon (area of visibility)
    try: poly_vis = poly.difference(building_geometry)
    except:
        pp = poly.buffer(0)
        poly_vis = pp.difference(building_geometry)      
    
    return poly_vis.area
    
def visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a' : []})):

    """
    The function calculates a 3d visibility score making use of precomputed 3d sight lines.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    
    sight_lines: LineString GeoDataFrame
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0
    buildings_gdf["3dvis"] = 0.0
    if "height" not in buildings_gdf.columns: return buildings_gdf
        
    #facade area (roughly computed)
    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: _facade_area(row["geometry"], row["height"]), axis = 1)
    if sight_lines.empty: return buildings_gdf
    
    # 3d visibility
    sight_lines = sight_lines.copy()
    sight_lines.drop(["Shape_Leng"], axis = 1, inplace = True, errors = "ignore")
    sight_lines["length"] = sight_lines["geometry"].length
    
    # average distance
    sight_lines = (sight_lines[["buildingID", "length", "nodeID"]].groupby(["buildingID", "nodeID"], as_index = False)["length"].max())
    avg = (sight_lines[["buildingID", "length"]].groupby(["buildingID"], as_index = False)["length"].mean())
    avg.rename(columns={"length":"mean_length"}, inplace=True)
    
    # count
    count = (sight_lines[["buildingID", "length"]].groupby(["buildingID"],as_index=False)["length"].count())
    count.rename(columns={"length":"n_slines"}, inplace=True)
    
    # max distance
    tmp = sight_lines.set_index("buildingID")
    distances = tmp.groupby("buildingID").agg(lambda copy: copy.values[copy["length"].values.argmax()])
    distances = distant.reset_index()
    
    # merging the data
    visibility_tmp = pd.merge(distances, avg, left_on = "buildingID", right_on = "buildingID")
    visibility = pd.merge(visibility_tmp, count, left_on= "buildingID", right_on = "buildingID")   
    
    # dropping and rename columns
    visibility.drop(["DIST_ALONG", "visible", "Visibility", "geometry"], axis = 1, errors = "ignore", inplace=True)
    visibility.rename(columns = {"length": "max_dist", "mean_length":"mean_dist"}, inplace = True) 
    
    # computing score on rescaled values
    visibility["max_dist"].fillna((tmp["max_dist"].min()), inplace = True) # longest sight_line to each building
    visibility["mean_dist"].fillna((tmp["mean_dist"].min()), inplace = True) # average distance sigh_lines to each buildings
    visibility["n_slines"].fillna((tmp["n_slines"].min()), inplace = True) # number of sigh_lines to each buildings
    col = ["max_dist", "mean_dist_dist", "n_slines"]                     
    for i in col: scaling_columnDF(visibility, i)
    # computing the 3d visibility score
    visibility["3dvis"] = tmp["max_dist_sc"]*0.5+tmp["mean_dist_sc"]*0.25+tmp["n_slines_sc"]*0.25

    # merging and building the final output
    buildings_gdf = pd.merge(buildings_gdf, visibility[["buildingID", "3dvis"]], on = "buildingID", how = "left") 

    return buildings_gdf

def _facade_area(building_geometry, building_height):

    """
    The function roughly computes the facade area of a building, given its geometry and height
     
    Parameters
    ----------
    building_geometry: Polygon
    building_height: float
   
    Returns
    -------
    float
    """
    
    envelope = building_geometry.envelope
    coords = mapping(envelope)["coordinates"][0]
    d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
    width = min(d)
    return width*building_height
    
def cultural_score_from_dataset(buildings_gdf, historical_elements_gdf, score = None):

    """
    The function computes a cultural score based on the number of features listed in historical/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the buildings object 
    of analysis.
    
    "score" indicates the attribute field containing scores assigned to historical buildings, if existing.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    historical_elements_gdf: Point or Polygon GeoDataFrame
    score: string
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    # spatial index
    sindex = historical_elements_gdf.sindex 
    buildings_gdf["cult"] = 0
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 
    
    buildings_gdf["cult"] = buildings_gdf.apply(lambda row: _compute_cultural_score_building(row["geometry"], sindex, score = score), axis = 1)
    return buildings_gdf
    
def _compute_cultural_score_building(building_geometry, building_land_use, historical_elements_gdf_sindex, score = None):

    """
    Compute pragmatic for a single building. It supports the function "pragmatic_meaning" 
     
    Parameters
    ----------
    buildings_geometry: Polygon
    building_land_use: string
    historical_elements_gdf_sindex: Rtree spatial index
    score: string
   
    Returns
    -------
    float
    """

    possible_matches_index = list(historical_elements_gdf_sindex.intersection(building_geometry.bounds)) # looking for possible candidates in the external GDF
    possible_matches = historical_elements_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]

    if (score == None): cs = len(pm) # score only based on number of intersecting elements
    elif len(pm) == 0: cs = 0
    else: cs = pm[score].sum() # otherwise sum the scores of the intersecting elements
    
    return cs

def cultural_score_from_OSM(building_gdf):

    """
    The function computes a cultural score simply based on a binary categorisation. This function exploits the tag "historic" in OSM buildings data.
    When such field is filled in OSM, the building is considered semantically meaningful. 
    Therefore, the score is either 0 or 1.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
   
    Returns
    -------
    GeoDataFrame
    """ 
    
    buildings_gdf = buildings_gdf.copy()    
    buildings_gdf["cult"] = 0
    if "historic" not in building_gdf.columns: return buildings_gdf
    building_gdf["historic"][building_gdfj["historic"].isnull()] = 0
    building_gdf["historic"][building_gdf["historic"] != 0] = 1
    building_gdf["cult"] = building_gdf["historic"]
        
    return buildings_gdf

def pragmatic_score(buildings_gdf, radius = 200):

    """
    Compute pragmatic score based on the frequency, and therefore unexpctdness, of a land_use class in an area around a building.
    The area is defined by the parameter "radius".
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    buffer: float
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()   
    buildings_gdf["nr"] = 1 # to count
    sindex = buildings_gdf.sindex # spatial index
    buildings_gdf["prag"] = buildings_gdf.apply(lambda row: _compute_prag_min(row.geometry, row.land_use, sindex, radius), axis = 1)
    
    return buildings_gdf
    
def _compute_pragmatic_meaning_building(building_geometry, building_land_use, buildings_gdf_sindex, radius):

    """
    Compute pragmatic for a single building. It supports the function "pragmatic_meaning" 
     
    Parameters
    ----------
    buildings_geometry: Polygon
    building_land_use: String
    buildings_gdf_sindex: Rtree Spatial index
    radius: float
   
    Returns
    -------
    float
    """

    buffer = building_geometry.buffer(radius)

    possible_matches_index = list(sindex.intersection(buffer.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    pm = possible_matches [possible_matches.intersects(buffer)]
    neigh = pm.groupby(["land_use"], as_index = True)["nr"].sum() 
    Nj = neigh.loc[building_land_use] # nr of neighbours with same land_use
    
    # Pj = Nj/N
    Pj = 1-(Nj/pm["nr"].sum()) # inverting the value
        
    return Pj
        
        
def compute_global_scores(buildings_gdf, g_cW, g_iW):
    """
    The function computes component and global scores, rescaling values when necessary and assigning weights to the different 
    properties measured.
    The user must provide two dictionaries:
    - g_cW: keys are component names (string), items are weights
    - g_iW: keys are index names (string), items are weights
    
    Example:
    g_cW = {"vScore": 0.50, "sScore" : 0.30, "cScore": 0.20, "pScore": 0.10}
    g_iW = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.30, "2dvis":0.30, "neigh": 0.20 , "road": 0.20}
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    g_cW, g_iW: dictionaries
   
    Returns
    -------
    GeoDataFrame
    """
    
    # scaling
    col = ["3dvis", "fac", "height", "area","2dvis", "cult", "prag"]
    col_inverse = ["neigh", "road"]
    
    if "height" not in buildings_gdf.columns: buildings_gdf['height'] = 0.0
    
    for i in col: 
        if buildings_gdf[i].max() == 0.0: buildings_gdf[i+"_sc"] = 0.0
        else: scaling_columnDF(buildings_gdf, i)
    
    for i in col_inverse: 
        if buildings_gdf[i].max() == 0.0: buildings_gdf[i+"_sc"] = 0.0
        else: scaling_columnDF(buildings_gdf, i, inverse = True) 
  
    # computing scores   
    buildings_gdf["vScore"] = buildings_gdf["fac_sc"]*g_iW["fac"] + buildings_gdf["height_sc"]*g_iW["height"] + buildings_gdf["3dvis"]*g_iW["3dvis"]
    buildings_gdf["sScore"] = buildings_gdf["area_sc"]*g_iW["area"] + buildings_gdf["neigh_sc"]*g_iW["neigh"] + buildings_gdf["2dvis_sc"]*g_iW["2dvis"] + buildings_gdf["road_sc"]*g_iW["road"]
    
    # rescaling components
    col = ["vScore", "sScore"]
    for i in col: 
        if buildings_gdf[i].max() == 0.0: buildings_gdf[i+"_sc"] = 0.0
        else: scaling_columnDF(buildings_gdf, i)
    
    buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    
    # final global score
    buildings_gdf["gScore"] = (buildings_gdf["vScore_sc"]*g_cW["vScore"] + buildings_gdf["sScore_sc"]*g_cW["sScore"] + 
                               buildings_gdf["cScore"]*g_cW["cScore"] + buildings_gdf["pScore"]*g_cW["pScore"])

    scaling_columnDF(buildings_gdf, "gScore")
    
    return buildings_gdf



def compute_local_scores(buildings_gdf, l_cW, l_iW, radius = 1500):

    """
    The function computes landmarkness at the local level. The components' weights may be different from the ones used to calculate the
    global score. The radius parameter indicates the extent of the area considered to rescale the landmarkness local score.
    - l_cW: keys are component names (string), items are weights.
    - l_iW: keys are index names (string), items are weights.
    
    # local landmarkness components weights
    l_cW = {"vScore": 0.25, "sScore" : 0.35, "cScore":0.10 , "pScore": 0.30}
    # local landmarkness indexes weights, cScore and pScore have only 1 index each
    l_iW = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.40, "2dvis": 0.00, "neigh": 0.30 , "road": 0.30}
    
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    l_cW, l_iW: dictionaries
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf.index = buildings_gdf.buildingID
    del buildings_gdf.index.name
    
    spatial_index = buildings_gdf.sindex # spatial index
    buildings_gdf["lScore"] = 0.0
    buildings_gdf["vScore_l"], buildings_gdf["sScore_l"] = 0.0, 0.0
                                          
    col = ["3dvis", "fac", "height", "area","2dvis", "cult","prag"]
    col_inverse = ["neigh", "road"]
   
    # recomputing the scores per each building in relation to its neighbours, in an area whose extent is regulated by the parameter "radius"
    buildings_gdf["lScore"] = buildings_gdf.apply(lambda row: _building_local_score(row["geometry"], row["buildingID"], buildings_gdf, sinde, radius), axis = 1)
    scaling_columnDF(buildings_gdf, "lScore")
    return buildings_gdf
    
def _building_local_score(building_geometry, buildingID, buildings_gdf, buildings_gdf_sindex,l_cW, l_iW, radius):

    """
    The function computes landmarkness at the local level for a single building. 

    
    Parameters
    ----------
    building_geometry: Polygon
    buildingID: int
    buildings_gdf: Polygon GeoDataFrame
    buildings_gdf_sindex: Rtree spatial index
    l_cW, l_iW: dictionaries
    radius: float, regulates the extension of the area wherein the scores are recomputed, around the building
   
    Returns
    -------
    GeoDataFrame
    """

    buffer = building_geometry.buffer(radius)
    possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index].copy()
    pm = possible_matches[possible_matches.intersects(buffer)]
    
    # rescaling the values 
    for i in col: scaling_columnDF(pm, i) 
    for i in col_inverse: scaling_columnDF(pm, i, inverse = True)
    
    # and recomputing scores
    pm["vScore_l"] =  pm["fac_sc"]*l_iW["fac"] + pm["height_sc"]*l_iW["height"] + pm["3dvis"]*l_iW["3dvis"]
    pm["sScore_l"] =  pm["area_sc"]*l_iW["area"]+ pm["neigh_sc"]*l_iW["neigh"] + pm["road_sc"]*l_iW["road"] + pm["2dvis_sc"]*l_iW["fac"]
    pm["cScore_l"] = pm["cult_sc"]
    pm["pScore_l"] = pm["prag_sc"]
    
    col_rs = ["vScore_l", "sScore_l"]
    for i in col_rs: scaling_columnDF(pm, i)
    pm["lScore"] =  pm["vScore_l_sc"]*l_cW["vScore"] + pm["sScore_l_sc"]*l_cW["sScore"] + pm["cScore_l"]*l_cW["cScore"] + pm["pScore_l"]*l_cW["pScore"]
    # return the so obtined score
    return float("{0:.3f}".format(pm["lScore"].loc[buildingID]))




