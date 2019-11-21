import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option('precision', 10)

import utilities as uf

"""
This set of functions is designed for extracting the computational Image of The City.
Computational landmarks can be extracted employing the following functions (see notebooks '2_Landmarks.ipynb' for usages and pipeline).

"""
        
	

def select_buildings(buildings, area_to_clip, height_field, base_field = None, area_obstructions = None):
    """    
    The function take a sets of buildings, returns two smaller GDFs of buildings: the case-study area, plus a larger area containing other 
    buildings, called 'obstructions' (for analyses which include adjacent buildings). If the area for clipping the obstructions is not
    provided a buffer from the case-study is used to build the obstructions GDF.
            
    Parameters
    ----------
    city_buildings, area_to_clip: GeoDataFrames
    height_field, base_field: strings height and base fields name in the original data-source
    area_obstructions: GeoDataFrame
    
    Returns
    -------
    GeoDataFrames
    """   
    # computing area, reassigning columns
    city_buildings['area'] = city_buildings['geometry'].area
    city_buildings['height'] = city_buildings[height_field]
    if (base is None): city_buildings['base'] = 0
    else: city_buildings['base'] = city_buildings[base]
    
    # dropping small buildings and buildings with null height
    city_buildings = city_buildings[city_buildings['area'] > 199]
    city_buildings = city_buildings[city_buildings['height'] >= 1]
    city_buildings = city_buildings[['height', 'base','geometry', 'area']]
    
    # creating ID
    city_buildings['buildingID'] = city_buildings.index.values.astype(int)
    
    # clipping obstructions
    if  (area_obstructions is None): area_obstructions = area_to_clip.geometry.loc[0].buffer(800)
    else: area_obstructions = area_obstructions.geometry.loc[0]
        
    city_buildings = city_buildings[(city_buildings['area'] > 199) & (city_buildings['height'] >= 1)]
    obstructions = city_buildings[city_buildings.geometry.within(area_obstructions)]
        
    # clipping buildings in the case-study area
    study_area_buildings = city_buildings[city_buildings.geometry.within(area_to_clip.geometry.loc[0])]
    study_area_buildings['r_height'] = study_area_buildings['height'] + study_area_buildings['base'] # relative_height

    return(study_area_buildings, obstructions)
   
def structural_properties(buildings_gdf, obstructions_gdf, street_gdf, buffer = 150):
    """
    The function extract properties that will be used for computing the structural score plus façade area (visibility score).
     
    Parameters
    ----------
    buildings_gdf, obstructions_gdf, street_gdf: GeoDataFrames
    buffer: float
   
    Returns
    -------
    GeoDataFrame
    """
    
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 
    
    # spatial index
    sindex = obstructions_gdf.sindex
    street_network = street_gdf.geometry.unary_union
    buildings_gdf['neigh'] = 0.0
    buildings_gdf['road'] = 0.0
    
    for row in buildings_gdf.itertuples():    
        g = row[ix_geo]
        buff = g.buffer(buffer)
        t = g.envelope
        coords = mapping(t)['coordinates'][0]
        d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]

        # width and length
        width = min(d)
        length = max(d)
        buildings_gdf.at[row[0], 'width'] =  width
        buildings_gdf.at[row[0], 'length'] = length
        
        # neighbours
        possible_neigh_index = list(sindex.intersection(buff.bounds))
        possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
        precise_neigh = possible_neigh[possible_neigh.intersects(buff)]
        buildings_gdf.at[row[0], 'neigh'] = len(precise_neigh)
        
        # distance from the road
        dist = g.distance(street_network)
        buildings_gdf.at[row[0], 'road'] = dist
    
    #facade area (roughly computed)
    if 'height' in buildings_gdf.columns: buildings_gdf['fac'] = buildings_gdf['height']*(buildings_gdf.width)
    buildings_gdf.drop(['width','length'], axis=1, inplace = True)
    
    return buildings_gdf

def advance_visibility(buildings_gdf, obstructions_gdf, distance = 300):
    """
    It creates a 2d polygon of visibility around each building in 'building_gdf'. The extent of this polygon is assigned as an advance
    visibility rough measure. The polygon is built constructing lines around the centroid, breaking them at obstructions and connecting 
    the new formed geometries to get the final polygon.
    It also returns, besides the updated 'buildings_gdf' a GDF containing the visibility polygons, for further analysis (usually with
    several incorrect geometries).
     
    Parameters
    ----------
    buildings_gdf, obstructions_gdf: GeoDataFrames
    distance: float
   
    Returns
    -------
    GeoDataFrames
    """
    
    visibility_polygons = buildings_gdf[['buildingID', 'geometry']].copy()
    buildings_gdf['a_vis'] = 0.0
    
    # sindex
    sindex = obstructions_gdf.sindex
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1
    counter = 0
    
    for row in buildings_gdf.itertuples():
        
        # indicates progress
        uf.print_row(row.Index)        
        # creating buffer
        origin = row[ix_geo].centroid
        exteriors = list(row[ix_geo].exterior.coords)
        no_holes = Polygon(exteriors)
        
        # identifying obstructions in an area of 2000 mt around the building
        possible_obstacles_index = list(sindex.intersection(origin.buffer(2000).bounds))
        possible_obstacles = obstructions_gdf.iloc[possible_obstacles_index]
        possible_obstacles = obstructions_gdf[obstructions_gdf.geometry != row[ix_geo]]
        possible_obstacles = obstructions_gdf[~obstructions_gdf.geometry.within(no_holes)]

        start = 0.0
        i = start
        list_lines = [] # list of lines
        
        # creating lines all around the building till a defined distance
        while(i <= 360):

            coords = uf.get_coord_angle([origin.x, origin.y], distance = distance, angle = i)
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
                try:
                    intersection = t[0].coords[0]
                except:
                    intersection = t.coords[0]
                
                lineNew = LineString([origin, Point(intersection)])
            
            # the line is not interrupted, keeping the original one
            else: lineNew = line 

            list_lines.append(lineNew)
            
            # increase the angle
            i = i+10
       
        # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
        list_points = [Point(origin)]
        for i in list_lines: list_points.append(Point(i.coords[1]))
        list_points.append(Point(origin))
        poly = Polygon([[p.x, p.y] for p in list_points])
        
        # subtracting th area of the building and computing the area of the polygon (area of visibility)
        try:
            poly_vis = poly.difference(row[ix_geo])
        except:
            pp = poly.buffer(0)
            poly_vis = pp.difference(row[ix_geo])      
        buildings_gdf.at[row[0],'a_vis'] = poly_vis.area
        
        """
        !! it does not work always - saving the polygon in a GDF containing visibility polygons. 
        It may create irregular multi part polygons.
        """
        try:
            if len(poly_vis) > 1: #MultiPolygon
                for i in range(0, len(poly_vis)): 
                    if (poly_vis[i].area < 100): del poly_vis[i]
        except:
            poly_vis = poly_vis

        visibility_polygons.at[row[0],'geometry'] = poly_vis
        
    return buildings_gdf, visibility_polygons

def reassign_sight_lines_to_new_nodes(nodes_gdf, sight_lines_gdf):
    
    sight_lines_gdf = sight_lines_gdf.copy()
    ix_geo = sight_lines_gdf.columns.get_loc("geometry")+1
    sindex = nodes_gdf.sindex
    to_drop = []
    
    for row in sight_lines_gdf.itertuples():
        start = row[ix_geo].coords[0]
        tmp = nodes_gdf[(nodes_gdf.x == start[0]) & (nodes_gdf.y == start[1])].copy()
        if len(tmp) != 0: sight_lines_gdf.at[row.Index, 'nodeID'] = tmp.iloc[0].nodeID
        else:
            point = Point(start)
            buffer = point.buffer(10)
            possible_matches_index = list(sindex.intersection(buffer.bounds))
            possible_matches = nodes_gdf.iloc[possible_matches_index]
            pm = possible_matches[possible_matches.intersects(buffer)]

            if len(pm) != 0:
                distance, index = uf.dist_to_gdf_point(point, pm)
                sight_lines_gdf.at[row.Index, 'nodeID'] = nodes_gdf.loc[index].nodeID
                new_line = list(row[ix_geo].coords)
                new_line[0] = (nodes_gdf.loc[index].x, nodes_gdf.loc[index].y)
                geo_line = LineString([coor for coor in new_line])
                sight_lines_gdf.at[row.Index, 'geometry'] = geo_line
            else:
                to_drop.append(row.Index)
    
    sight_lines_gdf.drop(to_drop, axis = 0, inplace = True)
    print("done") 
    return sight_lines_gdf

def visibility(buildings_gdf, sight_lines):
    """
    The function calculates a 3d visibility score making use of precomputed 3d sight lines
     
    Parameters
    ----------
    buildings_gdf, sight_lines: GeoDataFrames
   
    Returns
    -------
    GeoDataFrame
    """
    
    # cleaning the GDF, extracting count, max and mean length of sight-linesper each building
    sight_lines = sight_lines.copy()
    sight_lines.drop(['Shape_Leng'], axis = 1, inplace = True)
    sight_lines['length'] = sight_lines['geometry'].length
    
    # average distance
    sight_lines = (sight_lines[['buildingID', 'length', 'nodeID']].groupby(['buildingID', 'nodeID'], as_index = False)['length'].max())
    avg = (sight_lines[['buildingID', 'length']].groupby(['buildingID'], as_index = False)['length'].mean())
    avg.rename(columns={'length':'mean_length'}, inplace=True)
    
    # count
    count = (sight_lines[['buildingID', 'length']].groupby(['buildingID'],as_index=False)['length'].count())
    count.rename(columns={'length':'n_slines'}, inplace=True)
    
    # max distance
    tmp = sight_lines.set_index('buildingID')
    distant = tmp.groupby('buildingID').agg(lambda copy: copy.values[copy['length'].values.argmax()])
    distant = distant.reset_index()
    
    # merging the data
    visibility_tmp = pd.merge(distant, avg, left_on = 'buildingID', right_on = 'buildingID')
    visibility = pd.merge(visibility_tmp, count, left_on= 'buildingID', right_on = 'buildingID')   
    
    # dropping and rename columns
    visibility.drop(['DIST_ALONG', 'visible', 'Visibility', 'geometry'], axis = 1, errors = 'ignore', inplace=True)
    visibility.rename(columns = {'length':'dist','mean_length':'m_dist'}, inplace=True) 

    # merging and building the final output
    tmp = pd.merge(buildings_gdf, visibility[['buildingID', 'dist', 'm_dist', 'n_slines']], on = 'buildingID', how= 'left') 
    tmp['dist'].fillna((tmp['dist'].min()), inplace = True)
    tmp['m_dist'].fillna((tmp['m_dist'].min()), inplace=True)
    tmp['n_slines'].fillna((tmp['n_slines'].min()), inplace=True)
    
    # computing the 3d visibility score
    col = ['dist', 'm_dist', 'n_slines']                     
    for i in col: uf.scaling_columnDF(tmp, i)
    tmp['vis'] = tmp['dist_sc']*0.5+tmp['m_dist_sc']*0.25+tmp['n_slines_sc']*0.25
    
    return tmp

def cultural_meaning(buildings_gdf, cultural_gdf, score = None):
    """
    The function computes a cultural score based on the number of features listed in historical/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the building object 
    of analysis.
     
    Parameters
    ----------
    buildings_gdf, cultural_gdf: GeoDataFrames, buildings and external data-sets
   
    Returns
    -------
    GeoDataFrame
    """
    # spatial index
    sindex = cultural_gdf.sindex 
    buildings_gdf['cult'] = 0
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 
    
    for row in buildings_gdf.itertuples():
        g = row[ix_geo] # geometry
        possible_matches_index = list(sindex.intersection(g.bounds)) # looking for possible candidates in the external GDF
        possible_matches = cultural_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(g)]
        
        if (score == None): cm = len(precise_matches) # score only based on number of intersecting elements
        elif len(precise_matches) == 0: cm = 0
        else: cm = precise_matches[score].sum() # otherwise sum the scores of the intersecting elements
        
        buildings_gdf.at[row[0], 'cult'] = cm #cultural meaning
     
    return buildings_gdf

def classify_lu(buildings_gdf, land_use):
    """
    The function reclassifies land-use descriptors in a land-use field according to the categorisation presented below. 
     
    Parameters
    ----------
    buildings_gdf: GeoDataFrame
    land_use: string, the land use field
   
    Returns
    -------
    GeoDataFrame
    """
    
    # introducing classifications and possible entries
    
    university = ['university', 'college', 'research']
    commercial = ['bank', 'service',  'commercial',  'retail', 'Retail',  'pharmacy', 'commercial;educa', 'shop', 'Commercial',
                  'supermarket', 'offices', 'foundation', 'office', 'books', 'Commercial services', 'Commercial Land', 
                  'Mixed Use Res/Comm',  'Commercial Condo Unit']
    
    residential = [ 'apartments', None, 'NaN', 'residential','flats', 'no', 'houses', 'garage', 'garages', 'building', 
                  'roof', 'storage_tank', 'shed', 'silo',  'parking',  'toilets', 'picnic_site','hut', 'information', 'viewpoint',
                  'atm',   'canopy', 'smokestack', 'greenhouse', 'fuel', 'Residential Condo Unit', 'Apartments 4-6 Units', 
                  'Residential Two Family', 'Apartments 7 Units above', 'Residential Single Family', 'Condominium Parking', 
                  'Residential Three Family', 'Condominium Master', 'Residential Land']
    
    attractions = ['Attractions', 'museum',  'castle', 'cathedral', 'attraction','aquarium', 'monument',  'gatehouse',
                   'terrace', 'tower', 'Attraction And Leisure']
    hospitality = [ 'hotel',  'hostel', 'guest_house']
    eating_drinking = [ 'restaurant', 'fast_food', 'cafe', 'bar',  'pub', 'Accommodation, eating and drinking',]
    public = ['post_office', 'townhall', 'public_building',  'library','civic', 'courthouse', 'public', 'embassy',
              'Public infrastructure', 'community_centre', 'parking', 'dormitory', 'Exempt', 'Exempt 121A']
    library = ['library']
    sport = ['stadium', 'Sport and entertainment', 'Sports Or Exercise Facility']
    entertainment = [ 'exhibition_centr','theatre', 'cinema']
    education = ['school', 'kindergarten', 'Education', 'Education and health']
    religious = ['church', 'place_of_worship','convent', 'rectory', 'Religious Buildings']
    emergency_service = [ 'fire_station','police', 'Emergency Service']
    transport = [ 'station', 'train_station']
    medical_care = ['hospital', 'doctors', 'dentist','clinic','veterinary', 'Medical Care']
    industrial = [ 'industrial', 'factory', 'construction', 'Manufacturing and production',  'gasometer', 'data_center']
    cultural = ['club_house','gallery', 'arts_centre','Cultural Facility']
    military = ['general aviation', 'Barracks']
    transport = ['Transport', 'Road Transport', 'station', 'subway_entrance', 'bus_station']
    
    
    # reclassifying: replacing original values with relative categories
    buildings_gdf[land_use] = buildings_gdf[land_use].map( lambda x: 'university' if x in university
                                                              else 'commercial' if x in commercial
                                                              else 'residential' if x in residential
                                                              else 'attractions' if x in attractions
                                                              else 'library' if x in library
                                                              else 'hospitality' if x in hospitality
                                                              else 'eating_drinking' if x in eating_drinking
                                                              else 'public' if x in public
                                                              else 'sport' if x in sport
                                                              else 'entertainment' if x in entertainment
                                                              else 'education' if x in education
                                                              else 'religious' if x in religious
                                                              else 'emergency_service' if x in emergency_service
                                                              else 'industrial' if x in industrial
                                                              else 'cultural' if x in cultural
                                                              else 'transport' if x in transport
                                                              else 'medical_care' if x in medical_care
                                                              else 'military' if x in military
                                                              else 'other')
    
    buildings_gdf[land_use][buildings_gdf[land_use].str.contains('residential') | buildings_gdf[land_use].str.contains('Condominium') | buildings_gdf[land_use].str.contains('Residential')] = 'residential'
    
    buildings_gdf[land_use][buildings_gdf[land_use].str.contains('commercial') | 
                   buildings_gdf[land_use].str.contains('Commercial')] = 'residential'
    
    return(buildings_gdf)




def land_use_from_polygons(buildings_gdf, other_gdf, column, land_use_field):
    """
    It assigns land-use attributes to buildings in 'buildings_gdf', looking for possible matches in 'other_gdf' a polygons GDF.
    Possible matches here means the buildings in 'other gdf' whose area of interesection with the examined building (y), covers at least
    60% of the building's (y) area. The best match is chosen. 
     
    Parameters
    ----------
    buildings_gdf, other_gdf: GeoDataFrames, other_gdf is the GDF wherein looking for land_use attributes
    column: string, name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: name of the column in other_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf[column] = None
    
    # spatial index
    sindex = other_gdf.sindex
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 
    ix_geoPol = other_gdf.columns.get_loc("geometry")+1 
    
    for row in buildings_gdf.itertuples():

        g = row[ix_geo] #geometry
        possible_matches_index = list(sindex.intersection(g.bounds)) # looking for intersecting geometries
        possible_matches = other_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(g)]
        precise_matches['area'] = 0.0

        if (len(precise_matches) == 0): continue # no intersecting features in the other_gdf

        for row_C in precise_matches.itertuples(): # for each possible candidate, computing the extension of the area of intersection
            t = row_C[ix_geoPol]
            try:
                area_intersec = t.intersection(g).area
            except: 
                 continue
                    
            precise_matches.at[row_C[0], 'area'] = area_intersec
        
        # sorting the matches based on the extent of the area of intersection
        pm = precise_matches.sort_values(by='area', ascending=False).reset_index()
        
        # assigning the match land-use category if the area of intersection covers at least 60% of the building's area
        if (pm['area'].loc[0] > (g.area * 0.59)):
            main_use = pm[land_use_field].loc[0]
            buildings_gdf.at[row[0], column] = main_use
        else: continue
        
    return buildings_gdf


def land_use_from_points(buildings_gdf, other_gdf, column, land_use_field):
    """
    It assigns land-use attributes to buildings in 'buildings_gdf', looking for possible matches in 'other_gdf', a polygons GDF.
    Possible matches here means features in 'other gdf' which lies within the examined building's area.
     
    Parameters
    ----------
    buildings_gdf, other_gdf: GeoDataFrames, other_gdf is the GDF (Points) wherein looking for land_use attributes
    column: string, name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: name of the column in other_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """
        
    other_gdf['nr'] = 1
    buildings_gdf[column] = None
    sindex = other_gdf.sindex # spatial index

    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 

    for row in buildings_gdf.itertuples():

        g = row[ix_geo] # geometry

        possible_matches_index = list(sindex.intersection(g.bounds))
        possible_matches = other_gdf.iloc[possible_matches_index]

        if (len(possible_matches)==0): continue # no intersecting features in the other_gdf
        else:
            use = possible_matches.groupby([land_use_field],as_index=False)['nr'].sum().sort_values(by='nr', # counting nr of features
                                    ascending=False).reset_index()
        
        main_use = use[land_use_field].loc[0] # assigning the most represented land-use value within the building.
        buildings_gdf.at[row[0], column] = main_use
               
    return buildings_gdf


        
def pragmatic_meaning(buildings_gdf, radius = 200):
    """
    Compute pragmatic score based on the frequency, and therefore unexpctdness, of a land_use class in an area around a building.
    The area is defined by the parameter 'radius'.
     
    Parameters
    ----------
    buildings_gdf: GeoDataFrame
    buffer: float
   
    Returns
    -------
    GeoDataFrame
    """
        
    buildings_gdf['nr'] = 1 # to count
    sindex = buildings_gdf.sindex # spatial index
    buildings_gdf['prag'] = 0.0
	buildings_gdf['prag'] = buildings_gdf.apply(lambda row: compute_prag_min(row.geometry, row.land_use, radius))
	
	return buildings_gdf
	
	
def compute_pragmatic_meaning_building(building_geometry, building_land_use, radius)
    """
    Compute pragmatic for a single building. It supports the function 'pragmatic_meaning' 
     
    Parameters
    ----------
    buildings_geometry: Polygon
    building_land_use: String
	radius: float
   
    Returns
    -------
    float
    """


	building_geometry = row[ix_geo] #geometry
	buffer = building_geometry.buffer(radius)

	possible_matches_index = list(sindex.intersection(buffer.bounds))
	possible_matches = buildings_gdf.iloc[possible_matches_index]
	precise_matches = possible_matches [possible_matches.intersects(buffer)]
	neigh = precise_matches.groupby(['land_use'], as_index=True)['nr'].sum() 
	Nj = neigh.loc[building_land_use] # nr of neighbours with same land_use
	
	# Pj = Nj/N
	Pj = 1-(Nj/precise_matches['nr'].sum()) # inverting the value
        
    return Pj
        
        
def compute_scores(buildings_gdf, g_cW, g_iW):
    """
    The function computes component and global scores, rescaling values when necessary and assigning weights to the different 
    properties measured.
     
    Parameters
    ----------
    buildings_gdf: GeoDataFrame
   
    Returns
    -------
    GeoDataFrame
    """
    
    # scaling
    col = ['vis', 'fac', 'height', 'area','a_vis', 'cult','prag']
    col_inverse = ['neigh', 'road']
                                                                     
    for i in col: 
        if i not in buildings_gdf.columns:
            buildings_gdf[i] = 0.0
            buildings_gdf[i+'_sc'] = 0.0
        else: 
            if buildings_gdf[i].max() == 0.0: buildings_gdf[i+'_sc'] = 0.0
            else: uf.scaling_columnDF(buildings_gdf, i)
    
    for i in col_inverse: 
        if i not in buildings_gdf.columns:
            buildings_gdf[i] = 0.0
            buildings_gdf[i+'_sc'] = 0.0
        else: 
            if buildings_gdf[i].max() == 0.0: buildings_gdf[i+'_sc'] = 0.0
            else: uf.scaling_columnDF(buildings_gdf, i, inverse = True) 
  
    # computing scores   
    buildings_gdf['vScore'] = (buildings_gdf['fac_sc']*g_iW['fac'] + buildings_gdf['height_sc']*g_iW['height'] +
                                buildings_gdf['vis']*g_iW['vis'])
    buildings_gdf['sScore'] = (buildings_gdf['area_sc']*g_iW['area'] + buildings_gdf['neigh_sc']*g_iW['neigh'] + 
                               buildings_gdf['a_vis_sc']*g_iW['a_vis']+buildings_gdf['road_sc']*g_iW['road'])
    
    # rescaling components
    col = ['vScore', 'sScore']
    for i in col: 
        if buildings_gdf[i].max() == 0.0: buildings_gdf[i+'_sc'] = 0.0
        else: uf.scaling_columnDF(buildings_gdf, i)
    buildings_gdf['cScore'] = buildings_gdf['cult_sc']
    buildings_gdf['pScore'] = buildings_gdf['prag_sc']
    
    # final global score
    buildings_gdf['gScore'] = (buildings_gdf['vScore_sc']*g_cW['vScore'] + buildings_gdf['sScore_sc']*g_cW['sScore'] + 
                               buildings_gdf['cScore']*g_cW['cScore'] + buildings_gdf['pScore']*g_cW['pScore'])

    uf.scaling_columnDF(buildings_gdf, 'gScore')
    
    return buildings_gdf



def local_scores(buildings_gdf, l_cW, l_iW, buffer = 1500):
    """
    The function compute landmarkness at the local level. Here the components' weights are different from the ones used to calculate the
    global score. The buffer parameter indicates the extent of the area considered to rescale the landmarkness local score.
     
    Parameters
    ----------
    buildings_gdf: GeoDataFrame
    buffer: float, regulates the extension of the area wherein the scores are recomputed, around each building
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    spatial_index = buildings_gdf.sindex # spatial index
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1
    buildings_gdf['lScore'] = 0.0
    buildings_gdf['vScore_l'] = 0.0
    buildings_gdf['sScore_l'] = 0.0
                                          
                                          
    for i in col+col_inverse: 
        if i not in buildings_gdf.columns: buildings_gdf[i] = 0.0  
    col = ['vis', 'fac', 'height', 'area','a_vis', 'cult','prag']
    col_inverse = ['neigh', 'road']
   
    # recomputing the scores per each building in relation to its neighbours, in an area whose extent is regulated by 'buffer'
    for row in buildings_gdf.itertuples():
        b = row[ix_geo].centroid.buffer(buffer)
        possible_matches_index = list(spatial_index.intersection(b.bounds))
        possible_matches = buildings_gdf.iloc[possible_matches_index].copy()
        LL = possible_matches[possible_matches.intersects(b)]
        
        # rescaling the values 
        for i in col: uf.scaling_columnDF(LL, i) 
        for i in col_inverse: uf.scaling_columnDF(LL, i, inverse = True)
        
        # and recomputing scores
        LL['vScore_l'] =  LL['fac_sc']*l_iW['fac'] + LL['height_sc']*l_iW['height'] + LL['vis']*l_iW['vis']
        LL['sScore_l'] =  (LL['area_sc']*l_iW['area']+ LL['neigh_sc']*l_iW['neigh'] +
                           LL['road_sc']*l_iW['road'] + LL['a_vis_sc']*l_iW['fac']) 
        LL['cScore_l'] = LL['cult_sc']
        LL['pScore_l'] = LL['prag_sc']
        
        col_rs = ['vScore_l', 'sScore_l']
        for i in col_rs: uf.scaling_columnDF(LL, i)
        
        LL['lScore'] =  (LL['vScore_l_sc']*l_cW['vScore'] + LL['sScore_l_sc']*l_cW['sScore'] + 
                         LL['cScore_l']*l_cW['cScore'] + LL['pScore_l']*l_cW['pScore'])
        
        # assigning the so obtined score to the building
        localScore = float("{0:.3f}".format(LL['lScore'].loc[row[0]]))
        buildings_gdf.at[row[0], 'lScore'] = localScore
    
    uf.scaling_columnDF(buildings_gdf, 'lScore')
    return buildings_gdf



