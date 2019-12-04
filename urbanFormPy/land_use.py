import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option("precision", 10)

def classify_land_use(buildings_gdf, land_use):

    """
    The function reclassifies land-use descriptors in a land-use field according to the categorisation presented below. 
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
		the buildings GeoDataFrame
    land_use: string
		the land use field in the buildings_gdf
   
    Returns
    -------
    GeoDataFrame
    """
    
    # introducing classifications and possible entries
    university = ["university", "college", "research"]
    commercial = ["bank", "service",  "commercial",  "retail", "Retail",  "pharmacy", "commercial;educa", "shop", "Commercial",
                  "supermarket", "offices", "foundation", "office", "books", "Commercial services", "Commercial Land", 
                  "Mixed Use Res/Comm",  "Commercial Condo Unit"]
    
    residential = [ "apartments", None, "NaN", "residential","flats", "no", "houses", "garage", "garages", "building", 
                  "roof", "storage_tank", "shed", "silo",  "parking",  "toilets", "picnic_site","hut", "information", "viewpoint",
                  "atm",   "canopy", "smokestack", "greenhouse", "fuel", "Residential Condo Unit", "Apartments 4-6 Units", 
                  "Residential Two Family", "Apartments 7 Units above", "Residential Single Family", "Condominium Parking", 
                  "Residential Three Family", "Condominium Master", "Residential Land"]
    
    attractions = ["Attractions", "museum",  "castle", "cathedral", "attraction","aquarium", "monument",  "gatehouse",
                   "terrace", "tower", "Attraction And Leisure"]
    hospitality = [ "hotel",  "hostel", "guest_house"]
    eating_drinking = [ "restaurant", "fast_food", "cafe", "bar",  "pub", "Accommodation, eating and drinking",]
    public = ["post_office", "townhall", "public_building",  "library","civic", "courthouse", "public", "embassy",
              "Public infrastructure", "community_centre", "parking", "dormitory", "Exempt", "Exempt 121A"]
    library = ["library"]
    sport = ["stadium", "Sport and entertainment", "Sports Or Exercise Facility"]
    entertainment = [ "exhibition_centr","theatre", "cinema"]
    education = ["school", "kindergarten", "Education", "Education and health"]
    religious = ["church", "place_of_worship","convent", "rectory", "Religious Buildings"]
    emergency_service = [ "fire_station","police", "Emergency Service"]
    transport = [ "station", "train_station"]
    medical_care = ["hospital", "doctors", "dentist","clinic","veterinary", "Medical Care"]
    industrial = [ "industrial", "factory", "construction", "Manufacturing and production",  "gasometer", "data_center"]
    cultural = ["club_house","gallery", "arts_centre","Cultural Facility"]
    military = ["general aviation", "Barracks"]
    transport = ["Transport", "Road Transport", "station", "subway_entrance", "bus_station"]
    
    
    # reclassifying: replacing original values with relative categories
    buildings_gdf[land_use] = buildings_gdf[land_use].map( lambda x: "university" if x in university
                                                              else "commercial" if x in commercial
                                                              else "residential" if x in residential
                                                              else "attractions" if x in attractions
                                                              else "library" if x in library
                                                              else "hospitality" if x in hospitality
                                                              else "eating_drinking" if x in eating_drinking
                                                              else "public" if x in public
                                                              else "sport" if x in sport
                                                              else "entertainment" if x in entertainment
                                                              else "education" if x in education
                                                              else "religious" if x in religious
                                                              else "emergency_service" if x in emergency_service
                                                              else "industrial" if x in industrial
                                                              else "cultural" if x in cultural
                                                              else "transport" if x in transport
                                                              else "medical_care" if x in medical_care
                                                              else "military" if x in military
                                                              else "other")
    
    buildings_gdf[land_use][buildings_gdf[land_use].str.contains("residential") | buildings_gdf[land_use].str.contains("Condominium") | buildings_gdf[land_use].str.contains("Residential")] = "residential"
    buildings_gdf[land_use][buildings_gdf[land_use].str.contains("commercial") | buildings_gdf[land_use].str.contains("Commercial")] = "commercial"
    
    return buildings_gdf

def land_use_from_polygons(buildings_gdf, other_source_gdf, column, land_use_field):

    """
    It assigns land-use attributes to buildings in a buildings GeoDataFrame, looking for possible matches in "other_source_gdf", a Polygon GeoDataFrame
    Possible matches here means the buildings in "other_source_gdf" whose area of interesection with the examined building (y), covers at least
    60% of the building's (y) area. The best match is chosen. 
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
	
	other_source_gdf: Polygon GeoDataFrame
		the GeoDataFrame wherein looking for land_use attributes
    column: str
		name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: string, name of the column in other_source_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf[column] = None
    # spatial index
    sindex = other_source_gdf.sindex
    buildings_gdf[column] = buildings_gdf.apply(lambda row: _assign_land_use_from_polygons(row["geometry"], other_source_gdf, sindex, land_use_field))
    
    return buildings_gdf
    
def _assign_land_use_from_polygons(building_geometry, other_source_gdf, other_source_gdf_sindex, land_use_field):

    """
    It assigns land-use attributes to a building, looking for possible matches in "other_source_gdf".
     
    Parameters
    ----------
    buildings_geometry: Polygon
    other_source_gdf: Polygon GeoDataFrame
    other_source_gdf_sindex: Rtree spatial index
    land_use_field: string, name of the column in other_source_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    ix_geo = other_source_gdf.columns.get_loc("geometry")+1 
    
    possible_matches_index = list(other_source_gdf_sindex.intersection(building_geometry.bounds)) # looking for intersecting geometries
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    pm["area"] = 0.0
    if (len(pm) == 0): return None# no intersecting features in the other_source_gdf

    for row in pm.itertuples(): # for each possible candidate, computing the extension of the area of intersection
        other_geometry = row[ix_geo]
        try: overlapping_area = other_geometry.intersection(g).area
        except: continue
        pm.at[row.Index, "area"] = overlapping_area
        
        # sorting the matches based on the extent of the area of intersection
        pm = pm.sort_values(by="area", ascending=False).reset_index()
        # assigning the match land-use category if the area of intersection covers at least 60% of the building's area
        if (pm["area"].loc[0] >= (g.area * 0.60)): return pm[land_use_field].loc[0]

        else: continue
        
    return buildings_gdf

def land_use_from_points(buildings_gdf, other_source_gdf, column, land_use_field):

    """
    It assigns land-use attributes to buildings in "buildings_gdf", looking for possible matches in "other_source_gdf", a Point GeoDataFrame.
    Possible matches means features in "other_source_gdf" which lies within the examined building's area.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    other_source_gdf: Point GeoDataFrame - The GeoDataFrame wherein looking for land_use attributes
    column: string, name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: name of the column in other_source_gdf, wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()    
    other_source_gdf["nr"] = 1
    buildings_gdf[column] = None
    sindex = other_source_gdf.sindex
    buildings_gdf[column] = buildings_gdf.apply(lambda row: _assign_land_use_from_points(row["geometry"], other_source_gdf, sindex, land_use_field))
               
    return buildings_gdf

def _assign_land_use_from_points(building_geometry, other_source_gdf, other_source_gdf_sindex, land_use_field):

    """
    It assigns land-use attributes to a building, looking for possible matches in "other_source_gdf", a Point GeoDataFrame.
     
    Parameters
    ----------
    buildings_geometry: Polygon
    other_source_gdf: Point GeoDataFrame
    other_source_gdf_sindex: Rtree spatial index
    land_use_field: string, name of the column in other_source_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    GeoDataFrame
    """

    possible_matches_index = list(other_source_gdf_sindex.intersection(g.bounds))
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    
    if (len(pm)==0): return None # no intersecting features in the other_source_gdf
    else:
    # counting nr of features
        pm.groupby([land_use_field],as_index=False)["nr"].sum().sort_values(by="nr", ascending=False).reset_index()
        return use[land_use_field].loc[0]