## Natural roads extraction: the function has to be called twice for each segment, to check in both directions ###############
    """
    The concet of natural road regards the cognitive perception/representation of road entities, regardless changes in names or
    interruptions. Rather, different street segments are "mentally merged" according to continuity rules (here based on the deflection
    angle and the egoistic choice process).
    """
	
def natural_roads(streetID, naturalID, direction, nodes_gdf, edges_gdf): 
    """
    This function takes a direction "to" or "fr" and two GeoDataFrames, one for roads one for nodes (or junctions).
    Since this function works only for one segment at the time, in one direction, it has to be executed in a for loop
    while iterating through all the edges, both directions. See function below.
    
    Parameters
    ----------
    streetID: int, next road to examine
    naturalID: int, current naturalID 
    direction: string, {'to', 'fr'}
    edges_gdf: GeoDataFrame
    nodes_gdf: GeoDataFrame
    """
    # initialise variables
    angles = {}
    directions_dict = {}
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_nID = edges_gdf.columns.get_loc("naturalID")+1
    
    to_node, from_node = edges_gdf.loc[streetID]['u'], edges_gdf.loc[streetID]['v']
    geo = edges_gdf.loc[streetID]['geometry']
      
    # continue from the to_node or from the from_node    
    if (direction == "to"): intersecting = edges_gdf[(edges_gdf['u'] == to_node) | (edges_gdf['v'] == to_node)]
    else: intersecting = edges_gdf[(edges_gdf['u'] == from_node) | (edges_gdf['v'] == from_node)]
    if (len(intersecting) == 0): return
    
    # check all possible deflection angles with the intersecting roads identified
    for row_F in intersecting.itertuples():
        if ((streetID == row_F[0]) | (row_F[ix_nID] > 0)): continue
        to_node_F, from_node_F = row_F[ix_u], row_F[ix_v]
        geo_F = row_F[ix_geo]

        # where to go next, in case?
        if (to_node == to_node_F): towards = "fr"
        elif (to_node == from_node_F): towards = "to"
        elif (from_node == from_node_F): towards = "to"
        else: towards = "fr"

        # measuring deflection angle, adding it to the dictionary, if lower than 45 degrees
        deflection = uf.ang_geoline(geo, geo_F, degree = True, deflection = True)
        if (deflection >= 45): continue
        else:
            angles[row_F[0]] = deflection # dictionary with streetID and angle
            directions_dict[row_F[0]] = towards # dictionary with streetID and direction
    
    # No natural continuations
    if (len(angles) == 0):
        edges_gdf.set_value(streetID, 'naturalID', naturalID)
        return
   
    # selecting the best continuation and continuing in its direction
    else:
        angles_sorted = sorted(angles, key = angles.get) 
                                            
        # taking the streetID of the segment which form the gentlest angle with the segment examined
        matchID = angles_sorted[0] 
        edges_gdf.set_value(streetID, 'naturalID', naturalID)
        natural_roads(matchID, naturalID, directions_dict[matchID], nodes_gdf, edges_gdf)                    
                                                                                    
def identify_natural_roads(nodes_gdf, edges_gdf): 
    """
    Run the natural_roads function on an entire geodataframe of street segments.
    The geodataframes are supposed to be cleaned and can be obtained via the functions "get_fromOSM(place)" or "get_fromSHP(directory, 
    epsg)". Please clean the graph before running.
    
    Parameters
    ----------
    nodes_gdf, edges_gdf: GeoDataFrames
    
    Returns
    -------
    GeoDataFrames
    """
    edges_gdf['naturalID'] = 0
    ix_nID = edges_gdf.columns.get_loc("naturalID")+1
    
    if (not nodes_simplified(edges_gdf)) | (not edges_simplified(edges_gdf)): 
        raise StreetNetworkError('The street network is not simplified')
   
    edges_gdf.index = edges_gdf.streetID
    nodes_gdf.index = nodes_gdf.nodeID
    
    naturalID = 1  
    
    for row in edges_gdf.itertuples():
        if (row[ix_nID] > 0): continue # if already assigned to a natural road
        natural_roads(row.Index, naturalID, "fr", nodes_gdf, edges_gdf) 
        natural_roads(row.Index, naturalID, "to", nodes_gdf, edges_gdf) 
        naturalID = naturalID + 1
                                            
    return(nodes_gdf, edges_gdf)
    



class Error(Exception):
   """Base class for other exceptions"""
   pass
class StreetNetworkError(Error):
   """Raised when street network GDFs are not simplified"""
   pass
class epgsError(Error):
   """Raised when epsg code is not provided"""
   pass
    
