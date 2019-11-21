import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
pd.set_option('precision', 10)

import utilities as uf

def assign_buildings_to_nodes(nodes_gdf, buildings_gdf, buffer = 80):
    """
    The function simply resets the indexes of the two dataframes without losing the relative links
     
    Parameters
    ----------
    nodes_gdf, edges_gdf: GeoDataFrames, nodes and street segments
   
    Returns
    -------
    GeoDataFrames
    """
    
    spatial_index = buildings_gdf.sindex
    nodes_gdf['loc_land'] = 'None'
    nodes_gdf['loc_scor'] = 'None'
    ix_geo = nodes_gdf.columns.get_loc("geometry")+1

    for row in nodes_gdf.itertuples():
       
        g = row[ix_geo] #geometry
        b = g.buffer(buffer)    
         
        possible_matches_index = list(spatial_index.intersection(b.bounds))
        possible_matches = buildings_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(b)]
        
        if (len(precise_matches) == 0): continue
    
        precise_matches = precise_matches.sort_values(by = 'lScore_sc', ascending = False).reset_index()
        precise_matches = precise_matches.round({'lScore_sc':3})
        
        list_local = precise_matches['buildingID'].tolist()
        list_scores = precise_matches['lScore'].tolist()
        nodes_gdf.at[row[0], 'loc_land'] = list_local
        nodes_gdf.at[row[0], 'loc_scor'] = list_scores
    
    return nodes_gdf


def assign_global_anchors(nodes_gdf, buildings_gdf, buffer = 2000, threshold = 0.3):
    
    # keeping relevant landmarks and the sight lines that point at them
    
#     if smaller_area == True:
#         nodes_gdf['oldID'] = nodes_gdf['nodeID']
#         nodes_gdf['coordinates'] = nodes_gdf[['x', 'y']].apply(tuple, axis=1)
#         all_nodes['coordinates'] = all_nodes[['x', 'y']].apply(tuple, axis=1)
#         nodes_merged = pd.merge(nodes_gdf, all_nodes[['nodeID','coordinates']], left_on = 'coordinates', right_on = 'coordinates', 
#                                 how = 'left')
#         nodes_merged['nodeID'] = nodes_merged['nodeID_y'] 
#         nodes_merged.drop(['nodeID_x', 'nodeID_y', 'coordinates'], axis = 1, inplace = True)
#         node_gdf = nodes_merged
    
    relevant = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    relevant = relevant.round({'gScore_sc':3})

    nodes_gdf['anchors'] = None # orienting landmark for this destination
    nodes_gdf['distances'] = None # distances from them
    ix_nodeID = nodes_gdf.columns.get_loc("nodeID")+1
    ix_geo = nodes_gdf.columns.get_loc("geometry")+1
    
    sindex = relevant.sindex
    for row in nodes_gdf.itertuples():
        
        # anchors around the node (intended as a destination)
        b = row[ix_geo].buffer(buffer)
        possible_matches_index = list(sindex.intersection(b.bounds))
        possible_matches = relevant.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(b)]
        
        if len(precise_matches) > 0:    
            precise_matches['dist'] = precise_matches.apply(lambda row_rfn: row[ix_geo].distance(row_rfn.geometry), axis=1)
            precise_matches = precise_matches.round({'dist':3})
            precise_matches.sort_values(by = 'gScore_sc', ascending = False, inplace = True)
            anchors = precise_matches.iloc[0:7] # first seven rows of dataframe
            nodes_gdf.at[row[0], 'anchors'] =  anchors['buildingID'].tolist()
            nodes_gdf.at[row[0], 'distances'] = anchors['dist'].tolist()
            
#     if smaller_area == True:
#         nodes_gdf['nodeID'] = nodes_gdf['oldID']
#         nodes_gdf.drop(['oldID'], axis = 1, inplace = True)
    
    return nodes_gdf 

def assign_3dvisible_landmarks(nodes_gdf, buildings_gdf, sight_lines, threshold = 0.3):
    # per each node, the sight lines to the relevant landmarks are extracted.   
    
    ix_nodeID = nodes_gdf.columns.get_loc("nodeID")+1
    ix_geo = nodes_gdf.columns.get_loc("geometry")+1

    nodes_gdf['dist_land'] = None # visible landmarks from node
    nodes_gdf['dist_scor'] = None # their score
    relevant = buildings_gdf[buildings_gdf.gScore_sc >= threshold]
    relevant = relevant.round({'gScore_sc':3})
    index_relevant = buildings_gdf['buildingID'].values.astype(int) 
    sightLines_to_relevant = sight_lines[sight_lines['buildingID'].isin(index_relevant)]
    
    for row in nodes_gdf.itertuples():
        sight_node = sightLines_to_relevant[sightLines_to_relevant['nodeID'] == row[ix_nodeID]] 
        index_relevant_fromNode = list(sight_node['buildingID'].values.astype(int))
        relevant_fromNode = relevant[relevant['buildingID'].isin(index_relevant_fromNode)] 
        relevant_fromNode.sort_values(by = 'gScore_sc', ascending = False, inplace = True)
        
        if len(relevant_fromNode) > 0:
            relevant_list  = relevant_fromNode['buildingID'].tolist()
            scores_list = relevant_fromNode['gScore_sc'].tolist()
            while len(str(relevant_list)) > 254:
                del relevant_list[-1]
                del scores_list[-1]             
            
            nodes_gdf.at[row[0], 'dist_land'] =  relevant_list
            nodes_gdf.at[row[0], 'dist_scor'] = scores_list
    
    return nodes_gdf 


def advance_visibility_nodes(nodes_gdf, edges_gdf, buildings_gdf, distance = 300, field_view = True):
    
    visibility_polygons = pd.DataFrame(columns = ['nodeID', 'geometry'])
    if field_view == True: visibility_polygons = pd.DataFrame(columns = ['nodeID', 'from_Node', 'geometry'])
    
    degree_field_view = 140
    ix_x = nodes_gdf.columns.get_loc("x")+1 
    ix_y = nodes_gdf.columns.get_loc("y")+1 
    ix_nodeID = nodes_gdf.columns.get_loc("nodeID")+1 
    ix_geo = nodes_gdf.columns.get_loc("geometry")+1 
    
    index_v = edges_gdf.columns.get_loc("v")+1 
    index_u = edges_gdf.columns.get_loc("u")+1 
    
    to_sub = (360-degree_field_view)/2
    sindex = buildings_gdf.sindex
    coming_from = 0
    counter = 0
    
    for row in nodes_gdf.itertuples():
        uf.print_row(row.Index)
        origin = (float("{0:.10f}".format(row[ix_x])), float("{0:.10f}".format(row[ix_y])))
        possible_obstacles_index = list(sindex.intersection(row[ix_geo].buffer(distance*2).bounds))
        possible_obstacles = buildings_gdf.iloc[possible_obstacles_index]
        if field_view == True:
                
            segments = edges_gdf[(edges_gdf['v'] == row[ix_nodeID]) | (edges_gdf['u'] == row[ix_nodeID])]
            zeroCoord = uf.get_coord_angle(origin, distance = distance, angle = 0.5)
            
            for s in segments.itertuples():

                if s[index_v] == row[ix_nodeID]:     # take ndode u
                    otherCoord = (nodes_gdf.x[nodes_gdf['nodeID'] == s[index_u]].tolist()[0], 
                           nodes_gdf.y[nodes_gdf['nodeID'] == s[index_u]].tolist()[0])
                    coming_from = s[index_u]

                else: #takes node v
                    otherCoord = (nodes_gdf.x[nodes_gdf['nodeID'] ==  s[index_v]].tolist()[0], 
                           nodes_gdf.y[nodes_gdf['nodeID'] == s[index_v]].tolist()[0])
                    coming_from = s[index_v]

                line0 = ((origin), (zeroCoord))
                lineB = ((origin), (otherCoord))

                if otherCoord[0] > origin[0]: # east

                    diff = uf.ang(line0, lineB)
                    start = diff+to_sub
                    end = start+degree_field_view
                    if end >360: end = end-360

                else: # west
                    diff = 180-uf.ang(line0, lineB)
                    start = 180+diff+to_sub
                    if start > 360: start = start-360

                    end = start+degree_field_view 
                    if end > 360: end = end-360

                first_lineCoord = uf.get_coord_angle(origin, distance = distance, angle = start)
                last_lineCoord = uf.get_coord_angle(origin, distance = distance, angle = end)
                first_line = LineString([row[ix_geo], Point(first_lineCoord)])
                last_line = LineString([row[ix_geo], Point(last_lineCoord)])

                i = start
                list_lines = []

                while (((end > start) & ((i <= end) & (i >= start))) |
                       ((end < start) & (((i >= start) & (i > end)) | ((i < start) & (i <= end))))):

                    coords = uf.get_coord_angle(origin, distance = distance, angle = i)
                    line = LineString([row[ix_geo], Point(coords)])

                    obstacles = possible_obstacles[possible_obstacles.crosses(line)]
                    if len(obstacles) == 0: lineNew = line
                    else: 
                        ob = cascaded_union(obstacles.geometry)
                        t = line.intersection(ob)
                        try:
                            intersection = t[0].coords[0]
                        except:
                            intersection = t.coords[0]

                        lineNew = LineString([row[ix_geo], Point(intersection)])

                    list_lines.append(lineNew)
                    i = i+10
                    if i > 360: i = i - 360

                list_points = [Point(origin)]
                for i in list_lines: list_points.append(Point(i.coords[1]))
                list_points.append(Point(origin))
                poly = Polygon([[p.x, p.y] for p in list_points])
                
                visibility_polygons.loc[-1] = [row[ix_nodeID], coming_from, poly] 
                visibility_polygons.index = visibility_polygons.index + 1
            
        else:
            start = 0.0
            i = start
            list_lines = []
            
            while(i <= 360):
                coords = uf.get_coord_angle(origin, distance = distance, angle = i)
                line = LineString([row[ix_geo], Point(coords)])
                obstacles = possible_obstacles[possible_obstacles.crosses(line)]

                if len(obstacles == 0): lineNew = line
                else:
                    ob = cascaded_union(obstacles.geometry)
                    t = line.intersection(ob)
                    try:
                        intersection = t[0].coords[0]
                    except:
                        intersection = t.coords[0]

                    lineNew = LineString([row[ix_geo], Point(intersection)])

                list_lines.append(lineNew)
                i = i+10

            list_points = [Point(origin)]
            for i in list_lines: list_points.append(Point(i.coords[1]))
            list_points.append(Point(origin))
            poly = Polygon([[p.x, p.y] for p in list_points])
            
            visibility_polygons.loc[-1] = [row[ix_nodeID], poly] 
            visibility_polygons.index = visibility_polygons.index + 1
        
    
    visibility_polygons_gdf = gpd.GeoDataFrame(visibility_polygons.loc[:, visibility_polygons.columns != 'geometry'], 
                                               crs = nodes_gdf.crs, geometry = visibility_polygons['geometry'])
    return visibility_polygons_gdf
    
    

def visibility_matrix(buildings_gdf, visibility_polygons_gdf, nodes_gdf):
    
    columns = ['buildingID'] + nodes_gdf['nodeID'].astype(str).tolist()
    visibility_matrix = pd.DataFrame(columns = columns)
    visibility_matrix['buildingID'] = buildings_gdf['buildingID']

    index_buildingID = visibility_matrix.columns.get_loc("buildingID")+1  
    index_polygon = visibility_polygons_gdf.columns.get_loc("geometry")+1  
    ix_nodeID = visibility_polygons_gdf.columns.get_loc("nodeID")+1  
    sindex = visibility_polygons_gdf.sindex

    for row in visibility_matrix.itertuples(): 

        l_polygon = buildings_gdf['geometry'][buildings_gdf['buildingID'] == row[index_buildingID]].tolist()[0]
        b = l_polygon.buffer(200)
        possible_matches_index = list(sindex.intersection(b.bounds))
        possible_matches = visibility_polygons_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(b)]

        for p in precise_matches.itertuples():
            v_polygon = p[index_polygon]
            column = str(p[ix_nodeID])

            if ((l_polygon.intersects(v_polygon)) | (l_polygon.touches(v_polygon))):
                visibility_matrix.at[row[0], column] = True

    visibility_matrix.fillna(value = False, inplace=True)
    visibility_matrix = visibility_matrix.astype(int)
    
    return(visibility_matrix)

def visibility_matrix2d(buildings_gdf, nodes_gdf, v_dict):
    
    columns = ['buildingID'] + nodes_gdf['nodeID'].astype(str).tolist()
    matrix = pd.DataFrame(columns = columns)
    matrix['buildingID'] = buildings_gdf['buildingID']
    
    for n, buildings in v_dict.items():
        mask = matrix['buildingID'].isin(buildings)
        matrix.loc[mask, str(n)] = 1
        
    matrix.fillna(value = 0, inplace = True)
    return matrix

def visibility_2d(nodes_gdf, buildings_gdf, distance_along = 50, max_distance_node_to_building = 300):
    nodes_gdf = nodes_gdf.copy()
    buildings_gdf = buildings_gdf.copy()
    
    nodes_gdf.set_index('nodeID', drop = False, inplace = True, append = False)
    del nodes_gdf.index.name
    buildings_gdf.set_index('buildingID', drop = False, inplace = True, append = False)
    del buildings_gdf.index.name
    
    Nsindex = nodes_gdf.sindex
    Bsindex = buildings_gdf.sindex
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1
    ix_geoN = nodes_gdf.columns.get_loc("geometry")+1
    d = {el:[] for el in nodes_gdf.nodeID}
    buffer = max_distance_node_to_building
    interval = max_distance_node_to_building
    
    for row in buildings_gdf.itertuples(): 
        exteriors = row[ix_geo].exterior
        coords = list(exteriors.coords)
        no_holes = Polygon(coords)

        possible_obstacles_index = list(Bsindex.intersection(no_holes.buffer(buffer).bounds))
        possible_obstacles = buildings_gdf.iloc[possible_obstacles_index]
        obstacles = possible_obstacles[possible_obstacles.intersects(no_holes.buffer(buffer))]
        obstacles.drop(row.Index, axis = 0, inplace = True, errors = 'ignore')

        possible_nodes_index = list(Nsindex.intersection(no_holes.buffer(buffer).bounds))
        possible_nodes = nodes_gdf.iloc[possible_nodes_index]
        nodes_around = possible_nodes[possible_nodes.intersects(no_holes.buffer(buffer))]
        if len(nodes_around) == 0: continue

        new_ring = coords.copy()
        distance_so_far = 0

        for n, i in enumerate(coords):
            if (n == 0) | (n == len(coords)-1) : continue
            distance = Point(i).distance(Point(coords[n-1]))
            if distance < interval: 
                distance_so_far = distance_so_far + distance
                continue

            vertexes_to_add = int(distance/interval)
            index = new_ring.index(i)

            for v in range(0, vertexes_to_add):
                distance_along = distance_so_far + interval + (interval*v)
                next_vertex = exteriors.interpolate(distance_along)
                new_index = index+v
                new_ring.insert(new_index, next_vertex)

            distance_so_far = distance_so_far + distance

        new_ring = new_ring[:-1]
        no_obstacles = False
        if (len(obstacles) > 0): union = obstacles.unary_union
        else: no_obstacles = True
        
        for rowN in nodes_around.itertuples():
           
            for coord in new_ring:
                v = LineString([rowN[ix_geoN], Point(coord)])
                if no_obstacles == False: 
                    if v.intersects(union): continue        
                    
                self_intersection = v.intersection(exteriors)
                if (self_intersection.geom_type == 'Point') | (self_intersection.geom_type == 'GeometryCollection'):
                    d[rowN.Index] = d[rowN.Index]+[row.Index]
                    break                           
                else: continue
    print("done")                 
    return(d)
    