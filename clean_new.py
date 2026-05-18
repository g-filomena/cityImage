
def simplify_graph(nodes_gdf, edges_gdf, nodes_to_keep_regardless = []):
    """
    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).    
    
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
   
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()

    # Identify pseudo-nodes
    degree = nodes_degree(edges_gdf)
    pseudo_nodes = [node for node, deg in degree.items() if deg == 2]
    
    # Exclude nodes to keep regardless
    if nodes_to_keep_regardless:
        pseudo_nodes = [node for node in pseudo_nodes if node not in nodes_to_keep_regardless]

    if not pseudo_nodes:
        return nodes_gdf, edges_gdf
        
    for nodeID in pseudo_nodes:
            # Retrieve connected edges
            connected_edges = edges_gdf[(edges_gdf['u'] == nodeID) | (edges_gdf['v'] == nodeID)]
            if len(connected_edges) == 2:
                first_edge, second_edge = connected_edges.iloc[0], connected_edges.iloc[1]
                nodes_gdf, edges_gdf = merge_pseudo_edges(first_edge, second_edge, nodeID, nodes_gdf, edges_gdf)

    # Remove invalid edges where `u` == `v`
    edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']]

    return nodes_gdf, edges_gdf

def merge_pseudo_edges(first_edge, second_edge, nodeID, nodes_gdf, edges_gdf):
    """
    Merge pseudo-edges by updating node and edge information in the corresponding GeoDataFrames.

    Parameters
    ----------
    first_edge: Series
        The first pseudo-edge to be merged.
    second_edge: Series
        The second pseudo-edge to be merged.
    nodeID: int
        The nodeID of the node where the pseudo-edges meet.
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrames.
    """
    
    index_first, index_second = first_edge.edgeID, second_edge.edgeID
    line_coordsA = list(first_edge["geometry"].coords)
    line_coordsB = list(second_edge["geometry"].coords)

    # Determine merge order based on meeting points
    if first_edge["u"] == second_edge["u"]:
        edges_gdf.at[index_first, "u"] = first_edge["v"]
        edges_gdf.at[index_first, "v"] = second_edge["v"]
        line_coordsA.reverse()
    elif first_edge["u"] == second_edge["v"]:
        edges_gdf.at[index_first, "u"] = second_edge["u"]
    elif first_edge["v"] == second_edge["u"]:
        edges_gdf.at[index_first, "v"] = second_edge["v"]
    else:  # first_edge["v"] == second_edge["v"]
        edges_gdf.at[index_first, "v"] = second_edge["u"]
        line_coordsB.reverse()

    # Merge coordinates
    merged_line = line_coordsA + line_coordsB
    edges_gdf.at[index_first, "geometry"] = LineString(merged_line)

    # Drop the redundant edge and the pseudo-node
    edges_gdf.drop(index_second, inplace=True)
    nodes_gdf.drop(nodeID, inplace=True)

    return nodes_gdf, edges_gdf
 
def simplify_same_vertexes_edges(nodes_gdf, edges_gdf, preserve_direction):
    """
    This function is used to simplify edges that have the same start and end point (i.e. 'u' and 'v' values) 
    in the edges_gdf GeoDataFrame. It removes duplicate edges that have similar geometry, keeping only the one 
    with the longest length if one of them is 10% longer of the other. Otherwise generates a center line
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The clean street segments GeoDataFrames.
    """
    edges_gdf = edges_gdf.copy()
   
    # Create unique codes for edges based on `u` and `v`, accounting for direction
    if not preserve_direction:
        edges_gdf["code"] = np.where(
            edges_gdf['v'] >= edges_gdf['u'],
            edges_gdf['u'].astype(str) + "-" + edges_gdf['v'].astype(str),
            edges_gdf['v'].astype(str) + "-" + edges_gdf['u'].astype(str),
        )
    else:
        edges_gdf["code"] = edges_gdf['u'].astype(str) + "-" + edges_gdf['v'].astype(str)
        
    # Identify duplicates
    duplicated_codes = edges_gdf[edges_gdf.duplicated('code', keep=False)]

    if duplicated_codes.empty:
        return nodes_gdf, edges_gdf 
        
    # Group by edge codes and simplify duplicates
    to_drop = set()
    for code, group in duplicated_codes.groupby("code"):
        if len(group) > 1:
            # Keep the longest edge if one is 10% longer than the others
            max_length = group['length'].max()
            short_edges = group[group['length'] < max_length * 0.9]
            if not short_edges.empty:
                to_drop.update(short_edges.index)

            # If multiple edges remain, merge into a centerline
            remaining_edges = group.loc[group.index.difference(to_drop)]
            if len(remaining_edges) > 1:
                centerline = center_line(remaining_edges.geometry.tolist())
                edges_gdf.at[remaining_edges.index[0], 'geometry'] = centerline
                to_drop.update(remaining_edges.index[1:])    
        
    # Drop unnecessary edges
    edges_gdf = edges_gdf.drop(index=list(to_drop))

    # Keep only nodes referenced by edges
    valid_nodes = set(edges_gdf['u']).union(set(edges_gdf['v']))
    nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(valid_nodes)]

    return nodes_gdf, edges_gdf

def clean_edges(nodes_gdf, edges_gdf, preserve_direction = False):
    """
    Cleans the edges GeoDataFrame by removing self-loops, duplicate geometries, and redundant edges. 
    Ensures consistency with the nodes GeoDataFrame by retaining only referenced nodes.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        The GeoDataFrame containing nodes (junctions) with 'nodeID' and 'geometry'.
    edges_gdf : GeoDataFrame
        The GeoDataFrame containing edges (street segments) with 'u', 'v', and 'geometry'.
    preserve_direction : bool, optional
        If True, considers edges with reversed 'u' and 'v' as distinct. 
        If False, treats edges with reversed 'u' and 'v' as duplicates (default: False).

    Returns
    -------
    tuple
        A tuple containing the cleaned nodes GeoDataFrame and edges GeoDataFrame:
        - nodes_gdf : GeoDataFrame
            Cleaned nodes with only those referenced by edges.
        - edges_gdf : GeoDataFrame
            Cleaned edges with no duplicates, self-loops, or redundant geometries.
    """
   
    edges_gdf = edges_gdf.copy()

    # Generate unique edge codes
    edges_gdf["code"] = (
        edges_gdf["u"].astype(str) + "-" + edges_gdf["v"].astype(str)
        if preserve_direction
        else np.where(
            edges_gdf["v"] >= edges_gdf["u"],
            edges_gdf["u"].astype(str) + "-" + edges_gdf["v"].astype(str),
            edges_gdf["v"].astype(str) + "-" + edges_gdf["u"].astype(str),
        )
    )

    # Remove self-loops
    edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]
                        
  # Remove duplicate geometries
    geometries = edges_gdf["geometry"].apply(lambda geom: geom.wkb)
    edges_gdf = edges_gdf.loc[geometries.drop_duplicates().index]
    
    # Handle reversed coordinates if not preserving direction
    edges_gdf["coords"] = edges_gdf["geometry"].apply(lambda geom: list(geom.coords))
    if not preserve_direction:
        reversed_condition = (
            edges_gdf["u"].astype(str) + "-" + edges_gdf["v"].astype(str) != edges_gdf["code"]
        )
        edges_gdf.loc[reversed_condition, "coords"] = edges_gdf.loc[
            reversed_condition, "coords"
        ].apply(lambda x: x[::-1])

    edges_gdf["tmp"] = edges_gdf["coords"].apply(tuple)
    edges_gdf.drop_duplicates(subset=["tmp"], inplace=True)
    
    # Keep only nodes referenced in edges
    valid_nodes = set(edges_gdf["u"]).union(set(edges_gdf["v"]))
    nodes_gdf = nodes_gdf[nodes_gdf["nodeID"].isin(valid_nodes)]
    
    return nodes_gdf, edges_gdf

def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_islands = True, same_vertexes_edges = True, self_loops = False, fix_topology = False, 
                  preserve_direction = False, nodes_to_keep_regardless = []):
    """
    It calls a series of functions to clean nodes and edges GeoDataFrames.
    It handles:
        - pseudo-nodes;
        - duplicate-geometries (nodes and edges);
        - disconnected islands - optional;
        - edges with different geometry but same nodes - optional;
        - dead-ends - optional;
        - self-loops - optional;
        - toplogy issues - optional.
           
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    dead_ends: bool
        When true remove dead ends.
    remove_islands: bool
        When true checks the existence of disconnected islands within the network and deletes corresponding graph components.
    same_vertexes_edges: bool
        When true, it considers as duplicates couple of edges with same vertexes but different geometry. If one of the edges is 1% longer than the other, 
        the longer is deleted; otherwise a center line is built to replace the same vertexes edges.
    self_loops: bool
        When true, removes genuine self-loops.
    fix_topology: bool
        When true, it breaks lines at intersections with other lines in the streets GeoDataFrame.
    preserve_direction: bool
        When true, it does not consider segments with same coordinates list, but different directions, as identical. When false, it does and therefore
        considers them as duplicated geometries.
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).
    
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    nodes_gdf, edges_gdf = _prepare_dataframes(nodes_gdf, edges_gdf)  
    # removes fake self-loops wrongly coded by the data source
    nodes_gdf, edges_gdf = fix_self_loops(nodes_gdf, edges_gdf)  
    
    if dead_ends: 
        nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    if fix_topology: 
        nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf)
    
    cycle = 0
    while ((not is_edges_simplified(edges_gdf, preserve_direction) and same_vertexes_edges) |
            (not is_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless)) |
            (cycle == 0)):

        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = duplicate_nodes(nodes_gdf, edges_gdf)
        #eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
        
        nodes_gdf, edges_gdf = clean_edges(nodes_gdf, edges_gdf, preserve_direction) 
        
        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            nodes_gdf, edges_gdf = simplify_same_vertexes_edges(nodes_gdf, edges_gdf, preserve_direction)
  
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf, nodes_to_keep_regardless) 
        
        #eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf) # correct edges coordinates
    
    nodes_gdf.drop(['wkt'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf.drop(['coords', 'tmp', 'code', 'wkt', 'fixing', 'to_fix'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf['length'] = edges_gdf['geometry'].length
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    
    nodes_gdf.index.name = None
    edges_gdf.index.name = None
    
    return nodes_gdf, edges_gdf

def _add_fixed_edges(edges_gdf, to_fix_gdf):
    """
    Splits edges at specified intersection points and adds the resulting segments to the edges GeoDataFrame.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        The GeoDataFrame containing the existing street segments.
    to_fix_gdf : GeoDataFrame
        The GeoDataFrame containing edges that need to be split.

    Returns
    -------
    nodes_gdf, edges_gdf : tuple
        Updated GeoDataFrames of nodes and edges with fixed topology.
    """
    # Initialize list to store new edges
    new_edges = []

    def split_and_append(row):
        """
        Splits a LineString at intersection points and appends resulting segments to the new edges list.
        """
        # Split the LineString at given points
        split_points = [Point(coord) for coord in row["to_fix"]]
        split_segments = split_line_at_MultiPoint(row["geometry"], split_points)

        for segment in split_segments:
            # Create a new edge for each segment
            new_edge = row.copy()
            new_edge["geometry"] = segment
            new_edges.append(new_edge)

    # Apply the split and append process to each edge in `to_fix_gdf`
    to_fix_gdf.apply(split_and_append, axis=1)

    # Convert new edges to a GeoDataFrame
    new_edges_gdf = gpd.GeoDataFrame(new_edges, crs=edges_gdf.crs)

    # Concatenate the fixed edges with the original edges
    edges_gdf = pd.concat([edges_gdf, new_edges_gdf], ignore_index=True)

    # Recompute edge lengths
    edges_gdf["length"] = edges_gdf.geometry.length

    # Recreate the edge IDs
    edges_gdf["edgeID"] = edges_gdf.index

    # Update nodes and edges relationships
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)

    return nodes_gdf, edges_gdf
   
def fix_network_topology(nodes_gdf, edges_gdf):
    """
    Fix the network topology by splitting intersecting edges and adding fixed edges to the network.
    This function considers as segments to be fixed only segments that are actually fully intersecting, thus sharing coordinates, excluding their 
    from and to vertices coordinates, but withouth actually generating, in the given GeoDataFrame, the right number of features.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    
    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    edges_gdf.copy()
    edges_gdf['coords'] = edges_gdf.geometry.apply(lambda geom: list(geom.coords))
    # spatial index
    sindex = edges_gdf.sindex 

    def find_intersections(ix, line_geometry, coords):
        """
        Find intersection points between a line and other intersecting edges.

        Parameters
        ----------
        ix : int
            The index of the line to check for intersections.
        line_geometry : LineString
            The LineString geometry of the line to check for intersections.
        coords : list
            The list of coordinates of the line's geometry.

        Returns
        -------
        list
            A list of actual intersection points between the line and other intersecting edges.
        """
        
        possible_matches_index = list(spatial_index.intersection(line_geometry.bounds))
        possible_matches = edges_gdf.iloc[possible_matches_index]
        possible_matches = possible_matches[possible_matches.index != ix]
        
        tmp = tmp.drop(ix, axis = 0)
        union = tmp.unary_union
        
        # find actual intersections
        actual_intersections = []
        intersections = line_geometry.intersection(union)
        
        if intersections.is_empty or intersections is None:
            return []
        if intersections.geom_type == 'LineString':
            return []
        if intersections.geom_type == 'Point':
            intersections = [intersections]
        elif intersections.geom_type == 'MultiPoint':
            intersections = list(intersections.geoms)
               
        # from and to vertices of the given line
        segment_vertices = [coords[0], coords[-1]]
        # obtaining all the intersecting Points
        intersection_points = [intersection for intersection in intersections if intersection.geom_type == 'Point']
        
        # keeping intersections that are in the coordinate list of the given line, without actually coinciding with the from and to vertices
        for point in intersection_points: 
            if (point.coords[0] not in coords):
                pass
            if (point.coords[0] in segment_vertices): 
                pass 
            else: 
                actual_intersections.append(point)
        return actual_intersections
    
    # verify which street segment needs to be fixed
    edges_gdf['to_fix'] = edges_gdf.apply(lambda row: find_intersections(row.name, row.geometry, row.coords), axis=1)
    # verify which street segment needs to be fixed
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]   
    if len(to_fix) == 0:
        return edges_gdf    
    return _add_fixed_edges(edges_gdf, to_fix)
    
def fix_self_loops(nodes_gdf, edges_gdf):
    """
    Fix the network topology by removing (fake) self-loops and adding fixed edges.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    
    edges_gdf = edges_gdf.copy()
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # all the coordinates but the from and to vertices' ones.
    edges_gdf['coords'] = [coords[1:-1] for coords in edges_gdf.coords]
    
    # convert nodes_gdf['x'] and nodes_gdf['y'] to numpy arrays for faster computation
    x = list(nodes_gdf['x'])
    y = list(nodes_gdf['y'])
    # create a set of all coordinates in nodes. This essentially correspond to the from and to nodes of the edges currently in the edges_gdf
    nodes_set = set(zip(x, y))

    to_fix = []
    # loop through the coordinates in edges_gdf.coords and check if they are in the nodes_set. This means that one of the edges coords (not from and to),
    # coincide with some other edge from or to vertex (indicating some sort of loop) 
    for coords in edges_gdf.coords:
        fix_coords = []
        for coord in coords:
            if coord in nodes_set:
                fix_coords.append(coord)
        to_fix.append(fix_coords)

    # assign the results to self_loops['to_fix']
    edges_gdf['to_fix'] = to_fix
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]
    if len(to_fix) == 0:
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix)    
    
def remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID: str
        The name of the field containing the nodeIDs.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrame.
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf, nodeID)
    if not nx.is_connected(Ng):  
        largest_component = max(nx.connected_components(Ng), key=len)
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        to_keep = list(G.nodes())
        nodes_gdf = nodes_gdf[nodes_gdf[nodeID].isin(to_keep)]
        edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf[nodeID])) & (edges_gdf.v.isin(nodes_gdf[nodeID]))]
        
    return nodes_gdf, edges_gdf

def correct_edges(nodes_gdf, edges_gdf):
    """
    The function adjusts the edges LineString coordinates consistently with their relative u and v nodes' coordinates.
    It might be necessary to run the function after having cleaned the network.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
   
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """
    edges_gdf['geometry'] = edges_gdf.apply(lambda row: _update_line_geometry_coords(row['u'], row['v'], nodes_gdf, row['geometry']), axis=1)                            
    return edges_gdf

def _update_line_geometry_coords(u, v, nodes_gdf, line_geometry):
    """
    It supports the correct_edges function checks that the edges coordinates are consistent with their relative u and v nodes'coordinates.
    It can be necessary to run the function after having cleaned the network.
    
    Parameters
    ----------
    u: int
        The nodeID of the from node of the geometry.
    v: int
        The nodeID of the to node of the geometry.
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame .
    line_geometry: LineString
        A street segment geometry.
        
    Returns
    -------
    new_line_geometry: LineString
        The readjusted LineString, on the basis of the given u and v nodes.
    """
    line_coords = list(line_geometry.coords)
    line_coords[0] = (nodes_gdf.loc[u]['x'], nodes_gdf.loc[u]['y'])
    line_coords[-1] = (nodes_gdf.loc[v]['x'], nodes_gdf.loc[v]['y'])
    new_line_geometry = LineString([coor for coor in line_coords])
    return new_line_geometry
   