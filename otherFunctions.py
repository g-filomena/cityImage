def create_visibility_polygon(node, edge_angle, obstructions_gdf, obstructions_sindex, max_expansion_distance=600):
    distance_along = 10
    angles = np.arange(-45, 46, distance_along)
    coords = np.array([get_coord_angle([node.x, node.y], distance=max_expansion_distance, angle=edge_angle + i) for i in angles])
    lines = [LineString([node, Point(x)]) for x in coords]
    
    buffer = node.buffer(max_expansion_distance)
    possible_matches_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_matches = obstructions_gdf.iloc[possible_matches_index]
    obstacles = possible_matches[possible_matches.intersects(buffer)]
    obstacles = obstacles[~obstacles.geometry.touches(node)]
    
    if len(obstacles) > 0:
        ob = unary_union(obstacles.geometry)
        intersections = [line.intersection(ob) for line in lines]
        clipped_lines = [
            LineString([node, Point(intersection.geoms[0].coords[0])]) if isinstance(intersection, MultiLineString) and not intersection.is_empty else
            LineString([node, Point(intersection.coords[0])]) if isinstance(intersection, LineString) and not intersection.is_empty else
            LineString([node, Point(intersection.coords[0])]) if isinstance(intersection, Point) and not intersection.is_empty else
            line for intersection, line in zip(intersections, lines)
        ]
    else: 
        clipped_lines = lines
    
    poly = Polygon([[p.x, p.y] for p in [node] + [Point(line.coords[1]) for line in clipped_lines] + [node]])
    return poly.difference(unary_union(obstructions_gdf.geometry))

def fill_visibility_matrix(edges_gdf, obstructions_gdf, max_expansion_distance=600):
    """
    Iterates through all edges and fills a matrix where each row is formed with nodeID_to + "_" + nodeID_from,
    and columns are ID and visibility polygons.
    
    Parameters
    ----------
    edges_gdf: GeoDataFrame
        The edges GeoDataFrame containing LineString geometries.
    obstructions_gdf: GeoDataFrame
        Obstructions GeoDataFrame.
    max_expansion_distance: float
        The maximum distance from the node to expand the visibility cone.
    
    Returns
    -------
    DataFrame
        A DataFrame containing edge ID and visibility polygons.
    """

    obstructions_sindex = obstructions_gdf.sindex

    def process_edge(row):
        u, v = row['u'], row['v']
        poly1, poly2 = visibility_polygons_for_edge(row, obstructions_gdf, obstructions_sindex, max_expansion_distance)
        
        return pd.DataFrame({
            "ID": [f"{u}_{v}", f"{v}_{u}"],
            "Vispolygon": [poly1, poly2]
        })

    visibility_matrix = edges_gdf.apply(process_edge, axis=1).reset_index(drop=True)
    visibility_matrix = pd.concat(visibility_matrix.tolist(), ignore_index=True)
    
    # Convert to GeoDataFrame
    visibility_gdf = gpd.GeoDataFrame(visibility_matrix, geometry='geometry', crs=edges_gdf.crs)
    
    return visibility_gdf
    
 def visibility_polygons_for_edge(edge, obstructions_gdf, obstructions_sindex, max_expansion_distance=600):
    """
    Calculates two visibility polygons for an edge, one per direction.
    
    Parameters
    ----------
    edge: LineString
        The edge geometry.
    obstructions_gdf: GeoDataFrame
        Obstructions GeoDataFrame.
    max_expansion_distance: float
        The maximum distance from the node to expand the visibility cone.
    
    Returns
    -------
    tuple
        Two visibility polygons.
    """
    line = edge.geometry
    start, end = list(line.coords)[0], list(line.coords)[1]
    edge_angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
    node_start = Point(start)
    node_end = Point(end)
    poly1 = create_visibility_polygon(node_start, edge_angle, obstructions_gdf, obstructions_sindex, max_expansion_distance)
    poly2 = create_visibility_polygon(node_end, edge_angle + 180, obstructions_gdf, obstructions_sindex, max_expansion_distance)
    return poly1, poly2
	
	
def _assign_group_membership_to_islands(graph, edges_gdf):
    """
    Assign group membership to islands in the network by updating the 'group' attribute in the edges GeoDataFrame.

    Parameters
    ----------
    graph: NetworkX Graph
        The graph representing the network.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """  
    components = nx.connected_components(graph)
    for n, c in enumerate(islands):
        nodes_group = list(c)
        edges_gdf.loc[(edges_gdf.u.isin(nodes_group) & (edges_gdf.v.isin(nodes_group))),'group'] = n
        
    return edges_gdf
	
	def create_visibility_polygon(node, edge_angle, obstructions_gdf, obstructions_sindex, max_expansion_distance=600):
    distance_along = 10
    angles = np.arange(-45, 46, distance_along)
    coords = np.array([get_coord_angle([node.x, node.y], distance=max_expansion_distance, angle=edge_angle + i) for i in angles])
    lines = [LineString([node, Point(x)]) for x in coords]
    
    buffer = node.buffer(max_expansion_distance)
    possible_matches_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_matches = obstructions_gdf.iloc[possible_matches_index]
    obstacles = possible_matches[possible_matches.intersects(buffer)]
    obstacles = obstacles[~obstacles.geometry.touches(node)]
    
    if len(obstacles) > 0:
        ob = unary_union(obstacles.geometry)
        intersections = [line.intersection(ob) for line in lines]
        clipped_lines = [
            LineString([node, Point(intersection.geoms[0].coords[0])]) if isinstance(intersection, MultiLineString) and not intersection.is_empty else
            LineString([node, Point(intersection.coords[0])]) if isinstance(intersection, LineString) and not intersection.is_empty else
            LineString([node, Point(intersection.coords[0])]) if isinstance(intersection, Point) and not intersection.is_empty else
            line for intersection, line in zip(intersections, lines)
        ]
    else: 
        clipped_lines = lines
    
    poly = Polygon([[p.x, p.y] for p in [node] + [Point(line.coords[1]) for line in clipped_lines] + [node]])
    return poly.difference(unary_union(obstructions_gdf.geometry))

def fill_visibility_matrix(edges_gdf, obstructions_gdf, max_expansion_distance=600):
    """
    Iterates through all edges and fills a matrix where each row is formed with nodeID_to + "_" + nodeID_from,
    and columns are ID and visibility polygons.
    
    Parameters
    ----------
    edges_gdf: GeoDataFrame
        The edges GeoDataFrame containing LineString geometries.
    obstructions_gdf: GeoDataFrame
        Obstructions GeoDataFrame.
    max_expansion_distance: float
        The maximum distance from the node to expand the visibility cone.
    
    Returns
    -------
    DataFrame
        A DataFrame containing edge ID and visibility polygons.
    """

    obstructions_sindex = obstructions_gdf.sindex

    def process_edge(row):
        u, v = row['u'], row['v']
        poly1, poly2 = visibility_polygons_for_edge(row, obstructions_gdf, obstructions_sindex, max_expansion_distance)
        
        return pd.DataFrame({
            "ID": [f"{u}_{v}", f"{v}_{u}"],
            "Vispolygon": [poly1, poly2]
        })

    visibility_matrix = edges_gdf.apply(process_edge, axis=1).reset_index(drop=True)
    visibility_matrix = pd.concat(visibility_matrix.tolist(), ignore_index=True)
    
    # Convert to GeoDataFrame
    visibility_gdf = gpd.GeoDataFrame(visibility_matrix, geometry='geometry', crs=edges_gdf.crs)
    
    return visibility_gdf
    
    
 def visibility_polygons_for_edge(edge, obstructions_gdf, obstructions_sindex, max_expansion_distance=600):
    """
    Calculates two visibility polygons for an edge, one per direction.
    
    Parameters
    ----------
    edge: LineString
        The edge geometry.
    obstructions_gdf: GeoDataFrame
        Obstructions GeoDataFrame.
    max_expansion_distance: float
        The maximum distance from the node to expand the visibility cone.
    
    Returns
    -------
    tuple
        Two visibility polygons.
    """
    line = edge.geometry
    start, end = list(line.coords)[0], list(line.coords)[1]
    edge_angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
    node_start = Point(start)
    node_end = Point(end)
    poly1 = create_visibility_polygon(node_start, edge_angle, obstructions_gdf, obstructions_sindex, max_expansion_distance)
    poly2 = create_visibility_polygon(node_end, edge_angle + 180, obstructions_gdf, obstructions_sindex, max_expansion_distance)
    return poly1, poly2
    
    
    ### polygon handling
    
    

def aggregate_geometries(gdf, column_operations):
    """
    Aggregate overlapping geometries in a GeoDataFrame and perform specified operations on their attributes.

    Parameters
    ----------
    gdf: GeoDataFrame
        The input GeoDataFrame.
    column_operations: dict
        A dictionary where keys are column names and values are aggregation functions ('min', 'max', 'mean').

    Returns:
    ----------
    GeoDataFrame: 
        A new GeoDataFrame with aggregated geometries and attributes.
    """
    
    necessary = True

    while necessary:
        # Create a spatial index for faster lookup
        spatial_index = gdf.sindex
        to_remove_geometries = []

        # Loop through each geometry in the GeoDataFrame
        for idx, row in gdf.iterrows():
            # Find all geometries that contain the current geometry
            possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))
            containing_geometries = gdf.iloc[possible_matches_index]
            containing = containing_geometries[
                (containing_geometries.geometry.contains(row.geometry)) &
                (containing_geometries.index != idx) &
                (~containing_geometries.index.isin(to_remove_geometries))
            ]

            if len(containing) > 0:  # If the current geometry is contained by others
                # Get the container geometry (largest one by area)
                ix_container = containing.geometry.area.idxmax()
                row_container = gdf.loc[ix_container]

                # Update attributes based on the user-defined operations
                for column, operation in column_operations.items():
                    if operation == 'max':
                        gdf.loc[ix_container, column] = max(row[column], row_container[column])
                    elif operation == 'min':
                        gdf.loc[ix_container, column] = min(row[column], row_container[column])
                    elif operation == 'mean':
                        gdf.loc[ix_container, column] = (row[column] + row_container[column]) / 2

                # Add the current geometry to the list for removal
                to_remove_geometries.append(idx)

        # Break if no geometries were removed (no changes made)
        if len(to_remove_geometries) == 0:
            break

        # Remove aggregated geometries from the GeoDataFrame
        gdf = gdf[~gdf.index.isin(to_remove_geometries)]

    return gdf
    
def has_interior(poly):
    """Returns True if the polygon has interior rings (holes), False otherwise."""
    if poly.geom_type == "MultiPolygon":
        return false
    return len(poly.interiors) > 0

def is_exterior_inside_another(poly, other_poly):
    """Check if the exterior of one polygon is inside another polygon"""
    if not has_interior(poly):
        return False
    coords = list(other_poly.exterior.coords)
    coords.reverse()
    return any(coords == list(ring.coords) for ring in poly.interiors)
    
##########    
    import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

while True:
    gdf['ix'] = gdf.index
    sindex = gdf.sindex  # Spatial index for faster lookups

    gdf_right = gdf.copy()
    gdf_right['geo_right'] = gdf_right.geometry
    gdf['ch'] = gdf.geometry.convex_hull
    
    # **Step 1: Spatially join polygons that might be related**
    joined = gdf.set_geometry("ch").sjoin(gdf_right, predicate="intersects", how="left").query("ix_left != ix_right")
    
    # **Step 2: Identify containment relationships in bulk**
    # contained_pairs = joined[
    #     (joined.apply(lambda row: row["geometry"].contains(row["geo_right"]), axis=1)) |
    #     (joined.apply(lambda row: is_exterior_inside_another(row["geometry"], row["geo_right"]), axis=1)) |
    #     (joined.apply(lambda row: row["geometry"].convex_hull.contains(row["geo_right"]), axis=1))
    # ]

    contained_pairs = joined[
    joined.apply(lambda row: (
        row["geometry"].contains(row["geo_right"]) or
        is_exterior_inside_another(row["geometry"], row["geo_right"]) or
        row["geometry"].convex_hull.contains(row["geo_right"])
    ), axis=1)
]

    # **Step 3: If no more contained polygons, stop processing**
    if contained_pairs.empty:
        break

    print("size", len(contained_pairs))
    
    # **Step 4: Compute new min(base) and max(height) values in bulk**
    # Compute min(base_right) and max(height_right) for each ix_left
    min_base = contained_pairs.groupby("ix_left")["base_right"].min()
    min_height = contained_pairs.groupby("ix_left")["height_right"].max()

    # **Step 5: Apply updates efficiently**
    gdf.loc[min_base.index, "base"] = gdf.loc[min_base.index, "base"].combine(min_base, min)
    gdf.loc[min_height.index, "height"] = gdf.loc[min_height.index, "height"].combine(min_height, max)

    # Group contained geometries by ix_left and perform union
    merged_geometries = (
        contained_pairs.groupby("ix_left")["ix_right"]
        .apply(lambda group: unary_union(gdf.loc[group.tolist(), "geometry"]).buffer(0.001)))

    # break
    # Merge with the existing geometry of ix_left
    gdf.loc[merged_geometries.index, "geometry"] = gdf.loc[merged_geometries.index, "geometry"].combine(
        merged_geometries, lambda g1, g2: unary_union([g1, g2]))

    # **Step 7: Drop all contained polygons in one operation**
    gdf = gdf.drop(index=contained_pairs["ix_right"].unique())
    
    def merge_line_geometries(line_geometries):
    """
    Given a list of LineString geometries, this function reorders the geometries in the correct sequence based on their starting and ending points,
    and returns a merged LineString feature.
    
    Parameters:
    ----------
    line_geometries: List of LineString
        A list of LineString geometries to be merged.
        
    Returns:
    ----------
    LineString: 
        The merged LineString feature.
    """
    if not all(isinstance(line, LineString) for line in line_geometries):
        raise ValueError("Input must be a list of LineString geometries")
    if not all(isinstance(line, BaseGeometry) for line in line_geometries):
        raise ValueError("Input must be a list of valid geometries")
    if len(line_geometries) < 2:
        raise ValueError("At least 2 LineStrings are required to merge")
    
    # create a dictionary to store the "from" and "to" vertexes of each LineString as keys
    lines = {(line.coords[0], line.coords[-1]): line for line in line_geometries}
    # sort the line geometries based on their starting and ending coordinates
    line_geometries = sorted(line_geometries, key=lambda line: (line.coords[0], line.coords[-1]))
    # initialize an empty list to store the merged line geometries
    merged = []
    # rmove and store the first line geometry from the line_geometries list
    first_line = line_geometries.pop(0)
    # add the first line geometry to the merged list
    merged.append(first_line)

    # iterate over the remaining line geometries
    for line in line_geometries:
        # check if the last coordinate of the previously merged line is the same as the first coordinate of the current line
        if merged[-1].coords[-1] == line.coords[0]:
            # if so, add the current line to the merged list
            merged.append(line)
        # check if the last coordinate of the previously merged line is the same as the last coordinate of the current line
        elif merged[-1].coords[-1] == line.coords[-1]:
            # if so, add the reversed version of the current line to the merged list
            merged.append(LineString(list(reversed(line.coords))))
    
    return LineString(list(merged))