import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv
from tqdm import tqdm
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union
pd.set_option("display.precision", 3)

from .utilities import polygon_2d_to_3d 
from .angles import get_coord_angle
from .graph import graph_fromGDF

def visibility_polygon2d(building_geometry, obstructions_gdf, obstructions_sindex, max_expansion_distance = 600):
    """
    It creates a 2d polygon of visibility around a polygon geometry (e.g. building footprint) and computes its area. 
    This can be considered as a measure of 2d advance visibility. Such a polygon is built by constructing lines around the centroid of the building,
    breaking them at obstructions and connecting the new formed geometries to get the final polygon.
    The "max_expansion_distance" parameter indicates up to which distance from the building boundaries the visibility polygon can expand.
     
    Parameters
    ----------
    building_geometry: Polygon
        The building geometry.
    obstructions_gdf: Polygon GeoDataFrame
        The Polygon GeoDataFrame contianing the footprints of the obstructions (buildings).
    obstructions_sindex: Spatial Index
        The spatial index of the obstructions GeoDataFrame.
    max_expansion_distance: float
        It indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.

    Returns
    -------
    float
        The area of visibility.
    """
    # creating buffer
    distance_along = 10
    origin = building_geometry.centroid
    building_geometry = building_geometry.convex_hull if building_geometry.geom_type == 'MultiPolygon' else building_geometry
    max_expansion_distance += origin.distance(building_geometry.envelope.exterior)

    angles = np.arange(0, 360, distance_along)
    coords = np.array([get_coord_angle([origin.x, origin.y], distance=max_expansion_distance, angle=i) for i in angles])
    lines = [LineString([origin, Point(x)]) for x in coords]
    
    obstacles = obstructions_gdf[obstructions_gdf.crosses(unary_union(lines))]
    obstacles = obstacles[obstacles.geometry != building_geometry]
    obstacles = obstacles[~obstacles.geometry.within(building_geometry.convex_hull)]
    # creating lines all around the building till a defined distance
    
    if len(obstacles) > 0:
        ob = unary_union(obstacles.geometry)

        intersections = [line.intersection(ob) for line in lines]    
        clipped_lines = [LineString([origin, Point(intersection.geoms[0].coords[0])]) 
                         if ((type(intersection) == MultiLineString) & (not intersection.is_empty)) 
                         else LineString([origin, Point(intersection.coords[0])]) 
                         if ((type(intersection) == LineString) & (not intersection.is_empty))                               
                         else LineString([origin, Point(intersection[0].coords[0])]) 
                         if ((type(intersection) == Point) & (not intersection.is_empty))
                         else line for intersection, line in zip(intersections, lines)]
    # the line are not interrupted, keeping the original ones
    else:
        clipped_lines = lines

    # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
    poly = Polygon([[p.x, p.y] for p in [origin] + [Point(line.coords[1]) for line in clipped_lines ] + [origin]])
    poly_vis = poly.difference(building_geometry)
    if poly_vis.is_empty:
        poly_vis = poly.buffer(0).difference(building_geometry) 
    
    return poly_vis.area  
  
 
def compute_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, distance_min_observer_target = 300, chunk_size=500, consolidate = 
                        False, edges_gdf = None, tolerance = 0.0):
    """
    Computes the 3D sight lines between observer points in a Point GeoDataFrame (e.g. junction or nodes) and target buildings, based on a given distance along the buildings’ exterior and a minimum distance 
    between the observer and target (e.g. lines will not be constructed for observer points and 
    targets whose distance is lower than “distance_min_observer_target”). 
    
    The function performs 3d intervisibility checks in parallel by processing the set of sight lines in manageable chunks, based on the user cpu resources.
    
    Consolidation involves grouping spatially close observer nodes into clusters based on a specified tolerance. Nearby nodes are merged, with the derived cluster centroid 
    serving as the representative node. This process ensures that, for example, only one sight line is built for nodes that are close together, reducing computation time and generation of geometries.
    
    Note: the height of the building needs to represent the elevation of the building from the ground. The elevation of the ground, where the footprints lie on, should be passed through the column "base".
    
    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Point GeoDataFrame containing observer points (nodes) with 3D coordinates.
    buildings_gdf : GeoDataFrame
        GeoDataFrame containing building geometries, including attributes like base height and total height.
    distance_along : float, optional
        Distance interval for generating target points along building perimeters. Default is 200.
    distance_min_observer_target : float, optional
        Minimum distance required between observer and target points to create a sight line. Default is 300.
    chunk_size : int, optional
        Number of nodes to process in each chunk for parallel execution. Default is 500.
    consolidate : bool, optional
        Whether to consolidate nodes based on spatial proximity and tolerance. Default is False.
    edges_gdf : GeoDataFrame, optional
        GeoDataFrame representing edges between nodes, used for optional node consolidation. Default is None.
    tolerance : float, optional
        Tolerance for node consolidation, indicating the maximum allowable spatial distance for merging nodes. Default is 0.0.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the visible 3D sight lines, with columns for observer node IDs, building IDs,
        sight line geometry, and computed lengths.
    """
    if consolidate and edges_gdf is None:
        raise ValueError("Consolidation requires `edges_gdf` to be provided.")
        
    tmp_nodes, buildings_gdf = _prepare_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, consolidate = consolidate, edges_gdf = edges_gdf, tolerance = tolerance)
    
    buildings_sindex = buildings_gdf.sindex
    sight_lines_chunks = []
    # Divide nodes into manageable chunks
    node_chunks = np.array_split(tmp_nodes, max(1, len(tmp_nodes) // chunk_size))
    
    
    def _compute_distances_filter(row, min_distance):
        """
        Compute the distance between observer and target points and check if it's above the minimum distance.
        """
        observer = row.observer
        target = row.target
        distance = observer.distance(target)
        return distance >= min_distance

    for node_chunk in tqdm(node_chunks, desc="Processing node chunks"):
        # Create a temporary cartesian product for the current node chunk
        potential_sight_lines = pd.merge(node_chunk.assign(key=1), buildings_gdf.assign(key=1), on="key").drop("key", axis=1)

        # Compute distances and filter
        potential_sight_lines = potential_sight_lines[
            potential_sight_lines.apply(
                lambda row: _compute_distances_filter(row, distance_min_observer_target),
                axis=1
            )
        ]

        # If filtered potential sight lines are empty, skip processing
        if potential_sight_lines.empty:
            continue

        potential_sight_lines['start'] = [[observer.x, observer.y, observer.z] for observer in potential_sight_lines.observer]
        potential_sight_lines['stop'] =[[target.x, target.y, target.z] for target in potential_sight_lines.target]

        # Perform parallelized intervisibility check
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tqdm.pandas(desc="Checking visibility (parallelized)")
            potential_sight_lines["visible"] = list(
                executor.map(
                    lambda row: _intervisibility(row, buildings_gdf, buildings_sindex),
                    potential_sight_lines.itertuples(index=False)
                )
            )

        # Filter visible sight lines and append to the result
        potential_sight_lines = potential_sight_lines[potential_sight_lines["visible"] == True]
        sight_lines_chunks.append(potential_sight_lines)


    # Combine all chunks into a single GeoDataFrame
    sight_lines_tmp = pd.concat(sight_lines_chunks, ignore_index=True)
    sight_lines_tmp.reset_index(drop=True, inplace=True)
    
    sight_lines_tmp = sight_lines_tmp.drop(['geometry_x', 'geometry_y', 'start', 'stop', 'visible', 'base', 
                                    'height', 'building_3d', 'observer', 'z'], axis = 1)
    sight_lines_exploded = sight_lines_tmp.explode(column='nodeIDs', ignore_index=True)

    nodes_gdf['geometry'] = [Point(geometry.x, geometry.y, z) for geometry, z in zip(nodes_gdf['geometry'], nodes_gdf['z'])]
    
    # Merge with `nodes` GeoDataFrame to get the geometry of each nodeID
    sight_lines_exploded = sight_lines_exploded.merge(
        nodes_gdf[['nodeID', 'geometry']],
        left_on='nodeIDs',
        right_on='nodeID',
        suffixes=('', '_node')
    )
    
    # Step 3: Update the `observer` column with the merged geometry
    sight_lines_exploded['geometry'] = [LineString([observer, target]) for observer, target in zip(sight_lines_exploded['geometry'], 
                                                                                          sight_lines_exploded['target'])]
    sight_lines = gpd.GeoDataFrame(sight_lines_exploded, geometry='geometry', crs = nodes_gdf.crs)
    sight_lines['nodeID'] = sight_lines['nodeIDs']
    sight_lines.drop(['target', 'nodeIDs', 'nodeID_node'], axis = 1, inplace = True)
    sight_lines['length'] = sight_lines.geometry.length
    sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'], ascending=[False, False, False]).drop_duplicates(['buildingID', 'nodeID'], keep='first')
    sight_lines.reset_index(inplace = True, drop = True)
    
    return sight_lines 
    
def _intervisibility(row, buildings_gdf, buildings_sindex):
    """
    Check if a 3D line of sight between two points is obstructed by any buildings.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing observer and target points, along with their 3D coordinates.
    buildings_gdf : GeoDataFrame
        Polygon GeoDataFrame containing building geometries and associated attributes including base height and height from the base.
    buildings_sindex : rtree.index.Index
        Spatial index of the buildings GeoDataFrame for efficient spatial queries.

    Returns
    -------
    bool
        True if the line of sight is unobstructed (visible), False otherwise.
    """
    line2d = LineString([row.start, row.stop])
    possible_matches_index = list(buildings_sindex.intersection(line2d.buffer(5).bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    matches = possible_matches[possible_matches.intersects(line2d)]
    matches = matches[matches.buildingID != row.buildingID]
    if len(matches) == 0:
        return True

    for _, row_building in matches.iterrows():
        building_3d = row_building.building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(row.start, row.stop)
        if len(intersections) > 0:
            return False
            
    return True
    
def _prepare_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, consolidate = False, edges_gdf = None, tolerance = 0.0):
    """
    Prepare 3D sight lines by processing nodes and buildings into usable formats for visibility calculations.
    
    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Point GeoDataFrame containing observer points (nodes) with 3D coordinates.
    buildings_gdf : GeoDataFrame
        Polygon GeoDataFrame containing building geometries and associated attributes including base height and height from the base.
    distance_along : float, optional
        Distance interval for creating target points along building perimeters. Default is 200.
    consolidate : bool, optional
        Whether to consolidate nodes based on spatial proximity and tolerance. Default is False.
    edges_gdf : GeoDataFrame, optional
        GeoDataFrame representing edges between nodes, used for optional node consolidation. Default is None.
    tolerance : float, optional
        Tolerance for node consolidation, indicating the maximum allowable spatial distance for merging nodes. Default is 0.0.

    Returns
    -------
    tuple
        A tuple containing:
        - consolidated_nodes (GeoDataFrame): GeoDataFrame of processed and optionally consolidated observer nodes.
        - buildings_gdf (GeoDataFrame): GeoDataFrame with processed building geometries and target points.
    """
    nodes_gdf = nodes_gdf[['geometry', 'x', 'y', 'nodeID', 'z']].copy()
    nodes_gdf['geometry'] = nodes_gdf.apply(lambda row: Point(row['x'], row['y'], row['z']), axis=1)
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(
        lambda geom: geom if isinstance(geom, Polygon) else geom.convex_hull if geom.is_valid else geom.buffer(0)
    )

    # add a 'base' column to the buildings GeoDataFrame with a default value of 1.0, if not provided
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 1.0
    buildings_gdf['base'] = buildings_gdf['base'].where(buildings_gdf['base'] < 1.0, 1.0) # minimum base
    
    buildings_gdf = buildings_gdf[['geometry', 'buildingID', 'base', 'height']].copy()

    def add_height_to_line(exterior, base, height):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        return LineString([(x, y, height + base) for x, y in exterior.coords])

    def process_geometry_to_linestring(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString([point for poly in geometry.geoms for point in add_height_to_line(poly.exterior, base, height).coords])

    # create a Series with the height-extended LineString objects
    building_exteriors_roof = buildings_gdf.apply(lambda row: process_geometry_to_linestring(row.geometry, row.base, row.height), axis=1)
    # create a list with the number of intervals along the exterior LineString to divide the line into
    num_intervals = [max(1, int(exterior.length / distance_along)) for exterior in building_exteriors_roof]
    
    # create a new column with the list of targets along the exterior LineString
    def interpolate_targets(exterior, num_intervals):
        return [exterior.interpolate(min(exterior.length / 2, distance_along) * i) for i in range(num_intervals)]
    
    buildings_gdf['target'] = [interpolate_targets(exterior, intervals) for exterior, intervals in zip(building_exteriors_roof, num_intervals)]

    if consolidate:
        consolidated_nodes = _consolidate_nodes(nodes_gdf, edges_gdf, tolerance = tolerance)
    else:
        consolidated_nodes = nodes_gdf
        consolidated_nodes['geometry'] = [Point(geom.x, geom.y, z) for geom, z in zip(consolidated_nodes.geometry, consolidated_nodes['z'])]

    
    consolidated_nodes['observer'] = consolidated_nodes['geometry']
    # create a temporary dataframe with building targets
    buildings_gdf = buildings_gdf.explode('target')
    consolidated_nodes = consolidated_nodes.drop(["x", "y"], axis = 1)
    buildings_gdf = buildings_gdf[['buildingID', 'target', 'geometry', 'height', 'base']].copy()

    # create pyvista 3d objects for buildings
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['building_3d'] = [polygon_2d_to_3d(geo, base, height) for geo, base, height in 
                                    zip(buildings_gdf['geometry'], buildings_gdf['base'], buildings_gdf['height'])]

    return consolidated_nodes, buildings_gdf

def _consolidate_nodes(nodes_gdf, edges_gdf, tolerance = 40):
    """
    Consolidate nodes in a GeoDataFrame by clustering based on spatial proximity and resolving disconnected components.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Point GeoDataFrame containing observer points (nodes) with 3D coordinates.
    edges_gdf : GeoDataFrame
        GeoDataFrame representing edges between nodes, used to identify connectivity within clusters.
    tolerance : float, optional
        Maximum allowable spatial distance for merging nodes into clusters. Default is 40.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame of consolidated nodes, with updated geometries and attributes reflecting the merged clusters.
    """
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf.drop(['x', 'y'], axis = 1, inplace = True)
    nodes_gdf.index = nodes_gdf.nodeID
    nodes_gdf.index.name = None

    graph = graph_fromGDF(nodes_gdf, edges_gdf, nodeID = "nodeID")
    
    def _merge_nodes_geometries(nodes_gdf, tolerance):

        # buffer nodes then merge overlapping geometries
        merged = nodes_gdf.buffer(tolerance).unary_union

        # extract the member geometries if it's a multi-geometry
        merged = merged.geoms if hasattr(merged, "geoms") else merged
        return gpd.GeoSeries(merged, crs= nodes_gdf.crs)
    
    
    # STEP 1: Buffer nodes by tolerance, merge overlaps, and find centroids
    clusters = gpd.GeoDataFrame(geometry = _merge_nodes_geometries(nodes_gdf, tolerance), crs = nodes_gdf.crs)
    centroids = clusters.centroid
    clusters["x"] = centroids.x
    clusters["y"] = centroids.y

    # STEP 2: Attach nodes to their clusters by spatial join
    gdf = gpd.sjoin(nodes_gdf, clusters, how="left", predicate="within")
    gdf = gdf.drop(columns="geometry").rename(columns={"index_right": "cluster"})
    gdf["cluster"] = gdf["cluster"].astype(int)

    # STEP 3
    # if a cluster contains multiple components (i.e., it's not connected)
    # move each component to its own cluster (otherwise you will connect
    # nodes together that are not truly connected, e.g., nearby deadends or
    # surface streets with bridge).

    cluster_counter = gdf.cluster.max()+1
    for cluster_label, nodes_subset in gdf.groupby("cluster"):
        if len(nodes_subset) > 1:
            # identify all the (weakly connected) component in cluster
            wccs = list(nx.connected_components(graph.subgraph(nodes_subset.index)))
            if len(wccs) > 1:
                # if there are multiple components in this cluster
                for  wcc in wccs:
                    # set subcluster xy to the centroid of just these nodes
                    idx = list(wcc)
                    subcluster_centroid = nodes_gdf.loc[idx].unary_union.centroid
                    
                    gdf.loc[idx, "x"] = subcluster_centroid.x
                    gdf.loc[idx, "y"] = subcluster_centroid.y
                    # move to subcluster by appending suffix to cluster label
                    gdf.loc[idx, "cluster"] = cluster_counter
                    cluster_counter +=1

    # Assign unique integer labels to each consolidated node cluster
    gdf["cluster"] = gdf["cluster"].factorize()[0]

    # STEP 4: Group and consolidate nodes by cluster, keeping memory of nodeID origins
    consolidated_nodes = []

    for cluster_label, nodes_subset in gdf.groupby("cluster"):
        if len(nodes_subset) == 0:
            continue
        nodeIDs = nodes_subset.nodeID.to_list()
    
        # Calculate centroid of the geometries in nodes_subset
        cluster_centroid = nodes_gdf.loc[nodeIDs].geometry.unary_union.centroid
        z_mean = nodes_gdf.loc[nodeIDs].z.mean()
        
        consolidated_node = {
            "nodeIDs": nodeIDs,
            "x": cluster_centroid.x,
            "y": cluster_centroid.y,
            "z": z_mean,
            "nodeID" : cluster_label
        }
        consolidated_nodes.append(consolidated_node)

        consolidated_nodes_gdf = gpd.GeoDataFrame(
            consolidated_nodes,
            geometry = gpd.points_from_xy(
                [row["x"] for row in consolidated_nodes],
                [row["y"] for row in consolidated_nodes],
                z = [row["z"] for row in consolidated_nodes],
                crs= nodes_gdf.crs
            ))
    
    return(consolidated_nodes_gdf)