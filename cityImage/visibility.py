import os
import gc
import warnings
import psutil
import ast 

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, MultiPoint, GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree

import pyvista as pv
from tqdm import tqdm
from pathlib import Path

# Parallel/Distributed
import dask
from dask import delayed, compute
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project modules
from .angles import get_coord_angle
from .graph_consolidate import consolidate_nodes
from .utilities import gdf_multipolygon_to_polygon

pd.set_option("display.precision", 3)

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
        Obstructions GeoDataFrame.
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
    
def compute_3d_sight_lines(nodes_gdf: gpd.GeoDataFrame, target_buildings_gdf: gpd.GeoDataFrame, obstructions_buildings_gdf: gpd.GeoDataFrame, 
                           simplified_target_buildings: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame, city_name: str,
                           distance_along: float = 200, min_observer_target_distance: float = 300, height_relative_to_ground = False,
                           sight_lines_chunk_size: int = 500000, 
                           consolidate: bool = False, consolidate_tolerance: float = 0.0, 
                           num_workers: int = 20):
    """
    Computes 3D sight lines between observer nodes and target buildings, accounting for 2D and 3D obstructions.

    The computation is performed in memory-efficient chunks. For each chunk:
    1. Filters potential sight lines by minimum distance.
    2. Restricts obstruction checks to those near each target.
    3. Checks for 2D (planimetric) obstructions.
    4. Conducts visibility with full 3D intersection analysis.
    5. Outputs are merged and saved to disk per chunk, then concatenated and finalized.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Observer locations (usually points) with 3D coordinates and unique node IDs.
    target_buildings_gdf : GeoDataFrame
        Target building polygons (must have 3D geometry).
    obstructions_buildings_gdf : GeoDataFrame
        All buildings considered as potential obstructions (must have 3D geometry).
    simplified_target_buildings : GeoDataFrame
        Simplified versions of target buildings (used for finalization and snapping).
    edges_gdf : GeoDataFrame
        Edge geometries for network structure (used for consolidation).
    city_name : str
        Name prefix for exported chunk files.
    distance_along : float, optional
        Sampling step along building edges (default: 200).
    min_observer_target_distance : float, optional
        Minimum allowed observer-to-target distance (default: 300).
    sight_lines_chunk_size : int, optional
        Maximum number of sight lines to process per chunk (default: 500000).
    consolidate : bool, optional
        Whether to spatially consolidate observer nodes and targets (default: False).
    consolidate_tolerance : float, optional
        Tolerance for consolidation (default: 0.0).
    num_workers : int, optional
        Number of parallel workers to use for chunked and obstruction operations (default: 20).

    Returns
    -------
    sight_lines : GeoDataFrame
        Finalized GeoDataFrame of visible sight lines (3D LineString), with relevant attributes. If no lines are visible, returns empty GeoDataFrame.

    Notes
    -----
    - The function splits the computation into manageable chunks for memory efficiency.
    - Intermediate chunk files are written to disk and merged at the end.
    - Columns such as 'matchesIDs', 'observer_coords', and 'target_coords' are added temporarily and dropped in the final result.
    """
    
    # Step 0: Prepare data
    observers, targets, obstructions_gdf = _prepare_3d_sight_lines(nodes_gdf, target_buildings_gdf, obstructions_buildings_gdf,
                                                               distance_along = distance_along,
                                                               simplified_buildings = simplified_target_buildings,
                                                               consolidate=consolidate,
                                                               consolidate_tolerance=consolidate_tolerance,
                                                               edges_gdf=edges_gdf, height_relative_to_ground = height_relative_to_ground)
    
    num_observers = len(observers)
    num_targets = len(targets)
    projected_nr_sight_lines = num_observers * num_targets
    obstructions_sindex = obstructions_gdf.sindex   # GeoPandas SpatialIndex wrapper
    meshes = _build_meshes(obstructions_gdf) 
    
    if projected_nr_sight_lines <= sight_lines_chunk_size:
        observer_chunks = [observers]
    else:
        observers_per_chunk = max(1, sight_lines_chunk_size // num_targets)
        num_chunks = int(np.ceil(num_observers / observers_per_chunk))
        observer_chunks = np.array_split(observers, num_chunks)

    out_prefix="chunk_sight_lines"
    out_files = []
    
    for n, chunk in enumerate(observer_chunks):

        print(f"Starting chunk {n+1} of {len(observer_chunks)}")
        visibles = []

        print(" 01 - Filtering by distance")
        potential_sight_lines = filter_distance(chunk, targets, min_observer_target_distance)
        if potential_sight_lines.empty:
            continue

        print(" 02 - Checking 2d obstructions")
        visible_2d, obstructed_2d = obstructions_2d(potential_sight_lines, obstructions_gdf, obstructions_sindex, num_workers = num_workers)
        if not visible_2d.empty:
            visibles.append(visible_2d)
        
        print(" 03 - Checking 3d obstructions")
        visible_3d = pd.DataFrame()
        if not obstructed_2d.empty:
            obstructed_2d['geometry'] = [LineString([observer, target]) for observer, target in zip(obstructed_2d['observer_geo'], obstructed_2d['target_geo'])]
            visible_3d = obstructions_3d(obstructed_2d, obstructions_gdf, 'matchesIDs', meshes, simplified_target_buildings, num_workers=num_workers)
            if not visible_3d.empty:
                visibles.append(visible_3d)
        
        chunk_dir = Path("sight_lines_tmp")
        chunk_dir.mkdir(parents=True, exist_ok=True)  # create folder if not existing
        
        # Finalize columns and export
        if visibles:             
            cols_to_drop = ['visible', 'z']
            
            for df in visibles:
                df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
            
            chunk_sight_lines = pd.concat(visibles, ignore_index=True)
            chunk_sight_lines = _finalize_sight_lines(chunk_sight_lines, nodes_gdf, consolidate, simplified_target_buildings)

            # Build filename
            chunk_file = chunk_dir / f"{city_name}_{out_prefix}_{n}.gpkg"
            chunk_sight_lines.to_file(chunk_file)
            out_files.append(chunk_file)

            del chunk_sight_lines, visibles
            gc.collect()
            print(f"Chunk {n+1} processed and exported: {chunk_file}")

        visibles = []
        del chunk, potential_sight_lines, visible_2d, obstructed_2d, visible_3d
        gc.collect()
    
    print("All chunks processed.")
    tmp_sight_lines = gpd.GeoDataFrame(geometry=[], crs=obstructions_gdf.crs)
    if out_files:
        tmp_sight_lines = merge_gpkg_chunks_to_gdf(out_files, 'matchesIDs')
        tmp_sight_lines.drop(['visible'], axis=1, errors = 'ignore', inplace=True)
    else:
        print("No visible sight-lines")
        return tmp_sight_lines
  
    if consolidate:
        print(" 04 - Final check")
        sight_lines = _last_check(tmp_sight_lines, obstructions_gdf, obstructions_sindex, meshes, nodes_gdf, num_workers = num_workers)
    
    return sight_lines

## Preparation ###########################
def _prepare_3d_sight_lines(nodes_gdf, target_buildings_gdf, obstructions_gdf, distance_along = 200, simplified_buildings = gpd.GeoDataFrame, 
        consolidate = False, consolidate_tolerance = 0.0, edges_gdf = None, height_relative_to_ground = False):
    """
    Prepares observer points, target points, and obstruction geometries for 3D sight line analysis.

    This function processes the input node and building GeoDataFrames to:
        - Set up observer points with 3D coordinates,
        - Prepare target building roof points at regular intervals,
        - Optionally apply simplified building outlines and spatial consolidation,
        - Annotate each target with nearby building IDs for efficient obstruction queries,
        - Create 3D representations for all obstruction polygons.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Observer locations with columns ['geometry', 'x', 'y', 'nodeID', 'z'].
    target_buildings_gdf : GeoDataFrame
        Target building polygons with 'height' and 'base' attributes.
    obstructions_gdf : GeoDataFrame
        Building polygons considered as obstructions (must have 'height' and 'base').
    distance_along : float, optional
        Interval in CRS units for sampling target building roofs (default: 200).
    simplified_buildings : GeoDataFrame, optional
        Simplified building outlines, used if not empty (default: empty).
    consolidate : bool, optional
        If True, spatially merges close observer points (default: False).
    consolidate_tolerance : float, optional
        Distance tolerance for observer consolidation (default: 0.0).
    edges_gdf : GeoDataFrame, optional
        Network edges for observer consolidation.

    Returns
    -------
    observer_points_gdf : GeoDataFrame
        GeoDataFrame of observer points with 3D geometry.
    target_points : GeoDataFrame
        Points along target building roofs with associated nearby building IDs.
    obstructions_gdf : GeoDataFrame
        Obstructions as 3D polygons with updated geometry.
    """

    nodes_gdf = nodes_gdf[['geometry', 'x', 'y', 'nodeID', 'z']].copy()
    nodes_gdf.loc[nodes_gdf["z"] < -50, "z"] = 2
    nodes_gdf['geometry'] = nodes_gdf.apply(lambda row: Point(row['x'], row['y'], row['z']), axis=1)

    target_buildings_gdf = _prepare_buildings_gdf(target_buildings_gdf)
    obstructions_gdf = _prepare_buildings_gdf(obstructions_gdf)
    target_buildings_gdf = target_buildings_gdf[target_buildings_gdf.height > 5.0]
    target_points = _prepare_targets(target_buildings_gdf, distance_along, height_relative_to_ground)

    if simplified_buildings is not None and not simplified_buildings.empty:
        target_points, obstructions_gdf = _use_simplified_buildings(target_points, obstructions_gdf, simplified_buildings)
    else:
        target_points.drop('geometry', axis = 1, inplace = True)
  
    if consolidate:
        observer_points_gdf = consolidate_nodes(nodes_gdf, edges_gdf, consolidate_edges_too = False, tolerance = consolidate_tolerance)
    else:
        observer_points_gdf = nodes_gdf.copy()

    target_points.set_geometry('target_geo', inplace = True, crs = nodes_gdf.crs)
    observer_points_gdf['observer_geo'] = observer_points_gdf['geometry']
    observer_points_gdf = observer_points_gdf.drop(["x", "y", "z"], axis = 1)
    
    obstructions_gdf['building_3d'] = [polygon_2d_to_3d(geometry, base, height, extrude_from_sealevel = True, height_relative_to_ground = False) for geometry, base, height in 
                                       zip(obstructions_gdf['geometry'], obstructions_gdf['base'], obstructions_gdf['height'])]
    obstructions_gdf['geometry'] = obstructions_gdf['geometry'].apply(lambda geom: geom.exterior)
    
    return observer_points_gdf, target_points, obstructions_gdf

def _prepare_buildings_gdf(buildings_gdf):
    """
    Sets minimum base elevation, filters invalid heights, and standardizes columns.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Building footprints with 'geometry', 'buildingID', 'height', and (optionally) 'base' columns.

    Returns
    -------
    buildings_gdf : GeoDataFrame
        Cleaned building GeoDataFrame, indexed by 'buildingID', with non-null height and base >= 1.0.
    """      
    # add a 'base' column to the buildings GeoDataFrame with a default value of 1.0, if not provided
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 1.0
        
    buildings_gdf['base'] = buildings_gdf['base'].where(buildings_gdf['base'] > 1.0, 1.0) # minimum base
    buildings_gdf = buildings_gdf[buildings_gdf["height"].notna() & (buildings_gdf["height"] != None)]
    buildings_gdf = buildings_gdf[['buildingID', 'geometry', 'height', 'base']].copy()
    buildings_gdf.index = buildings_gdf.buildingID
    buildings_gdf.index.name = None
    return buildings_gdf
    
def _prepare_targets(target_buildings_gdf, distance_along, height_relative_to_ground = False):
    """
    Generates 3D target points along the roof perimeter of each building at specified intervals.

    Parameters
    ----------
    target_buildings_gdf : GeoDataFrame
        Target building polygons with 'geometry', 'height', and 'base' columns.
    distance_along : float
        Interval in CRS units for placing points along the roof edge.

    Returns
    -------
    target_points : GeoDataFrame
        Exploded GeoDataFrame with 3D points along each building's roof edge.
    """

    target_buildings = target_buildings_gdf.copy()
    target_buildings['geometry'] = target_buildings_gdf['geometry'].apply(
        lambda geom: geom if isinstance(geom, Polygon) else geom.convex_hull if geom.is_valid else geom.buffer(0)
    )
    
    def extrude_roof(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString([point for poly in geometry.geoms for point in add_height_to_line(poly.exterior, base, height).coords])
    
    def add_height_to_line(exterior, base, height, height_relative_to_ground = height_relative_to_ground):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        return LineString([(coord[0], coord[1], height + base) for coord in exterior.coords])
        
    # create the roof
    building_tops = target_buildings.apply(lambda row: extrude_roof(row.geometry, row.base, row.height), axis=1)
    target_buildings["target_geo"] = [downsample_coords(line, distance_along) for line in building_tops]

    # 3. explode into individual vertices → Points
    target_points = target_buildings.explode("target_geo", ignore_index=True)
    target_points["target_geo"] = target_points["target_geo"].apply(Point)
    return target_points
  
def downsample_coords(line, distance_along):
    coords = np.array(line.coords)
    if coords.shape[0] == 0:
        return []
    selected, dist = [], 0.0
    for i in range(1, len(coords)):
        seg_len = np.linalg.norm(coords[i] - coords[i-1])
        dist += seg_len
        if dist >= distance_along:
            selected.append(tuple(coords[i]))
            dist = 0.0
    if not selected:
        mid = np.array(line.interpolate(line.length / 2.0).coords[0])
        dists = np.linalg.norm(coords - mid, axis=1)
        selected = [tuple(coords[np.argmin(dists)])]
    return selected
  
def _use_simplified_buildings(target_points, obstructions_gdf, simplified_buildings):
    """
    Associates each target point and obstruction polygon with a simplified building outline,
    then computes aggregated attributes for each simplified building.

    Parameters
    ----------
    target_points : GeoDataFrame
        3D points representing target locations on building roofs.
    obstructions_gdf : GeoDataFrame
        Building polygons considered as obstructions.
    simplified_buildings : GeoDataFrame
        Simplified building outlines.

    Returns
    -------
    simplified_target_points : GeoDataFrame
        Simplified buildings with aggregated target information and 3D centroid points.
    simplified_obstructions : GeoDataFrame
        Simplified buildings with aggregated obstruction attributes.
    """

    simplified_buildings = gdf_multipolygon_to_polygon(simplified_buildings)
    # Create a spatial index for the simplified buildings
    simplified_tree = STRtree(simplified_buildings.geometry)
    
    def assign_geometry_to_simplified_building(geometry):
        # First, check if the target is within or touches any simplified building using spatial index
        touching_buildings = simplified_tree.query(geometry)
        
        # Check for touch or within condition
        for simplified_building in touching_buildings: #provided these are IDs  
            simplified_building_geo = simplified_buildings.loc[simplified_building].geometry
            if geometry.within(simplified_building_geo) or geometry.touches(simplified_building_geo) or geometry.intersects(simplified_building_geo):
                return simplified_building
        
        return None
    
    # Apply the optimized function to each target
    target_points['simplifiedID'] = target_points['target_geo'].apply(lambda x: assign_geometry_to_simplified_building(x))
    obstructions_gdf['simplifiedID'] = obstructions_gdf['geometry'].apply(lambda x: assign_geometry_to_simplified_building(x))

    # Function to get building IDs, average height, and average base of targets assigned to the same simplified building
    def get_attributes(simplifiedID, gdf):
        # Filter targets assigned to the simplified building
        detailed_buildings = gdf[gdf['simplifiedID'] == simplifiedID].copy()  
        # Extract building IDs of these targets
        buildingIDs = detailed_buildings['buildingID'].tolist() if not detailed_buildings.empty else []
    
        # Check if 'target_geo' exists and extract target geometries
        if 'target_geo' in detailed_buildings.columns:
            target_GEOs = detailed_buildings['target_geo'].tolist() if not detailed_buildings.empty else []
        else:
            target_GEOs = None
    
        # Calculate the average height and base of these targets
        height = detailed_buildings['height'].mean() if not detailed_buildings.empty else np.nan
        base = detailed_buildings['base'].mean() if not detailed_buildings.empty else np.nan
    
        return pd.Series([buildingIDs, target_GEOs, base, height])

    simplified_target_points = simplified_buildings.copy()
    simplified_obstructions = simplified_buildings.copy()
    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_target_points[['buildingIDs', 'target_GEOs', 'base', 'height']] = simplified_buildings.index.to_series().apply(
        get_attributes, gdf=target_points).apply(pd.Series)

    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_obstructions[['buildingIDs', 'target_GEOs', 'base', 'height']] = simplified_buildings.index.to_series().apply(
        get_attributes, gdf=obstructions_gdf).apply(pd.Series)

    simplified_target_points = simplified_target_points[simplified_target_points['buildingIDs'].apply(len) > 0]
    simplified_target_points = simplified_target_points[simplified_target_points['height'].notna()]
    simplified_target_points['buildingID'] = simplified_target_points.index
    
    simplified_obstructions.drop(['buildingIDs', 'target_GEOs'], axis = 1, inplace = True)    
    simplified_obstructions = simplified_obstructions[simplified_obstructions['height'].notna()]
    simplified_obstructions['buildingID'] = simplified_obstructions.index
    
    simplified_target_points['target_geo'] = simplified_target_points.apply(lambda row: Point(row['geometry'].centroid.x, 
                                                                                      row['geometry'].centroid.y, row['height'] + row['base']), axis=1)
    simplified_target_points.drop(['geometry', 'base', 'height', 'FID'], axis = 1, inplace = True, errors = 'ignore')
    return simplified_target_points, simplified_obstructions
    
def merge_gpkg_chunks_to_gdf(out_files, potential_obstructions_column):
    """
    Reads multiple GeoPackage chunk files and merges them into a single GeoDataFrame.

    Parameters
    ----------
    out_files : list of str
        List of file paths to the .gpkg files (one per chunk).

    Returns
    -------
    final_gdf : GeoDataFrame
        Combined GeoDataFrame containing all features from the input files.
    """

    # Read each chunk one at a time and append to a list, then concat all together
    gdfs = []
    for f in out_files:
        gdfs.append(gpd.read_file(f))
    final_gdf = pd.concat(gdfs, ignore_index=True)
    return final_gdf
    
## Step 0 ###########################
def filter_distance(chunk, targets, min_observer_target_distance):
    """
    Filters potential observer-target pairs by minimum distance and generates sight line geometries.

    Parameters
    ----------
    chunk : GeoDataFrame
        Observer points (usually a subset/chunk of all observers).
    targets : GeoDataFrame
        Target points to be paired with observers.
    min_observer_target_distance : float
        Minimum allowed 2D distance between observer and target (in CRS units).

    Returns
    -------
    potential_lines : GeoDataFrame
        GeoDataFrame of observer-target pairs meeting the distance threshold,
        with new 'geometry' (LineString) and coordinate columns.
    """

    # Prepare data
    potential_lines = _prepare_chunk_data(chunk, targets)
    # Assign 2D coordinates to new columns directly
    potential_lines['observer_coords'] = [(p.x, p.y) for p in potential_lines['observer_geo']]
    potential_lines['target_coords'] = [(p.x, p.y) for p in potential_lines['target_geo']]
   
    # Extract 2D coordinates and then distance filter
    observer_coords = np.array(potential_lines['observer_coords'].to_list())
    target_coords = np.array(potential_lines['target_coords'].to_list())
       
    # Query pairs with distance >= threshold
    distances = np.linalg.norm(observer_coords - target_coords, axis=1)
    mask = distances >= min_observer_target_distance
    potential_lines = potential_lines.loc[mask]

    # Extract 2D coordinates and then distance filter
    observer_coords = np.array(potential_lines['observer_coords'].to_list())
    target_coords = np.array(potential_lines['target_coords'].to_list())
    line_geometries = [LineString([start, stop]) for start, stop in zip(observer_coords, target_coords)]
    potential_lines['geometry'] = line_geometries

    return potential_lines
    
def _prepare_chunk_data(chunk: gpd.GeoDataFrame, targets: gpd.GeoDataFrame):
    """
    Computes the cartesian product of observers and targets for a chunk, omitting geometry columns for efficiency.

    Parameters
    ----------
    chunk : GeoDataFrame
        Observer points (one chunk of all observers).
    targets : GeoDataFrame
        Target points.

    Returns
    -------
    potential_lines : DataFrame
        DataFrame of all observer-target pairs (cartesian product) with all non-geometry attributes.
    """

    # Drop geometry column early to reduce memory usage
    chunk_data = chunk.drop('geometry', axis=1).assign(key=1)
    targets_data = targets.assign(key=1)
    
    # Compute cartesian product efficiently
    potential_lines = pd.merge(chunk_data, targets_data, on="key").drop("key", axis=1)
    print("Num potential lines is", len(potential_lines), "; Nodes in chunk:", len(chunk_data), "Num Targets", len(targets_data))
    return potential_lines

## Step 1 ###########################    
def obstructions_2d(potential_sight_lines, obstructions_gdf, obstructions_sindex, num_workers=20):
    """
    Identifies 2D (planimetric) obstructions between each observer-target sight line and surrounding buildings.

    Uses spatial batch processing and parallel computation to:
      1. Find which obstructions could intersect each sight line (via spatial index).
      2. Check actual intersections and compute obstruction locations for each line.
      3. Split results into visible and potentially obstructed sight lines.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines (with geometry).
    obstructions_gdf : GeoDataFrame
        Building polygons (must include 'buildingID', 'geometry', 'height', and 'base').
    num_workers : int, optional
        Number of parallel threads for batch computation (default: 20).

    Returns
    -------
    visible : GeoDataFrame
        Sight lines with no 2D obstructions.
    potentially_obstructed : GeoDataFrame
        Sight lines that intersect at least one obstruction in 2D.
    """
  
    potential_sight_lines = potential_sight_lines.copy()
    sub_chunks = _define_batches(potential_sight_lines)
    tasks = [_find_obstructions_2d_delayed(sub_chunk, obstructions_gdf, obstructions_sindex) for sub_chunk in sub_chunks]
    results = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)
    
    visibles = [r[0] for r in results]
    potentially_obstructeds = [r[1] for r in results]
    visible = pd.concat(visibles, ignore_index=True)
    potentially_obstructed = pd.concat(potentially_obstructeds, ignore_index=True)
    return visible, potentially_obstructed

def _define_batches(potential_sight_lines, min_size=50, max_size=2000, mem_fraction=0.1, row_weight_kb=10_000):
    """
    Splits a GeoDataFrame of sight lines into memory-aware batches for efficient parallel processing.

    Batch size is based on available system memory and estimated row weight.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Data to be divided into batches.
    min_size : int, optional
        Minimum number of rows per batch (default: 50).
    max_size : int, optional
        Maximum number of rows per batch (default: 2000).
    mem_fraction : float, optional
        Fraction of available system memory to use for each batch (default: 0.1).
    row_weight_kb : int, optional
        Estimated memory usage per row, in kilobytes (default: 10,000).

    Yields
    ------
    GeoDataFrame
        The next batch of rows.
    """

    available_mem = psutil.virtual_memory().available
    rows_per_batch = int(available_mem * mem_fraction / row_weight_kb)
    rows_per_batch = max(min_size, min(rows_per_batch, max_size))
    n = len(potential_sight_lines)
    for start in range(0, n, rows_per_batch):
        yield potential_sight_lines.iloc[start: start + rows_per_batch]

def _find_obstructions_2d(sub_chunk, obstructions_gdf, obstructions_sindex):
    """
    For each sight line in the batch, checks for actual 2D intersections with obstructions,
    using a prebuilt GeoPandas.sindex (SpatialIndex).

    Parameters
    ----------
    sub_chunk : GeoDataFrame
        Batch of candidate sight lines, must have 'geometry' and typically 'buildingID'.
    obstructions_gdf : GeoDataFrame
        Obstructions with 'buildingID' and 'geometry'.
    sindex : geopandas.sindex.SpatialIndex
        Pre-built spatial index of obstructions_gdf.

    Returns
    -------
    visible : GeoDataFrame
        Lines without any true 2D obstructions.
    obstructed : GeoDataFrame
        Lines intersecting ≥1 obstruction; includes 'obstructionIDs' column.
    """

    # Vectorised spatial query – true intersections if backend supports predicate
    pairs = obstructions_sindex.query(sub_chunk.geometry, predicate="crosses")
    poly_ids = obstructions_gdf["buildingID"].to_numpy()

    # Group buildingIDs by line index
    obstruction_dict = {}
    for line_idx, poly_idx in zip(pairs[0], pairs[1]):
        bid = poly_ids[poly_idx]
        obstruction_dict.setdefault(line_idx, []).append(bid)

    # Assign intersecting buildingIDs per row
    obstructionIDs = [
        [int(bid) for bid in obstruction_dict.get(i, [])]
        for i in range(len(sub_chunk))
    ]
    sub_chunk["matchesIDs"] = obstructionIDs

    mask = sub_chunk["matchesIDs"].map(len) > 0
    visible = sub_chunk.loc[~mask].reset_index(drop=True).copy()
    obstructed = sub_chunk.loc[mask].reset_index(drop=True)

    return visible, obstructed
     
_find_obstructions_2d_delayed = delayed(_find_obstructions_2d)
     
## Step 3 ########################### 
def obstructions_3d(potential_sight_lines, obstructions_gdf, potential_obstructions_column, meshes, simplified_target_buildings, num_workers):
    """
    Checks 3D visibility of each sight line using detailed ray tracing with PyVista.

    Each sight line is tested for intersection with potential obstructions (buildings) using mesh ray tracing.
    Parallelized for speed using ThreadPoolExecutor.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines with observer/target coordinates and a column listing potential obstruction IDs.
    obstructions_gdf : GeoDataFrame
        Buildings with 'building_3d' mesh geometry.
    potential_obstructions_column : str
        Name of the column in `potential_sight_lines` with a list of candidate obstruction building IDs.
    num_workers : int
        Number of parallel threads to use for the visibility checks.

    Returns
    -------
    GeoDataFrame
        Only sight lines which are unobstructed in 3D, as determined by PyVista ray tracing.
    """
    potential_sight_lines = potential_sight_lines.copy()
    # Remove the current buildingID from the list of potential obstructions for each sight line
    
    if (not simplified_target_buildings.empty):
        potential_sight_lines[potential_obstructions_column] = (
            potential_sight_lines.apply(
                lambda r: [bid for bid in r[potential_obstructions_column] if bid != r.buildingID],
                axis=1
            )
        )
    
    potential_sight_lines['observer_coords'] = [ (line.coords[0][0], line.coords[0][1], line.coords[0][2]) for line in potential_sight_lines['geometry']]
    potential_sight_lines['target_coords'] = [(line.coords[-1][0], line.coords[-1][1], line.coords[-1][2]) for line in potential_sight_lines['geometry']]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda row: _intervisibility(row, meshes, potential_obstructions_column),
                    potential_sight_lines.itertuples(index=False),
                ),
                total=len(potential_sight_lines),
                desc="Checking intervisibility"
            )
        )
        
    potential_sight_lines["visible"] = results
    # Filter visible sight lines and append to the result
    potential_sight_lines = potential_sight_lines[potential_sight_lines["visible"] == True]
    return potential_sight_lines

def _process_one(bid, solid):
    mesh = pv.wrap(solid).extract_surface().triangulate()
    return bid, mesh

def _build_meshes(obstructions_gdf, n_jobs=None):
    print("Building obstructions meshes")
    ids = obstructions_gdf["buildingID"].values
    solids = obstructions_gdf["building_3d"].values
    tasks = list(zip(ids, solids))

    meshes = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_process_one, bid, solid) for bid, solid in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Building meshes"):
            bid, mesh = f.result()
            meshes[bid] = mesh
    return meshes
       
def _intervisibility(row, meshes, potential_obstructions_column):
    """
    Checks if a 3D line of sight is unobstructed using prebuilt PyVista meshes.

    Parameters
    ----------
    row : namedtuple
        A single row from the potential sight lines DataFrame, containing observer and target coordinates,
        and the list of potential obstruction building IDs.
    meshes : dict
        Prebuilt dictionary {buildingID: pv.PolyData} with triangulated surfaces.
    potential_obstructions_column : str
        Name of the column in `row` containing candidate obstruction building IDs.

    Returns
    -------
    bool
        True if no intersections are found, False if any obstruction blocks the line.
    """
    if len(getattr(row, potential_obstructions_column)) == 0:
        return True
    
    p0 = row.observer_coords
    p1 = row.target_coords

    for buildingID in getattr(row, potential_obstructions_column):
        mesh = meshes.get(buildingID)
        points, intersections = mesh.ray_trace(p0, p1)
        if len(intersections) > 0:
            return False
    return True

def _last_check(potential_sight_lines, obstructions_gdf, obstructions_sindex, meshes, nodes_gdf, num_workers):
    
    visible_2d, obstructed = _find_obstructions_2d(potential_sight_lines, obstructions_gdf, obstructions_sindex)
    visbile_3d = obstructions_3d(obstructed, obstructions_gdf, 'matchesIDs', meshes, gpd.GeoDataFrame, num_workers=num_workers)
    sight_lines_tmp = pd.concat([visible_2d, visbile_3d], ignore_index=True)
    return _finalize_sight_lines(sight_lines_tmp, nodes_gdf, False, gpd.GeoDataFrame)

def _finalize_sight_lines(sight_lines_tmp, nodes_gdf, consolidate, simplified_buildings):
    """
    Finalizes, cleans, and merges sight lines after visibility analysis.

    This function handles the final step in sight line construction by:
      - Exploding node pairs if node consolidation was performed.
      - (If using simplified buildings) Exploding all building-target pairs from simplified outlines.
      - Rebuilding correct LineString geometries for each observer-target pair.
      - Dropping intermediate columns used in previous steps.
      - Sorting and deduplicating the result so each (buildingID, nodeID) pair appears only once, keeping the longest line.

    Parameters
    ----------
    sight_lines_tmp : DataFrame or GeoDataFrame
        Temporary results table with observer/target columns, node/building references, and possibly intermediate attributes.
    nodes_gdf : GeoDataFrame
        Reference nodes with 3D coordinates, including 'nodeID', 'geometry', and (optionally) 'z'.
    consolidate : bool
        Whether node consolidation (merging/clustered nodes) was performed.
    simplified_buildings : GeoDataFrame
        If non-empty, provides simplified outlines for additional sight line explosion and assignment.

    Returns
    -------
    sight_lines : GeoDataFrame
        Cleaned, deduplicated GeoDataFrame of final sight lines, with correct geometry, observer and target references, and length.
    """
    nodes_gdf['geometry'] = [Point(geometry.x, geometry.y, z) for geometry, z in zip(nodes_gdf['geometry'], nodes_gdf['z'])]     

    if consolidate:
        sight_lines_tmp = sight_lines_tmp.explode(column='old_nodeIDs', ignore_index=True)
        sight_lines_tmp['nodeID'] = sight_lines_tmp['old_nodeIDs']
        geom_map = nodes_gdf.set_index('nodeID')['geometry']
        sight_lines_tmp['observer_geo'] = sight_lines_tmp['nodeID'].map(geom_map)

    if (simplified_buildings is not None) and (not simplified_buildings.empty):
        sight_lines_tmp['pair'] = sight_lines_tmp.apply(
            lambda row: list(zip(row['buildingIDs'], row['target_GEOs'])), axis=1
        )
        sight_lines_tmp_exploded = sight_lines_tmp.explode('pair', ignore_index=True)
        sight_lines_tmp_exploded['buildingID'] = sight_lines_tmp_exploded['pair'].apply(lambda x: x[0])
        sight_lines_tmp_exploded['target_geo'] = sight_lines_tmp_exploded['pair'].apply(lambda x: x[1])
        sight_lines_tmp = sight_lines_tmp_exploded.copy()
    
    if "observer_geo" in sight_lines_tmp.columns:
        sight_lines_tmp['geometry'] = [LineString([observer, target]) for observer, target in zip(sight_lines_tmp['observer_geo'], 
                                                                                                  sight_lines_tmp['target_geo'])]
                                                                                                                                                                   
    sight_lines_tmp = sight_lines_tmp.drop(['old_nodeIDs', 'buildingIDs', 'target_GEOs', 'pair', 'observer_geo', 'target_geo', 'observer_coords',
                                           'target_coords', 'matchesIDs'], errors='ignore', axis=1)
    sight_lines = gpd.GeoDataFrame(sight_lines_tmp, geometry='geometry', crs=nodes_gdf.crs)
    sight_lines['length'] = sight_lines.geometry.length
    
    if not consolidate:
        sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'], ascending=[False, False, False]).drop_duplicates(
            ['buildingID', 'nodeID'], keep='first')
        sight_lines.reset_index(inplace=True, drop=True)
    
    sight_lines['nodeID'] = sight_lines['nodeID'].astype(int)
    sight_lines['buildingID'] = sight_lines['buildingID'].astype(int)
    return sight_lines
  
def polygon_2d_to_3d(building_polygon, base, height, extrude_from_sealevel = True, height_relative_to_ground = False):
    """
    Convert a 2D polygon (building) to a 3D geometry by extruding it with a specified height and base.
    
    Parameters
    ----------
    building_polygon : shapely.geometry.Polygon
        2D polygon representing the building footprint to be extruded.
    base : float
        The base height of the 3D polygon (the height from which extrusion starts).
    height : float
        The total height of the 3D polygon after extrusion.
    extrude_from_zero : bool, optional, default=True
        If True, the extrusion starts from the zero height (i.e., sea level). If False, the extrusion starts from the `base` height.
    height_from_base : bool, optional, default=True
        If True, the `height` is considered to be above the `base`. If False, it is considered above sea level (i.e., height is reduced by `base`).
        
    Returns
    -------
    pyvista.PolyData
        The resulting 3D polygon (building) after extrusion.
    """
    
    def reorient_coords(xy):

        value = 0
        for i in range(len(xy)):
            x1, y1 = xy[i]
            x2, y2 = xy[(i+1)%len(xy)]
            value += (x2-x1)*(y2+y1)
        if value > 0:
            return xy
        else:
            return xy[::-1]
    
    poly_points = building_polygon.exterior.coords
       
    # Reorient the coordinates of the polygon
    xy = reorient_coords(poly_points)
    # Create 3D coordinates with the base height or "fictious" ground
    # Determine the starting height and adjust the extrusion height based on the flags
    
    if extrude_from_sealevel:
        # If extruding from zero (e.g., ground level), set the base height to 0
        building_base = 0.0
        if not height_relative_to_ground:
            # If height is computed from base (e.g., above ground level), add base to height
            height = height + base
    else:
        # If extruding from the given base, set the base height to the given base value
        building_base = base
        if height_relative_to_ground:
            # If height is computed from sea level (not the base), subtract the base from height
            height = height - base
   
    xyz_base = [(x, y, building_base) for x, y in xy]
    # Create faces of the polygon
    faces = [len(xyz_base), *range(len(xyz_base))]
    # Create the 3D polygon using pyvista
    polygon = pv.PolyData(xyz_base, faces=faces)
    
    # Extrude the 3D polygon to the specified height 
    return polygon.extrude((0, 0, height), capping=True)