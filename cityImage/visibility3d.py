"""Optional 3D sight-line generation utilities for cityImage.

This module preserves the existing cityImage 3D sight-line workflow while
keeping it out of the lightweight core. Heavy optional dependencies such as
PyVista, Dask, psutil, and tqdm are imported lazily by the functions that need
them, so importing :mod:`cityImage` or resolving its public API does not require
the 3D stack to be installed.
"""

from __future__ import annotations

import gc
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

from .geometry import gdf_multipolygon_to_polygon
from .network_topology import consolidate_nodes

pd.set_option("display.precision", 3)

_VISIBILITY3D_OPTIONAL_DEPENDENCIES = {
    "dask": "dask",
    "psutil": "psutil",
    "pyvista": "pyvista",
    "tqdm": "tqdm",
    "vtk": "pyvista",  # VTK ships with pyvista; installing the extra provides both
}


def _import_optional_dependency(import_name: str):
    """Import an optional visibility3d dependency with a clear error message."""
    try:
        return import_module(import_name)
    except ImportError as exc:
        package_name = _VISIBILITY3D_OPTIONAL_DEPENDENCIES.get(import_name, import_name)
        raise ImportError(
            "cityImage visibility3d functionality requires optional dependency "
            f"'{package_name}'. Install the 'visibility3d' extra with: "
            'python -m pip install -e ".[visibility3d]"'
        ) from exc


def _require_visibility3d_dependencies(*import_names: str) -> None:
    """Raise a clear error if required optional 3D dependencies are missing."""
    if not import_names:
        import_names = tuple(_VISIBILITY3D_OPTIONAL_DEPENDENCIES)

    missing = []
    for import_name in import_names:
        try:
            import_module(import_name)
        except ImportError:
            missing.append(_VISIBILITY3D_OPTIONAL_DEPENDENCIES.get(import_name, import_name))

    if missing:
        raise ImportError(
            "cityImage visibility3d functionality requires optional dependencies: "
            f"{', '.join(missing)}. Install the 'visibility3d' extra with: "
            'python -m pip install -e ".[visibility3d]"'
        )


def _tqdm(*args, **kwargs):
    """Return tqdm.tqdm lazily."""
    return _import_optional_dependency("tqdm").tqdm(*args, **kwargs)


def compute_3d_sight_lines(
    nodes_gdf: gpd.GeoDataFrame,
    target_buildings_gdf: gpd.GeoDataFrame,
    obstructions_buildings_gdf: gpd.GeoDataFrame,
    simplified_target_buildings: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    city_name: str,
    distance_along: float = 200,
    min_observer_target_distance: float = 300,
    height_relative_to_ground=False,
    sight_lines_chunk_size: int = 500000,
    consolidate: bool = False,
    consolidate_tolerance: float = 0.0,
    num_workers: int = 20,
):
    """Compute visible 3D sight lines between observer nodes and target buildings.

    The workflow samples target points along building roofs, generates candidate
    observer-target lines, removes lines that are too short, filters 2D
    obstructions, and checks remaining candidate lines with PyVista ray tracing.
    Large candidate sets are processed in chunks and written temporarily to
    GeoPackage files before being merged.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        Observer nodes. Required columns are ``geometry``, ``x``, ``y``, ``nodeID``,
        and ``z``.
    target_buildings_gdf : geopandas.GeoDataFrame
        Buildings to use as visibility targets. Required columns are
        ``buildingID``, ``geometry``, ``height``, and optionally ``base``.
    obstructions_buildings_gdf : geopandas.GeoDataFrame
        Buildings to use as possible obstructions. Required columns are
        ``buildingID``, ``geometry``, ``height``, and optionally ``base``.
    simplified_target_buildings : geopandas.GeoDataFrame or None
        Optional simplified target outlines. When provided, detailed target
        buildings are mapped back to simplified geometries for final sight-line
        assignment.
    edges_gdf : geopandas.GeoDataFrame
        Network edges used only when ``consolidate=True``.
    city_name : str
        Prefix used for temporary chunk files.
    distance_along : float, default 200
        Sampling distance along target-building roof edges.
    min_observer_target_distance : float, default 300
        Minimum 2D distance allowed between observer and target.
    height_relative_to_ground : bool, default False
        Whether building heights are already relative to local ground/base height.
    sight_lines_chunk_size : int, default 500000
        Maximum number of candidate sight lines processed per chunk.
    consolidate : bool, default False
        Whether to consolidate observer nodes before sight-line generation.
    consolidate_tolerance : float, default 0.0
        Spatial tolerance used for node consolidation.
    num_workers : int, default 20
        Number of parallel workers used for obstruction and mesh operations.

    Returns
    -------
    geopandas.GeoDataFrame
        Visible 3D sight lines. If no visible lines are found, an empty
        GeoDataFrame with the obstruction CRS is returned.
    """

    _require_visibility3d_dependencies()

    # Step 0: Prepare data
    observers, targets, obstructions_gdf = _prepare_3d_sight_lines(
        nodes_gdf,
        target_buildings_gdf,
        obstructions_buildings_gdf,
        distance_along=distance_along,
        simplified_buildings=simplified_target_buildings,
        consolidate=consolidate,
        consolidate_tolerance=consolidate_tolerance,
        edges_gdf=edges_gdf,
        height_relative_to_ground=height_relative_to_ground,
    )

    num_observers = len(observers)
    num_targets = len(targets)
    projected_nr_sight_lines = num_observers * num_targets
    obstructions_sindex = obstructions_gdf.sindex  # GeoPandas SpatialIndex wrapper
    # Meshes and their VTK ray-casting locators are built lazily, only for buildings that
    # actually turn up as 2D-obstruction candidates (usually a small fraction of the city).
    meshes = _MeshCache(obstructions_gdf)

    if projected_nr_sight_lines <= sight_lines_chunk_size:
        observer_chunks = [observers]
    else:
        observers_per_chunk = max(1, sight_lines_chunk_size // num_targets)
        num_chunks = int(np.ceil(num_observers / observers_per_chunk))
        observer_chunks = np.array_split(observers, num_chunks)

    out_prefix = "chunk_sight_lines"
    out_files = []

    for n, chunk in enumerate(observer_chunks):
        print(f"Starting chunk {n + 1} of {len(observer_chunks)}")
        visibles = []

        print(" 01 - Filtering by distance")
        potential_sight_lines = filter_distance(chunk, targets, min_observer_target_distance)
        if potential_sight_lines.empty:
            continue

        print(" 02 - Checking 2d obstructions")
        visible_2d, obstructed_2d = obstructions_2d(
            potential_sight_lines, obstructions_gdf, obstructions_sindex, num_workers=num_workers
        )
        if not visible_2d.empty:
            visibles.append(visible_2d)

        print(" 03 - Checking 3d obstructions")
        visible_3d = pd.DataFrame()
        if not obstructed_2d.empty:
            obstructed_2d["geometry"] = [
                LineString([observer, target])
                for observer, target in zip(
                    obstructed_2d["observer_geo"], obstructed_2d["target_geo"], strict=False
                )
            ]
            visible_3d = obstructions_3d(
                obstructed_2d,
                obstructions_gdf,
                "matchesIDs",
                meshes,
                simplified_target_buildings,
                num_workers=num_workers,
            )
            if not visible_3d.empty:
                visibles.append(visible_3d)

        chunk_dir = Path("sight_lines_tmp")
        chunk_dir.mkdir(parents=True, exist_ok=True)  # create folder if not existing

        # Finalize columns and export
        if visibles:
            cols_to_drop = ["visible", "z"]

            for df in visibles:
                df.drop(
                    columns=[col for col in cols_to_drop if col in df.columns],
                    inplace=True,
                    errors="ignore",
                )

            chunk_sight_lines = pd.concat(visibles, ignore_index=True)
            chunk_sight_lines = _finalize_sight_lines(
                chunk_sight_lines, nodes_gdf, consolidate, simplified_target_buildings
            )

            # Build filename
            chunk_file = chunk_dir / f"{city_name}_{out_prefix}_{n}.gpkg"
            chunk_sight_lines.to_file(chunk_file)
            out_files.append(chunk_file)

            del chunk_sight_lines, visibles
            gc.collect()
            print(f"Chunk {n + 1} processed and exported: {chunk_file}")

        visibles = []
        del chunk, potential_sight_lines, visible_2d, obstructed_2d, visible_3d
        gc.collect()

    print("All chunks processed.")
    tmp_sight_lines = gpd.GeoDataFrame(geometry=[], crs=obstructions_gdf.crs)
    if out_files:
        tmp_sight_lines = merge_gpkg_chunks_to_gdf(out_files, "matchesIDs")
        tmp_sight_lines.drop(["visible"], axis=1, errors="ignore", inplace=True)
    else:
        print("No visible sight-lines")
        return tmp_sight_lines

    if consolidate:
        print(" 04 - Final check")
        sight_lines = _last_check(
            tmp_sight_lines,
            obstructions_gdf,
            obstructions_sindex,
            meshes,
            nodes_gdf,
            num_workers=num_workers,
        )
    else:
        sight_lines = tmp_sight_lines

    return sight_lines


## Preparation ###########################
def _prepare_3d_sight_lines(
    nodes_gdf,
    target_buildings_gdf,
    obstructions_gdf,
    distance_along=200,
    simplified_buildings=None,
    consolidate=False,
    consolidate_tolerance=0.0,
    edges_gdf=None,
    height_relative_to_ground=False,
):
    """Prepare observers, target points, and obstruction geometries.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        Observer nodes with ``geometry``, ``x``, ``y``, ``nodeID``, and ``z``.
    target_buildings_gdf : geopandas.GeoDataFrame
        Target buildings with ``buildingID``, ``geometry``, ``height``, and
        optionally ``base``.
    obstructions_gdf : geopandas.GeoDataFrame
        Buildings considered as potential obstructions.
    distance_along : float, default 200
        Sampling distance along target-building roof edges.
    simplified_buildings : geopandas.GeoDataFrame or None, default None
        Optional simplified building outlines.
    consolidate : bool, default False
        Whether to consolidate observer nodes.
    consolidate_tolerance : float, default 0.0
        Tolerance used for observer consolidation.
    edges_gdf : geopandas.GeoDataFrame or None, default None
        Network edges required when observer consolidation is enabled.
    height_relative_to_ground : bool, default False
        Whether building heights are already relative to local ground/base height.

    Returns
    -------
    tuple
        ``(observer_points_gdf, target_points, obstructions_gdf)`` prepared for
        3D sight-line generation.
    """

    if simplified_buildings is None:
        simplified_buildings = gpd.GeoDataFrame(geometry=[], crs=target_buildings_gdf.crs)

    nodes_gdf = nodes_gdf[["geometry", "x", "y", "nodeID", "z"]].copy()
    nodes_gdf.loc[nodes_gdf["z"] < -50, "z"] = 2
    nodes_gdf["geometry"] = nodes_gdf.apply(lambda row: Point(row["x"], row["y"], row["z"]), axis=1)

    target_buildings_gdf = _prepare_buildings_gdf(target_buildings_gdf)
    obstructions_gdf = _prepare_buildings_gdf(obstructions_gdf)
    target_buildings_gdf = target_buildings_gdf[target_buildings_gdf.height > 5.0]
    target_points = _prepare_targets(
        target_buildings_gdf, distance_along, height_relative_to_ground
    )

    if simplified_buildings is not None and not simplified_buildings.empty:
        target_points, obstructions_gdf = _use_simplified_buildings(
            target_points, obstructions_gdf, simplified_buildings
        )
    else:
        target_points.drop("geometry", axis=1, inplace=True)

    if consolidate:
        observer_points_gdf = consolidate_nodes(
            nodes_gdf, edges_gdf, consolidate_edges_too=False, tolerance=consolidate_tolerance
        )
    else:
        observer_points_gdf = nodes_gdf.copy()

    target_points.set_geometry("target_geo", inplace=True, crs=nodes_gdf.crs)
    observer_points_gdf["observer_geo"] = observer_points_gdf["geometry"]
    observer_points_gdf = observer_points_gdf.drop(["x", "y", "z"], axis=1)

    obstructions_gdf["building_3d"] = [
        polygon_2d_to_3d(
            geometry, base, height, extrude_from_sealevel=True, height_relative_to_ground=False
        )
        for geometry, base, height in zip(
            obstructions_gdf["geometry"],
            obstructions_gdf["base"],
            obstructions_gdf["height"],
            strict=False,
        )
    ]
    obstructions_gdf["geometry"] = obstructions_gdf["geometry"].apply(lambda geom: geom.exterior)

    return observer_points_gdf, target_points, obstructions_gdf


def _prepare_buildings_gdf(buildings_gdf):
    """Clean and standardize a building GeoDataFrame for 3D processing.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        Building footprints with ``buildingID``, ``geometry``, ``height``, and
        optionally ``base``.

    Returns
    -------
    geopandas.GeoDataFrame
        Building table indexed by ``buildingID`` with non-null heights and a
        minimum base elevation of 1.0.
    """
    # add a 'base' column to the buildings GeoDataFrame with a default value of 1.0, if not provided
    if "base" not in buildings_gdf.columns:
        buildings_gdf["base"] = 1.0

    buildings_gdf["base"] = buildings_gdf["base"].where(
        buildings_gdf["base"] > 1.0, 1.0
    )  # minimum base
    buildings_gdf = buildings_gdf[buildings_gdf["height"].notna()]
    buildings_gdf = buildings_gdf[["buildingID", "geometry", "height", "base"]].copy()
    buildings_gdf.index = buildings_gdf.buildingID
    buildings_gdf.index.name = None
    return buildings_gdf


def _prepare_targets(target_buildings_gdf, distance_along, height_relative_to_ground=False):
    """Generate 3D target points along target-building roof perimeters.

    Parameters
    ----------
    target_buildings_gdf : geopandas.GeoDataFrame
        Target buildings with ``geometry``, ``height``, and ``base`` columns.
    distance_along : float
        Sampling distance along each roof perimeter.
    height_relative_to_ground : bool, default False
        Whether height values are relative to local ground/base height.

    Returns
    -------
    geopandas.GeoDataFrame
        Exploded target-point table with a ``target_geo`` geometry column.
    """

    target_buildings = target_buildings_gdf.copy()
    target_buildings["geometry"] = target_buildings_gdf["geometry"].apply(
        lambda geom: (
            geom
            if isinstance(geom, Polygon)
            else geom.convex_hull
            if geom.is_valid
            else geom.buffer(0)
        )
    )

    def extrude_roof(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString(
                [
                    point
                    for poly in geometry.geoms
                    for point in add_height_to_line(poly.exterior, base, height).coords
                ]
            )

    def add_height_to_line(
        exterior, base, height, height_relative_to_ground=height_relative_to_ground
    ):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        return LineString([(coord[0], coord[1], height + base) for coord in exterior.coords])

    # create the roof
    building_tops = target_buildings.apply(
        lambda row: extrude_roof(row.geometry, row.base, row.height), axis=1
    )
    target_buildings["target_geo"] = [
        downsample_coords(line, distance_along) for line in building_tops
    ]

    # 3. explode into individual vertices → Points
    target_points = target_buildings.explode("target_geo", ignore_index=True)
    target_points["target_geo"] = target_points["target_geo"].apply(Point)
    return target_points


def downsample_coords(line, distance_along):
    """Sample representative coordinates along a LineString.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Source line geometry.
    distance_along : float
        Minimum accumulated distance between selected coordinates.

    Returns
    -------
    list[tuple]
        Coordinates sampled along the line. If no coordinate reaches the
        requested spacing, the coordinate closest to the line midpoint is
        returned.
    """
    coords = np.array(line.coords)
    if coords.shape[0] == 0:
        return []
    selected, dist = [], 0.0
    for i in range(1, len(coords)):
        seg_len = np.linalg.norm(coords[i] - coords[i - 1])
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
    # Create a spatial index for the simplified buildings. STRtree.query returns
    # *positions* into the geometry array passed to the constructor, NOT index labels:
    # translate positions back to the frame's index explicitly (the previous
    # label-based .loc lookup only worked while the index happened to be a RangeIndex).
    simplified_tree = STRtree(simplified_buildings.geometry.values)
    simplified_labels = simplified_buildings.index.to_numpy()

    def assign_geometries_to_simplified_buildings(geometries):
        """First intersecting simplified-building label per input geometry (or None)."""
        geometries = np.asarray(geometries, dtype=object)
        # 'intersects' subsumes the previous within/touches/intersects triple.
        input_positions, tree_positions = simplified_tree.query(
            geometries, predicate="intersects"
        )
        first_match = {}
        for geometry_position, tree_position in zip(input_positions, tree_positions, strict=False):
            if geometry_position not in first_match:
                first_match[geometry_position] = simplified_labels[tree_position]
        return [first_match.get(position) for position in range(len(geometries))]

    target_points["simplifiedID"] = assign_geometries_to_simplified_buildings(
        target_points["target_geo"]
    )
    obstructions_gdf["simplifiedID"] = assign_geometries_to_simplified_buildings(
        obstructions_gdf["geometry"]
    )

    # Function to get building IDs, average height, and average base of targets assigned to the same simplified building
    def get_attributes(simplifiedID, gdf):
        # Filter targets assigned to the simplified building
        detailed_buildings = gdf[gdf["simplifiedID"] == simplifiedID].copy()
        # Extract building IDs of these targets
        buildingIDs = (
            detailed_buildings["buildingID"].tolist() if not detailed_buildings.empty else []
        )

        # Check if 'target_geo' exists and extract target geometries
        if "target_geo" in detailed_buildings.columns:
            target_GEOs = (
                detailed_buildings["target_geo"].tolist() if not detailed_buildings.empty else []
            )
        else:
            target_GEOs = None

        # Calculate the average height and base of these targets
        height = detailed_buildings["height"].mean() if not detailed_buildings.empty else np.nan
        base = detailed_buildings["base"].mean() if not detailed_buildings.empty else np.nan

        return pd.Series([buildingIDs, target_GEOs, base, height])

    simplified_target_points = simplified_buildings.copy()
    simplified_obstructions = simplified_buildings.copy()
    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_target_points[["buildingIDs", "target_GEOs", "base", "height"]] = (
        simplified_buildings.index.to_series()
        .apply(get_attributes, gdf=target_points)
        .apply(pd.Series)
    )

    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_obstructions[["buildingIDs", "target_GEOs", "base", "height"]] = (
        simplified_buildings.index.to_series()
        .apply(get_attributes, gdf=obstructions_gdf)
        .apply(pd.Series)
    )

    simplified_target_points = simplified_target_points[
        simplified_target_points["buildingIDs"].apply(len) > 0
    ]
    simplified_target_points = simplified_target_points[simplified_target_points["height"].notna()]
    simplified_target_points["buildingID"] = simplified_target_points.index

    simplified_obstructions.drop(["buildingIDs", "target_GEOs"], axis=1, inplace=True)
    simplified_obstructions = simplified_obstructions[simplified_obstructions["height"].notna()]
    simplified_obstructions["buildingID"] = simplified_obstructions.index

    simplified_target_points["target_geo"] = simplified_target_points.apply(
        lambda row: Point(
            row["geometry"].centroid.x, row["geometry"].centroid.y, row["height"] + row["base"]
        ),
        axis=1,
    )
    simplified_target_points.drop(
        ["geometry", "base", "height", "FID"], axis=1, inplace=True, errors="ignore"
    )
    return simplified_target_points, simplified_obstructions


def merge_gpkg_chunks_to_gdf(out_files, potential_obstructions_column):
    """Merge temporary GeoPackage sight-line chunks.

    Parameters
    ----------
    out_files : sequence of path-like
        GeoPackage chunk files written by ``compute_3d_sight_lines``.
    potential_obstructions_column : str
        Kept for API compatibility with the chunking workflow.

    Returns
    -------
    geopandas.GeoDataFrame
        Combined sight-line chunks.
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

    import shapely

    # Prepare data
    potential_lines = _prepare_chunk_data(chunk, targets)

    # Vectorised 2D coordinate extraction (drops z) and distance filter.
    observer_coords = shapely.get_coordinates(
        np.asarray(potential_lines["observer_geo"], dtype=object)
    )
    target_coords = shapely.get_coordinates(
        np.asarray(potential_lines["target_geo"], dtype=object)
    )
    distances = np.linalg.norm(observer_coords - target_coords, axis=1)
    mask = distances >= min_observer_target_distance
    potential_lines = potential_lines.loc[mask]

    # Vectorised 2D line construction for the planimetric obstruction check.
    segments = np.stack([observer_coords[mask], target_coords[mask]], axis=1)
    potential_lines["geometry"] = shapely.linestrings(segments)

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
    chunk_data = chunk.drop("geometry", axis=1).assign(key=1)
    targets_data = targets.assign(key=1)

    # Compute cartesian product efficiently
    potential_lines = pd.merge(chunk_data, targets_data, on="key").drop("key", axis=1)
    print(
        "Num potential lines is",
        len(potential_lines),
        "; Nodes in chunk:",
        len(chunk_data),
        "Num Targets",
        len(targets_data),
    )
    return potential_lines


## Step 1 ###########################
def obstructions_2d(potential_sight_lines, obstructions_gdf, obstructions_sindex, num_workers=20):
    """Split candidate sight lines by 2D planimetric obstruction status.

    Parameters
    ----------
    potential_sight_lines : geopandas.GeoDataFrame
        Candidate sight lines with 2D line geometry.
    obstructions_gdf : geopandas.GeoDataFrame
        Obstruction geometries with a ``buildingID`` column.
    obstructions_sindex : geopandas.sindex.SpatialIndex
        Spatial index built from ``obstructions_gdf``.
    num_workers : int, default 20
        Number of worker threads used by Dask.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Visible lines and lines that require 3D obstruction testing.
    """

    _require_visibility3d_dependencies("dask", "psutil")
    dask = _import_optional_dependency("dask")

    potential_sight_lines = potential_sight_lines.copy()
    sub_chunks = _define_batches(potential_sight_lines)
    tasks = [
        dask.delayed(_find_obstructions_2d)(sub_chunk, obstructions_gdf, obstructions_sindex)
        for sub_chunk in sub_chunks
    ]
    results = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)

    visibles = [r[0] for r in results]
    potentially_obstructeds = [r[1] for r in results]
    visible = pd.concat(visibles, ignore_index=True)
    potentially_obstructed = pd.concat(potentially_obstructeds, ignore_index=True)
    return visible, potentially_obstructed


def _define_batches(
    potential_sight_lines, min_size=50, max_size=25_000, mem_fraction=0.1, row_weight_kb=10_000
):
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
        Maximum number of rows per batch (default: 25,000). The previous 2,000 cap was
        always binding (the memory estimate is very conservative), which fragmented the
        2D check into thousands of tiny Dask tasks whose scheduling overhead dominated.
    mem_fraction : float, optional
        Fraction of available system memory to use for each batch (default: 0.1).
    row_weight_kb : int, optional
        Estimated memory usage per row, in kilobytes (default: 10,000).

    Yields
    ------
    GeoDataFrame
        The next batch of rows.
    """

    psutil = _import_optional_dependency("psutil")
    available_mem = psutil.virtual_memory().available
    rows_per_batch = int(available_mem * mem_fraction / row_weight_kb)
    rows_per_batch = max(min_size, min(rows_per_batch, max_size))
    n = len(potential_sight_lines)
    for start in range(0, n, rows_per_batch):
        yield potential_sight_lines.iloc[start : start + rows_per_batch]


def _find_obstructions_2d(sub_chunk, obstructions_gdf, obstructions_sindex):
    """Find 2D obstruction candidates for a batch of sight lines.

    Parameters
    ----------
    sub_chunk : geopandas.GeoDataFrame
        Batch of candidate sight lines.
    obstructions_gdf : geopandas.GeoDataFrame
        Obstruction geometries with a ``buildingID`` column.
    obstructions_sindex : geopandas.sindex.SpatialIndex
        Spatial index built from ``obstructions_gdf``.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Lines with no 2D crossings and lines with candidate obstruction IDs in
        ``matchesIDs``.
    """

    # Vectorised spatial query – true intersections if backend supports predicate
    pairs = obstructions_sindex.query(sub_chunk.geometry, predicate="crosses")
    poly_ids = obstructions_gdf["buildingID"].to_numpy()

    # Group buildingIDs by line index
    obstruction_dict = {}
    for line_idx, poly_idx in zip(pairs[0], pairs[1], strict=False):
        bid = poly_ids[poly_idx]
        obstruction_dict.setdefault(line_idx, []).append(bid)

    # Assign intersecting buildingIDs per row
    obstructionIDs = [
        [int(bid) for bid in obstruction_dict.get(i, [])] for i in range(len(sub_chunk))
    ]
    sub_chunk["matchesIDs"] = obstructionIDs

    mask = sub_chunk["matchesIDs"].map(len) > 0
    visible = sub_chunk.loc[~mask].reset_index(drop=True).copy()
    obstructed = sub_chunk.loc[mask].reset_index(drop=True)

    return visible, obstructed


## Step 3 ###########################
def obstructions_3d(
    potential_sight_lines,
    obstructions_gdf,
    potential_obstructions_column,
    meshes,
    simplified_target_buildings,
    num_workers,
):
    """Check 3D visibility for sight lines with candidate obstructions.

    Parameters
    ----------
    potential_sight_lines : geopandas.GeoDataFrame
        Candidate sight lines with 3D geometry and a candidate-obstruction column.
    obstructions_gdf : geopandas.GeoDataFrame
        Obstruction building table. Kept for workflow consistency.
    potential_obstructions_column : str
        Column containing lists of candidate obstruction building IDs.
    meshes : dict
        Mapping from building IDs to triangulated PyVista meshes.
    simplified_target_buildings : geopandas.GeoDataFrame or None
        Optional simplified target outlines. When provided, the current target
        building is removed from each candidate obstruction list.
    num_workers : int
        Number of worker threads used for ray tracing.

    Returns
    -------
    geopandas.GeoDataFrame
        Sight lines that remain visible after 3D ray-tracing checks.
    """
    _require_visibility3d_dependencies("tqdm")
    tqdm = _import_optional_dependency("tqdm").tqdm

    potential_sight_lines = potential_sight_lines.copy()
    # Remove the current buildingID from the list of potential obstructions for each sight line

    if simplified_target_buildings is not None and not simplified_target_buildings.empty:
        potential_sight_lines[potential_obstructions_column] = potential_sight_lines.apply(
            lambda r: [bid for bid in r[potential_obstructions_column] if bid != r.buildingID],
            axis=1,
        )

    potential_sight_lines["observer_coords"] = [
        (line.coords[0][0], line.coords[0][1], line.coords[0][2])
        for line in potential_sight_lines["geometry"]
    ]
    potential_sight_lines["target_coords"] = [
        (line.coords[-1][0], line.coords[-1][1], line.coords[-1][2])
        for line in potential_sight_lines["geometry"]
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda row: _intervisibility(row, meshes, potential_obstructions_column),
                    potential_sight_lines.itertuples(index=False),
                ),
                total=len(potential_sight_lines),
                desc="Checking intervisibility",
            )
        )

    potential_sight_lines["visible"] = results
    # Filter visible sight lines and append to the result
    potential_sight_lines = potential_sight_lines[potential_sight_lines["visible"]]
    return potential_sight_lines


class _MeshCache:
    """Lazy per-building triangulated meshes and VTK OBB ray-casting locators.

    Two performance properties compared to building every mesh upfront:

    - meshes are built **on first use only**, so obstructions that never appear as a
      2D-crossing candidate (usually the vast majority of a city) are never triangulated;
    - each building gets a **cached, pre-built ``vtkOBBTree``** reused across all its
      ray checks. PyVista's ``mesh.ray_trace`` constructs a fresh OBB tree on *every
      call*, which dominated the 3D step when thousands of candidate lines crossed the
      same tall building.

    Locators are built under a lock; afterwards ``IntersectWithLine`` traversals are
    read-only, so concurrent queries from the worker threads are safe.
    """

    def __init__(self, obstructions_gdf):
        self._solids = dict(
            zip(obstructions_gdf["buildingID"], obstructions_gdf["building_3d"], strict=False)
        )
        self._locators = {}
        self._lock = threading.Lock()

    def get_locator(self, buildingID):
        """Return the (cached) OBB locator for a building, or None when unknown."""
        locator = self._locators.get(buildingID)
        if locator is not None:
            return locator
        solid = self._solids.get(buildingID)
        if solid is None:
            return None

        with self._lock:
            locator = self._locators.get(buildingID)  # racing thread may have built it
            if locator is not None:
                return locator
            pv = _import_optional_dependency("pyvista")
            vtk = _import_optional_dependency("vtk")
            mesh = pv.wrap(solid).extract_surface().triangulate()
            locator = vtk.vtkOBBTree()
            locator.SetDataSet(mesh)
            locator.BuildLocator()
            self._locators[buildingID] = locator
            return locator


def _process_one(bid, solid):
    """Build a triangulated PyVista mesh for one obstruction solid.

    Parameters
    ----------
    bid : hashable
        Building identifier.
    solid : object
        PyVista-compatible solid geometry.

    Returns
    -------
    tuple
        ``(bid, mesh)`` where ``mesh`` is a triangulated surface mesh.
    """
    pv = _import_optional_dependency("pyvista")
    mesh = pv.wrap(solid).extract_surface().triangulate()
    return bid, mesh


def _build_meshes(obstructions_gdf, n_jobs=None):
    """Build PyVista meshes for obstruction buildings.

    Parameters
    ----------
    obstructions_gdf : geopandas.GeoDataFrame
        Obstruction table containing ``buildingID`` and ``building_3d`` columns.
    n_jobs : int or None, default None
        Number of worker processes. ``None`` lets ``ProcessPoolExecutor`` choose.

    Returns
    -------
    dict
        Mapping from ``buildingID`` to triangulated PyVista mesh.
    """
    _require_visibility3d_dependencies("pyvista", "tqdm")
    tqdm = _import_optional_dependency("tqdm").tqdm

    print("Building obstructions meshes")
    ids = obstructions_gdf["buildingID"].values
    solids = obstructions_gdf["building_3d"].values
    tasks = list(zip(ids, solids, strict=False))

    meshes = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_process_one, bid, solid) for bid, solid in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Building meshes"):
            bid, mesh = f.result()
            meshes[bid] = mesh
    return meshes


def _intervisibility(row, meshes, potential_obstructions_column):
    """Return whether one 3D sight line is unobstructed.

    Parameters
    ----------
    row : namedtuple
        Row from the candidate sight-line table. It must contain observer and target
        coordinate attributes and the candidate-obstruction list.
    meshes : dict
        Mapping from building IDs to triangulated PyVista meshes.
    potential_obstructions_column : str
        Name of the attribute containing candidate obstruction IDs.

    Returns
    -------
    bool
        ``True`` when no candidate mesh intersects the ray, otherwise ``False``.
    """
    candidates = getattr(row, potential_obstructions_column)
    if len(candidates) == 0:
        return True

    p0 = row.observer_coords
    p1 = row.target_coords

    get_locator = getattr(meshes, "get_locator", None)
    if get_locator is not None:
        # _MeshCache path: pre-built OBB locators, one ray cast per candidate.
        for buildingID in candidates:
            locator = get_locator(buildingID)
            if locator is None:
                continue
            if locator.IntersectWithLine(p0, p1, None, None) != 0:
                return False
        return True

    # Plain mapping of pyvista meshes (kept for API compatibility with callers that
    # pass their own dict): ray_trace rebuilds its locator per call, so this is slow.
    for buildingID in candidates:
        mesh = meshes.get(buildingID)
        if mesh is None:
            continue

        _, intersections = mesh.ray_trace(p0, p1)
        if len(intersections) > 0:
            return False
    return True


def _last_check(
    potential_sight_lines, obstructions_gdf, obstructions_sindex, meshes, nodes_gdf, num_workers
):
    """Run a final 2D/3D obstruction check on consolidated sight lines.

    Parameters
    ----------
    potential_sight_lines : geopandas.GeoDataFrame
        Candidate sight lines to re-check.
    obstructions_gdf : geopandas.GeoDataFrame
        Building obstructions with 2D and 3D obstruction attributes.
    obstructions_sindex : geopandas.sindex.SpatialIndex
        Spatial index built from ``obstructions_gdf``.
    meshes : dict
        Mapping from building IDs to triangulated PyVista meshes.
    nodes_gdf : geopandas.GeoDataFrame
        Original observer nodes used to finalize sight-line geometries.
    num_workers : int
        Number of worker threads for 3D intervisibility checks.

    Returns
    -------
    geopandas.GeoDataFrame
        Finalized sight lines that pass the last obstruction check.
    """
    empty_simplified = gpd.GeoDataFrame(geometry=[], crs=nodes_gdf.crs)

    visible_2d, obstructed = _find_obstructions_2d(
        potential_sight_lines, obstructions_gdf, obstructions_sindex
    )
    visible_3d = obstructions_3d(
        obstructed,
        obstructions_gdf,
        "matchesIDs",
        meshes,
        empty_simplified,
        num_workers=num_workers,
    )
    sight_lines_tmp = pd.concat([visible_2d, visible_3d], ignore_index=True)
    return _finalize_sight_lines(sight_lines_tmp, nodes_gdf, False, empty_simplified)


def _finalize_sight_lines(sight_lines_tmp, nodes_gdf, consolidate, simplified_buildings):
    """Finalize, clean, and deduplicate visible sight lines.

    Parameters
    ----------
    sight_lines_tmp : pandas.DataFrame or geopandas.GeoDataFrame
        Temporary sight-line results with observer/target references.
    nodes_gdf : geopandas.GeoDataFrame
        Original observer nodes with ``nodeID``, ``geometry``, and ``z``.
    consolidate : bool
        Whether observer node consolidation was used.
    simplified_buildings : geopandas.GeoDataFrame or None
        Optional simplified building table used to explode simplified target
        assignments back to detailed building IDs.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned visible sight lines with ``nodeID``, ``buildingID``, ``geometry``,
        and ``length``.
    """
    # Work on a copy: this frame is the caller's original nodes GeoDataFrame, and
    # replacing its geometry with 3D points in place would leak out of this function.
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["geometry"] = [
        Point(geometry.x, geometry.y, z)
        for geometry, z in zip(nodes_gdf["geometry"], nodes_gdf["z"], strict=False)
    ]
    oldIDs_column = "old_nodeID"

    if consolidate:
        sight_lines_tmp = sight_lines_tmp.explode(column=oldIDs_column, ignore_index=True)
        sight_lines_tmp["nodeID"] = sight_lines_tmp[oldIDs_column]
        geom_map = nodes_gdf.set_index("nodeID")["geometry"]
        sight_lines_tmp["observer_geo"] = sight_lines_tmp["nodeID"].map(geom_map)

    if (simplified_buildings is not None) and (not simplified_buildings.empty):
        sight_lines_tmp["pair"] = sight_lines_tmp.apply(
            lambda row: list(zip(row["buildingIDs"], row["target_GEOs"], strict=False)), axis=1
        )
        sight_lines_tmp_exploded = sight_lines_tmp.explode("pair", ignore_index=True)
        sight_lines_tmp_exploded["buildingID"] = sight_lines_tmp_exploded["pair"].apply(
            lambda x: x[0]
        )
        sight_lines_tmp_exploded["target_geo"] = sight_lines_tmp_exploded["pair"].apply(
            lambda x: x[1]
        )
        sight_lines_tmp = sight_lines_tmp_exploded.copy()

    if "observer_geo" in sight_lines_tmp.columns:
        sight_lines_tmp["geometry"] = [
            LineString([observer, target])
            for observer, target in zip(
                sight_lines_tmp["observer_geo"], sight_lines_tmp["target_geo"], strict=False
            )
        ]

    sight_lines_tmp = sight_lines_tmp.drop(
        [
            oldIDs_column,
            "buildingIDs",
            "target_GEOs",
            "pair",
            "observer_geo",
            "target_geo",
            "observer_coords",
            "target_coords",
            "matchesIDs",
        ],
        errors="ignore",
        axis=1,
    )
    sight_lines = gpd.GeoDataFrame(sight_lines_tmp, geometry="geometry", crs=nodes_gdf.crs)
    sight_lines["length"] = sight_lines.geometry.length

    if not consolidate:
        sight_lines = sight_lines.sort_values(
            ["buildingID", "nodeID", "length"], ascending=[False, False, False]
        ).drop_duplicates(["buildingID", "nodeID"], keep="first")
        sight_lines.reset_index(inplace=True, drop=True)

    sight_lines["nodeID"] = sight_lines["nodeID"].astype(int)
    sight_lines["buildingID"] = sight_lines["buildingID"].astype(int)
    return sight_lines


def polygon_2d_to_3d(
    building_polygon, base, height, extrude_from_sealevel=True, height_relative_to_ground=False
):
    """Extrude a 2D building polygon into a 3D PyVista solid.

    Parameters
    ----------
    building_polygon : shapely.geometry.Polygon
        Building footprint to extrude.
    base : float
        Base elevation of the building.
    height : float
        Building height or absolute top elevation, depending on
        ``height_relative_to_ground`` and ``extrude_from_sealevel``.
    extrude_from_sealevel : bool, default True
        If ``True``, extrusion starts from zero. If ``False``, extrusion starts
        from ``base``.
    height_relative_to_ground : bool, default False
        If ``True``, ``height`` is interpreted as height above ``base``. If
        ``False``, ``height`` is interpreted as absolute height above sea level
        when appropriate.

    Returns
    -------
    pyvista.PolyData
        Extruded 3D building solid.
    """

    pv = _import_optional_dependency("pyvista")

    def reorient_coords(xy):

        value = 0
        for i in range(len(xy)):
            x1, y1 = xy[i]
            x2, y2 = xy[(i + 1) % len(xy)]
            value += (x2 - x1) * (y2 + y1)
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
