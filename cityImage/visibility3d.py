"""Optional 3D sight-line generation utilities for cityImage.

This module keeps the cityImage 3D sight-line workflow out of the lightweight
core. The visibility test is closed-form (see :func:`_analytic_obstructions_3d`),
so the only heavy dependencies are Dask and psutil, used to batch the 2D
planimetric pre-filter; both are imported lazily by the functions that need them,
so importing :mod:`cityImage` or resolving its public API does not require the
``visibility3d`` extra to be installed.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .network_topology import consolidate_nodes

pd.set_option("display.precision", 3)

_VISIBILITY3D_OPTIONAL_DEPENDENCIES = {
    "dask": "dask",
    "dask.diagnostics": "dask",
    "psutil": "psutil",
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


class _ProgressLogger:
    """Console progress and timing for :func:`compute_3d_sight_lines`, in one place.

    Holds the run's start time and chunk count so callers do not thread them through.
    When ``enabled`` is ``False`` every reporting method is a no-op (``step`` still times
    the block but prints nothing), so the compute path is silent unless the caller opts in
    via ``compute_3d_sight_lines(..., verbose=True)``. ``format_wall_time`` stays static and
    usable without an instance; ``chunk`` and ``total`` report against the run start.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total_start = time.perf_counter()
        self.n_chunks = 0

    @staticmethod
    def format_wall_time(seconds: float) -> str:
        """Format a duration in seconds as a compact h/m/s string."""
        if seconds >= 3600:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
        if seconds >= 60:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
        return f"{seconds:.1f}s"

    @contextmanager
    def step(self, label):
        """Time a one-off step, then (when enabled) print its wall time on a single line."""
        start = time.perf_counter()
        yield
        if not self.enabled:
            return
        elapsed = self.format_wall_time(time.perf_counter() - start)
        print(f"{label} [{elapsed}]", flush=True)

    def chunk(self, done, n_lines, n_records, t_2d, t_3d):
        """One compact status line per observer chunk: progress bar, per-step times, ETA."""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.total_start
        width = 24
        filled = int(width * done / self.n_chunks) if self.n_chunks else 0
        bar = "#" * filled + "-" * (width - filled)
        eta = ""
        if 0 < done < self.n_chunks:
            eta = f" ETA {self.format_wall_time(elapsed / done * (self.n_chunks - done))}"
        print(
            f"chunk {done:>3}/{self.n_chunks} [{bar}] {n_lines:>8,} lines  "
            f"2d {t_2d:4.0f}s 3d {t_3d:4.0f}s -> {n_records:>6,} sight lines  "
            f"| elapsed {self.format_wall_time(elapsed)}{eta}",
            flush=True,
        )

    def total(self):
        """Print the total wall time since the run started."""
        if not self.enabled:
            return
        elapsed = self.format_wall_time(time.perf_counter() - self.total_start)
        print(f"compute_3d_sight_lines total wall time: {elapsed}", flush=True)


def compute_3d_sight_lines(
    nodes_gdf: gpd.GeoDataFrame,
    target_buildings_gdf: gpd.GeoDataFrame,
    obstructions_buildings_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    city_name: str,
    distance_along: float = 200,
    min_observer_target_distance: float = 300,
    max_observer_target_distance: float | None = None,
    sight_lines_chunk_size: int = 500000,
    consolidate: bool = False,
    consolidate_tolerance: float = 0.0,
    num_workers: int = 20,
    verbose: bool = False,
):
    """Compute visible 3D sight lines between observer nodes and target buildings.

    The workflow samples target points along building roofs, generates candidate
    observer-target lines, removes lines that are too short, filters 2D
    obstructions, and resolves the remaining candidates with a closed-form
    (analytic) 3D visibility test against the vertical-extrusion obstructions.
    Large candidate sets are processed in chunks and written temporarily to
    GeoPackage files before being merged.

    Building ``height`` is treated as above-ground and added to ``base`` (terrain
    elevation, defaulting to a 1.0 minimum) to obtain the absolute roof elevation
    used for both targets and occluders.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        Observer nodes. Required columns are ``geometry``, ``x``, ``y`` and
        ``nodeID``; ``z`` (observer elevation) is optional and defaults to 0
        (ground level) when the column is absent.
    target_buildings_gdf : geopandas.GeoDataFrame
        Buildings to use as visibility targets. Required columns are
        ``buildingID``, ``geometry``, ``height``, and optionally ``base``.
    obstructions_buildings_gdf : geopandas.GeoDataFrame
        Buildings to use as possible obstructions. Required columns are
        ``buildingID``, ``geometry``, ``height``, and optionally ``base``.
    edges_gdf : geopandas.GeoDataFrame
        Network edges used only when ``consolidate=True``.
    city_name : str
        Prefix used for temporary chunk files.
    distance_along : float, default 200
        Sampling distance along target-building roof edges.
    min_observer_target_distance : float, default 300
        Minimum 2D distance allowed between observer and target.
    max_observer_target_distance : float or None, default None
        Maximum 2D distance allowed between observer and target. When set, candidate
        observer-target pairs are generated with a KD-tree radius query, so only near
        pairs are ever materialised (rather than the full observer x target cartesian
        product). Sight lines longer than this are never considered: in a dense city
        they are dominated by the (rare, mostly obstructed) long lines that account for
        the bulk of the 2D-obstruction cost. ``None`` reproduces the historical
        unbounded behaviour.
    sight_lines_chunk_size : int, default 500000
        Maximum number of candidate sight lines processed per chunk.
    consolidate : bool, default False
        Whether to consolidate observer nodes before sight-line generation.
    consolidate_tolerance : float, default 0.0
        Spatial tolerance used for node consolidation.
    num_workers : int, default 20
        Number of parallel Dask workers used for the 2D obstruction check.
    verbose : bool, default False
        When ``True``, print per-chunk progress (a bar, per-step 2D/3D times, and an ETA)
        plus step and total wall times to stdout. When ``False`` (the default) the compute
        path is silent, leaving progress reporting to the caller.

    Returns
    -------
    geopandas.GeoDataFrame
        Visible 3D sight lines. If no visible lines are found, an empty
        GeoDataFrame with the obstruction CRS is returned.
    """

    # The 3D visibility test is closed-form (see _analytic_obstructions_3d); only the
    # 2D batching stack (dask + psutil) is required.
    _require_visibility3d_dependencies("dask", "psutil")
    progress = _ProgressLogger(enabled=verbose)

    # Step 0: Prepare data
    with progress.step(" 00 - Preparing observers, targets and obstructions"):
        observers, targets, obstructions_gdf = _prepare_3d_sight_lines(
            nodes_gdf,
            target_buildings_gdf,
            obstructions_buildings_gdf,
            distance_along=distance_along,
            consolidate=consolidate,
            consolidate_tolerance=consolidate_tolerance,
            edges_gdf=edges_gdf,
        )

    num_observers = len(observers)
    num_targets = len(targets)
    # With a distance cap, a typical observer only pairs with the fraction of targets
    # inside the cap radius, so far fewer candidate lines are generated per observer.
    # Estimate that fraction from the target footprint area to size chunks realistically
    # (an uncapped run keeps the historical estimate of "every observer x every target").
    est_pairs_per_observer = num_targets
    if max_observer_target_distance is not None and num_targets:
        target_xy_all = _points_xy(targets["target_geo"])
        span_x = float(np.ptp(target_xy_all[:, 0])) or 1.0
        span_y = float(np.ptp(target_xy_all[:, 1])) or 1.0
        cap_fraction = min(1.0, np.pi * max_observer_target_distance ** 2 / (span_x * span_y))
        est_pairs_per_observer = max(1, int(num_targets * cap_fraction))
    projected_nr_sight_lines = num_observers * est_pairs_per_observer
    obstructions_sindex = obstructions_gdf.sindex  # GeoPandas SpatialIndex wrapper
    # Footprint polygons + roof heights for the closed-form 3D visibility test (the
    # obstruction buildings are vertical extrusions, so no ray tracing is needed).
    occluders = _AnalyticOccluders(obstructions_gdf)

    if projected_nr_sight_lines <= sight_lines_chunk_size:
        observer_chunks = [observers]
    else:
        observers_per_chunk = max(1, sight_lines_chunk_size // est_pairs_per_observer)
        num_chunks = int(np.ceil(num_observers / observers_per_chunk))
        # Split by row position and index back with .iloc so each chunk stays a GeoDataFrame.
        # np.array_split(observers, ...) would coerce the frame to bare ndarrays, which later
        # break in filter_distance (chunk.drop(columns="geometry")).
        observer_chunks = [
            observers.iloc[positions]
            for positions in np.array_split(np.arange(num_observers), num_chunks)
        ]

    out_prefix = "chunk_sight_lines"
    out_files = []

    progress.n_chunks = len(observer_chunks)
    for n, chunk in enumerate(observer_chunks):
        visibles = []

        potential_sight_lines = filter_distance(
            chunk, targets, min_observer_target_distance, max_observer_target_distance
        )
        n_lines = len(potential_sight_lines)
        if potential_sight_lines.empty:
            progress.chunk(n + 1, 0, 0, 0.0, 0.0)
            continue

        t0 = time.perf_counter()
        visible_2d, obstructed_2d = obstructions_2d(
            potential_sight_lines, obstructions_gdf, obstructions_sindex, num_workers=num_workers
        )
        t_2d = time.perf_counter() - t0
        if not visible_2d.empty:
            visibles.append(visible_2d)

        t0 = time.perf_counter()
        visible_3d = pd.DataFrame()
        if not obstructed_2d.empty:
            import shapely

            # Vectorised 3D segment construction (observer/target are 3D Points).
            observer_xyz = shapely.get_coordinates(
                np.asarray(obstructed_2d["observer_geo"], dtype=object), include_z=True
            )
            target_xyz = shapely.get_coordinates(
                np.asarray(obstructed_2d["target_geo"], dtype=object), include_z=True
            )
            obstructed_2d["geometry"] = shapely.linestrings(
                np.stack([observer_xyz, target_xyz], axis=1)
            )
            visible_3d = _analytic_obstructions_3d(obstructed_2d, occluders, "matchesIDs")
            if not visible_3d.empty:
                visibles.append(visible_3d)
        t_3d = time.perf_counter() - t0

        chunk_dir = Path("sight_lines_tmp")
        chunk_dir.mkdir(parents=True, exist_ok=True)  # create folder if not existing

        # Finalize columns and export
        n_records = 0
        if visibles:
            for df in visibles:
                df.drop(
                    columns=[col for col in ("visible", "z") if col in df.columns],
                    inplace=True,
                    errors="ignore",
                )
            chunk_sight_lines = pd.concat(visibles, ignore_index=True)
            chunk_sight_lines = _finalize_sight_lines(
                chunk_sight_lines, nodes_gdf, consolidate
            )
            n_records = len(chunk_sight_lines)
            chunk_file = chunk_dir / f"{city_name}_{out_prefix}_{n}.gpkg"
            chunk_sight_lines.to_file(chunk_file)
            out_files.append(chunk_file)
            del chunk_sight_lines

        del visibles, chunk, potential_sight_lines, visible_2d, obstructed_2d, visible_3d
        gc.collect()
        progress.chunk(n + 1, n_lines, n_records, t_2d, t_3d)
    tmp_sight_lines = gpd.GeoDataFrame(geometry=[], crs=obstructions_gdf.crs)
    if out_files:
        with progress.step(" -- Merging chunk files"):
            tmp_sight_lines = merge_gpkg_chunks_to_gdf(out_files, "matchesIDs")
            tmp_sight_lines.drop(["visible"], axis=1, errors="ignore", inplace=True)
    else:
        print("No visible sight-lines")
        progress.total()
        return tmp_sight_lines

    if consolidate:
        with progress.step(" 04 - Final check"):
            sight_lines = _last_check(
                tmp_sight_lines,
                obstructions_gdf,
                obstructions_sindex,
                occluders,
                nodes_gdf,
                progress,
                num_workers=num_workers,
            )
    else:
        sight_lines = tmp_sight_lines

    progress.total()
    return sight_lines


## Preparation ###########################
def _prepare_3d_sight_lines(
    nodes_gdf,
    target_buildings_gdf,
    obstructions_gdf,
    distance_along=200,
    consolidate=False,
    consolidate_tolerance=0.0,
    edges_gdf=None,
):
    """Prepare observers, target points, and obstruction geometries.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        Observer nodes with ``geometry``, ``x``, ``y``, ``nodeID``; ``z`` is
        optional and defaults to 0 (ground level) when the column is absent.
    target_buildings_gdf : geopandas.GeoDataFrame
        Target buildings with ``buildingID``, ``geometry``, ``height``, and
        optionally ``base``.
    obstructions_gdf : geopandas.GeoDataFrame
        Buildings considered as potential obstructions.
    distance_along : float, default 200
        Sampling distance along target-building roof edges.
    consolidate : bool, default False
        Whether to consolidate observer nodes.
    consolidate_tolerance : float, default 0.0
        Tolerance used for observer consolidation.
    edges_gdf : geopandas.GeoDataFrame or None, default None
        Network edges required when observer consolidation is enabled.

    Returns
    -------
    tuple
        ``(observer_points_gdf, target_points, obstructions_gdf)`` prepared for
        3D sight-line generation.
    """

    if "z" not in nodes_gdf.columns:
        # No observer elevation supplied (e.g. OSM nodes with no DTM sampled): treat
        # observers as standing at ground level. Mirrors the pipeline convention that
        # node z is 0 when no terrain raster is available. assign() returns a copy, so
        # the caller's frame is never mutated.
        nodes_gdf = nodes_gdf.assign(z=0.0)
    nodes_gdf = nodes_gdf[["geometry", "x", "y", "nodeID", "z"]].copy()
    nodes_gdf.loc[nodes_gdf["z"] < -50, "z"] = 2
    nodes_gdf["geometry"] = gpd.points_from_xy(nodes_gdf["x"], nodes_gdf["y"], nodes_gdf["z"])

    target_buildings_gdf = _prepare_buildings_gdf(target_buildings_gdf)
    obstructions_gdf = _prepare_buildings_gdf(obstructions_gdf)
    target_buildings_gdf = target_buildings_gdf[target_buildings_gdf.height > 5.0]
    target_points = _prepare_targets(target_buildings_gdf, distance_along)
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

    # The analytic 3D test works directly on footprint rings + roof elevations, so no 3D
    # solids are built; reduce each obstruction to its exterior ring here.
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


def _prepare_targets(target_buildings_gdf, distance_along):
    """Generate 3D target points along target-building roof perimeters.

    Parameters
    ----------
    target_buildings_gdf : geopandas.GeoDataFrame
        Target buildings with ``geometry``, ``height``, and ``base`` columns.
    distance_along : float
        Sampling distance along each roof perimeter.

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

    def add_height_to_line(exterior, base, height):
        # Absolute roof elevation = terrain base + above-ground height.
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


def _points_xy(point_series):
    """Vectorised 2D coordinates (dropping z) for a series/array of shapely Points."""
    import shapely

    return shapely.get_coordinates(np.asarray(point_series, dtype=object))


## Step 0 ###########################
def filter_distance(
    chunk, targets, min_observer_target_distance, max_observer_target_distance=None
):
    """
    Filters potential observer-target pairs by distance and generates sight line geometries.

    Parameters
    ----------
    chunk : GeoDataFrame
        Observer points (usually a subset/chunk of all observers).
    targets : GeoDataFrame
        Target points to be paired with observers.
    min_observer_target_distance : float
        Minimum allowed 2D distance between observer and target (in CRS units).
    max_observer_target_distance : float or None, default None
        Maximum allowed 2D distance between observer and target (in CRS units). When
        set, candidate pairs are found with a KD-tree radius query so only pairs within
        the cap are ever materialised; ``None`` keeps every pair beyond the minimum
        (the full cartesian product), reproducing the historical behaviour.

    Returns
    -------
    potential_lines : GeoDataFrame
        GeoDataFrame of observer-target pairs meeting the distance thresholds,
        with new 'geometry' (LineString) and coordinate columns.
    """

    import shapely

    # Vectorised 2D coordinates (drops z) — one small array per side, never per pair.
    observer_attrs = chunk.drop(columns="geometry")
    observer_xy = _points_xy(observer_attrs["observer_geo"])
    target_xy = _points_xy(targets["target_geo"])

    min_sq = float(min_observer_target_distance) ** 2

    if max_observer_target_distance is not None:
        # KD-tree radius query: only pairs within the cap are enumerated, so we never
        # materialise the observer x target cartesian product (the source of both the
        # runtime blow-up on long lines and the memory pressure on large cities).
        from scipy.spatial import cKDTree

        observer_tree = cKDTree(observer_xy)
        target_tree = cKDTree(target_xy)
        near = observer_tree.sparse_distance_matrix(
            target_tree, float(max_observer_target_distance), output_type="coo_matrix"
        )
        observer_pos, target_pos, dist = near.row, near.col, near.data
        keep = dist >= float(min_observer_target_distance)
        observer_pos, target_pos = observer_pos[keep], target_pos[keep]
    else:
        # Broadcast the pairwise distance filter instead of materialising the cartesian
        # product with a pandas cross merge (which duplicated every attribute column —
        # including object-dtype Points — for all observer x target pairs and dominated
        # per-chunk runtime on real cities).
        dx = observer_xy[:, 0][:, None] - target_xy[None, :, 0]
        dy = observer_xy[:, 1][:, None] - target_xy[None, :, 1]
        keep = (dx * dx + dy * dy) >= min_sq
        observer_pos, target_pos = np.nonzero(keep)

    # Materialise only the surviving pairs, by row position (fast .take, no merge).
    left = observer_attrs.take(observer_pos).reset_index(drop=True)
    right = pd.DataFrame(targets).take(target_pos).reset_index(drop=True)
    potential_lines = pd.concat([left, right], axis=1)

    # Vectorised 2D line construction for the planimetric obstruction check.
    segments = (
        np.stack([observer_xy[observer_pos], target_xy[target_pos]], axis=1)
        if len(observer_pos)
        else np.empty((0, 2, 2))
    )
    potential_lines["geometry"] = shapely.linestrings(segments)

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


def _obstruction_candidates_slice(
    endpoint_xyz, obstructions_sindex, bounds, tops, poly_ids
):
    """Candidate obstruction building IDs per sight line, for one slice of lines.

    A fully vectorised bounding-box broad phase: for each line we keep the buildings whose
    bounding box the line segment actually enters (a Liang–Barsky slab clip) and whose
    extruded top rises above the sight line over that entry interval. Every other
    bounding-box overlap is dropped because it provably cannot block the line. This prunes
    each line's thousands of bounding-box overlaps down to the few tens of buildings along
    its corridor. The result is a *superset* of the true footprint crossers; the exact
    crossing-and-height decision is made analytically in :func:`_analytic_obstructions_3d`.

    Parameters
    ----------
    endpoint_xyz : numpy.ndarray, shape (n_lines, 2, 3)
        Observer/target endpoints (x, y, z) for the slice's lines.
    obstructions_sindex : geopandas.sindex.SpatialIndex
        Spatial index over the obstruction geometries.
    bounds : numpy.ndarray, shape (n_obstructions, 4)
        ``[minx, miny, maxx, maxy]`` per obstruction (cached).
    tops : numpy.ndarray, shape (n_obstructions,)
        ``height + base`` per obstruction (cached).
    poly_ids : numpy.ndarray
        ``buildingID`` per obstruction (positional order of the index).

    Returns
    -------
    list[list[int]]
        Candidate obstruction building IDs for each line in the slice.
    """
    import shapely

    n_lines = endpoint_xyz.shape[0]
    xy0, xy1 = endpoint_xyz[:, 0, :2], endpoint_xyz[:, 1, :2]
    z0, z1 = endpoint_xyz[:, 0, 2], endpoint_xyz[:, 1, 2]

    lines = shapely.linestrings(endpoint_xyz[:, :, :2])
    # Bounding-box broad phase only (no exact predicate): fast tree traversal.
    line_pos, tree_pos = obstructions_sindex.query(lines, predicate=None)
    if len(line_pos) == 0:
        return [[] for _ in range(n_lines)]

    p0 = xy0[line_pos]
    delta = xy1[line_pos] - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (bounds[tree_pos, :2] - p0) / delta
        t_high = (bounds[tree_pos, 2:] - p0) / delta
        t_min = np.minimum(t_low, t_high)
        t_max = np.maximum(t_low, t_high)
    # Axis-parallel segments produce NaN slabs; make them non-restrictive (no constraint).
    t_min = np.where(np.isnan(t_min), -np.inf, t_min)
    t_max = np.where(np.isnan(t_max), np.inf, t_max)
    t_enter = np.maximum(t_min.max(axis=1), 0.0)
    t_exit = np.minimum(t_max.min(axis=1), 1.0)

    # Keep bbox-entering pairs that could block (top above the line's lowest z over the
    # entry interval). Missing/degenerate z (NaN) conservatively keeps the pair.
    enters_box = t_enter <= t_exit
    pair_z0 = z0[line_pos]
    pair_dz = z1[line_pos] - pair_z0
    z_lowest = np.minimum(pair_z0 + t_enter * pair_dz, pair_z0 + t_exit * pair_dz)
    can_block = ~(np.isfinite(z_lowest) & (z_lowest > tops[tree_pos] + 1e-6))
    keep = enters_box & can_block
    line_pos = line_pos[keep]
    tree_pos = tree_pos[keep]

    obstruction_dict = {}
    for line_idx, poly_idx in zip(line_pos, tree_pos, strict=False):
        obstruction_dict.setdefault(line_idx, []).append(int(poly_ids[poly_idx]))
    return [obstruction_dict.get(i, []) for i in range(n_lines)]


def _find_obstructions_2d(sub_chunk, obstructions_gdf, obstructions_sindex, slice_size=750):
    """Find candidate obstructions for a batch of sight lines (bounding-box broad phase).

    Splits each candidate line into a list of possible obstruction building IDs and
    partitions the batch into ``visible`` (no candidate obstruction — visible without any
    ray casting) and ``obstructed`` (has candidates, needs the 3D check). Candidate
    finding uses a bounding-box + vertical prune rather than an exact ``crosses`` test;
    see :func:`_obstruction_candidates_slice`. Lines are processed in slices of
    ``slice_size`` so the bounding-box broad phase never materialises more than a bounded
    number of (line, building) pairs at once.

    Parameters
    ----------
    sub_chunk : geopandas.GeoDataFrame
        Batch of candidate sight lines. Only the line ``geometry`` (with
        ``observer_geo``/``target_geo`` as a fallback for missing z) is read here. The
        target's own ``buildingID`` is *not* stripped from the candidate obstructions, so a
        line reaching a roof point across the target's own footprint can be occluded by that
        footprint — the physically correct result for a vertical-extrusion building. The
        exact crossing-and-height decision is deferred to :func:`_analytic_obstructions_3d`.
    obstructions_gdf : geopandas.GeoDataFrame
        Obstruction geometries with ``buildingID`` and (for the prune) ``height``/``base``.
    obstructions_sindex : geopandas.sindex.SpatialIndex
        Spatial index built from ``obstructions_gdf``.
    slice_size : int, default 750
        Number of lines whose bounding-box candidates are materialised at once.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Lines with no candidate obstruction and lines with candidate IDs in ``matchesIDs``.
    """

    import shapely

    # Bounds/tops are identical for every batch and chunk; compute once per frame.
    # (Concurrent dask threads may race on first use — the value is idempotent.)
    prune_cache = obstructions_gdf.attrs.get("_ci_v3d_prune_cache")
    if prune_cache is None:
        prune_cache = (
            obstructions_gdf.bounds.to_numpy(),
            (obstructions_gdf["height"] + obstructions_gdf["base"]).to_numpy(),
            obstructions_gdf["buildingID"].to_numpy(),
        )
        obstructions_gdf.attrs["_ci_v3d_prune_cache"] = prune_cache
    bounds, tops, poly_ids = prune_cache

    # Endpoints (x, y, z). Per-chunk lines are planimetric with z in observer_geo/
    # target_geo; the consolidation re-check passes 3D lines directly.
    endpoint_xyz = shapely.get_coordinates(
        np.asarray(sub_chunk.geometry, dtype=object), include_z=True
    ).reshape(-1, 2, 3)
    if np.isnan(endpoint_xyz[:, :, 2]).any() and "observer_geo" in sub_chunk.columns:
        endpoint_xyz[:, 0, 2] = shapely.get_coordinates(
            np.asarray(sub_chunk["observer_geo"], dtype=object), include_z=True
        )[:, 2]
        endpoint_xyz[:, 1, 2] = shapely.get_coordinates(
            np.asarray(sub_chunk["target_geo"], dtype=object), include_z=True
        )[:, 2]

    obstructionIDs = []
    for start in range(0, len(sub_chunk), slice_size):
        stop = start + slice_size
        obstructionIDs.extend(
            _obstruction_candidates_slice(
                endpoint_xyz[start:stop],
                obstructions_sindex,
                bounds,
                tops,
                poly_ids,
            )
        )

    sub_chunk["matchesIDs"] = obstructionIDs

    mask = sub_chunk["matchesIDs"].map(len) > 0
    visible = sub_chunk.loc[~mask].reset_index(drop=True).copy()
    obstructed = sub_chunk.loc[mask].reset_index(drop=True)

    return visible, obstructed


## Step 3 ###########################
class _AnalyticOccluders:
    """Footprint polygons and roof heights for the analytic 3D visibility test.

    The obstruction buildings are vertical extrusions (flat roofs), so a sight line is
    occluded by a building exactly when the line's planimetric projection crosses the
    building footprint AND the line dips to or below the roof somewhere over that crossing.
    That is a closed-form test on the footprint polygon and the ``height + base`` roof
    elevation — no triangulated mesh or ray tracing is needed. This holds the per-building
    footprint polygon and roof top keyed by ``buildingID``.
    """

    def __init__(self, obstructions_gdf):
        import shapely

        geoms = obstructions_gdf.geometry.to_numpy()
        polys = np.empty(len(geoms), dtype=object)
        for i, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                polys[i] = None
            elif geom.geom_type == "Polygon":
                polys[i] = geom
            else:  # exterior ring / linestring (as produced by _prepare_3d_sight_lines)
                polys[i] = shapely.Polygon(geom)
        ids = obstructions_gdf["buildingID"].to_numpy()
        tops = (obstructions_gdf["height"] + obstructions_gdf["base"]).to_numpy(dtype=float)
        self.poly_by_id = dict(zip(ids, polys, strict=False))
        self.top_by_id = dict(zip(ids, tops, strict=False))


def _analytic_obstructions_3d(
    potential_sight_lines, occluders, potential_obstructions_column, eps=1e-6
):
    """Return the visible subset of candidate sight lines, tested analytically.

    For every (line, candidate-building) pair, intersect the line's 2D projection with the
    building footprint; where that intersection has positive length the line crosses the
    footprint, and the building occludes the line when the line's height over the crossing
    falls to or below the roof (``height + base``). A line is visible when none of its
    candidate buildings occlude it. This reproduces the ray-traced visibility exactly for
    vertical-extrusion buildings, and — unlike a mesh ray cast — a building that merely
    touches the line at an endpoint (its shared roof corner) yields a zero-length
    intersection and correctly does not occlude.

    Parameters
    ----------
    potential_sight_lines : geopandas.GeoDataFrame
        Candidate sight lines with 3D ``geometry`` and a candidate-obstruction column.
    occluders : _AnalyticOccluders
        Footprint polygons and roof tops keyed by ``buildingID``.
    potential_obstructions_column : str
        Column holding, per line, the list of candidate obstruction building IDs.
    eps : float, default 1e-6
        Minimum crossing length (in CRS units) that counts as a real footprint crossing.

    Returns
    -------
    geopandas.GeoDataFrame
        The visible sight lines (rows of ``potential_sight_lines`` that no building blocks).
    """
    import shapely

    n = len(potential_sight_lines)
    if n == 0:
        return potential_sight_lines.copy()

    endpoint = shapely.get_coordinates(
        np.asarray(potential_sight_lines.geometry, dtype=object), include_z=True
    ).reshape(-1, 2, 3)
    lines2d = shapely.linestrings(endpoint[:, :, :2])

    matches = potential_sight_lines[potential_obstructions_column].tolist()
    counts = np.fromiter((len(m) for m in matches), dtype=np.int64, count=n)
    max_candidates = int(counts.max()) if n else 0
    if max_candidates == 0:
        return potential_sight_lines.copy()

    blocked = np.zeros(n, dtype=bool)
    # Early-exit rounds: a line is occluded as soon as ONE candidate blocks it, so test the
    # r-th candidate of every still-unresolved line together (vectorised), drop the ones it
    # blocks, and advance. In a dense city most lines block on an early candidate, so this
    # touches a few candidates per line instead of exhaustively intersecting all of them.
    for r in range(max_candidates):
        active = np.nonzero(~blocked & (counts > r))[0]
        if active.size == 0:
            break
        bids = np.fromiter((matches[i][r] for i in active), dtype=np.int64, count=active.size)
        polys = np.array([occluders.poly_by_id.get(b) for b in bids], dtype=object)
        tops = np.array([occluders.top_by_id.get(b, np.inf) for b in bids], dtype=float)

        inter = shapely.intersection(lines2d[active], polys)
        real = shapely.length(inter) > eps
        if not real.any():
            continue

        ridx = np.nonzero(real)[0]
        coords, cidx = shapely.get_coordinates(inter[ridx], return_index=True)
        real_lines = active[ridx]
        p_obs = endpoint[real_lines, 0][cidx]
        p_tgt = endpoint[real_lines, 1][cidx]
        direction = p_tgt[:, :2] - p_obs[:, :2]
        seglen2 = np.einsum("ij,ij->i", direction, direction)
        seglen2 = np.where(seglen2 == 0.0, 1.0, seglen2)
        t = np.einsum("ij,ij->i", coords - p_obs[:, :2], direction) / seglen2
        z = p_obs[:, 2] + t * (p_tgt[:, 2] - p_obs[:, 2])
        # Height is linear in t, so the line's minimum over a crossing is at one of the
        # intersection vertices: the candidate blocks if any vertex sits at/under its roof.
        below = z <= tops[ridx][cidx] + eps
        crossing_blocked = np.zeros(ridx.size, dtype=bool)
        np.logical_or.at(crossing_blocked, cidx, below)
        blocked[real_lines[crossing_blocked]] = True

    return potential_sight_lines.loc[~blocked].reset_index(drop=True)


def _last_check(
    potential_sight_lines,
    obstructions_gdf,
    obstructions_sindex,
    occluders,
    nodes_gdf,
    progress,
    num_workers,
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
    occluders : _AnalyticOccluders
        Footprint polygons and roof tops for the analytic visibility test.
    nodes_gdf : geopandas.GeoDataFrame
        Original observer nodes used to finalize sight-line geometries.
    progress : _ProgressLogger
        Progress reporter from the parent run; its ``step`` timers print only when the run
        was started with ``verbose=True``.
    num_workers : int
        Unused; kept for signature stability with earlier callers.

    Returns
    -------
    geopandas.GeoDataFrame
        Finalized sight lines that pass the last obstruction check.
    """
    with progress.step("   04a - Re-checking 2d obstructions (consolidated lines)"):
        visible_2d, obstructed = _find_obstructions_2d(
            potential_sight_lines, obstructions_gdf, obstructions_sindex
        )
    with progress.step("   04b - Re-checking 3d obstructions (consolidated lines)"):
        visible_3d = _analytic_obstructions_3d(obstructed, occluders, "matchesIDs")
    sight_lines_tmp = pd.concat([visible_2d, visible_3d], ignore_index=True)
    return _finalize_sight_lines(sight_lines_tmp, nodes_gdf, False)


def _finalize_sight_lines(sight_lines_tmp, nodes_gdf, consolidate):
    """Finalize, clean, and deduplicate visible sight lines.

    Parameters
    ----------
    sight_lines_tmp : pandas.DataFrame or geopandas.GeoDataFrame
        Temporary sight-line results with observer/target references.
    nodes_gdf : geopandas.GeoDataFrame
        Original observer nodes with ``nodeID`` and ``geometry``; ``z`` is optional
        and defaults to 0 (ground level) when the column is absent.
    consolidate : bool
        Whether observer node consolidation was used.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned visible sight lines with ``nodeID``, ``buildingID``, ``geometry``,
        and ``length``.
    """
    # Work on a copy: this frame is the caller's original nodes GeoDataFrame, and
    # replacing its geometry with 3D points in place would leak out of this function.
    nodes_gdf = nodes_gdf.copy()
    if "z" not in nodes_gdf.columns:
        nodes_gdf["z"] = 0.0  # ground level when no observer elevation was supplied
    nodes_gdf["geometry"] = gpd.points_from_xy(
        nodes_gdf.geometry.x, nodes_gdf.geometry.y, nodes_gdf["z"]
    )
    oldIDs_column = "old_nodeID"

    if consolidate:
        sight_lines_tmp = sight_lines_tmp.explode(column=oldIDs_column, ignore_index=True)
        sight_lines_tmp["nodeID"] = sight_lines_tmp[oldIDs_column]
        geom_map = nodes_gdf.set_index("nodeID")["geometry"]
        sight_lines_tmp["observer_geo"] = sight_lines_tmp["nodeID"].map(geom_map)

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
            "observer_geo",
            "target_geo",
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
