"""Core building landmark and imageability scoring functions.

This module contains the scientific cityImage building-level scoring logic.
It deliberately excludes data-loading helpers such as OSM download wrappers.
External libraries should prepare/download data; cityImage should score already-prepared GeoDataFrames.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from .buildings import known_heights
from .data_utils import scaling_columnDF

pd.set_option("display.precision", 3)


# Process-pool workers for the 2D advance-visibility isovist. The per-building isovist is
# CPU-bound and GIL-limited (threads plateau at ~4x), so real speed-up needs processes.
# Each worker keeps one copy of the obstruction set (shipped once when the pool starts) and
# builds its spatial index once; building geometries stream in as small tasks. Kept at
# module level and picklable so the "spawn" start method works on every platform.
_STRUCTURAL_OBSTRUCTIONS = None


def _init_structural_worker(obstructions_gdf):
    global _STRUCTURAL_OBSTRUCTIONS
    _STRUCTURAL_OBSTRUCTIONS = obstructions_gdf
    _ = obstructions_gdf.sindex  # build the (cached) index once per worker


def _structural_visibility_task(args):
    from .visibility2d import visibility_polygon2d

    geometry, max_expansion_distance = args
    return visibility_polygon2d(geometry, _STRUCTURAL_OBSTRUCTIONS, None, max_expansion_distance)


def _advance_visibility_areas(
    geometries, obstructions_gdf, sindex, max_expansion_distance, workers
):
    """2D advance-visibility area per building geometry, in insertion order.

    Runs across a ``spawn`` process pool when ``workers > 1``; falls back to a serial pass
    for a single worker, a trivial input, or if the pool cannot start (the result is
    identical either way — only the execution changes).
    """
    from .visibility2d import visibility_polygon2d

    def _serial():
        return [
            visibility_polygon2d(geometry, obstructions_gdf, sindex, max_expansion_distance)
            for geometry in geometries
        ]

    if not workers or workers <= 1 or len(geometries) <= 1:
        return _serial()

    import multiprocessing as mp

    tasks = [(geometry, max_expansion_distance) for geometry in geometries]
    chunksize = max(1, len(tasks) // (workers * 10))
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_structural_worker,
            initargs=(obstructions_gdf,),
        ) as executor:
            return list(executor.map(_structural_visibility_task, tasks, chunksize=chunksize))
    except Exception as exc:  # environment-dependent; never fail the whole stage over it
        import warnings

        warnings.warn(
            f"structural_score: parallel visibility failed ({exc!r}); running serially.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _serial()


def structural_score(
    buildings_gdf,
    obstructions_gdf,
    edges_gdf,
    advance_vis_expansion_distance=300,
    neighbours_radius=150,
    workers=1,
):
    """
    The function computes the "Structural Landmark Component" sub-scores of each building.
    It considers:
    - distance from the street network:
    - advance 2d visibility polygon;
    - number of neighbouring buildings in a given radius.

    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    edges_gdf: LineString GeoDataFrame
        Street segmetns GeoDataFrame.
    obstructions_gdf: Polygon GeoDataFrame
        Obstructions GeoDataFrame.
    advance_vis_expansion_distance: float
        2d advance visibility - it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    neighbours_radius: float
        Neighbours - search radius for other adjacent buildings.
    workers: int
        Number of processes used for the (CPU-bound) 2D advance-visibility computation.
        ``1`` runs serially; higher values parallelise it across a process pool. The result
        is identical regardless of the worker count.

    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """
    buildings_gdf = buildings_gdf.copy()
    assert_all_polygons(buildings_gdf)
    if buildings_gdf.empty:
        buildings_gdf["road"] = pd.Series(dtype=float, index=buildings_gdf.index)
        buildings_gdf["2dvis"] = pd.Series(dtype=float, index=buildings_gdf.index)
        buildings_gdf["neigh"] = pd.Series(dtype=int, index=buildings_gdf.index)
        return buildings_gdf

    # remove z coordinates if they are there already - issue with 2dvis
    if len(buildings_gdf.geometry.iloc[0].exterior.coords[0]) == 3:
        buildings_gdf["geometry"] = buildings_gdf["geometry"].apply(
            lambda g: type(g)([(x, y) for x, y, *_ in g.exterior.coords])
        )

    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.union_all()

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = _advance_visibility_areas(
        list(buildings_gdf.geometry),
        obstructions_gdf,
        sindex,
        advance_vis_expansion_distance,
        workers,
    )
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(
        lambda row: _number_neighbours(row, obstructions_gdf, sindex, radius=neighbours_radius)
    )

    return buildings_gdf


def _number_neighbours(geometry, obstructions_gdf, obstructions_sindex, radius):
    """
    The function counts the number of neighbours, in a GeoDataFrame, around a given geometry, within a
    search radius.

    Parameters
    ----------
    geometry: Shapely Geometry
        The geometry for which neighbors are counted.
    obstructions_gdf: GeoDataFrame
        The GeoDataFrame containing the obstructions.
    obstructions_sindex: Spatial Index
        The spatial index of the obstructions GeoDataFrame.
    radius: float
        The search radius for neighboring buildings.

    Returns
    -------
    int
        The number of neighbors.
    """
    buffer = geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)


def visibility_score(buildings_gdf, sight_lines=None, method="longest"):
    """Calculate visibility landmark sub-scores.

    Adds:
    - fac: approximate facade area;
    - 3dvis: 3D visibility score, derived from sight-line lengths; 0 for a building no sight
      line reaches (not visible, or too short to be a target) when another building is reached.
      When no building with a height is reached (``sight_lines`` None or empty, or matching none
      of them), ``3dvis`` is NaN for every building: there is no visibility to score.

    Heights are read through ``known_heights``. A building without a height gets NaN for both,
    so it stays out of their rescaling in the landmark scores.
    """
    if sight_lines is None:
        sight_lines = pd.DataFrame()

    buildings_gdf = buildings_gdf.copy()
    if "height" in buildings_gdf.columns:
        buildings_gdf["height"] = known_heights(buildings_gdf["height"])
        known = buildings_gdf["height"].notna()
    else:
        known = pd.Series(False, index=buildings_gdf.index)

    buildings_gdf["fac"] = np.nan
    if known.any():
        buildings_gdf.loc[known, "fac"] = [
            _facade_area(geometry, height)
            for geometry, height in zip(
                buildings_gdf.geometry[known], buildings_gdf.loc[known, "height"], strict=True
            )
        ]

    buildings_gdf["3dvis"] = np.nan
    if not known.any() or sight_lines.empty:
        return buildings_gdf

    sight_lines = sight_lines.copy()
    sight_lines["nodeID"] = sight_lines["nodeID"].astype(int)
    sight_lines["buildingID"] = sight_lines["buildingID"].astype(int)
    sight_lines["length"] = sight_lines.geometry.length

    stats = sight_lines.groupby("buildingID").agg({"length": ["mean", "max", "count"]})
    stats.columns = stats.columns.droplevel(0)
    stats.rename(columns={"count": "nr_lines"}, inplace=True)

    for column in ["max", "mean", "nr_lines"]:
        stats[column] = stats[column].fillna(stats[column].min())
        stats[column + "_sc"] = scaling_columnDF(stats[column])

    if method == "longest":
        stats["3dvis"] = stats["max_sc"]
    elif method == "combined":
        stats["3dvis"] = (
            stats["max_sc"] * 0.5 + stats["mean_sc"] * 0.25 + stats["nr_lines_sc"] * 0.25
        )
    else:
        raise ValueError("method must be either 'longest' or 'combined'")

    # Mapped by buildingID rather than merged, so the scored frame keeps the caller's index.
    reached = buildings_gdf["buildingID"].isin(stats.index) & known
    if reached.any():
        buildings_gdf["3dvis"] = (
            buildings_gdf["buildingID"].map(stats["3dvis"]).fillna(0.0).where(known)
        )

    return buildings_gdf


def _facade_area(building_geometry, building_height):
    """
    Compute the approximate facade area of a building given its geometry and height.

    Parameters
    ----------
    building_geometry: Polygon
        The geometry of the building.
    building_height: float
        The height of the building.

    Returns
    -------
    float
        The computed approximate facade area of the building.
    """
    envelope = building_geometry.envelope
    coords = mapping(envelope)["coordinates"][0]
    d = [
        (Point(coords[0])).distance(Point(coords[1])),
        (Point(coords[1])).distance(Point(coords[2])),
    ]
    width = min(d)
    return width * building_height


def _is_historic(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() not in {"0", "no", "false", "", "none", "nan"}


def cultural_score(
    buildings_gdf,
    historic_elements_gdf=None,
    score_column: str | None = None,
    from_OSM: bool = False,
):
    """Compute a cultural landmark component per building.

    ``cult`` counts the historic elements intersecting each building, or sums their
    ``score_column``; with ``from_OSM=True`` it is 1 for a building with a ``historic`` tag. A
    building with nothing gets 0 when another building has something; when no building has
    anything (no historic layer, no element intersecting a building, no ``historic`` tag, or
    every sum 0), ``cult`` is NaN for every building: there is no cultural information to score.
    """
    buildings_gdf = buildings_gdf.copy()
    cult = _cultural_values(buildings_gdf, historic_elements_gdf, score_column, from_OSM)
    buildings_gdf["cult"] = cult if (cult > 0).any() else np.nan
    return buildings_gdf


def _cultural_values(buildings_gdf, historic_elements_gdf, score_column, from_OSM):
    """Each building's cultural value (see ``cultural_score``), 0 where it has nothing."""
    cult = pd.Series(0.0, index=buildings_gdf.index)

    if from_OSM:
        if "historic" not in buildings_gdf.columns:
            raise ValueError("from_OSM=True requires buildings_gdf to contain a 'historic' column")
        return buildings_gdf["historic"].apply(_is_historic).astype(float)

    if historic_elements_gdf is None or len(historic_elements_gdf) == 0:
        return cult

    if buildings_gdf.crs != historic_elements_gdf.crs:
        raise ValueError(
            "CRS mismatch: buildings_gdf and historic_elements_gdf must have the same CRS"
        )

    right_cols = ["geometry"]
    if score_column is not None:
        if score_column not in historic_elements_gdf.columns:
            raise ValueError(f"score_column '{score_column}' not found in historic_elements_gdf")
        right_cols.append(score_column)

    left = buildings_gdf[["geometry"]]
    left = left[left.geometry.notna()].copy()
    right = historic_elements_gdf[right_cols]
    right = right[right.geometry.notna()].copy()
    if left.empty or right.empty:
        return cult

    try:
        joined = gpd.sjoin(left, right, how="inner", predicate="intersects")
    except TypeError:
        joined = gpd.sjoin(left, right, how="inner", op="intersects")
    if joined.empty:
        return cult

    if score_column is None:
        values = joined.groupby(joined.index).size().astype(float)
    else:
        scores = pd.to_numeric(joined[score_column], errors="coerce").fillna(0.0)
        values = scores.groupby(joined.index).sum().astype(float)
    return values.reindex(buildings_gdf.index, fill_value=0.0)


def pragmatic_score(
    buildings_gdf,
    land_uses_column: str = "land_uses",
    overlaps_column: str = "land_uses_overlap",
    search_radius: float = 200,
    default_land_use: str = "unclassified",
):
    """Compute a pragmatic landmark component from semantic land-use labels.

    Missing or empty land-use labels are treated as ``default_land_use`` and
    aligned with a full overlap weight of ``[1.0]``. A building with no other building within
    ``search_radius`` is as unexpected as can be: 1.
    """
    gdf = buildings_gdf.copy()

    def _as_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, set):
            return list(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if v is None:
            return []
        try:
            if pd.isna(v):
                return []
        except Exception:
            pass
        return [v]

    if land_uses_column not in gdf.columns:
        gdf[land_uses_column] = pd.Series(
            [[default_land_use] for _ in range(len(gdf))],
            index=gdf.index,
            dtype="object",
        )

    if overlaps_column not in gdf.columns:
        gdf[overlaps_column] = pd.Series(
            [[] for _ in range(len(gdf))],
            index=gdf.index,
            dtype="object",
        )

    gdf[land_uses_column] = gdf[land_uses_column].apply(lambda v: _as_list(v) or [default_land_use])

    if gdf.empty:
        gdf["prag"] = pd.Series(dtype=float, index=gdf.index)
        return gdf

    def _weights_for_row(row):
        labels = row[land_uses_column]
        weights = _as_list(row[overlaps_column])

        if len(labels) == 0:
            return []
        if len(weights) != len(labels):
            return [1.0 / len(labels)] * len(labels)

        try:
            weights = [float(x) for x in weights]
        except Exception:
            return [1.0 / len(labels)] * len(labels)

        total = float(np.nansum(weights))
        if not np.isfinite(total) or total <= 0:
            return [1.0 / len(labels)] * len(labels)

        return [x / total for x in weights]

    gdf["_ci_row_id"] = np.arange(len(gdf))
    gdf["_w_list"] = gdf.apply(_weights_for_row, axis=1)
    gdf[overlaps_column] = pd.Series(gdf["_w_list"].tolist(), index=gdf.index, dtype="object")
    gdf["_lu_w"] = gdf.apply(
        lambda r: list(zip(r[land_uses_column], r["_w_list"], strict=False)),
        axis=1,
    )

    gdf_exploded = gdf.explode("_lu_w", ignore_index=False)
    gdf_exploded[land_uses_column] = gdf_exploded["_lu_w"].apply(
        lambda x: x[0] if isinstance(x, tuple) else x
    )
    gdf_exploded["_w"] = gdf_exploded["_lu_w"].apply(
        lambda x: float(x[1]) if isinstance(x, tuple) else 1.0
    )

    sindex = gdf_exploded.sindex

    def _unexpectedness(row_id, building_geometry, building_label):
        buf = building_geometry.buffer(search_radius)
        candidate_idx = list(sindex.intersection(buf.bounds))
        possible = gdf_exploded.iloc[candidate_idx]
        matches = possible[possible.intersects(buf)]
        matches = matches[matches["_ci_row_id"] != row_id]

        total_w = float(matches["_w"].sum())
        if total_w <= 0:  # no neighbour
            return 1.0

        Nj_w = float(matches.loc[matches[land_uses_column] == building_label, "_w"].sum())
        return 1.0 - (Nj_w / total_w)

    gdf_exploded["prag_temp"] = gdf_exploded.apply(
        lambda row: _unexpectedness(row["_ci_row_id"], row.geometry, row[land_uses_column]),
        axis=1,
    )

    scores = gdf_exploded.groupby("_ci_row_id")["prag_temp"].max()
    gdf["prag"] = gdf["_ci_row_id"].map(scores).astype(float)

    return gdf.drop(columns=["_ci_row_id", "_w_list", "_lu_w"], errors="ignore")


# The indexes of each component. The visual component is computed only where a building has a
# height; the cultural and pragmatic components are their single index.
VISUAL_INDEXES = ("fac", "height", "3dvis")
STRUCTURAL_INDEXES = ("area", "neigh", "2dvis", "road")
COMPONENT_INDEXES = {
    "vScore": VISUAL_INDEXES,
    "sScore": STRUCTURAL_INDEXES,
    "cScore": ("cult",),
    "pScore": ("prag",),
}
# Indexes where a lower value makes a building stand out.
INVERSE_INDEXES = ("neigh", "road")


def _with_known_heights(buildings_gdf):
    """A copy with ``height``, where there is one, read through ``known_heights``: NaN where
    missing, unreadable or not above zero. A frame without a height column is left without one.

    Where the layer has heights, the visual score is computed and a building without one gets 0:
    an unknown height never makes a building stand out, and the building can still be a landmark
    through its other components. It is not an obstruction to 3D sight lines either. Heights are
    the caller's to supply; drop such buildings beforehand to leave them out of the scores.
    """
    buildings_gdf = buildings_gdf.copy()
    if "height" in buildings_gdf.columns:
        buildings_gdf["height"] = known_heights(buildings_gdf["height"])
    return buildings_gdf


def _has_heights(buildings_gdf):
    """Whether at least one building has a height (read by ``_with_known_heights``)."""
    return "height" in buildings_gdf.columns and buildings_gdf["height"].notna().any()


def _component_scores(buildings_gdf, indexes_weights, components_weights, suffix=""):
    """Rescale the indexes over ``buildings_gdf`` and write each component and its rescaled
    value (``<component><suffix>`` and ``..._sc``); return the weighted sum of the components.

    A component is written only when at least one of its indexes has a value; the visual one
    only when a building has a height. Rescaling runs over the known values: a NaN index stays
    NaN and never moves the others' scale, and a building without a height has no visual score.
    A NaN counts as 0 only in the weighted sums, after rescaling, where it adds nothing.
    """
    has_heights = _has_heights(buildings_gdf)
    total = pd.Series(0.0, index=buildings_gdf.index)
    for component, weight in components_weights.items():
        if component == "vScore" and not has_heights:
            continue
        # The visual indexes of a building without a height are unknown, whatever the frame holds.
        if component == "vScore":
            rows = buildings_gdf["height"].notna()
        else:
            rows = pd.Series(True, index=buildings_gdf.index)
        indexes = [
            index
            for index in COMPONENT_INDEXES.get(component, ())
            if index in buildings_gdf.columns and buildings_gdf[index].where(rows).notna().any()
        ]
        if not indexes:
            continue
        for index in indexes:
            buildings_gdf[f"{index}_sc"] = scaling_columnDF(
                buildings_gdf[index].where(rows), inverse=index in INVERSE_INDEXES
            )
        if len(COMPONENT_INDEXES[component]) == 1:
            score = buildings_gdf[f"{indexes[0]}_sc"]
        else:
            score = sum(
                buildings_gdf[f"{index}_sc"].fillna(0.0) * indexes_weights[index]
                for index in indexes
            )
        if component == "vScore":
            score = score.where(rows)
        buildings_gdf[f"{component}{suffix}"] = score
        buildings_gdf[f"{component}{suffix}_sc"] = scaling_columnDF(score)
        total = total + buildings_gdf[f"{component}{suffix}_sc"].fillna(0.0) * weight
    return total


def compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights):
    """Compute component and global landmarkness scores.

    Indexes and components are rescaled over the whole city (see ``_component_scores``): a
    building without a height has no visual score and gets nothing from that component.
    """
    buildings_gdf = _with_known_heights(buildings_gdf)

    if not (abs(sum(global_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Global components weights must sum to 1.0")

    gScore = _component_scores(buildings_gdf, global_indexes_weights, global_components_weights)
    buildings_gdf["gScore"] = gScore
    buildings_gdf["gScore_sc"] = scaling_columnDF(gScore)
    return buildings_gdf


def compute_local_scores(
    buildings_gdf, local_indexes_weights, local_components_weights, rescaling_radius=1500
):
    """
    The function computes landmarkness at the local level. The components' weights may be different from the ones used to calculate the
    global score. The radius parameter indicates the extent of the area considered to rescale the landmarkness local score.
    - local_indexes_weights: keys are index names (string), items are weights.
    - local_components_weights: keys are component names (string), items are weights.

    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        The input GeoDataFrame containing buildings information.
    local_indexes_weights: dict
        Dictionary with index names (string) as keys and weights as values.
    local_components_weights: dict
        Dictionary with component names (string) as keys and weights as values.

    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame, with ``lScore`` and ``lScore_sc``. Each building's
        indexes and components are rescaled over its neighbourhood (see ``_component_scores``).

    Examples
    --------
    >>> # local landmarkness indexes weights, cScore and pScore have only 1 index each
    >>> local_indexes_weights = {
    ...     "3dvis": 0.50,
    ...     "fac": 0.30,
    ...     "height": 0.20,
    ...     "area": 0.40,
    ...     "2dvis": 0.00,
    ...     "neigh": 0.30,
    ...     "road": 0.30,
    ... }
    >>> # local landmarkness components weights
    >>> local_components_weights = {"vScore": 0.25, "sScore": 0.35, "cScore": 0.10, "pScore": 0.30}
    """

    # A copy, so the score columns are not added to the caller's frame.
    buildings_gdf = _with_known_heights(buildings_gdf)
    sindex = buildings_gdf.sindex  # spatial index

    # Validate that local_components_weights sum to 1.0
    if not (abs(sum(local_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Local components weights must sum to 1.0")

    buildings_gdf["lScore"] = 0.0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {
            executor.submit(
                _building_local_score,
                row["geometry"],
                idx,
                buildings_gdf,
                sindex,
                local_components_weights,
                local_indexes_weights,
                rescaling_radius,
            ): idx
            for idx, row in buildings_gdf.iterrows()
        }
        for future in concurrent.futures.as_completed(future_scores):
            buildingID = future_scores[future]
            buildings_gdf.loc[buildingID, "lScore"] = future.result()

    buildings_gdf["lScore_sc"] = scaling_columnDF(buildings_gdf["lScore"])
    return buildings_gdf


def _building_local_score(
    building_geometry,
    buildingID,
    buildings_gdf,
    buildings_gdf_sindex,
    local_components_weights,
    local_indexes_weights,
    radius,
):
    """
    The function computes landmarkness at the local level for a single building: its components
    rescaled over the buildings within ``radius`` (see ``_component_scores``).

    Parameters
    ----------
    building_geometry  Polygon
        The geometry of the building.
    buildingID: int
        The ID of the building.
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame containing the buildings.
    buildings_gdf_sindex: Spatial Index
        The spatial index of the buildings GeoDataFrame.
    local_components_weights: dictionary
        The weights assigned to local-level components.
    local_indexes_weights: dictionary
        The weights assigned to local-level indexes.
    radius: float
        The radius that regulates the area around the building within which the scores are recomputed.

    Returns
    -------
    score : float
        The computed local-level landmarkness score for the building.
    """
    buffer = building_geometry.buffer(radius)
    matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    matches = buildings_gdf.iloc[matches_index].copy()
    matches = matches[matches.intersects(buffer)]

    lScore = _component_scores(
        matches, local_indexes_weights, local_components_weights, suffix="_l"
    )
    return lScore.loc[buildingID]


def assert_all_polygons(gdf: gpd.GeoDataFrame):
    """Raise TypeError if a GeoDataFrame contains non-Polygon geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame expected to contain only Shapely Polygon geometries.
    """

    invalid = gdf[~gdf.geometry.apply(lambda g: isinstance(g, (Polygon)))]
    if not invalid.empty:
        raise TypeError(
            f"Found non-polygon geometries: {invalid.geometry.geom_type.unique().tolist()}"
        )
