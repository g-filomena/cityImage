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

from .data_utils import scaling_columnDF

pd.set_option("display.precision", 3)


def structural_score(
    buildings_gdf,
    obstructions_gdf,
    edges_gdf,
    advance_vis_expansion_distance=300,
    neighbours_radius=150,
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

    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """
    from .visibility2d import visibility_polygon2d

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
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(
        lambda row: visibility_polygon2d(
            row, obstructions_gdf, sindex, max_expansion_distance=advance_vis_expansion_distance
        )
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
    - fac: approximate facade area, computed whenever height is available;
    - 3dvis: 3D visibility score, derived from sight-line lengths when provided.
    """
    if sight_lines is None:
        sight_lines = pd.DataFrame()

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0

    has_height = "height" in buildings_gdf.columns
    if has_height and not buildings_gdf.empty:
        buildings_gdf["fac"] = buildings_gdf.apply(
            lambda row: _facade_area(row["geometry"], row["height"])
            if pd.notnull(row["height"])
            else 0.0,
            axis=1,
        )

    if not has_height or sight_lines.empty:
        buildings_gdf["3dvis"] = 0.0
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

    stats.reset_index(inplace=True)
    buildings_gdf = pd.merge(
        buildings_gdf,
        stats[["buildingID", "3dvis"]],
        on="buildingID",
        how="left",
    )
    buildings_gdf["3dvis"] = buildings_gdf["3dvis"].where(
        pd.notnull(buildings_gdf["3dvis"]),
        0.0,
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
    """Compute a cultural landmark component per building."""
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["cult"] = 0.0

    if from_OSM:
        if "historic" not in buildings_gdf.columns:
            raise ValueError("from_OSM=True requires buildings_gdf to contain a 'historic' column")
        buildings_gdf["cult"] = (
            buildings_gdf["historic"].apply(_is_historic).astype("int8").astype(float)
        )
        return buildings_gdf

    if historic_elements_gdf is None or len(historic_elements_gdf) == 0:
        return buildings_gdf

    if buildings_gdf.crs != historic_elements_gdf.crs:
        raise ValueError(
            "CRS mismatch: buildings_gdf and historic_elements_gdf must have the same CRS"
        )

    left = buildings_gdf[["geometry"]].copy()
    right_cols = ["geometry"]

    if score_column is not None:
        if score_column not in historic_elements_gdf.columns:
            raise ValueError(f"score_column '{score_column}' not found in historic_elements_gdf")
        right_cols.append(score_column)

    right = historic_elements_gdf[right_cols].copy()

    left = left[left.geometry.notna()].copy()
    right = right[right.geometry.notna()].copy()

    if left.empty or right.empty:
        return buildings_gdf

    try:
        joined = gpd.sjoin(left, right, how="inner", predicate="intersects")
    except TypeError:
        joined = gpd.sjoin(left, right, how="inner", op="intersects")

    if joined.empty:
        return buildings_gdf

    if score_column is None:
        counts = joined.groupby(joined.index).size().astype(float)
        buildings_gdf["cult"] = counts.reindex(buildings_gdf.index, fill_value=0.0)
        return buildings_gdf

    vals = pd.to_numeric(joined[score_column], errors="coerce").fillna(0.0)
    sums = vals.groupby(joined.index).sum().astype(float)
    buildings_gdf["cult"] = sums.reindex(buildings_gdf.index, fill_value=0.0)
    return buildings_gdf


def pragmatic_score(
    buildings_gdf,
    land_uses_column: str = "land_uses",
    overlaps_column: str = "land_uses_overlap",
    search_radius: float = 200,
    default_land_use: str = "unclassified",
):
    """Compute a pragmatic landmark component from semantic land-use labels.

    Missing or empty land-use labels are treated as ``default_land_use`` and
    aligned with a full overlap weight of ``[1.0]``.
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

    gdf[land_uses_column] = gdf[land_uses_column].apply(
        lambda v: _as_list(v) or [default_land_use]
    )

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
        if not candidate_idx:
            return 0.0

        possible = gdf_exploded.iloc[candidate_idx]
        matches = possible[possible.intersects(buf)]
        matches = matches[matches["_ci_row_id"] != row_id]

        if matches.empty:
            return 0.0

        total_w = float(matches["_w"].sum())
        if total_w <= 0:
            return 0.0

        Nj_w = float(matches.loc[matches[land_uses_column] == building_label, "_w"].sum())
        return 1.0 - (Nj_w / total_w)

    gdf_exploded["prag_temp"] = gdf_exploded.apply(
        lambda row: _unexpectedness(row["_ci_row_id"], row.geometry, row[land_uses_column]),
        axis=1,
    )

    scores = gdf_exploded.groupby("_ci_row_id")["prag_temp"].max()
    gdf["prag"] = gdf["_ci_row_id"].map(scores).fillna(0.0).astype(float)

    return gdf.drop(columns=["_ci_row_id", "_w_list", "_lu_w"], errors="ignore")


def compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights):
    """Compute component and global landmarkness scores."""
    buildings_gdf = buildings_gdf.copy()

    cols = {
        "direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"],
        "inverse": ["neigh", "road"],
    }

    if not (abs(sum(global_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Global components weights must sum to 1.0")

    compute_vScore = (
        "vScore" in global_components_weights
        and "height" in buildings_gdf.columns
        and buildings_gdf["height"].max() > 0.0
    )

    for col in cols["direct"] + cols["inverse"]:
        if col in buildings_gdf.columns:
            buildings_gdf[col + "_sc"] = scaling_columnDF(
                buildings_gdf[col],
                inverse=(col in cols["inverse"]),
            )

    if compute_vScore:
        buildings_gdf["vScore"] = sum(
            buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
            for col in ["fac", "height", "3dvis"]
            if f"{col}_sc" in buildings_gdf
        )
        buildings_gdf["vScore_sc"] = scaling_columnDF(buildings_gdf["vScore"])

    buildings_gdf["sScore"] = sum(
        buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
        for col in ["area", "neigh", "2dvis", "road"]
        if f"{col}_sc" in buildings_gdf
    )
    buildings_gdf["sScore_sc"] = scaling_columnDF(buildings_gdf["sScore"])

    buildings_gdf["cScore"] = (
        buildings_gdf["cult_sc"] if "cult_sc" in buildings_gdf.columns else 0.0
    )
    buildings_gdf["pScore"] = (
        buildings_gdf["prag_sc"] if "prag_sc" in buildings_gdf.columns else 0.0
    )

    if "cult_sc" in buildings_gdf.columns:
        buildings_gdf["cScore_sc"] = buildings_gdf["cult_sc"]
    if "prag_sc" in buildings_gdf.columns:
        buildings_gdf["pScore_sc"] = buildings_gdf["prag_sc"]

    buildings_gdf["gScore"] = sum(
        buildings_gdf[f"{component}_sc"] * global_components_weights[component]
        for component in global_components_weights
        if f"{component}_sc" in buildings_gdf and (component != "vScore" or compute_vScore)
    )
    buildings_gdf["gScore_sc"] = scaling_columnDF(buildings_gdf["gScore"])
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
        The updated buildings GeoDataFrame.

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

    sindex = buildings_gdf.sindex  # spatial index

    # Validate that local_components_weights sum to 1.0
    if not (abs(sum(local_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Local components weights must sum to 1.0")

    # Initialize scores conditionally
    compute_vScore = (
        "vScore" in local_components_weights
        and "height" in buildings_gdf.columns
        and buildings_gdf["height"].max() > 0.0
    )
    if compute_vScore:
        buildings_gdf["vScore_l"] = 0.0  # Initialize only if valid height data exists

    buildings_gdf["sScore_l"] = 0.0
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
    The function computes landmarkness at the local level for a single building.

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

    cols = {
        "direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"],
        "inverse": ["neigh", "road"],
    }

    buffer = building_geometry.buffer(radius)
    matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    matches = buildings_gdf.iloc[matches_index].copy()
    matches = matches[matches.intersects(buffer)]

    # Rescale all values dynamically if the column exists in matches
    for column in cols["direct"] + cols["inverse"]:
        if column in matches.columns:
            matches[column + "_sc"] = scaling_columnDF(
                matches[column], inverse=(column in cols["inverse"])
            )

    # Compute structural score (sScore)
    if "sScore" in local_components_weights:
        matches["sScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col]
            for col in ["area", "2dvis", "neigh", "road"]
            if f"{col}_sc" in matches
        )

    # Recomputing visual scores only if "height" is valid
    # Determine if vScore should be computed
    compute_vScore = (
        "vScore" in local_components_weights
        and "height" in matches.columns
        and matches["height"].max() > 0.0
    )

    if compute_vScore:
        matches["vScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col]
            for col in ["fac", "height", "3dvis"]
            if f"{col}_sc" in matches
        )

    # Compute cultural and pragmatic scores if defined
    if "cScore" in local_components_weights and "cult_sc" in matches.columns:
        matches["cScore_l"] = matches["cult_sc"]
    if "pScore" in local_components_weights and "prag_sc" in matches.columns:
        matches["pScore_l"] = matches["prag_sc"]

    # Rescale component scores dynamically
    for component in local_components_weights:
        if f"{component}_l" in matches and (component != "vScore" or compute_vScore):
            matches[f"{component}_l_sc"] = scaling_columnDF(matches[f"{component}_l"])

    # Compute the final local score
    matches["lScore"] = sum(
        matches[f"{component}_l_sc"] * local_components_weights[component]
        for component in local_components_weights
        if f"{component}_l_sc" in matches and (component != "vScore" or compute_vScore)
    )

    # Return the local score for the specified building
    return round(matches.loc[buildingID, "lScore"], 3)


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