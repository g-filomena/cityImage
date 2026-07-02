"""Pedestrian network filtering and construction.

This module preserves the cityImage-specific pedestrian filtering semantics
previously embedded in the old OSM loading route, while delegating raw OSM
feature retrieval to OSMnx and generic line-to-network construction to
``cityImage.network``.

The core entry point is ``pedestrian_network_from_osm_features``: pass a
GeoDataFrame of OSM highway features that you downloaded elsewhere, and this
module filters it into a cityImage-style pedestrian network.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import pandas as pd

from .network import network_from_lines

EXCLUDED_HIGHWAY_VALUES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "busway",
    "construction",
    "proposed",
    "raceway",
}

EXCLUDED_ACCESS_VALUES = {
    "private",
    "no",
}

# OSM ``foot=*`` values that grant public pedestrian access. ``official`` is a
# signposted legal designation (a stronger ``designated``). Restricted values
# such as ``permit`` and ``customers`` are deliberately excluded: like
# ``private``, they do not grant general-public access. Note "promiscuous" is not
# an OSM value; the intended concept is ``permissive`` (access tolerated but
# revocable), which is already covered.
PEDESTRIAN_OK_FOOT_VALUES = {
    "yes",
    "designated",
    "permissive",
    "destination",
    "official",
}

# OSM ``foot=*`` values that mean pedestrians should not use this way. Like
# ``foot=no`` they take precedence over a permissive general ``access`` and drop
# the way. ``use_sidepath`` means "walk on the parallel path, not here";
# ``discouraged`` is legal but advised against; ``private`` restricts access.
FOOT_NO_VALUES = {
    "no",
    "use_sidepath",
    "discouraged",
    "private",
}

# OSM-style affirmative spellings. OSM prefers "yes", but sloppy data also uses
# these variants; treat them all as an affirmative value.
OSM_YES_VALUES = {
    "yes",
    "true",
    "1",
    "y",
}

# A sidewalk counts as present when it is tagged affirmatively or with a
# side/position ("separate" means it is mapped as its own way). This also covers
# sub-keys such as ``sidewalk:left=1`` because their tokens are collected too.
SIDEWALK_EVIDENCE_VALUES = OSM_YES_VALUES | {
    "both",
    "left",
    "right",
    "separate",
}

# Highways that are inherently walkable and kept regardless of any evidence.
INHERENTLY_PEDESTRIAN_HIGHWAYS = {
    "footway",
    "pedestrian",
    "path",
    "steps",
    "corridor",
    "track",
    "bridleway",
}

# Pedestrian-priority streets (woonerf-style), kept as walkable regardless of
# evidence.
KEEP_REGARDLESS = {
    "living_street",
}

# Highways kept unconditionally, but whose ``ped`` column records whether there
# is actual pedestrian evidence: "yes" with a usable foot tag or mapped sidewalk,
# otherwise "noEvidence". Covers residential streets and major through-roads,
# where walkability is assumed but uncertain without evidence.
KEEP_WITH_EVIDENCE_FLAG = {
    "residential",
    "primary",
    "secondary",
    "tertiary",
}

# Values written to the ``ped`` column of the resulting edges.
PED_YES = "yes"
PED_NO_EVIDENCE = "noEvidence"

# How to reconcile separately-mapped sidewalks (``highway=footway`` +
# ``footway=sidewalk``) with road centrelines, which can otherwise describe the
# same street twice as parallel edges:
#   - "keep_both":   keep both, no de-duplication (may double-count streets).
#   - "centrelines": one edge per street — drop separately-mapped
#                    ``footway=sidewalk`` ways and keep road centrelines. Suits
#                    Image-of-the-City / centrality analysis.
#   - "sidewalks":   walking-surface network — keep sidewalks/crossings and drop
#                    the centreline of any road that declares ``sidewalk=separate``
#                    (its sidewalk is mapped elsewhere). Suits fine pedestrian
#                    simulation/routing.
SIDEWALK_POLICIES = ("keep_both", "centrelines", "sidewalks")
DEFAULT_SIDEWALK_POLICY = "keep_both"


def _as_tokens(value: Any) -> list[str]:
    """Return a normalised list of lowercase OSM tag tokens."""
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []

    if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_as_tokens(item))
        return tokens

    if pd.isna(value):
        return []

    token = str(value).strip().lower()
    return [token] if token else []


def _first_token(value: Any) -> str | None:
    """Return the first normalised OSM tag token, if available."""
    tokens = _as_tokens(value)
    return tokens[0] if tokens else None


def _truthy_osm_yes(value: Any) -> bool:
    """Return True for OSM-style affirmative values (yes/true/1/y)."""
    return _first_token(value) in OSM_YES_VALUES


def _ensure_columns(gdf: gpd.GeoDataFrame, columns: Iterable[str]) -> gpd.GeoDataFrame:
    """Ensure all requested columns exist, filled with NA values when absent."""
    gdf = gdf.copy()
    for column in columns:
        if column not in gdf.columns:
            gdf[column] = pd.NA
    return gdf


def _line_geometries_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only LineString and MultiLineString geometries."""
    return gdf[gdf.geometry.geom_type.isin({"LineString", "MultiLineString"})].copy()


def _pedestrian_status(
    row: pd.Series, sidewalk_policy: str = DEFAULT_SIDEWALK_POLICY
) -> str | None:
    """Classify an OSM highway row for the pedestrian network.

    Parameters
    ----------
    row
        A row of OSM highway features.
    sidewalk_policy
        How to reconcile separately-mapped sidewalks with road centrelines; one
        of :data:`SIDEWALK_POLICIES`.

    Returns
    -------
    str or None
        - ``"yes"``: confidently walkable — inherently pedestrian, a
          walkable-by-default street, or carrying explicit pedestrian evidence
          (a usable ``foot`` tag or a mapped sidewalk).
        - ``"noEvidence"``: a residential street or major through-road kept
          without pedestrian evidence; walkability is uncertain.
        - ``None``: the row should be dropped from the network.
    """
    highway_tokens = set(_as_tokens(row.get("highway")))
    highway = _first_token(row.get("highway"))

    if not highway_tokens:
        return None

    if highway_tokens & EXCLUDED_HIGHWAY_VALUES:
        return None

    if _truthy_osm_yes(row.get("area")):
        return None

    foot_tokens = set(_as_tokens(row.get("foot")))
    if foot_tokens & FOOT_NO_VALUES:
        return None

    has_explicit_foot_access = bool(foot_tokens & PEDESTRIAN_OK_FOOT_VALUES)

    # A generic access restriction excludes the way unless pedestrians are
    # explicitly allowed via the mode-specific ``foot`` tag, which takes
    # precedence over ``access`` in OSM.
    access_tokens = set(_as_tokens(row.get("access")))
    if access_tokens & EXCLUDED_ACCESS_VALUES and not has_explicit_foot_access:
        return None

    sidewalk_tokens = set()
    for column, value in row.items():
        if str(column).startswith("sidewalk"):
            sidewalk_tokens.update(_as_tokens(value))

    has_sidewalk_evidence = bool(sidewalk_tokens & SIDEWALK_EVIDENCE_VALUES)
    has_evidence = has_explicit_foot_access or has_sidewalk_evidence

    footway_tokens = set(_as_tokens(row.get("footway")))
    is_separately_mapped_sidewalk = "footway" in highway_tokens and "sidewalk" in footway_tokens
    is_inherently_pedestrian = bool(highway_tokens & INHERENTLY_PEDESTRIAN_HIGHWAYS)

    # Sidewalk-vs-centreline de-duplication.
    if sidewalk_policy == "centrelines" and is_separately_mapped_sidewalk:
        # The road centreline already represents this street; drop the parallel
        # sidewalk way.
        return None
    if (
        sidewalk_policy == "sidewalks"
        and "separate" in sidewalk_tokens
        and not is_inherently_pedestrian
        and highway != "cycleway"
    ):
        # The sidewalk is mapped as its own way; drop the redundant road centreline.
        return None

    if highway == "cycleway":
        # A cycleway joins the pedestrian network when pedestrians are allowed
        # (foot tag) or a sidewalk is mapped alongside it; otherwise it is dropped.
        return PED_YES if has_evidence else None

    if is_inherently_pedestrian:
        return PED_YES

    if highway_tokens & KEEP_REGARDLESS:
        return PED_YES

    if highway_tokens & KEEP_WITH_EVIDENCE_FLAG:
        return PED_YES if has_evidence else PED_NO_EVIDENCE

    # Ambiguous ways (service, unclassified, road) and any unrecognised highway
    # type are kept only with explicit pedestrian evidence, otherwise dropped.
    return PED_YES if has_evidence else None


def _is_pedestrian_row(row: pd.Series) -> bool:
    """Return True when an OSM highway row should remain in the pedestrian network."""
    return _pedestrian_status(row) is not None


def filter_pedestrian_osm_features(
    highways_gdf: gpd.GeoDataFrame,
    sidewalk_policy: str = DEFAULT_SIDEWALK_POLICY,
) -> gpd.GeoDataFrame:
    """Filter OSM highway features using cityImage pedestrian-network semantics.

    Parameters
    ----------
    highways_gdf
        GeoDataFrame of OSM features, usually downloaded with an OSMnx
        ``features_from_*``/``geometries_from_*`` call using
        ``tags={"highway": True}``.
    sidewalk_policy
        How to reconcile separately-mapped sidewalks with road centrelines; one
        of :data:`SIDEWALK_POLICIES`. Defaults to ``"keep_both"`` (no
        de-duplication). Use ``"centrelines"`` for one edge per street or
        ``"sidewalks"`` for a walking-surface network.

    Returns
    -------
    geopandas.GeoDataFrame
        Line/MultiLine features that remain after pedestrian filtering, with an
        added ``ped`` column: ``"yes"`` for confidently walkable ways and
        ``"noEvidence"`` for residential/major roads kept without pedestrian
        evidence.
    """
    if not isinstance(highways_gdf, gpd.GeoDataFrame):
        raise TypeError("highways_gdf must be a GeoDataFrame")

    if sidewalk_policy not in SIDEWALK_POLICIES:
        raise ValueError(
            f"sidewalk_policy must be one of {SIDEWALK_POLICIES}, got {sidewalk_policy!r}"
        )

    if highways_gdf.empty:
        return highways_gdf.copy()

    required_filter_columns = [
        "highway",
        "area",
        "foot",
        "footway",
        "cycleway",
        "access",
    ]

    gdf = _ensure_columns(highways_gdf, required_filter_columns)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = _line_geometries_only(gdf)

    if gdf.empty:
        return gdf

    gdf["ped"] = gdf.apply(_pedestrian_status, axis=1, sidewalk_policy=sidewalk_policy)
    return gdf[gdf["ped"].notna()].copy()


def pedestrian_network_from_osm_features(
    highways_gdf: gpd.GeoDataFrame,
    crs: Any,
    sidewalk_policy: str = DEFAULT_SIDEWALK_POLICY,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a cityImage pedestrian network from already-downloaded OSM features.

    Raw OSM download is deliberately not owned by this function. Use OSMnx or
    another data source to retrieve highway features, then pass them here.

    ``sidewalk_policy`` selects how separately-mapped sidewalks and road
    centrelines are reconciled; see :func:`filter_pedestrian_osm_features`.
    """
    filtered = filter_pedestrian_osm_features(highways_gdf, sidewalk_policy=sidewalk_policy)

    if filtered.empty:
        empty_edges = gpd.GeoDataFrame(columns=["edgeID", "u", "v", "length"], geometry=[], crs=crs)
        empty_nodes = gpd.GeoDataFrame(columns=["nodeID", "x", "y"], geometry=[], crs=crs)
        return empty_nodes, empty_edges

    metadata_columns = [
        "name",
        "highway",
        "ped",
        "lit",
        "foot",
        "footway",
        "cycleway",
        "access",
        "surface",
        "width",
    ]
    metadata_columns.extend(
        column for column in filtered.columns if str(column).startswith("sidewalk")
    )

    other_columns = []
    seen = set()
    for column in metadata_columns:
        if column in seen or column not in filtered.columns:
            continue
        # Skip metadata columns that are entirely empty (e.g. tag columns
        # back-filled with NA when the source lacked them), so the edges do not
        # carry all-null passthrough columns.
        if filtered[column].isna().all():
            continue
        other_columns.append(column)
        seen.add(column)

    return network_from_lines(filtered, crs, other_columns=other_columns)


def _call_osmnx_features(ox: Any, method_name: str, *args: Any, **kwargs: Any) -> gpd.GeoDataFrame:
    """Call modern OSMnx feature functions with fallback for older versions."""
    modern = getattr(ox, f"features_from_{method_name}", None)
    if modern is not None:
        return modern(*args, **kwargs)

    legacy = getattr(ox, f"geometries_from_{method_name}", None)
    if legacy is not None:
        return legacy(*args, **kwargs)

    raise ImportError(
        "Installed OSMnx version does not expose features_from_* or geometries_from_* APIs"
    )


def pedestrian_network_from_osm(
    query: Any = None,
    *,
    crs: Any,
    download_method: str = "OSMplace",
    distance: float = 500,
    address: str | None = None,
    point: tuple[float, float] | None = None,
    polygon: Any = None,
    sidewalk_policy: str = DEFAULT_SIDEWALK_POLICY,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download OSM highway features with OSMnx and build a pedestrian network.

    This is a thin optional convenience wrapper. The cityImage-specific part is
    the pedestrian filtering; OSM acquisition remains delegated to OSMnx.

    Parameters
    ----------
    query
        Place name for ``download_method="OSMplace"``. For address/point/polygon
        methods, prefer the explicit keyword arguments.
    crs
        Target projected CRS for the output network.
    download_method
        One of ``"OSMplace"``, ``"distance_from_address"``,
        ``"distance_from_point"``, or ``"polygon"``.
    distance
        Distance in metres for address/point downloads.
    address, point, polygon
        Explicit spatial inputs for the corresponding download methods.
    sidewalk_policy
        How to reconcile separately-mapped sidewalks with road centrelines; see
        :func:`filter_pedestrian_osm_features`.
    """
    import osmnx as ox

    tags = {"highway": True}

    if download_method == "OSMplace":
        if query is None:
            raise ValueError("query must be provided when download_method='OSMplace'")
        features = _call_osmnx_features(ox, "place", query, tags=tags)
    elif download_method == "distance_from_address":
        address_query = address or query
        if address_query is None:
            raise ValueError(
                "address or query must be provided when download_method='distance_from_address'"
            )
        features = _call_osmnx_features(ox, "address", address_query, tags=tags, dist=distance)
    elif download_method == "distance_from_point":
        point_query = point or query
        if point_query is None:
            raise ValueError(
                "point or query must be provided when download_method='distance_from_point'"
            )
        features = _call_osmnx_features(ox, "point", point_query, tags=tags, dist=distance)
    elif download_method == "polygon":
        polygon_query = polygon or query
        if polygon_query is None:
            raise ValueError("polygon or query must be provided when download_method='polygon'")
        features = _call_osmnx_features(ox, "polygon", polygon_query, tags=tags)
    else:
        raise ValueError(
            "download_method must be one of: OSMplace, distance_from_address, "
            "distance_from_point, polygon"
        )

    return pedestrian_network_from_osm_features(features, crs, sidewalk_policy=sidewalk_policy)
