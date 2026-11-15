"""One call from a place, or from your own layers, to the five elements of the Image of the City.

``image_of_the_city`` follows Filomena, Verstegen & Manley (2019), *A computational approach to
'The Image of the City'*, Cities 89, 14-25, with the steps of the example notebooks; each step is an
existing cityImage function, so the pipeline adds no method of its own:

- **nodes**: betweenness centrality of the street nodes (``Bc``), on the primal graph weighted by
  length (``network_from_osm``, ``clean_network``, ``calculate_centrality``);
- **paths**: angular betweenness of the streets (``Ab``): betweenness of the dual graph, whose
  links are weighted by the angle between two streets;
- **districts**: modularity partitions of the dual graph of the drive network, weighted by angle
  (``identify_regions``), as polygons (``polygonise_partitions``) and on the street nodes, with
  their gateways (``amend_nodes_membership``, ``find_gateways``);
- **edges**: the barriers (``barriers_from_osm``), and the streets crossing them
  (``assign_structuring_barriers``);
- **landmarks**: the global landmark score of the buildings (``score_cityimage_buildings``).

Every layer keeps all its features with their scores. As in the paper, the scores of nodes, paths
and landmarks are rescaled to 0-1 over the study area and ranked in Lynch's ranges (``lynch_bins``:
0.125, 0.25, 0.5, 0.75): ``lynch_class`` is 1 for the strongest elements (over 0.75), then 2,
3 and 4 (0.125 to 0.25), and 0 below the lowest range, so that ``lynch_class.isin([1, 2])`` keeps
only the strongest elements. Districts and edges are not ranked.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .barriers import assign_structuring_barriers
from .buildings import known_heights
from .centrality import calculate_centrality
from .data_utils import scaling_columnDF
from .graph import dual_gdf, dual_graph_fromGDF, graph_fromGDF
from .network_topology import clean_network
from .regions import (
    INVALID_DISTRICT,
    amend_nodes_membership,
    district_to_nodes_from_polygons,
    find_gateways,
    identify_regions,
    polygonise_partitions,
)
from .schema import validate_buildings_gdf, validate_edges_gdf, validate_nodes_gdf
from .scoring import LandmarkScoringConfig, score_cityimage_buildings

ELEMENTS = ("paths", "nodes", "edges", "districts", "landmarks")
# Lynch's ranges, the share of interviewees mentioning an element (12.5-25%, 25-50%, 50-75%, over
# 75%), applied to the rescaled scores as in Filomena, Verstegen & Manley (2019).
LYNCH_BINS = (0.125, 0.25, 0.5, 0.75)


def _default_clean_kwargs() -> dict[str, Any]:
    return {
        "dead_ends": False,
        "remove_islands": True,
        "same_vertexes_edges": True,
        "self_loops": True,
    }


def _default_structural_kwargs() -> dict[str, Any]:
    return {"advance_vis_expansion_distance": 300, "neighbours_radius": 150}


@dataclass(frozen=True)
class CityImageConfig:
    """Parameters of ``image_of_the_city``; the defaults follow the 2019 paper and the notebooks.

    Attributes
    ----------
    network_type
        OSMnx network type of the street network the paths and nodes are computed on.
    clean_kwargs
        Keyword arguments of ``clean_network``, for downloaded and given networks alike. Dead
        ends are kept: removing them cascades (a street left without exit is removed in turn),
        which on a small or clipped area removes most of the network. A dead end has low
        betweenness, so it is rarely a path or a node.
    lynch_bins
        Increasing thresholds in (0, 1] on the rescaled scores of nodes, paths and landmarks:
        ``lynch_class`` is 1 for a score reaching the highest, 2 for one reaching only the one
        below, and so on; 0 for a score below the lowest.
    straightness
        Also compute the straightness centrality of the nodes (``Sc``). It holds every
        shortest-path distance in memory: on a large city it is slow and memory-hungry.
    district_network
        ``"drive"`` partitions the drive network (downloaded from the same place) into districts,
        as the paper partitions a wider network for districts; ``"same"`` partitions the street network of the paths. Without a place
        to download from, the street network of the paths is used.
    district_weight
        Dual-graph weight of the partition: ``"rad"`` (angular change, as in the paper),
        ``"length"`` or ``None`` (topological).
    min_district_size
        Partitions with this many streets or fewer are left out, and their nodes join a
        neighbouring district.
    random_state
        Seed of the modularity partition; ``None`` draws a different partition on each call.
    include_primary_barriers, include_secondary_barriers, parks_min_area
        Passed to ``barriers_from_osm``.
    buildings_min_area
        Buildings smaller than this (m²) are not downloaded (``buildings_from_osm``).
    landmark_scoring
        Weights and radii of the landmark scores; the defaults are those of the paper (Table 1).
    structural_kwargs
        Keyword arguments of ``structural_score``: neighbours within 150 m as in the paper.
        ``workers`` (e.g. 4) computes the 2D advance visibility in that many processes, several
        times faster on thousands of buildings. The processes start with "spawn": in a script,
        run the call under ``if __name__ == "__main__":`` (not needed in a notebook).
    """

    network_type: str = "walk"
    clean_kwargs: Mapping[str, Any] = field(default_factory=_default_clean_kwargs)
    lynch_bins: tuple[float, ...] = LYNCH_BINS
    straightness: bool = False
    district_network: str = "drive"
    district_weight: str | None = "rad"
    min_district_size: int = 10
    random_state: int | None = 0
    include_primary_barriers: bool = True
    include_secondary_barriers: bool = False
    parks_min_area: float = 100000
    buildings_min_area: float | None = 200
    landmark_scoring: LandmarkScoringConfig = field(default_factory=LandmarkScoringConfig)
    structural_kwargs: Mapping[str, Any] = field(default_factory=_default_structural_kwargs)

    def __post_init__(self) -> None:
        bins = list(self.lynch_bins)
        if not bins or bins != sorted(set(bins)) or not 0 < bins[0] <= bins[-1] <= 1:
            raise ValueError("lynch_bins must be increasing thresholds in (0, 1]")
        if self.district_network not in {"drive", "same"}:
            raise ValueError("district_network must be 'drive' or 'same'")


@dataclass
class CityImage:
    """The five elements of the Image of the City, with the configuration that produced them.

    ``paths`` are the streets and ``nodes`` the street nodes, all of them, with their centrality
    (``Ab``, ``Bc``), its rescaled value (``Ab_sc``, ``Bc_sc``) and ``lynch_class``; ``edges``
    are the barriers; ``districts`` the district polygons; ``landmarks`` the buildings with their
    scores (``gScore``, ``gScore_sc``) and ``lynch_class``. An element that was not requested, or
    could not be computed, is ``None``.
    """

    paths: gpd.GeoDataFrame | None
    nodes: gpd.GeoDataFrame | None
    edges: gpd.GeoDataFrame | None
    districts: gpd.GeoDataFrame | None
    landmarks: gpd.GeoDataFrame | None
    config: CityImageConfig
    metadata: dict[str, Any]

    def layers(self) -> dict[str, gpd.GeoDataFrame]:
        """The computed elements, by name."""
        return {
            name: layer
            for name in ELEMENTS
            if (layer := getattr(self, name)) is not None and not layer.empty
        }

    def to_file(self, path: str) -> None:
        """Write every computed element to a GeoPackage, one layer each (``paths``, ...).

        List-like values (e.g. a building's land uses) are written as comma-separated text.
        """
        for name, layer in self.layers().items():
            _gpkg_ready(layer).to_file(path, layer=name, driver="GPKG")

    def plot(self, **kwargs: Any) -> Any:
        """Map the five elements on one figure; needs the ``plot`` extra.

        Keyword arguments go to ``cityImage.plotting.plot_city_image``. Returns the figure.
        """
        from .plotting import plot_city_image

        return plot_city_image(self, **kwargs)

    def explore(self, **kwargs: Any) -> Any:
        """An interactive web map of the five elements, one layer each; needs the
        ``interactive`` extra.

        Keyword arguments go to ``cityImage.interactive.explore_city_image``. Returns the
        ``folium.Map``, shown inline in a notebook; ``.save("map.html")`` writes it to a file.
        """
        from .interactive import explore_city_image

        return explore_city_image(self, **kwargs)


def image_of_the_city(
    place: Any = None,
    *,
    crs: Any = None,
    download_method: str = "OSMplace",
    distance: float | None = None,
    nodes: gpd.GeoDataFrame | None = None,
    edges: gpd.GeoDataFrame | None = None,
    buildings: gpd.GeoDataFrame | None = None,
    barriers: gpd.GeoDataFrame | None = None,
    elements: Iterable[str] = ELEMENTS,
    config: CityImageConfig | None = None,
) -> CityImage:
    """Compute the computational Image of the City of a place, or of your own layers.

    Parameters
    ----------
    place
        What to download from OpenStreetMap: a place name, an address, a ``(lat, lon)`` point or
        a polygon, as ``download_method`` says (see ``network_from_osm``). Layers given as
        GeoDataFrames are not downloaded.
    crs
        Projected CRS of the result. ``None`` takes the local UTM zone of the downloaded network,
        or the CRS of ``edges``.
    download_method, distance
        As in ``network_from_osm``.
    nodes, edges
        A street network in cityImage schema (e.g. from ``network_from_file``), instead of
        downloading one. It is cleaned with ``config.clean_kwargs``.
    buildings, barriers
        Buildings and barriers in cityImage schema, instead of downloading them.
    elements
        The elements to compute, any of ``"paths"``, ``"nodes"``, ``"edges"``, ``"districts"``,
        ``"landmarks"``.
    config
        Parameters of each step; see ``CityImageConfig``.

    Returns
    -------
    CityImage
        The elements, with the configuration and the metadata (place, CRS, cityImage version,
        date) that make the result reproducible.

    Warns
    -----
    UserWarning
        When an element cannot be computed (no buildings, no barriers, no districts) or a part of
        it is left out (no building heights: no visual landmark component).
    """
    config = config or CityImageConfig()
    requested = _requested_elements(elements)
    if (nodes is None) != (edges is None):
        raise ValueError("nodes and edges must be given together")
    if place is None and nodes is None:
        raise ValueError("give a place to download, or nodes and edges")

    from . import __version__

    metadata: dict[str, Any] = {
        "place": place if isinstance(place, str) else (None if place is None else repr(place)),
        "download_method": download_method if place is not None else None,
        "distance": distance,
        "elements": list(requested),
        "cityImage_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if nodes is None or edges is None:  # both, checked above
        from .osm import network_from_osm

        nodes, edges = network_from_osm(
            place,
            download_method=download_method,
            network_type=config.network_type,
            crs=crs,
            distance=distance,
        )
        metadata["network"] = f"OSM ({config.network_type})"
    else:
        _require_schema(validate_nodes_gdf(nodes), "nodes")
        _require_schema(validate_edges_gdf(edges), "edges")
        if crs is not None:
            nodes, edges = nodes.to_crs(crs), edges.to_crs(crs)
        metadata["network"] = "given"
    if edges.empty:
        raise ValueError("the street network is empty")
    nodes, edges = clean_network(nodes, edges, **dict(config.clean_kwargs))
    crs = edges.crs
    metadata["crs"] = crs.to_string() if crs is not None else None

    # Nodes: betweenness on the primal graph. Paths: angular betweenness on the dual graph.
    graph = graph_fromGDF(nodes, edges)
    nodes = nodes.copy()
    nodes["Bc"] = nodes["nodeID"].map(calculate_centrality(graph, "betweenness", weight="length"))
    if config.straightness:
        sc = calculate_centrality(graph, "straightness", weight="length")
        nodes["Sc"] = nodes["nodeID"].map(sc)
    nodes = _rank(nodes, "Bc", config.lynch_bins)
    nodes_dual, edges_dual = dual_gdf(nodes, edges, crs)
    dual_graph = dual_graph_fromGDF(nodes_dual, edges_dual)
    edges = edges.copy()
    edges["Ab"] = edges["edgeID"].map(calculate_centrality(dual_graph, "betweenness", weight="rad"))
    edges = _rank(edges, "Ab", config.lynch_bins)

    result_edges = None
    if "edges" in requested:
        if barriers is None and place is not None:
            from .osm import barriers_from_osm

            barriers = barriers_from_osm(
                place,
                download_method=download_method,
                crs=crs,
                distance=distance,
                include_primary=config.include_primary_barriers,
                include_secondary=config.include_secondary_barriers,
                parks_min_area=config.parks_min_area,
            )
        if barriers is None or barriers.empty:
            _warn("edges", "no barriers were found or given")
        else:
            result_edges = _barriers_within(barriers.to_crs(crs), edges)
            edges = assign_structuring_barriers(edges, result_edges)

    districts = None
    if "districts" in requested:
        district_edges = edges
        district_graph = dual_graph
        if config.district_network == "drive" and place is not None:
            district_edges, district_graph = _drive_dual_graph(
                place, download_method, distance, crs, config
            )
            metadata["district_network"] = "OSM (drive)"
        else:
            metadata["district_network"] = "paths network"
        nodes, districts = _districts(nodes, edges, district_edges, district_graph, config)

    landmarks = None
    if "landmarks" in requested:
        if buildings is None and place is not None:
            from .osm import buildings_from_osm

            kwargs: dict[str, Any] = {"min_area": config.buildings_min_area}
            if distance is not None:
                kwargs["distance"] = distance
            buildings = buildings_from_osm(
                place, download_method=download_method, crs=crs, **kwargs
            )
        if buildings is None or buildings.empty:
            _warn("landmarks", "no buildings were found or given")
        else:
            _require_schema(validate_buildings_gdf(buildings), "buildings")
            landmarks = _landmarks(buildings.to_crs(crs), edges, config)

    return CityImage(
        paths=edges if "paths" in requested else None,
        nodes=nodes if "nodes" in requested else None,
        edges=result_edges,
        districts=districts,
        landmarks=landmarks,
        config=config,
        metadata=metadata,
    )


def _requested_elements(elements: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(elements))
    unknown = set(requested) - set(ELEMENTS)
    if unknown:
        raise ValueError(f"unknown elements {sorted(unknown)}; choose from {list(ELEMENTS)}")
    return requested


def _require_schema(report: Any, name: str) -> None:
    if not report.ok:
        missing = ", ".join(report.missing_columns)
        raise ValueError(f"{name} does not match the cityImage schema; missing: {missing}")


def _warn(element: str, reason: str) -> None:
    warnings.warn(f"{element} not computed: {reason}", UserWarning, stacklevel=3)


def _rank(gdf: gpd.GeoDataFrame, score: str, bins: tuple[float, ...]) -> gpd.GeoDataFrame:
    """Rescale ``score`` to 0-1 (``<score>_sc``) and rank it in Lynch's ranges.

    ``lynch_class`` is 1 for a rescaled score reaching the highest of ``bins``, counting up as
    the ranges go down, and 0 for a score below the lowest (or NaN).
    """
    gdf = gdf.copy()
    gdf[f"{score}_sc"] = scaling_columnDF(gdf[score])
    gdf["lynch_class"] = lynch_classes(gdf[f"{score}_sc"], bins)
    return gdf


def lynch_classes(scaled: Any, bins: tuple[float, ...] = LYNCH_BINS) -> np.ndarray:
    """The Lynch class of each rescaled (0-1) score: 1 for a score reaching the highest of
    ``bins``, counting up as the ranges go down, and 0 for a score below the lowest (or NaN)."""
    reached = np.digitize(pd.Series(scaled, dtype=float).fillna(-1.0).to_numpy(), bins)
    return np.where(reached > 0, len(bins) + 1 - reached, 0).astype(int)


def _barriers_within(barriers: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The barriers clipped to the study area, with a ``barrierID``.

    A river or a railway runs on far beyond the study area; only its stretch within it is an
    edge of this image. The study area is the concave hull of the street network, 50 m wider,
    and a barrier crossing its outline is cut there.
    """
    area = shapely.concave_hull(edges.union_all(), ratio=0.3).buffer(50.0)
    within = gpd.clip(barriers, area)
    within = within[~within.geometry.is_empty].copy()
    if "barrierID" not in within.columns:
        within = within.reset_index(drop=True)
        within["barrierID"] = within.index.astype(int)
    return within


def _drive_dual_graph(
    place: Any, download_method: str, distance: float | None, crs: Any, config: CityImageConfig
) -> tuple[gpd.GeoDataFrame, Any]:
    from .osm import network_from_osm

    nodes, edges = network_from_osm(
        place, download_method=download_method, network_type="drive", crs=crs, distance=distance
    )
    nodes, edges = clean_network(nodes, edges, **dict(config.clean_kwargs))
    nodes_dual, edges_dual = dual_gdf(nodes, edges, crs)
    return edges, dual_graph_fromGDF(nodes_dual, edges_dual)


def _districts(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    district_edges: gpd.GeoDataFrame,
    district_graph: Any,
    config: CityImageConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    """Partition ``district_edges``, polygonise the partitions and assign them to ``nodes``."""
    weight = config.district_weight
    column = f"p_{weight or 'topo'}"
    partitioned = identify_regions(
        district_graph, district_edges, weight=weight, random_state=config.random_state
    )
    sizes = partitioned[column].value_counts()
    kept = partitioned[
        partitioned[column].isin(sizes[sizes > config.min_district_size].index)
        & (partitioned[column] != INVALID_DISTRICT)
    ]
    if kept.empty:
        _warn("districts", f"no partition has more than {config.min_district_size} streets")
        return nodes, None

    polygons = polygonise_partitions(kept, column)
    nodes = district_to_nodes_from_polygons(nodes, polygons, column)
    nodes = amend_nodes_membership(nodes, edges, column, config.min_district_size)
    nodes = find_gateways(nodes, edges, column)
    nodes = nodes.rename(columns={column: "district"})
    polygons = polygons.rename(columns={column: "district"})
    polygons["district"] = polygons["district"].astype(int)
    # Districts whose nodes all joined a neighbour (too small, or cut off) are left out.
    polygons = polygons[polygons["district"].isin(nodes["district"])]
    return nodes, polygons.reset_index(drop=True)


def _landmarks(
    buildings: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, config: CityImageConfig
) -> gpd.GeoDataFrame:
    """Global landmark scores of ``buildings``, ranked in Lynch's ranges."""
    if "height" not in buildings.columns or known_heights(buildings["height"]).isna().all():
        warnings.warn(
            "landmarks: no building has a height, so the visual component is left out of the "
            "landmark score; give buildings with a 'height' column to include it",
            UserWarning,
            stacklevel=3,
        )
    cultural_from_osm = "historic" in buildings.columns
    scored = score_cityimage_buildings(
        buildings,
        config=config.landmark_scoring,
        compute_global=True,
        compute_local=False,
        edges_gdf=edges,
        obstructions_gdf=buildings,
        compute_structural=True,
        cultural_from_osm=cultural_from_osm,
        structural_kwargs=config.structural_kwargs,
    )
    return _rank(scored, "gScore", config.lynch_bins)


def _gpkg_ready(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A copy whose list-like and dict values are text, which a GeoPackage can store."""
    out = gdf.copy()
    for column in out.columns:
        if column == out.geometry.name or out[column].dtype != object:
            continue
        out[column] = out[column].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple, set)) else x
        )
        out[column] = out[column].apply(lambda x: str(x) if isinstance(x, dict) else x)
    return out
