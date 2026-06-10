"""District/region and gateway semantics.

This module keeps the cityImage-specific district/gateway semantics while
making community detection an explicitly delegated operation.

Preferred delegated workflow:

```python
partition = community.best_partition(dual_graph, weight="topo")
regions = ci.regions_from_dual_partition(partition, dual_graph, edges_gdf, column="p_topo")
```

Convenience wrappers ``identify_regions`` and ``identify_regions_primal`` are
retained, but they lazy-import ``python-louvain``/``community`` and only wrap its
``best_partition`` result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import polygonize_full, unary_union

pd.set_option("display.precision", 3)

INVALID_DISTRICT = 999999


def _best_partition(graph: Any, weight: str) -> dict[Any, int]:
    """Delegate modularity/community detection to python-louvain."""
    try:
        import community
    except ImportError as exc:
        raise ImportError(
            "Region identification requires the optional 'community' dependency "
            "from python-louvain. Install the regions extra or pass a precomputed "
            "partition to regions_from_dual_partition/regions_from_primal_partition."
        ) from exc

    return community.best_partition(graph, weight=weight)


def _graph_from_gdfs(nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame, node_id_column: str = "nodeID"):
    """Build a NetworkX graph using cityImage graph semantics, imported lazily."""
    from .graph import graph_fromGDF

    return graph_fromGDF(nodes_gdf, edges_gdf, node_id_column)


def _dual_id_dict(partition: Mapping[Any, int], dual_graph: Any, attribute: str) -> dict[Any, int]:
    """Map a dual-graph node partition to a primal edge attribute value."""
    regions: dict[Any, int] = {}
    for node, district in partition.items():
        regions[dual_graph.nodes[node][attribute]] = district
    return regions


def _min_distance_geometry_gdf(geometry: Any, gdf: gpd.GeoDataFrame) -> tuple[float, Any]:
    """Return minimum distance and index label of the nearest geometry row."""
    if gdf.empty:
        raise ValueError("Cannot calculate nearest geometry from an empty GeoDataFrame")

    distances = gdf.geometry.distance(geometry)
    index = distances.idxmin()
    return float(distances.loc[index]), index


def _geometry_union(geometries: gpd.GeoSeries) -> Any:
    """Union geometries with GeoPandas/Shapely compatibility."""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return unary_union(list(geometries))


def regions_from_dual_partition(
    partition: Mapping[Any, int],
    dual_graph: Any,
    edges_gdf: gpd.GeoDataFrame,
    *,
    column: str = "p_topo",
) -> gpd.GeoDataFrame:
    """Assign a precomputed dual-graph partition to primal street segments.

    Parameters
    ----------
    partition
        Mapping from dual-graph node IDs to district/community IDs.
    dual_graph
        Dual graph whose nodes expose an ``edgeID`` attribute.
    edges_gdf
        Primal street-segment GeoDataFrame containing an ``edgeID`` column.
    column
        Output district column name.
    """
    edge_partition = _dual_id_dict(partition, dual_graph, "edgeID")
    regions = edges_gdf.copy()
    regions[column] = regions["edgeID"].map(edge_partition)
    return regions


def regions_from_primal_partition(
    partition: Mapping[Any, int],
    nodes_gdf: gpd.GeoDataFrame,
    *,
    column: str = "p_topo",
) -> gpd.GeoDataFrame:
    """Assign a precomputed primal-graph partition to street nodes."""
    regions = nodes_gdf.copy()
    regions[column] = regions["nodeID"].map(partition)
    return regions


def identify_regions(dual_graph: Any, edges_gdf: gpd.GeoDataFrame, weight: str | None = None) -> gpd.GeoDataFrame:
    """Identify edge-based regions using delegated python-louvain partitioning.

    This convenience function preserves the old cityImage behaviour but the
    community-detection algorithm itself is delegated to python-louvain.
    """
    if weight is None:
        weight = "topo"

    partition = _best_partition(dual_graph, weight=weight)
    return regions_from_dual_partition(
        partition,
        dual_graph,
        edges_gdf,
        column=f"p_{weight}",
    )


def identify_regions_primal(graph: Any, nodes_gdf: gpd.GeoDataFrame, weight: str | None = None) -> gpd.GeoDataFrame:
    """Identify node-based regions using delegated python-louvain partitioning."""
    if weight is None:
        weight = "topo"

    partition = _best_partition(graph, weight=weight)
    return regions_from_primal_partition(partition, nodes_gdf, column=f"p_{weight}")


def polygonise_partitions(
    edges_gdf: gpd.GeoDataFrame,
    column: str,
    convex_hull: bool = True,
    buffer: float = 30,
) -> gpd.GeoDataFrame:
    """Create district polygons from edge-based partition labels.

    This preserves the old geometry pipeline: union partition edges,
    ``polygonize_full``, union the result, buffer, and optionally return the
    convex hull.
    """
    polygons = []
    partition_ids = []

    for partition_id in edges_gdf[column].unique():
        partition_edges = edges_gdf[edges_gdf[column] == partition_id]
        polygonised = polygonize_full(_geometry_union(partition_edges.geometry))
        polygon = unary_union(polygonised).buffer(buffer)
        polygons.append(polygon.convex_hull if convex_hull else polygon)
        partition_ids.append(partition_id)

    return gpd.GeoDataFrame(
        {column: partition_ids},
        geometry=polygons,
        crs=edges_gdf.crs,
    )


def district_to_nodes_from_edges(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> gpd.GeoDataFrame:
    """Assign edge-based district IDs to nodes using nearest street segment."""
    nodes_gdf = nodes_gdf.copy()
    sindex = edges_gdf.sindex

    nodes_gdf[column] = nodes_gdf.apply(
        lambda row: _assign_district_to_node(row["geometry"], edges_gdf, sindex, column),
        axis=1,
    )
    nodes_gdf[column] = nodes_gdf[column].astype(int)
    return nodes_gdf


def _assign_district_to_node(
    node_geometry: Any,
    edges_gdf: gpd.GeoDataFrame,
    sindex: Any,
    column: str,
) -> int:
    """Assign one node to the district of its nearest edge."""
    search_area = node_geometry.buffer(100)
    possible_matches_index = list(sindex.intersection(search_area.bounds))
    possible_matches = edges_gdf.iloc[possible_matches_index].copy()
    _, nearest_index = _min_distance_geometry_gdf(node_geometry, possible_matches)
    return int(edges_gdf.loc[nearest_index][column])


def districts_to_edges_from_nodes(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> gpd.GeoDataFrame:
    """Assign node-based district IDs to edges.

    The output preserves the old columns:
    ``{column}_uv``, ``{column}_u``, and ``{column}_v``.
    """
    edges_gdf = edges_gdf.copy()
    edges_gdf[f"{column}_uv"] = INVALID_DISTRICT
    edges_gdf[f"{column}_u"] = INVALID_DISTRICT
    edges_gdf[f"{column}_v"] = INVALID_DISTRICT

    edges_gdf[[f"{column}_uv", f"{column}_u", f"{column}_v"]] = edges_gdf.apply(
        lambda row: _assign_district_to_edge(row["edgeID"], nodes_gdf, edges_gdf, column),
        axis=1,
        result_type="expand",
    )

    return edges_gdf


def _assign_district_to_edge(
    edge_id: Any,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> tuple[int, int, int]:
    """Return edge district assignment from the districts of endpoint nodes."""
    edge = edges_gdf.loc[edge_id]
    district_u = int(nodes_gdf.loc[edge.u][column])
    district_v = int(nodes_gdf.loc[edge.v][column])
    district_uv = district_u if district_u == district_v else INVALID_DISTRICT
    return district_uv, district_u, district_v


def district_to_nodes_from_polygons(
    nodes_gdf: gpd.GeoDataFrame,
    partitions_gdf: gpd.GeoDataFrame,
    column: str,
) -> gpd.GeoDataFrame:
    """Assign polygon-based district IDs to nodes using nearest polygon."""
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf[column] = nodes_gdf.apply(
        lambda row: _assign_district_to_node_from_polygons(row["geometry"], partitions_gdf, column),
        axis=1,
    )
    nodes_gdf[column] = nodes_gdf[column].astype(int)

    return nodes_gdf


def _assign_district_to_node_from_polygons(
    node_geometry: Any,
    partitions_gdf: gpd.GeoDataFrame,
    column: str,
) -> int:
    """Assign one node to the nearest district polygon."""
    _, nearest_index = _min_distance_geometry_gdf(node_geometry, partitions_gdf)
    return int(partitions_gdf.loc[nearest_index][column])


def amend_nodes_membership(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
    min_size_district: int = 10,
) -> gpd.GeoDataFrame:
    """Amend node membership based on connectivity and minimum district size."""
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf = _check_disconnected_districts(nodes_gdf, edges_gdf, column, min_size_district)

    while INVALID_DISTRICT in nodes_gdf[column].unique():
        current_nodes_gdf = nodes_gdf
        nodes_gdf[column] = nodes_gdf.apply(
            lambda row, current_nodes_gdf=current_nodes_gdf: _amend_node_membership(
                row["nodeID"], current_nodes_gdf, edges_gdf, column
            ),
            axis=1,
        )
        nodes_gdf = _check_disconnected_districts(nodes_gdf, edges_gdf, column, min_size_district)

    return nodes_gdf


def _amend_node_membership(
    node_id: Any,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> int:
    """Amend one invalid node to the most plausible neighbouring district."""
    if nodes_gdf.loc[node_id][column] != INVALID_DISTRICT:
        return int(nodes_gdf.loc[node_id][column])

    tmp_edges = edges_gdf[(edges_gdf.u == node_id) | (edges_gdf.v == node_id)].copy()
    unique_nodes = list(np.unique(tmp_edges[["u", "v"]].values))
    unique_nodes.remove(node_id)

    tmp_nodes = nodes_gdf[
        (nodes_gdf.nodeID.isin(unique_nodes)) & (nodes_gdf[column] != INVALID_DISTRICT)
    ].copy()

    if tmp_nodes.empty:
        return INVALID_DISTRICT

    districts_sorted = tmp_nodes[column].value_counts(sort=True, ascending=False)
    if len(districts_sorted) == 1:
        return int(districts_sorted.idxmax())

    if districts_sorted.iloc[0] > districts_sorted.iloc[1]:
        return int(districts_sorted.idxmax())

    candidate_districts = list(districts_sorted.iloc[0:2].index)
    tmp_nodes = tmp_nodes[tmp_nodes[column].isin(candidate_districts)]
    _, closest_index = _min_distance_geometry_gdf(nodes_gdf.loc[node_id].geometry, tmp_nodes)
    return int(tmp_nodes.loc[closest_index][column])


def _check_disconnected_districts(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
    min_size: int = 10,
) -> gpd.GeoDataFrame:
    """Mark too-small/disconnected districts as invalid."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("amend_nodes_membership requires networkx") from exc

    nodes_gdf = nodes_gdf.copy()

    for district in nodes_gdf[column].unique():
        if district == INVALID_DISTRICT:
            continue

        tmp_nodes = nodes_gdf[nodes_gdf[column] == district].copy()
        tmp_edges = edges_gdf[
            edges_gdf.u.isin(tmp_nodes.nodeID) & edges_gdf.v.isin(tmp_nodes.nodeID)
        ].copy()

        if len(tmp_nodes) < min_size:
            nodes_gdf.loc[nodes_gdf.nodeID.isin(tmp_nodes.nodeID), column] = INVALID_DISTRICT
            continue

        tmp_graph = _graph_from_gdfs(tmp_nodes, tmp_edges, "nodeID")

        if not nx.is_connected(tmp_graph):
            largest_component = max(nx.connected_components(tmp_graph), key=len)
            component_graph = tmp_graph.subgraph(largest_component)
            to_check = [node for node in list(tmp_nodes.nodeID) if node not in list(component_graph.nodes())]
            nodes_gdf.loc[nodes_gdf.nodeID.isin(to_check), column] = INVALID_DISTRICT

    return nodes_gdf


def find_gateways(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> gpd.GeoDataFrame:
    """Identify nodes lying on a district boundary."""
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["gateway"] = nodes_gdf.apply(
        lambda row: _gateway(row["nodeID"], nodes_gdf, edges_gdf, column),
        axis=1,
    )
    return nodes_gdf


def _gateway(
    node_id: Any,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    column: str,
) -> int:
    """Return 1 if a node connects to another district, otherwise 0."""
    connected_edges = edges_gdf[(edges_gdf.u == node_id) | (edges_gdf.v == node_id)].copy()
    connected_nodes = nodes_gdf[
        nodes_gdf.nodeID.isin(connected_edges.u) | nodes_gdf.nodeID.isin(connected_edges.v)
    ].copy()

    if len(connected_nodes[column].unique()) > 1:
        return 1
    return 0
