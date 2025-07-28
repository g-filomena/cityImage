import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, LineString
from shapely.ops import unary_union
pd.set_option("display.precision", 3)

from .graph import graph_fromGDF

def consolidate_nodes(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', consolidate_edges_too = False, tolerance=20):
    """
    Consolidates nodes in a spatial network that are within a given distance (tolerance), preserving topology and unclustered nodes.

    Nodes within `tolerance` distance are clustered together and represented by a single consolidated node at the cluster centroid.
    For clusters containing disconnected components, each connected component is further split into its own consolidated node.
    Optionally, edges can be updated to reference the new consolidated node IDs and geometries.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of nodes, must include columns 'nodeID' and 'geometry'. If present, 'z' is averaged for clusters.
    edges_gdf : GeoDataFrame
        GeoDataFrame of edges for checking network connectivity.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    consolidate_edges_too : bool, optional
        If True, also returns the updated edges GeoDataFrame (default: False).
    tolerance : float, optional
        Distance threshold for clustering nodes (in CRS units). Nodes within this distance are merged (default: 20).

    Returns
    -------
    consolidated_nodes_gdf : GeoDataFrame
        GeoDataFrame of consolidated nodes. Columns include:
            - 'old_nodeIDs': list of merged node IDs
            - 'x', 'y': centroid coordinates
            - 'z' (optional): averaged elevation for the cluster
            - 'nodeID': new node ID
            - 'geometry': consolidated node Point geometry
    consolidated_edges_gdf : GeoDataFrame (optional)
        Only returned if `consolidate_edges_too` is True.
        Edges with endpoints mapped to new consolidated node IDs and geometries
    """
    
    nodes_gdf = nodes_gdf.copy().set_index(nodeID_column, drop=False)
    nodes_gdf.index.name = None
    nodes_gdf.drop(columns=["x", "y"], inplace=True, errors="ignore")
    graph = graph_fromGDF(nodes_gdf, edges_gdf, nodeID_colum = nodeID_column)

    # Step 1: Cluster nodes within tolerance
    clusters = nodes_gdf.buffer(tolerance).unary_union
    clusters = clusters.geoms if hasattr(clusters, "geoms") else [clusters]
    clusters = gpd.GeoDataFrame(geometry=gpd.GeoSeries(clusters, crs=nodes_gdf.crs))
    clusters["x"] = clusters.geometry.centroid.x
    clusters["y"] = clusters.geometry.centroid.y

    # Step 2: Assign nodes to clusters
    gdf = gpd.sjoin(nodes_gdf, clusters, how="left", predicate="within").drop(columns="geometry")
    gdf.rename(columns={"index_right": "new_"+nodeID_column}, inplace=True)
    new_nodeID = gdf.new_nodeID.max() + 1
    
    # Step 3: Split non-connected components in clusters
    for cluster_label, nodes_subset in gdf.groupby("new_nodeID"):
        if len(nodes_subset) > 1:  # Skip unclustered nodes
            wccs = list(nx.connected_components(graph.subgraph(nodes_subset.index)))
            if len(wccs) > 1:
                for wcc in wccs:
                    idx = list(wcc)
                    subcluster_centroid = nodes_gdf.loc[idx].geometry.unary_union.centroid
                    gdf.loc[idx, ["x", "y"]] = subcluster_centroid.x, subcluster_centroid.y
                    gdf.loc[idx, "new_nodeID"] = new_nodeID
                    new_nodeID += 1

    # Step 4: Consolidate nodes, but preserve unclustered ones
    consolidated_nodes = []
    has_z = 'z' in nodes_gdf.columns

    for new_nodeID, nodes_subset in gdf.groupby("new_nodeID"):
        old_nodeIDs = nodes_subset[nodeID_colum].to_list()
        cluster_x, cluster_y = nodes_subset.iloc[0][["x", "y"]]

        new_node = {
            "old_nodeIDs": old_nodeIDs,
            "x": cluster_x,
            "y": cluster_y,
            nodeID_column: new_nodeID,
        }

        if has_z:
            new_node["z"] = nodes_gdf.loc[old_nodeIDs, "z"].mean() if len(old_nodeIDs) > 1 else nodes_gdf.loc[old_nodeIDs[0], "z"]
        
        consolidated_nodes.append(new_node)

    # Convert list of dicts to DataFrame
    consolidated_nodes_df = pd.DataFrame(consolidated_nodes)

    # Create final GeoDataFrame
    consolidated_nodes_gdf = gpd.GeoDataFrame(
        consolidated_nodes_df,
        geometry=gpd.points_from_xy(
            consolidated_nodes_df["x"],
            consolidated_nodes_df["y"],
            consolidated_nodes_df["z"] if "z" in consolidated_nodes_df.columns else None
        ),
        crs=nodes_gdf.crs
    )

    if consolidate_edges_too:
        return consolidated_nodes_gdf, consolidate_edges(edges_gdf, consolidated_nodes_gdf)
        
    return consolidated_nodes_gdf
    
def consolidate_edges(edges_gdf, consolidated_nodes_gdf, nodeID_colum):
    """
    Reassigns edge endpoints ('u', 'v') and updates geometries to match consolidated nodes.

    This function replaces the 'u' and 'v' node IDs in each edge with new node IDs
    based on the consolidation mapping found in `consolidated_nodes_gdf`.
    It also updates the edge geometry by snapping the first and last coordinates of
    each edge's LineString to the positions of the new (consolidated) node geometries.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        GeoDataFrame of edges, containing columns 'u', 'v', 'geometry', and 'edgeID'.
        The 'u' and 'v' columns should refer to old node IDs.
    consolidated_nodes_gdf : GeoDataFrame
        GeoDataFrame of consolidated nodes
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
        
    Returns
    -------
    consolidated_edges : GeoDataFrame
        Updated GeoDataFrame of edges with:
            - 'u' and 'v': replaced by new node IDs
            - 'geometry': LineString updated to start/end at the consolidated node locations
        Self-loop edges (where u == v) are removed. The index is set to 'edgeID'.
    """
    
    oldIDs_column = "old_"+nodeID_colum
    # Create a mapping from old_nodeIDs to their corresponding nodeID and geometry
    nodes_mapping = (
        consolidated_nodes_gdf.explode(oldIDs_column)[[oldIDs_column, "geometry", nodeID_colum]]
        .set_index(oldIDs_column)
    )

    def _update_edge(row):

        old_u, old_v, geom = row["u"], row["v"], row["geometry"]
        
        # Map old_u and old_v to their corresponding new nodeIDs
        new_u_id = nodes_mapping.loc[old_u, nodeID_colum]
        new_v_id = nodes_mapping.loc[old_v, nodeID_colum]

        # Get the new geometries for u and v
        new_u_geom = nodes_mapping.loc[old_u, "geometry"]
        new_v_geom = nodes_mapping.loc[old_v, "geometry"]

        # Update the geometry (replace first and last coordinates)
        if isinstance(geom, LineString):
            new_coords = [new_u_geom.coords[0]] + list(geom.coords[1:-1]) + [new_v_geom.coords[0]]
            geom = LineString(new_coords)

        return pd.Series({"u": new_u_id, "v": new_v_id, "geometry": geom})

    # Apply updates to the edges
    consolidated_edges = edges_gdf.copy()
    consolidated_edges[["u", "v", "geometry"]] = consolidated_edges.apply(_update_edge, axis=1)
    consolidated_edges = consolidated_edges[consolidated_edges.u != consolidated_edges.v]
    consolidated_edges.index = consolidated_edges['edgeID']
    consolidated_edges.index.name = None
    
    return consolidated_edges