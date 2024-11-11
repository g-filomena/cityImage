.. _api_ref:
.. automodule:: cityImage
.. currentmodule:: cityImage

cityImage API reference
=======================

angles
------
.. autosummary::
   :toctree: api/

   get_coord_angle
   angle_line_geometries


Barriers
--------------
.. autosummary::
   :toctree: api/

   road_barriers
   water_barriers
   railway_barriers
   park_barriers
   along_water
   along_within_parks
   barriers_along
   assign_structuring_barriers
   get_barriers

centrality
----------

.. autosummary::
   :toctree: api/
   
   calculate_centrality
   reach_centrality
   straightness_centrality
   weight_nodes
   append_edges_metrics


clean
-----
.. autosummary::
   :toctree: api/

   clean_network
   simplify_graph
   duplicate_nodes
   fix_dead_ends
   is_nodes_simplified
   is_edges_simplified
   merge_pseudo_edges
   simplify_same_vertexes_edges
   clean_edges
   fix_network_topology
   fix_self_loops
   remove_disconnected_islands
   correct_edges

colors
------
.. autosummary::
   :toctree: api/

   random_colors_list
   rand_cmap
   kindlmann
   normalize
   lighten_color

graph
-----
.. autosummary::
   :toctree: api/

   graph_fromGDF
   multiGraph_fromGDF
   dual_gdf
   dual_graph_fromGDF
   dual_id_dict
   nodes_degree

land_use
--------
.. autosummary::
   :toctree: api/

   classify_land_use
   land_use_from_other_gdf

landmarks
---------
.. autosummary::
   :toctree: api/

   get_buildings_fromFile
   get_buildings_fromOSM
   structural_score
   number_neighbours
   visibility_score
   facade_area
   get_historical_buildings_fromOSM
   cultural_score
   pragmatic_score
   compute_global_scores
   compute_local_scores
   
load
----
.. autosummary::
   :toctree: api/

   get_network_fromOSM
   get_network_fromFile
   get_network_fromGDF
   obtain_nodes_gdf
   join_nodes_edges_by_coordinates
   reset_index_graph_gdfs

plot
----
.. autosummary::
   :toctree: api/

   plot_gdf
   plot_grid_gdfs_column
   plot_grid_gdf_columns
   
regions
-------
.. autosummary::
   :toctree: api/

   identify_regions
   identify_regions_primal
   polygonise_partitions
   district_to_nodes_from_edges
   districts_to_edges_from_nodes
   district_to_nodes_from_polygons
   amend_nodes_membership
   find_gateways

utilities
---------
.. autosummary::
   :toctree: api/

   downloader
   scaling_columnDF
   dict_to_df
   center_line
   min_distance_geometry_gdf
   split_line_at_MultiPoint
   merge_line_geometries
   envelope_wgs
   convex_hull_wgs
   rescale_ranges
   gdf_from_geometries
   line_at_centroid
   sum_at_centroid
   polygons_gdf_multiparts_to_singleparts
   fix_multiparts_LineString_gdf
   remove_lists_columns
   polygon_2d_to_3d

visibility
----------
.. autosummary::
   :toctree: api/

   visibility_polygon2d
   compute_3d_sight_lines
   intervisibility
