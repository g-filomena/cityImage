.. _api_ref:

cityImage API reference
=======================

This page documents the public API after the core refactor. Generic file IO,
OSM acquisition, plotting internals, and low-level helper functions are not
exposed as top-level user API unless they preserve cityImage-specific output
semantics.

.. currentmodule:: cityImage

Core schema and adapters
------------------------

.. autosummary::
   :toctree: api/

   SchemaError
   SchemaReport
   require_columns
   require_geometry
   missing_columns
   ensure_building_schema_defaults
   validate_nodes_gdf
   validate_edges_gdf
   validate_buildings_gdf
   standardize_nodes_gdf
   standardize_edges_gdf
   standardize_buildings_gdf
   adapt_nodes_gdf
   adapt_edges_gdf
   adapt_buildings_gdf

IO and OSM bridge API
---------------------

.. autosummary::
   :toctree: api/

   network_from_file
   buildings_from_file
   features_from_osm
   network_from_osm
   buildings_from_osm
   barriers_from_osm
   network_from_lines
   pedestrian_network_from_osm
   pedestrian_network_from_osm_features
   filter_pedestrian_osm_features

Angles and graph semantics
--------------------------

.. autosummary::
   :toctree: api/

   get_coord_angle
   angle_line_geometries
   graph_fromGDF
   multiGraph_fromGDF
   dual_gdf
   dual_graph_fromGDF
   dual_id_dict
   from_nx_to_gdf
   nodes_degree

Network topology
----------------

.. autosummary::
   :toctree: api/

   clean_network
   simplify_graph
   clean_duplicate_nodes
   clean_duplicate_edges
   clean_same_vertexes_edges
   fix_dead_ends
   fix_fake_self_loops
   fix_network_topology
   remove_disconnected_islands
   correct_edge_geometries
   consolidate_nodes
   consolidate_edges

Centrality
----------

.. autosummary::
   :toctree: api/

   calculate_centrality
   reach_centrality
   straightness_centrality
   append_edges_metrics

Barriers
--------

.. autosummary::
   :toctree: api/

   barrier_osm_feature_tags
   barriers_from_osm_features
   road_barriers_from_osm_features
   water_barriers_from_osm_features
   railway_barriers_from_osm_features
   park_barriers_from_osm_features
   along_water
   along_within_parks
   barriers_along
   assign_structuring_barriers

Regions and districts
---------------------

.. autosummary::
   :toctree: api/

   identify_regions
   identify_regions_primal
   regions_from_dual_partition
   regions_from_primal_partition
   polygonise_partitions
   district_to_nodes_from_edges
   districts_to_edges_from_nodes
   district_to_nodes_from_polygons
   amend_nodes_membership
   find_gateways

Land use
--------

.. autosummary::
   :toctree: api/

   classify_land_use
   classify_land_uses_raws_into_OSMgroups
   classify_land_uses_intoDMAs
   derive_land_uses_raw_fromOSM
   land_use_from_other_gdf
   land_use_from_points
   land_use_from_polygons
   classify_sparse_land_uses
   attach_sparse_land_uses
   find_land_use_values_matching
   find_unclassified_tokens_OSM_groups

Landmarks and imageability scoring
----------------------------------

.. autosummary::
   :toctree: api/

   structural_score
   visibility_score
   facade_area
   number_neighbours
   cultural_score
   pragmatic_score
   compute_global_scores
   compute_local_scores
   score_cityimage_buildings
   score_buildings_global
   score_buildings_local
   score_building_components
   validate_score_weights
   LandmarkScoringConfig

Building height
---------------

.. autosummary::
   :toctree: api/

   buildings_height_from_dem_dtm
   assign_building_heights_from_other_gdf
   assign_height_from_dtm

Visibility
----------

.. autosummary::
   :toctree: api/

   visibility_polygon2d
   compute_3d_sight_lines
   obstructions_2d
   obstructions_3d
   filter_distance
   downsample_coords
   polygon_2d_to_3d
   merge_gpkg_chunks_to_gdf

Geometry and small utilities
----------------------------

.. autosummary::
   :toctree: api/

   center_line
   split_line_at_MultiPoint
   fix_multiparts_LineString_gdf
   gdf_multipolygon_to_polygon
   scaling_columnDF

Plotting
--------

Plotting is optional and imported lazily. Use the plotting entry points below
rather than internal axis/legend helpers.

.. autosummary::
   :toctree: api/

   Plot
   MultiPlot
   plot_gdf
   plot_grid_gdfs_column
   plot_grid_gdf_columns
   rand_cmap
   kindlmann
   normalize
   lighten_color
   random_colors_list
