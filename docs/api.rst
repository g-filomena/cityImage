.. _api_ref:

cityImage API reference
=======================

This page documents the public API. cityImage leaves file reading and CRS
handling to GeoPandas, OpenStreetMap downloads to OSMnx, graph algorithms to
NetworkX, igraph and python-louvain, raster sampling to rasterio and
rasterstats, the parallel 3D sight lines to Dask and static plots to Matplotlib
and mapclassify. The functions listed here are cityImage's own: networks,
barriers, regions, land use, landmarks, visibility and scores. Internal helpers
are not listed.

.. currentmodule:: cityImage

The Image of the City in one call
---------------------------------

``image_of_the_city`` chains the steps below into the five elements of Lynch's
Image of the City. ``explore_city_image`` needs the ``interactive`` extra,
``plot_city_image`` the ``plot`` extra.

.. autosummary::
   :toctree: api/

   image_of_the_city
   CityImage
   CityImageConfig
   explore_city_image
   plot_city_image
   landmark_weight_sensitivity

Schema and adapters
-------------------

.. autosummary::
   :toctree: api/

   SchemaError
   SchemaReport
   missing_columns
   require_columns
   require_geometry
   require_land_use_lists
   ensure_building_schema_defaults
   validate_nodes_gdf
   validate_edges_gdf
   validate_buildings_gdf
   standardize_nodes_gdf
   standardize_edges_gdf
   standardize_buildings_gdf
   standardize_cityimage_inputs

IO and OSM bridge API
---------------------

These helpers preserve cityImage output schemas while delegating acquisition and
file IO to GeoPandas and OSMnx.

.. autosummary::
   :toctree: api/

   network_from_file
   buildings_from_file
   features_from_osm
   network_from_osm
   buildings_from_osm
   barriers_from_osm
   network_from_lines

Buildings
---------

.. autosummary::
   :toctree: api/

   select_buildings_by_study_area

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

Centrality wrappers keep cityImage node/edge output semantics while delegating
centrality algorithms to graph libraries.

.. autosummary::
   :toctree: api/

   calculate_centrality
   reach_centrality
   straightness_centrality
   weight_nodes
   append_edges_metrics

Barriers
--------

Barrier helpers convert roads, waterways, coastlines, railways, and parks into
the Lynchian ``edge`` semantics used by cityImage.

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

Land-use functions cover OSM-derived land-use semantics and sparse/non-OSM
attribute workflows.

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

These functions implement the cityImage landmark and imageability model:
structural, visual, cultural, and pragmatic salience, followed by global/local
score composition.

.. autosummary::
   :toctree: api/

   LandmarkScoringConfig
   structural_score
   visibility_score
   cultural_score
   pragmatic_score
   compute_global_scores
   compute_local_scores
   score_building_components
   score_buildings_global
   score_buildings_local
   score_cityimage_buildings
   validate_score_weights

Building height and terrain elevation
-------------------------------------

Height helpers are optional. Raster workflows require the ``height`` extra.

The **DTM** (bare-earth terrain) gives node ``z`` and building ``base``; a
**surface** model (DSM, with the roofs on it, often distributed as *DEM*)
compared with the DTM gives the building heights. For a city without building
heights, ``assign_elevations_from_rasters`` does all of this in one call.

.. autosummary::
   :toctree: api/

   assign_elevations_from_rasters
   buildings_base_from_dtm
   buildings_height_from_dem_dtm
   assign_building_heights_from_other_gdf
   assign_height_from_dtm

Visibility
----------

2D visibility is part of the core geospatial workflow. 3D sight-line workflows
are optional and require the ``visibility3d`` extra, or ``all``.

.. autosummary::
   :toctree: api/

   visibility_polygon2d
   compute_3d_sight_lines
   obstructions_2d
   filter_distance
   downsample_coords
   merge_gpkg_chunks_to_gdf

Geometry and small utilities
----------------------------

Only cityImage-specific geometry helpers are exposed. Generic envelope,
convex-hull, distance, file-loading, and arbitrary dataframe helpers were
removed from the public API; use GeoPandas/Shapely/Pandas directly for those.

.. autosummary::
   :toctree: api/

   center_line
   split_line_at_MultiPoint
   fix_multiparts_LineString_gdf
   gdf_multipolygon_to_polygon
   scaling_columnDF

Plotting
--------

Plotting is optional and imported lazily. Use these public plotting entry points
rather than internal axis, legend, or colorbar helpers.

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
