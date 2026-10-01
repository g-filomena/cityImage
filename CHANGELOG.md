# Changelog

All notable changes to **cityImage** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) from 2.0.0 onwards.

Entries marked **⚠ behaviour** change the output of an existing call with the same arguments.

## [2.2.0] — 2026-09-28

### Network cleaning — reworked

The same-vertex, dead-end and pseudo-node steps of `clean_network` were rewritten around one rule:
**parallel streets stay parallel, no node is ever added, and a street is only reduced when it is
mapped more than once.** See the [network topology guide](https://github.com/g-filomena/cityImage/blob/master/docs/network_topology.md)
for a visual walk-through of every case.

#### Added
- `same_vertexes_tolerance` (default `5.0`, CRS units) on `clean_network` and
  `clean_same_vertexes_edges`: the largest Hausdorff distance between two edges joining the same
  nodes for them to count as one street mapped twice.
- `nodes_to_keep_regardless` on `fix_dead_ends`: dead-end peeling stops at these nodes.
- `self_loops` on `clean_duplicate_edges` (default `False`). `clean_network` passes its own
  `self_loops` through.
- `preserve_direction` on `simplify_graph`, passed by `clean_network(preserve_direction=True)`: a
  pseudo-node is kept where merging would lose the direction of a one-way street (one segment
  one-way and the other not, or two one-ways that do not run on from one into the other), so
  `oneway` stays exact on the cleaned edges. Without it, as before, a merged edge takes the
  `oneway` covering most of its length.

#### Changed
- **⚠ behaviour** `clean_same_vertexes_edges` no longer drops the shorter of two edges between
  the same nodes. Edges are grouped as one street when their lengths are within 10 % **and** they
  lie within `same_vertexes_tolerance` of each other; matches chain transitively, and each group
  becomes its middle copy: with an odd number of copies the most central edge as mapped, with an
  even number the centre line of the two most central (a pair is still averaged, as before).
  Different streets between the same two junctions (a crescent and a straight road, the two sides
  of a block) are kept as parallel edges.
- **⚠ behaviour** `clean_network(self_loops=False)` keeps loop streets: a street that leaves a
  junction and returns to it survives as one self-loop edge. `clean_duplicate_edges` used to
  remove every self-loop whatever `self_loops` said. The default, `self_loops=True` (previously
  `False`), removes them, so a call without the argument gives the same result as before.
- **⚠ behaviour** `clean_duplicate_edges` called on its own keeps self-loops unless
  `self_loops=True`.
- **⚠ behaviour** `clean_network` fixes topology (`fix_topology=True`) *before* removing dead ends
  and islands. A street joined to the network only at an un-noded shared vertex was previously
  seen as a dead end or an island and deleted.
- **⚠ behaviour** `fix_dead_ends` removes a dead-end street back to the junction where it meets the
  rest of the network (repeating until none is left), stops at `nodes_to_keep_regardless`, and
  leaves a loop-free component as it is instead of peeling it to nothing.
- `simplify_graph` merges the two segments of a loop street correctly when they share both ends,
  and a merged edge takes, per attribute, the non-null value covering the most length. Column dtypes
  are kept.
- `center_line` averages lines at the same fractions of their length, so lines with different
  vertex counts are averaged whole instead of being truncated to the shorter coordinate list,
  and keeps `z` when every line has it.

#### Fixed
- **⚠ behaviour** Splitting an edge (`fix_fake_self_loops`, which `clean_network` always runs, and
  `fix_network_topology`) no longer rebuilds the nodes from the edge end points. Previously any
  split renumbered every `nodeID` and `edgeID` and dropped node columns other than the
  coordinates, so `nodes_to_keep_regardless` protected the wrong nodes. Existing IDs, columns and
  dtypes are now kept: a split edge keeps its `edgeID` on its first piece, the other pieces get new
  IDs, and a new junction gets the next `nodeID`. A new junction's `z` is interpolated along the
  split edge; its other columns are missing (integer and boolean node columns become nullable).
  Pieces are oriented by the edge's geometry, so an edge stored against its `u`/`v` labels is
  split correctly, and an edge end is matched to its node even when the node's point is off by
  float noise. A vertex repeated at the end of a line no longer yields a zero-length edge.
  When nothing is split, both functions return copies rather than the frames passed in.
- Edges are split at their own vertices, so 3D edges keep their `z`. `fix_network_topology`
  used to fail on 3D edges, and `fix_fake_self_loops` skipped them (it compared coordinates
  including `z`). `fix_fake_self_loops` also no longer needs `x` / `y` node columns.
- `network_from_lines` / `network_from_file` join 3D lines to their nodes (u/v were all NaN), and
  `clean_network` accepts 3D line geometries.
- With `self_loops=True`, a node whose last edge was a removed self-loop is dropped instead of
  being left in the output with no edges.
- Merging segments in `simplify_graph` no longer fails on `int32` (and other non-`int64`)
  attribute columns, and keeps each column's dtype.
- **⚠ behaviour** With pandas 2, a missing text value (a street without a `name`, say) came out of
  `clean_network`, and every function ending in `convert_numeric_columns`, as the string `"None"`
  or `"nan"`; it stays missing now, as it already did with pandas 3.
- User edge columns named `fixing` or `to_fix` are no longer ignored by attribute merging or dropped
  from the output.
- `remove_disconnected_islands` accepts an empty network.

### Graphs, dual graphs and centrality
- **⚠ behaviour** `graph_fromGDF` keeps the *shortest* of parallel edges (it previously kept
  whichever came last), so shortest-path measures are exact. `multiGraph_fromGDF` keeps all; use
  it for one-way routing, since a two-way street mapped as two opposing one-ways between the same
  nodes keeps only one direction in a `Graph`.
- `graph_fromGDF` and `multiGraph_fromGDF` keep each edge's `u` and `v` as attributes, so the
  direction of a one-way street (`u` → `v`) survives in the undirected graph.
  `multiGraph_fromGDF` keeps parallel streets that share a `key` (`network_from_lines` keys every
  edge 0) instead of keeping the last one read.
- **⚠ behaviour** `dual_gdf` writes one row per pair of adjacent segments, with a new `oneway`
  column: 0 when the move is allowed both ways, 1 when only `u` → `v` is (with `oneway=True`,
  respecting one-way streets; the row points in the allowed direction). `dual_graph_fromGDF`
  stays an undirected `networkx.Graph` whose edges keep the row's `u`, `v` and `oneway`, so a
  model can follow one-way moves. With `oneway=True`, a primal `oneway` of True, 1 or `"yes"` is
  one-way and False, 0, `"no"` or missing is two-way (any case); any other value, such as OSM's
  `-1` or `reversible`, raises a `ValueError` (it was silently misread).
- `dual_gdf` no longer fails when no two segments meet (a single street), and finds each segment's
  neighbours through the junctions instead of scanning every segment, so its time grows with the
  network rather than with its square (about 9× faster on 2,000 edges).
- With a `Graph`, `append_edges_metrics` gives the parallel streets the graph leaves out (it
  keeps the shortest of each pair) 0, the exact value for shortest-path measures, instead of NaN;
  any other NaN stays. It also takes a `MultiGraph` (`multiGraph_fromGDF`), keyed `(u, v, key)`;
  weighted by length, betweenness is the same on either. `calculate_centrality` accepts a
  `MultiGraph` too.
- `weight_nodes` and `append_edges_metrics` match rows by `nodeID`/`edgeID` rather than by index
  label (a `KeyError`, or extra rows, when the index was not the IDs).

### Buildings and heights
- **⚠ behaviour** One reading of building heights (`known_heights`): a height is a number of
  metres above zero, read from numbers or strings such as `"12 m"` or `"12,5"`; a missing,
  unreadable, zero or negative value is unknown (NaN). The building schema
  (`standardize_buildings_gdf`, and so every building loader and `score_building_components`),
  the landmark scores and the 3D sight lines all read heights this way.
- **⚠ behaviour** `buildings_from_file` drops the buildings whose height is lower than
  `min_height`, as the loader did before the 2.x API refactor; a building without a height is
  kept with a NaN height. A file without a height field no longer gets `min_height` for every
  building: every height is NaN and the landmark scores leave the visual component out. Heights
  are the caller's to supply.
- **⚠ behaviour** `buildings_from_osm` keeps OSM `height` tags only with the new
  `keep_osm_heights=True`. By default every height is NaN and the visual component is left out
  for every building, as before the refactor.
- `gdf_multipolygon_to_polygon` keeps the ID column unless a MultiPolygon is split, so
  `buildings_from_file` keeps the file's building IDs.
- `assign_building_heights_from_other_gdf` gives a detailed building's height to its best match
  only, not to every building it overlaps.

### Landmark scores
- **⚠ behaviour** `visibility_score` gives a building without a height a NaN `fac` and `3dvis`
  (they were 0; a zero or negative height gave a zero or negative facade area), so it stays out
  of their rescaling. `3dvis` is NaN for every building when no building with a height is
  reached by a sight line (none passed, an empty result, or lines to other buildings); it was 0.
  When a building is reached, the others still get 0. It keeps the caller's index.
- **⚠ behaviour** The landmark scores (`score_buildings_global` / `score_buildings_local`,
  `compute_global_scores` / `compute_local_scores`) rescale over known values only: a NaN index
  stays NaN and never moves the other buildings' scale, and a NaN counts as 0 only in the
  weighted sums that follow the rescaling. A building without a height has no visual score
  (`vScore`, and its visual indexes, NaN), which adds nothing to `gScore`/`lScore`; it received a
  NaN score. Every building stays in the output and can be a landmark through its other
  components; drop such buildings beforehand to leave them out. A component (`vScore`,
  `sScore`, `cScore`, `pScore`) and its indexes' `_sc` columns are written only when computed;
  `cScore`/`pScore`/`sScore` were written as 0 without their indexes. The scores never add a
  `height` column. `lScore` is no longer rounded to 3 decimals, so no two buildings tie by
  rounding. Global and local scores share one implementation.
- **⚠ behaviour** `score_buildings_local` / `compute_local_scores` no longer write `sScore_l`
  and `vScore_l`, which were always 0; the local score is `lScore` (and `lScore_sc`).
- **⚠ behaviour** `scaling_columnDF` keeps NaN as NaN when the other values are all equal; it
  returned 0 (1 with `inverse=True`) for them.
- **⚠ behaviour** `cultural_score` gives `cult` NaN for every building when no building has
  anything: no historic layer, no historic element intersecting a building, no `historic` tag
  with `from_OSM=True`, or every `score_column` sum 0; it was 0. When a building has something,
  the others still get 0.
- **⚠ behaviour** `pragmatic_score` gives a building with no other building within
  `search_radius` a `prag` of 1 (as unexpected as can be); it was 0.
- `score_buildings_local` no longer writes into the caller's frame.

### 3D visibility
- **⚠ behaviour** `observer_height` (default `1.6` m) on `compute_3d_sight_lines` and the
  sight-line helpers: sight lines start at eye level (node `z` + `observer_height`) rather than at
  node `z`.
- **⚠ behaviour** Observers stand at node `z` as given: a `z` below -50 is no longer read as DTM
  nodata and moved to 2 (nodata is the DTM reader's to handle, as `assign_height_from_dtm` does
  with `min_valid_elev`). Where some nodes have a `z`, the nodes without one are left out.
  Without elevations (no `z` column, or every value missing) observers stand at 0, the ground of
  a building without a `base`. A warning is logged when only the nodes or only the buildings have
  elevations.
- **⚠ behaviour** `network_from_lines` / `network_from_file` and `network_from_osm` give the nodes
  of a 2D network `z = 0` (ground level, as a building without a `base`) instead of 2, so eyes are
  at `observer_height` above the buildings' ground.
- **⚠ behaviour** `compute_3d_sight_lines` leaves out, as targets and obstructions, the buildings
  without a height (a zero or unreadable one included), and raises a `ValueError` when no
  building has one. A building's `base` is used as given, 0 when missing; it was raised to at
  least 1.0, which put the roofs 1 m high over ground-level observers and moved low-lying terrain.
  Targets are the buildings at least `min_target_height` tall (new argument, default 5 m); they
  were the ones taller than 5 m. The building preparation no longer writes its `base` defaults
  into the caller's frame.
- `compute_3d_sight_lines(max_observer_target_distance=...)` imported scipy, which is not a
  dependency, and failed where it was not installed. The radius query now uses shapely's `STRtree`.
- `compute_3d_sight_lines` writes its chunk files to a temporary subfolder of `tmp_dir` (new
  argument, default the working directory), removed afterwards, instead of leaving them in
  `./sight_lines_tmp`.
- `compute_3d_sight_lines(verbose=True)` logs its progress at INFO on the
  `cityImage.visibility3d` logger instead of printing it, one line each time the progress bar
  advances. `verbose=True` sets that logger to INFO and gives it a console handler when logging is
  not configured, so the progress shows either way. "No visible sight-lines" is logged too.

### 2D visibility
- **⚠ behaviour** 2D advance visibility (`visibility_polygon2d`, `2dvis`) covers the whole ring
  of rays: the slice between the 350° and 0° rays was left out.

### Barriers
- **⚠ behaviour** Roads and railways tagged `tunnel=no` are no longer dropped as tunnels.
- **⚠ behaviour** Park barriers are closed with `buffer(10).buffer(-10)`: the outline stays on the
  park's edge (it used to sit 10 m outside it), and parks less than 20 m apart merge.
- `barrier_osm_feature_tags` requests only the tag values the extractors keep (motorways rather
  than every `highway`, parks rather than every `leisure` feature) and takes `include_primary`,
  `include_secondary` and `keep_light_rail`.
- `barriers_from_osm` now forwards `include_primary` / `include_secondary` to the download query.
- `barriers_from_osm`, `barriers_from_osm_features` and the per-type builders project to the local
  UTM zone when `crs` is None and the input is in longitude/latitude; the barrier rules were
  applied in degrees. Input without a CRS stays in its own units.
- `along_within_parks` no longer writes into the caller's frame.

### Pedestrian networks
- `service=alley` ways are kept (`ped="noEvidence"`, or `"yes"` with pedestrian evidence) instead of
  being dropped as generic service roads.
- `network_from_osm(network_type="walk")` / `pedestrian_network_from_osm` project to the local UTM
  zone when `crs` is None, and validate `distance` like the other network types instead of passing
  `None` to OSMnx.

### Districts
- **⚠ behaviour** `identify_regions` and `identify_regions_primal` take `random_state` (default `0`)
  and pass it to python-louvain, so the same input gives the same districts on every call.
  `random_state=None` restores the previous random partition.
- **⚠ behaviour** `amend_nodes_membership` raises a `ValueError`, instead of looping forever, when
  the network is not connected (remove its islands first), is smaller than `min_size_district`,
  has no district of that size, has nodes that cannot be amended, or does not settle within one
  pass per node.
- `districts_to_edges_from_nodes` matches rows by `nodeID` rather than by index label (silently
  wrong districts when the index was not the IDs), and `district_to_nodes_from_edges` falls back
  to the nearest edge anywhere when none is within 100 m.

### Documentation
- New [network topology guide](https://github.com/g-filomena/cityImage/blob/master/docs/network_topology.md):
  every case `clean_network` handles, with diagrams, and how its steps depend on each other.
- This changelog, also in the documentation.
- The example notebooks describe the current cleaning options; the API reference no longer lists
  `obstructions_3d` and `polygon_2d_to_3d`, which stopped being public in 2.1.0.

### Project
- Releases are published to PyPI by `.github/workflows/publish.yml` through trusted publishing when
  a `v*` tag is pushed.
- Regression tests for every change above, including dead ends, parallel and duplicate edges, loop
  streets, pseudo-node simplification, edge splitting, one-way streets, barrier tag queries,
  `center_line`, the dual graph, heights and the landmark scores.
- `clean_network` is checked on the real York, Ontario network (a central subset, and the whole
  town as `slow` tests) against the rules of the guide: no vertex invented except on centre lines,
  no orphan node, no pseudo-node or street mapped twice left, IDs kept, and a second pass changing
  nothing.
- `pytest-timeout` in the `test` and `dev` extras, so a loop that never ends fails its test.
- The live-OSM tests (`-m network`) use the first Overpass endpoint that answers (the public one,
  then two mirrors), with a 60 s request timeout, and report a test that cannot reach any as
  skipped rather than failed.

## [2.1.1] — 2026-09-26

### Fixed
- `consolidate_nodes` only forms a cluster when every member lies within `tolerance` of every
  other (no more chaining along closely spaced nodes). Clusters are placed at the mean of their
  members, the result no longer depends on row order, and `consolidate_edges` keeps the
  dimensionality (2D/3D) of each edge's geometry.

### Changed
- PyVista and tqdm are no longer dependencies anywhere (environment, CI, docs); the
  `visibility3d` extra is `dask` + `psutil`.
- The 3D visibility progress logger writes one compact, in-place status line.
- Ruff pre-commit configuration added; README logo uses an absolute URL so it renders on PyPI.
- CI actions bumped (`actions/checkout` 7, `codecov/codecov-action` 7, `github/codeql-action` 4,
  `mamba-org/setup-micromamba` 3).

## [2.1.0] — 2026-07-22

### Changed
- 3D sight-line visibility is computed analytically instead of by PyVista mesh ray casting: a
  sight line is obstructed when its 2D projection crosses a building footprint and the line dips
  to or below the roof over that crossing. Candidates are pruned with a vectorised bounding-box
  and vertical test before the exact check.
- `compute_3d_sight_lines` takes `max_observer_target_distance` to limit candidate pairs.
- `obstructions_3d` and `polygon_2d_to_3d` (PyVista-based) are no longer public.
- `structural_score` takes `workers` for parallel 2D visibility; 2D visibility ray clipping is
  vectorised.

### Fixed
- `compute_3d_sight_lines` no longer crashes on large cities (observer chunks were coerced to
  arrays by `np.array_split`).
- 3D sight-line preparation is about 35× faster (lazy per-building extrusion, broadcast distance
  filter, vectorised segment construction).
- `matplotlib.cm.get_cmap` removal handled (`plt.get_cmap`).

### Project
- Offline test suite expanded (coverage ≈ 92 %), Codecov reporting in CI, test fixtures moved from
  Shapefile to GeoPackage.

## [2.0.1] — 2026-07-19

### Fixed
- `regions._graph_from_gdfs` still passed a third positional argument to `graph_fromGDF` after the
  column-parameter removal, which broke the districts stage
  (`amend_nodes_membership` → `_check_disconnected_districts`).

## [2.0.0] — 2026-07-18

A ground-up refactor of the package layout and public API.

### Breaking
- **Fixed schema.** The `nodeID_column` / `edgeID_column` parameters are removed from the network,
  topology, graph and height functions; inputs must use the `nodeID`, `edgeID`, `u`, `v` schema.
  Use the `validate_*` / `standardize_*` helpers to bring other data to it.
- **Module layout.** The old modules (`graph_clean`, `graph_consolidate`, `graph_topology`,
  `graph_load`, `graph_centrality`, `buildings_*`, `land_use_*`, `plot`, `utilities`, …) are
  replaced by `schema`, `adapters`, `io`, `osm`, `pedestrian`, `network`, `network_topology`,
  `graph`, `angles`, `centrality`, `barriers`, `regions`, `landuse/`, `buildings`, `height`,
  `landmarks`, `scoring`, `visibility2d`, `visibility3d` and `plotting/`.
- **Dependencies.** `igraph` and `python-louvain` are core dependencies; the `centrality` and
  `regions` extras are gone. `osmnx` is core. Optional extras are `plot`, `height`,
  `visibility3d` and `all`. Python ≥ 3.10.
- `land_use_field` renamed to `land_uses_raw_field`.

### Added
- Lazy public API: `import cityImage` loads no submodule or heavy dependency until a symbol is used;
  optional dependencies raise an `ImportError` naming the extra to install.
- Pedestrian network filtering from OSM (`pedestrian`), with a `ped` column (`"yes"` /
  `"noEvidence"`) and a `sidewalk_policy` to reconcile separately mapped sidewalks with road
  centrelines.
- `buildings_base_from_dtm` and `assign_elevations_from_rasters`.
- Sphinx documentation on ReadTheDocs (HTML, PDF, EPUB) with notebooks and an autosummary API
  reference; ruff linting/formatting; CI with unit, full-optional and network jobs.

### Changed
- `fix_network_topology` nodes the network only at shared, un-noded vertices, so bridges, tunnels
  and 2D-only crossings are left intact; pseudo-node simplification and network cleaning are
  faster.

### Fixed
- `.loc` lookups by ID in `barriers`, `regions`, `network_topology` and `visibility3d` failed on
  frames with non-contiguous IDs and a plain `RangeIndex` (e.g. GeoPackage reloads).

## Earlier releases

Versions before 2.0.0 were not tagged consistently and their commit history is terse; the list
below records what the repository shows.

| Version | Date | Notes |
| --- | --- | --- |
| 1.2.3 | 2026-05 | Packaging moved to `pyproject.toml`; `buildings_*` and `land_use_*` modules replace `landmarks.py`, `land_use.py` and `visibility.py`. |
| 1.2.2 | 2025-08 | `graph_clean`, `graph_consolidate`, `graph_topology`, `graph_load` and `graph_centrality` replace `clean.py`, `load.py` and `centrality.py`. |
| 1.21 | 2024-07 | `visibility.py` module added; documentation notebooks. |
| 1.10 | 2024-04 | Documentation and notebooks. |
| 1.0 | 2023-06 | `colors.py` added; `simplify_streets`, `simplify_junctions` and `transport_network` modules removed. |
| 0.14 | 2022-01 | Modules include `clean`, `simplify_streets`, `simplify_junctions`, `transport_network`, `barriers`, `regions`, `landmarks`, `land_use`. |
| 0.12 | 2020-06 | Travis CI, requirement fixes. |
| 0.1 – 0.11 | 2019-11 | Initial releases, following Filomena, Verstegen & Manley (2019), *A computational approach to The Image of the City*, Cities 89. |

[Unreleased]: https://github.com/g-filomena/cityImage/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/g-filomena/cityImage/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/g-filomena/cityImage/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/g-filomena/cityImage/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/g-filomena/cityImage/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/g-filomena/cityImage/releases/tag/v2.0.0
