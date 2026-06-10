# cityImage module ownership map

This file records the current post-refactor ownership boundary.

Status meanings:

- `keep`: cityImage owns the semantics.
- `keep thin`: cityImage owns the boundary/output semantics but delegates algorithms, acquisition, or file IO.
- `keep small`: small helper module retained because multiple owned modules need it.

| Module | Owner | Status | Optional deps | Delegates to | Keep reason |
|---|---|---|---|---|---|
| `cityImage/__init__.py` | public API | keep | none | lazy import map | Top-level import contract and curated public surface. |
| `cityImage/adapters.py` | schema/adapters | keep | none | pandas/geopandas conventions | Normalises user data into cityImage column/schema expectations. |
| `cityImage/schema.py` | schema/adapters | keep | none | pandas/geopandas conventions | Defines and validates the internal nodes/edges/buildings schema. |
| `cityImage/io.py` | file IO bridge API | keep thin | none | GeoPandas | Preserves file-based notebook/user pipelines while delegating file reading. |
| `cityImage/osm.py` | OSM bridge API | keep thin | osmnx | OSMnx | Preserves OSM notebook/user pipelines while delegating acquisition. |
| `cityImage/network.py` | network construction | keep thin | none | GeoPandas/Shapely | Converts line data into cityImage nodes/edges schema. |
| `cityImage/pedestrian.py` | pedestrian network acquisition | keep thin | osmnx | OSMnx | Pedestrian-specific OSM/network workflow boundary and filtering semantics. |
| `cityImage/network_topology.py` | network topology | keep | none | shapely/geopandas/networkx | Custom geometry/topology repair operations not safely delegated wholesale. |
| `cityImage/angles.py` | graph semantics | keep | none | math/shapely only | Preserves custom endpoint-oriented street-segment angle semantics. |
| `cityImage/graph.py` | graph core | keep | none | NetworkX | Prepared GDF to graph and dual graph semantics. |
| `cityImage/centrality.py` | centrality outputs | keep thin | igraph | igraph | Preserves centrality output semantics used by scoring; delegates algorithms. |
| `cityImage/regions.py` | regions/districts | keep | python-louvain/community | NetworkX/community-louvain/Shapely | Keeps district/gateway semantics and mapping between primal/dual outputs. |
| `cityImage/barriers.py` | barriers | keep | none | Shapely/GeoPandas | Keeps custom barrier extraction/crossing semantics. |
| `cityImage/landmarks.py` | landmarks | keep | none | Shapely/GeoPandas | Hard Lynchian landmark scoring and local landmark semantics. |
| `cityImage/scoring.py` | imageability scoring | keep | none | NumPy/Pandas | Combines Lynchian/imageability indicators into score outputs. |
| `cityImage/height.py` | building height | keep thin | rasterio/rasterstats | rasterstats/rasterio | Thin raster/zonal-stat boundary plus cityImage building-height schema. |
| `cityImage/visibility2d.py` | 2D visibility | keep | none | Shapely/GeoPandas | Preserves 2D visibility semantics. |
| `cityImage/visibility3d.py` | 3D visibility | keep | pyvista/dask/psutil/tqdm | PyVista/Dask | Owns 3D sight-line workflow and output schema; delegates mesh/ray operations. |
| `cityImage/geometry.py` | geometry helpers | keep small | none | Shapely/GeoPandas | Small custom helpers used by topology/visibility/regions. |
| `cityImage/data_utils.py` | data helpers | keep small | none | Pandas/NumPy | Small scaling/conversion helpers used internally. |
| `cityImage/landuse/` | land-use package | keep | none | GeoPandas/Pandas | Derivation, classification, sparse representation, and assignment of land-use semantics. |
| `cityImage/plotting/` | plotting package | keep thin | matplotlib/mapclassify | Matplotlib/mapclassify | Optional static plotting entry points isolated from core import. |
