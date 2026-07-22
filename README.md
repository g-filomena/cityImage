[![Actions Status](https://github.com/g-filomena/cityImage/actions/workflows/tests.yml/badge.svg)](https://github.com/g-filomena/cityImage/actions)
[![PyPI version](https://badge.fury.io/py/cityImage.svg)](https://badge.fury.io/py/cityImage)
[![Documentation Status](https://readthedocs.org/projects/cityimage/badge/?version=latest)](https://cityimage.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/g-filomena/cityImage/branch/master/graph/badge.svg)](https://codecov.io/gh/g-filomena/cityImage)

<img src="https://raw.githubusercontent.com/g-filomena/cityImage/master/docs/_static/logo.png" alt="cityImage logo" width="33%">

# cityImage

**cityImage** is a Python package for analysing urban legibility and extracting a
computational version of Kevin Lynch's *Image of the City* from geospatial data.

The package works with user-provided GeoPandas datasets and with OpenStreetMap
data acquired through OSMnx. The refactored API keeps cityImage-specific schema,
topology, barrier, district, landmark, and scoring semantics while delegating
generic acquisition, spatial joins, plotting, graph algorithms, and raster
operations to established libraries.

For full documentation and examples, see the
[cityImage documentation](https://cityimage.readthedocs.io/en/latest/).

## What cityImage does

cityImage operationalises Lynchian urban elements as reproducible geospatial
workflows:

- **Paths and nodes** from street-network structure, centrality, graph topology,
  and optional angular/dual-graph semantics.
- **Edges** from structuring barriers such as water, railways, large parks, and
  major roads.
- **Districts** from network partitions, polygonisation, and gateway detection.
- **Landmarks** from structural, visual, cultural, and pragmatic salience.
- **Imageability scores** from composable landmark and urban-element indicators.

The package is designed as a semantic layer over geospatial data rather than as
a replacement for general-purpose GIS, graph, or OSM tooling.

## How cityImage differs from other geospatial libraries

cityImage deliberately delegates generic operations to specialised libraries:

- **GeoPandas/Shapely** handle geometry containers, CRS conversion, spatial
  predicates, overlay-like operations, and file IO.
- **OSMnx** handles OpenStreetMap acquisition and raw OSM graph/feature access.
- **NetworkX/iGraph** handle graph representation and graph algorithms.
- **python-louvain** handles modularity-based community detection.
- **Rasterio/rasterstats** handle raster sampling and zonal statistics.
- **Dask** parallelises batching in the optional 3D sight-line workflow.

The specificity of cityImage is different: it defines the **urban-image
semantics** that sit on top of those tools. In practice this means cityImage
focuses on:

- stable nodes/edges/buildings schemas used across the package;
- conversion of raw files or OSM outputs into those schemas;
- graph cleaning and topology repair where the cityImage node/edge relationship
  must be preserved;
- dual/primal graph transformations with street-segment angle semantics;
- barrier extraction and assignment as Lynchian edge semantics;
- district and gateway logic from network partitions;
- landmark/imageability component scores and final score composition;
- example-ready workflows that keep the same conceptual outputs across cities.

So, for example, OSMnx can acquire a walk network, GeoPandas can spatially join
layers, and NetworkX can compute graph measures. cityImage connects those pieces
into a reproducible workflow that returns *paths, nodes, edges, districts,
landmarks,* and imageability scores in a consistent schema.

## Background

The methods are based on:

Filomena, G., Verstegen, J. A., & Manley, E. (2019).
[A computational approach to The Image of the City](https://doi.org/10.1016/j.cities.2019.01.006).
*Cities*, 89, 14–25.

## Installation

Core install:

```bash
pip install cityImage
```

The core install includes graph centrality (igraph) and region detection
(python-louvain), and keeps only heavier, workflow-specific dependencies out of the
base environment. Use extras for those:

```bash
pip install "cityImage[plot]"         # plotting helpers (matplotlib, mapclassify)
pip install "cityImage[height]"       # DEM/DTM height helpers
pip install "cityImage[visibility3d]" # 3D sight-line workflow
pip install "cityImage[all]"          # all optional runtime dependencies
```

For development:

```bash
pip install -e ".[all,test,docs,dev]"
```

### Why 3D visibility is optional

The 3D visibility workflow depends on additional libraries — Dask for parallel
batching and psutil for memory-aware chunking. These are useful for
line-of-sight and obstruction workflows, but they are not required for core
network construction, land-use assignment, barriers, districts, landmark
scoring, or 2D visibility. For that reason, `cityImage.visibility3d` should
remain an optional extra.

This keeps the default installation lighter and makes it possible to use
cityImage in environments where the 3D sight-line extras are unnecessary or
difficult to install.

## Main API areas

The current API separates cityImage-owned semantics from external libraries.
Submodules marked *(extra: X)* need the corresponding optional install (see
[Installation](#installation)); the rest are available with the core install.

- `cityImage.schema` and `cityImage.adapters`: stable node/edge/building schemas,
  validation, and input standardisation.
- `cityImage.io`: file/GeoPandas loading into cityImage schemas.
- `cityImage.osm`: OSMnx acquisition into cityImage schemas.
- `cityImage.pedestrian`: pedestrian-network filtering and construction from OSM
  highway features.
- `cityImage.network` and `cityImage.network_topology`: street-network
  construction, cleaning, simplification, and topology repair.
- `cityImage.graph` and `cityImage.angles`: primal/dual graph semantics and
  angular relationships.
- `cityImage.centrality`: node/edge centrality wrappers (iGraph-based measures;
  iGraph is a core dependency).
- `cityImage.barriers`: natural and artificial barriers such as rivers,
  railways, parks, and major roads.
- `cityImage.regions`: districts and gateways from network partitions
  (modularity-based detection via python-louvain, a core dependency).
- `cityImage.landuse`: land-use derivation, classification, sparse
  representation, and assignment.
- `cityImage.buildings`: building selection and study-area helpers.
- `cityImage.height`: DEM/DTM-based building heights. *(extra: height)*
- `cityImage.landmarks` and `cityImage.scoring`: Lynchian landmark and
  imageability scoring.
- `cityImage.visibility2d`: 2D visibility workflows.
- `cityImage.visibility3d`: optional 3D sight-line workflows. *(extra: visibility3d)*
- `cityImage.plotting`: optional static plotting helpers. *(extra: plot)*

## Minimal example

```python
import cityImage as ci

nodes, edges = ci.network_from_osm(
    "Susa, Italy",
    download_method="OSMplace",
    network_type="walk",
    crs="EPSG:32632",
)

buildings = ci.buildings_from_osm(
    "Susa, Italy",
    download_method="OSMplace",
    crs="EPSG:32632",
)

barriers = ci.barriers_from_osm(
    "Susa, Italy",
    download_method="OSMplace",
    crs="EPSG:32632",
)
```

## Development checks

```bash
ruff check cityImage tests scripts
pytest -m "not network" -ra
python -m build
twine check dist/*
```

## License

cityImage is distributed under the GNU General Public License v3.0 or later.
