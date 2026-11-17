# About *cityImage*

**cityImage** is developed by [Gabriele Filomena, Geographic Data Science Lab, University of Liverpool](https://www.liverpool.ac.uk/environmental-sciences/staff/gabriele-filomena/).

The package originated from research on the computational formulation of Kevin Lynch's *Image of the City*. Its development mainly took place during work at [The Bartlett Centre for Advanced Spatial Analysis](https://www.ucl.ac.uk/bartlett/casa/bartlett-centre-advanced-spatial-analysis), University College London, and at the [Institute for Geoinformatics](https://www.uni-muenster.de/Geoinformatics/), University of Münster, under the supervision and support of Ed Manley and Judith Verstegen, respectively.

## Purpose

`cityImage` derives the elements of Lynch's *Image of the City* from geospatial data: paths and nodes from the street network and its centrality, edges from barriers such as water, railways, parks and major roads, districts from partitions of the network, with their gateways, and landmarks from the structural, visual, cultural and pragmatic salience of buildings.

It builds on GeoPandas, Shapely, OSMnx, NetworkX, igraph and rasterio rather than replacing them. Data acquisition, spatial operations, graph algorithms and raster processing are theirs; the schemas, the network topology, the scores and the Lynch classes are cityImage's.

## Design

- `cityImage.io` reads local files into the cityImage schemas; `cityImage.osm` does the same with OSMnx downloads.
- `cityImage.network` and `cityImage.network_topology` build and clean street networks, keeping the node and edge identifiers.
- `cityImage.landuse`, `cityImage.barriers`, `cityImage.regions`, `cityImage.landmarks` and `cityImage.scoring` derive the elements and their scores.
- Plotting, interactive maps, building heights from rasters and 3D sight lines need the optional extras `plot`, `interactive`, `height` and `visibility3d`, so the core install stays light.

## Terms and conditions

All content on this site is licensed under a [Creative Commons Attribution license](https://creativecommons.org/licenses/by/3.0/).
