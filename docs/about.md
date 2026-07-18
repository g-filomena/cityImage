# About *cityImage*

**cityImage** is developed by [Gabriele Filomena, Geographic Data Science Lab, University of Liverpool](https://www.liverpool.ac.uk/environmental-sciences/staff/gabriele-filomena/).

The package originated from research on the computational formulation of Kevin Lynch's *Image of the City*. Its development mainly took place during work at [The Bartlett Centre for Advanced Spatial Analysis](https://www.ucl.ac.uk/bartlett/casa/bartlett-centre-advanced-spatial-analysis), University College London, and at the [Institute for Geoinformatics](https://www.uni-muenster.de/Geoinformatics/), University of Münster, under the supervision and support of Ed Manley and Judith Verstegen, respectively.

## Purpose

`cityImage` provides reproducible geospatial workflows for identifying computational analogues of Lynchian urban elements:

- paths and nodes from street-network structure and graph measures;
- edges from barriers such as water, railways, parks, and major roads;
- districts from network partitions and gateway logic;
- landmarks from structural, visual, cultural, and pragmatic salience;
- imageability scores from composable cityImage indicators.

The package is not a replacement for GeoPandas, Shapely, OSMnx, NetworkX, iGraph, rasterio, or PyVista. Instead, it defines a semantic layer on top of those libraries. Generic data acquisition, spatial operations, graph algorithms, raster processing, and mesh/ray operations are delegated to specialised packages; cityImage keeps the schema, topology, Lynchian classification, and scoring semantics consistent across workflows.

## Design after the refactor

The current API separates cityImage-owned semantics from external-library responsibilities:

- `cityImage.io` bridges local geospatial files into cityImage schemas.
- `cityImage.osm` bridges OSMnx outputs into cityImage schemas.
- `cityImage.network` and `cityImage.network_topology` preserve node/edge/topology semantics.
- `cityImage.landuse`, `cityImage.barriers`, `cityImage.regions`, `cityImage.landmarks`, and `cityImage.scoring` implement the core urban-image logic.
- `cityImage.visibility3d` remains optional because 3D sight-line computation requires heavier mesh and parallel-processing dependencies.

This design keeps the core installation lighter while retaining advanced optional workflows for plotting, building-height estimation, and 3D visibility.

## Terms and conditions

All content on this site is licensed under a [Creative Commons Attribution license](https://creativecommons.org/licenses/by/3.0/).
