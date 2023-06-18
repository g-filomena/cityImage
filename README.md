[![CodeFactor](https://www.codefactor.io/repository/github/g-filomena/cityimage/badge)](https://www.codefactor.io/repository/github/g-filomena/cityimage)
[![Actions Status](https://github.com/g-filomena/cityimage/workflows/tests/badge.svg)](https://github.com//g-filomena/cityimage/actions?query=workflow%3Atests)
[![codecov](https://codecov.io/gh/g-filomena/cityImage/branch/master/graph/badge.svg)](https://codecov.io/gh/g-filomena/cityImage)

# cityImage

**A tool for analysing urban legibility and extracting The Computational Image of the City**

This repository provides a set of functions to extract salient urban features in line with the definitions laid down by Kevin Lynch in [The Image of The City](https://mitpress.mit.edu/books/image-city) using open and freely available geospatial datasets.

The tools are written in Python and requires:

* [OSMNx](https://osmnx.readthedocs.io/en/stable/), 
* [PyVista](https://docs.pyvista.org/version/stable/) and related dependencies.
* [python-louvain](https://github.com/taynaud/python-louvain)
* [mapclassify] (https://github.com/pysal/mapclassify)

It is built [GeoPandas](https://github.com/geopandas/geopandas), [NetworkX](https://github.com/networkx/networkx), and [Shapely](https://github.com/shapely/shapely)

The methods are fully documented in *A Computational approach to The Image of the City* by Filomena, Verstegen, and Manley, published in [Cities](https://doi.org/10.1016/j.cities.2019.01.006).

## How to install *cityImage*
.. code-block:: shell

    pip install cityImage