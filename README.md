[![CodeFactor](https://www.codefactor.io/repository/github/g-filomena/cityimage/badge)](https://www.codefactor.io/repository/github/g-filomena/cityimage)
[![Actions Status](https://github.com/g-filomena/cityimage/workflows/tests/badge.svg)](https://github.com//g-filomena/cityimage/actions?query=workflow%3Atests)
[![codecov](https://codecov.io/gh/g-filomena/cityImage/branch/master/graph/badge.svg)](https://codecov.io/gh/g-filomena/cityImage)
[![PyPI version](https://badge.fury.io/py/cityImage.svg)](https://badge.fury.io/py/cityImage)
[![Documentation Status](https://readthedocs.org/projects/cityimage/badge/?version=latest)](https://cityimage.readthedocs.io/en/latest/?badge=latest)

# cityImage

**A tool for analysing urban legibility and extracting The Computational Image of the City**
For full documentation and examples see [the user manual](https://cityimage-docs.readthedocs.io/en/latest/).

This repository provides a set of functions to extract salient urban features in line with the definitions laid down by Kevin Lynch in [The Image of The City](https://mitpress.mit.edu/books/image-city) using open and freely available geospatial datasets.
The methods are fully documented in *A Computational approach to The Image of the City* by Filomena, Verstegen, and Manley, published in [Cities](https://doi.org/10.1016/j.cities.2019.01.006).

The tools are written in Python and requires:

* [OSMNx](https://osmnx.readthedocs.io/en/stable/).
* [PyVista](https://docs.pyvista.org/version/stable/).
* [python-louvain](https://github.com/taynaud/python-louvain).
* [mapclassify](https://github.com/pysal/mapclassify).

It is built on [GeoPandas](https://github.com/geopandas/geopandas), [NetworkX](https://github.com/networkx/networkx), and [Shapely](https://github.com/shapely/shapely).

## How to install *cityImage*

    pip install cityImage
