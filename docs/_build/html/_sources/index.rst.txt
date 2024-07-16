.. cityImage documentation master file, created by
   sphinx-quickstart on Thu Feb  1 11:53:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cityImage's documentation
=========================
Introduction
------------
`cityImage` is a Python package that supports the extraction of The Image of the City by using geospatial datasets, either provided by the user or directly downloaded from OpenStreetMap. 

Theoretical considerations
--------------------------
The Image of the City is a community mental representation of a city resulting from the overlap of people's individual images of the city. The term *image of the city* coincides, to some extent, with other notions such as cognitive map, mental images, image schemata and so on, advanced to conceptualise cognitive representations of the urban space. 
In a nuthsell, these are cognitive mental models employed to move within the urban space and interact with its elements. Lynch identified the existence of five urban elements that, despite individual nuances and differences, are shared across the citizens and the visitors of a certain city: paths, nodes, edges, districts and landmarks.

Whereas in his original method Lynch employed qualitative interviews to identify the image of the city of Los Angeles, Boston and New Jersey, this library allows the identification of such shared salient elements from geospatial datasets.
For more details on Lynch’s work and the geo-computational formulation you can refer to: Filomena, G., Verstegen, J. A., & Manley, E. (2019). [A computational approach to The Image of the City. Cities, 89, 14–25](https://doi.org/10.1016/j.cities.2019.01.006).

How to install
--------------
from PyPI using pip:

	pip install cityImage
	
Main purposes
-------------
The set of different functions implemented in `cityImage` enable to identify the five Lynchan elements:

 - Crucial **Nodes** and **Paths** on the basis of betweenness centrality measures computed on the street network. Paths can be identified both from a primal and a dual graph representation of the street network, thus making use of angularity measures.
 - Urban regions (**districts**) by means of network community detection techniques (i.e. modularity optimisation).
 - Natural and artificial barriers (**edges**) such as rivers, railway structures and main roads.
 - Computational **landmarks** on the basis of visual, structural, cultural and pragmatic salience.

Additional functions
--------------------
The library, moreover, presents a set of novel spatial methods and algorithms, otherwise not implemented in python environments:

 - It support straightforward scraping from OSM of buildings and street network data into `GeoPandas` GeoDataframes.
 - It allows cleaning and simplifying grpah representations of the street network (module `clean`). 
 - It allows straightforward operations on dual graph representations of the street network (modules `load` and `graph`).
 - It computes 3d sight-lines towards buildings, and computes actual 3d visibility, considering a set of obstructions, by exploiting the capabilities of the `pyvista` package. This is a valid alternative to sighlines computation in ArcGis.
 - It provides the user with ready-to-use visualisation tools that support comparison of metrics of interest across cities (module `plot`), such as accessibility values, centrality values, etc.

The examples in the rest of the documentation, present the capabilities of the library.

Documentation content
---------------------

.. toctree::
   :maxdepth: 1

   Home <self>
   notebooks/userGuide.rst
                

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`