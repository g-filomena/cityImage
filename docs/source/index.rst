cityImage |version|
===================
cityImage is a Python package that supports the extraction of *The Image of the City* by using geospatial datasets, either provided by the user or directly downloaded from OpenStreetMap. The Image of the City is a sort of mental representation of a city employed to move within the urban space and interact with its elements. The term overlaps, to some extent, with other notions such as cognitive map, mental images, image schemata and so on, advanced to indicate cognitive representations of the urban space. Lynch identified the existence of five urban elements that, despite individual nuances and differences, are shared across the citizens and the visitors of a certain city: paths, nodes, edges, districts and landmarks. 

Whereas in his original method Lynch employed qualitative interviews to identify the image of the city of Los Angeles, Boston and New Jersey, such library allows the identification of such shared salient elements from geospatial datasets. 

For more details on Lynch’s work and the geo-computational formulation you can refer to:
Filomena, G., Verstegen, J. A., & Manley, E. (2019). A computational approach to ‘The Image of the City.’ *Cities*, 89, 14–25. https://doi.org/10.1016/j.cities.2019.01.006.

In particular, the set of functions enable to identify:

* Crucial nodes and paths on the basis of betweenness centrality measures computed on the street network.
* Paths can be identified both from a primal and a dual graph representation of the street network, thus making use of angularity measures.
* Urban regions, districts, by means of network community detection techniques (i.e. modularity optimisation).
* Natural and artificial (severing) barriers (edges) such as rivers, railway structures and main roads.
* Computational landmarks on the basis of visual, structural, cultural and pragmatic salience.

The library, moreover, presents the following characteristics:
* It allows cleaning the street network
* It allows simplifying dual lines and roundabouts.
* It allows straightforward operations on the dual graph.
* It provides the user with ready to use visualisation tools that support comparison across cities and the depiction of the computational image of the city.

Installation
------------

 
Usage and examples
------------------
The repository `Computational-Image-of-the-City` shows the functionality of the library; different examples for extracting the five elements and for different case-studies are included.

__Computational-Image-of-the-City\: https://github.com/g-filomena/Computational-Image-of-the-City

cityImage is built on top of OSMNx, GeoPandas, NetworkX, as well as maplotlib.

Usage examples and demonstrations of these features are in the `examples`_ GitHub repo. More feature development details are in the `change log`_. Read the `journal article`_ for further technical details. 
The library functions are detailed in the `user reference`_.

.. _user reference: cityImage.html

User reference
--------------

.. toctree::
   :maxdepth: 2

   cityImage


License
-------

The project is licensed under the MIT license.


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`