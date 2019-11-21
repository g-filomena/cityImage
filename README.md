# The Computational Image of the City 
[![DOI](https://zenodo.org/badge/130851801.svg)](https://zenodo.org/badge/latestdoi/130851801)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/g-filomena/Computational-Image-of-the-City/master)

**A tool for extracting the Computational Image of the City from geospatial datasets**


<img src="image_of_the_city_boston.jpg" alt="Image of the City (© The MIT Press)" />

## Introduction

This repository provides a set of functions to extract salient urban features in line with the definitions laid down by Kevin Lynch in [The Image of The City](https://mitpress.mit.edu/books/image-city) using open and freely available geospatial datasets.

The tools are written in Python and uses a series of libraries, including [Geopandas](http://geopandas.org), [OSMNx](https://osmnx.readthedocs.io/en/stable/), and related dependencies.

The methods are fully documented in *A Computational approach to ‘The Image of the City’* by Filomena, Verstegen, and Manley, published in [Cities](https://doi.org/10.1016/j.cities.2019.01.006).

## Functions

The functions are divided in three scripts, each contributing methods for extracting different elements:

* **[street_network_functions.py](street_network_functions.py)** - which provides methods for extraction of *Nodes, Paths and Districts*, as well as basic street network operations (loading or downloading network, cleaning, simplifcation, etc.).  
* **[landmarks_functions.py](landmarks_functions.py)** - which allows for extraction of landmarks. Note that this function relies on ArcGIS functions, provided in [sight_lines.py](sight_lines.py) for computation of 3D sight lines.
* **[utilities.py](utilities.py)** - which contains a variety of functions for handling and plotting data.

The pipeline and the usage of the functions are detailed in the Jupyter notebooks **[1_Nodes_paths_districts.ipynb](1_Nodes_paths_districts.ipynb)** and **[2_Landmarks.ipynb](2_Landmarks.ipynb)**.

