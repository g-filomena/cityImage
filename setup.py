from setuptools import setup, find_packages

with open("README.md", "r") as readme_file: readme = readme_file.read()

requirements = ["osmnx>=0.11", "seaborn>=0.10.0",  "matplotlib>=3.1", "networkx>=2.4", "python-louvain>=0.13", "Shapely==1.7.0", "Rtree>=0.9"]
    
setup(
    name="cityImage",
    description="A package for studying urban form and obtaining the computational Image of the City",
    long_description = readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version="0.14",
    author="Gabriele Filomena",
    author_email="gabriele.filomena@uni-muenster.de",
    install_requires=requirements,
    keywords=["pip","urban Form Analysis","Computational Image of the City", "Kevin Lynch", "cognitive map"],
    )
