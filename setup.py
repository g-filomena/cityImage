from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = ["osmnx>=1.9.1", "python-louvain>=0.16", "pyvista>=0.43.2", "mapclassify>=2.5.0", "igraph>=0.11.6", "tqdm>=4.65.0"]
    
setup(
    name="cityImage",
    description="A package for studying urban form and obtaining the computational Image of the City",
    long_description = readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version="1.21",
    author="Gabriele Filomena",
    author_email="gabriele.filomena@liverpool.ac.uk",
    install_requires=requirements,
    keywords=["urban Form Analysis","Computational Image of the City", "Kevin Lynch", "cognitive maps"],
    )
