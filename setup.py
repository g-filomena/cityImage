from setuptools import setup, find_packages

with open("README.md", "r") as readme_file: readme = readme_file.read()

requirements = ["osmnx>=0.10", "seaborn>=0.9.0",  "descartes>=1.1", "matplotlib>=3.1",
    "networkx>=2.4", "osmnx>=0.10", "pysal>=2.1", "python-louvain>=0.13"]
	
setup(
    name='urbanFormPy',
    description='A package for studying urban Form and obtaining the computational Image of the City',
    long_description = readme,
    long_description_content_type="text/markdown",
	packages=find_packages('urbanFormPy'),  # include all packages under src
    package_dir={'':'urbanFormPy'},   # tell distutils packages are under src
	version='0.1',
    author='Gabriele Filomena',
    author_email='gabriele.filomena@uni-muenster.de',
	install_requires=requirements,
    keywords=['pip','urban Form Analysis','Computational Image of the City'],
    )
