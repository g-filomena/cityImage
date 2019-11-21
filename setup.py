from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["osmnx>=0.10", "seaborn>=0.9.0"]
setup(
    name='urbanFormPy',
    packages=['urbanFormPy'],
    description='A package for studying urban Form and obtaining the computational Image of the City',
    long_description=readme,
    long_description_content_type="text/markdown",
	packages=find_packages(),
	version='0.1',
    url='http://github.com/urbanFormPy',
    author='Gabriele Filomena',
    author_email='gabriele.filomena@uni-muenster.de',
	install_requires=requirements,
    keywords=['pip','urban Form Analysis','Computational Image of the City'],
    )
