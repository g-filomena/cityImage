# Contributing to cityImage

Thanks for your interest in cityImage. Bug reports, documentation fixes, new examples and code
are all welcome.

## Reporting a bug or asking for a feature

Open an [issue](https://github.com/g-filomena/cityImage/issues/new/choose) and pick the bug
report or feature request form. For a bug, a short snippet that reproduces it on a small area
(a few streets, or a small GeoDataFrame you can paste) makes it much faster to fix.

## Setting up a development environment

The geospatial stack installs most reliably from conda-forge:

```bash
git clone https://github.com/g-filomena/cityImage.git
cd cityImage
conda env create -f environment.yml
conda activate cityimage
pip install -e ".[all,test,docs,dev]"
```

With plain pip, `pip install -e ".[all,test,docs,dev]"` alone also works where wheels are
available for your platform.

On Windows, run everything inside the activated environment (`conda activate cityimage`, or
`conda run -n cityimage ...`). Calling the environment's `python.exe` directly leaves its DLLs
off the `PATH`, and native extensions such as matplotlib's can crash.

## Checks to run before opening a pull request

```bash
ruff check cityImage tests scripts
ruff format --check cityImage tests scripts
mypy
pytest -m "not network" -ra
```

The same checks run in CI on Python 3.10 to 3.13. Lines are at most 100 characters;
`ruff format cityImage tests scripts` fixes formatting for you.

## Writing tests

- Put tests in `tests/`, named `test_<module>_<topic>.py`.
- Prefer small synthetic GeoDataFrames built in the test, or the fixtures in `tests/fixtures/`.
- A test that downloads from OpenStreetMap must be marked `network`, so the default offline
  run skips it:

  ```python
  import pytest

  pytestmark = pytest.mark.network  # whole module
  # or
  @pytest.mark.network                 # a single test
  ```

- A test that takes more than a few seconds should also be marked `slow`.
- Run only the offline tests with `pytest -m "not network"`, and the slow ones out too with
  `pytest -m "not network and not slow"`.

## Documentation notebooks

The notebooks in `docs/notebooks` are stored with their outputs; the documentation build does not
run them. Those that read local files find their data in `docs/data`, which is not in the
repository: `python scripts/fetch_docs_data.py` downloads it from Zenodo.

## Code organisation

Each module owns one part of the Image of the City and delegates generic work (OSM acquisition,
geometry, graph algorithms, raster operations) to established libraries. See the
[module ownership map](docs/development/module_ownership.md) before adding a new module or
moving functions between modules.

## Pull requests

- Keep a pull request to one change, and describe what it changes and why.
- Add or update tests for any change in behaviour.
- Add a line to `CHANGELOG.md` under `## [Unreleased]` for user-visible changes.

By contributing you agree that your contribution is licensed under the project's
[license](LICENSE), and that you follow the [code of conduct](CODE_OF_CONDUCT.md).
