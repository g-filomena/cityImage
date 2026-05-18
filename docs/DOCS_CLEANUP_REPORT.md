# Docs cleanup report

Updated files:

- `about.md`: spelling and wording cleanup.
- `api.rst`: removed legacy/deleted API names and reorganised public API around the refactored modules.
- `index.rst`: updated introduction, install instructions, and module descriptions for `io.py` / `osm.py` / split visibility.
- `Makefile`: kept standard Sphinx targets and added explicit `clean` / `html` targets.
- `module_ownership.md`: added `io.py`, `osm.py`, `visibility2d.py`; updated public API wording after hard cleanup.
- `requirements.txt`: uses `cityImage[all]` for optional API docs, removes redundant theme dependency, adds `ipykernel`.
- `conf.py`: version now follows `cityImage.__version__`, fixed path setup, excludes checkpoints, disables notebook execution during Sphinx builds.

Notes:

- These files are ready to overwrite the docs folder originals.
- Notebook runtime testing should stay separate from Sphinx build.
