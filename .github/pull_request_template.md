## What this changes

<!-- What the pull request changes and why. Link the issue it fixes, if any (e.g. "Fixes #12"). -->

## Checklist

- [ ] `ruff check cityImage tests scripts` and `ruff format --check cityImage tests scripts` pass
- [ ] `mypy` passes
- [ ] `pytest -m "not network"` passes
- [ ] Tests added or updated for the change in behaviour
- [ ] Tests that download from OpenStreetMap are marked `network`
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` for user-visible changes
