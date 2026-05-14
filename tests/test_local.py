"""Manual/local integration tests are intentionally disabled in pytest.

The old `test_local.py` duplicated `test_cityImage.py`, doubling live OSM calls in
network runs. Keep local exploratory workflows outside pytest, or add them under
another filename that does not match `test_*.py`.
"""

import pytest

pytest.skip(
    "Duplicate local integration workflow disabled; use tests/test_cityImage.py for scheduled network smoke tests.",
    allow_module_level=True,
)
