"""Tests for lightweight top-level imports and lazy optional dependencies."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_import_cityimage_does_not_eagerly_import_heavy_optional_modules():
    """Importing cityImage should not import optional feature modules."""
    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys
        import cityImage

        forbidden = [
            "cityImage.buildings_height",
            "cityImage.buildings_visibility",
            "cityImage.graph_centrality",
            "cityImage.graph_load",
            "cityImage.plot",
            "cityImage.regions",
            "osmnx",
            "pyvista",
            "rasterstats",
            "igraph",
            "community",
            "matplotlib",
        ]
        loaded = [name for name in forbidden if name in sys.modules]
        if loaded:
            raise AssertionError(f"Optional modules imported eagerly: {loaded}")
        """
    )
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_lazy_top_level_access_still_exposes_core_land_use_helpers():
    """The historical import style still works for core helpers."""
    import cityImage as ci

    assert callable(ci.classify_sparse_land_uses)
    assert callable(ci.classify_land_use)
    assert callable(ci.validate_buildings_gdf)
