"""Import, optional dependency, and static refactor-boundary tests.

Merged from:
- test_lazy_import.py
- test_plotting_package.py
- test_optional_dependency_boundaries.py
- test_refactor_boundaries_static.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.audit_optional_import_boundaries import run_audit as run_optional_import_audit
from scripts.audit_refactor_boundaries import run_audit as run_refactor_boundary_audit


def test_import_cityimage_does_not_eagerly_import_heavy_optional_modules():
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
            "cityImage.plotting",
            "cityImage.plotting.static",
            "cityImage.plotting.colors",
            "pyvista",
            "rasterstats",
            "igraph",
            "community",
            "matplotlib",
            "matplotlib.pyplot",
        ]
        loaded = [name for name in forbidden if name in sys.modules]
        if loaded:
            raise AssertionError(f"Optional modules imported eagerly: {loaded}")
        """
    )
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_lazy_top_level_access_still_exposes_core_helpers():
    import cityImage as ci

    assert callable(ci.classify_sparse_land_uses)
    assert callable(ci.classify_land_use)
    assert callable(ci.validate_buildings_gdf)


def test_plotting_symbols_are_available_from_optional_subpackage_when_matplotlib_is_installed():
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")

    from cityImage.plotting import Plot, kindlmann, plot_gdf, rand_cmap

    assert callable(plot_gdf)
    assert Plot is not None
    assert callable(kindlmann)
    assert callable(rand_cmap)


def test_optional_dependency_static_boundary_passes_for_production_code():
    repo_root = Path(__file__).resolve().parents[1]

    findings = run_optional_import_audit(repo_root, [repo_root / "cityImage"])

    assert findings == []


def test_refactor_boundary_static_audit_passes():
    repo_root = Path(__file__).resolve().parents[1]

    findings = run_refactor_boundary_audit(
        repo_root,
        [
            repo_root / "cityImage",
            repo_root / "tests",
            repo_root / "scripts",
        ],
    )

    assert findings == []


def test_core_import_does_not_import_visibility3d_or_pyvista():
    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys
        import cityImage

        forbidden = [
            "cityImage.visibility3d",
            "pyvista",
            "dask",
        ]
        loaded = [name for name in forbidden if name in sys.modules]
        if loaded:
            raise AssertionError(f"3D visibility dependencies imported eagerly: {loaded}")
        """
    )

    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_visibility3d_symbol_resolves_only_when_optional_dependency_is_available():
    pytest = __import__("pytest")
    pytest.importorskip("pyvista")

    import cityImage as ci

    assert callable(ci.compute_3d_sight_lines)
