"""Optional dependency boundary tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.audit_optional_import_boundaries import run_audit


def test_optional_dependency_static_boundary_passes_for_production_code():
    repo_root = Path(__file__).resolve().parents[1]

    findings = run_audit(repo_root, [repo_root / "cityImage"])

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
