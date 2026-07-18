"""Static optional-dependency boundary audit for cityImage.

The hard refactor makes most heavy dependencies optional. This audit checks that
optional dependencies are imported only from the modules that own those optional
features.

It scans production code only by default.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

OPTIONAL_IMPORT_ROOTS = {
    "dask",
    "mapclassify",
    "matplotlib",
    "mpl_toolkits",
    "psutil",
    "pyvista",
    "rasterio",
    "rasterstats",
    "tqdm",
}

ALLOWED_BY_ROOT = {
    "dask": {"cityImage/visibility3d.py"},
    "mapclassify": {"cityImage/plotting/colors.py", "cityImage/plotting/static.py"},
    "matplotlib": {"cityImage/plotting/colors.py", "cityImage/plotting/static.py"},
    "mpl_toolkits": {"cityImage/plotting/static.py"},
    "psutil": {"cityImage/visibility3d.py"},
    "pyvista": {"cityImage/visibility3d.py"},
    "rasterio": {"cityImage/height.py"},
    "rasterstats": {"cityImage/height.py"},
    "tqdm": {"cityImage/visibility3d.py"},
}


def _root_module(module_name: str) -> str:
    """Return the top-level import root."""
    return module_name.split(".", maxsplit=1)[0]


def _iter_python_files(paths: list[Path]) -> list[Path]:
    """Return Python files under the supplied files/directories."""
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return sorted(set(files))


def _relative_posix(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path relative to repository root."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_allowed(root_name: str, rel_path: str) -> bool:
    """Return True when an optional dependency is imported from its owner module."""
    return rel_path in ALLOWED_BY_ROOT[root_name]


def find_optional_boundary_violations(root: Path, paths: list[Path]) -> list[str]:
    """Return human-readable optional dependency boundary violations."""
    findings: list[str] = []

    for path in _iter_python_files(paths):
        rel_path = _relative_posix(path, root)

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(f"{rel_path}: syntax error while parsing: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = _root_module(alias.name)
                    if root_name in OPTIONAL_IMPORT_ROOTS and not _is_allowed(root_name, rel_path):
                        findings.append(
                            f"{rel_path}:{node.lineno}: optional import {alias.name!r} "
                            f"is only allowed in {sorted(ALLOWED_BY_ROOT[root_name])}"
                        )

            elif isinstance(node, ast.ImportFrom) and node.module:
                root_name = _root_module(node.module)
                if root_name in OPTIONAL_IMPORT_ROOTS and not _is_allowed(root_name, rel_path):
                    findings.append(
                        f"{rel_path}:{node.lineno}: optional from-import {node.module!r} "
                        f"is only allowed in {sorted(ALLOWED_BY_ROOT[root_name])}"
                    )

    return findings


def run_audit(root: Path, paths: list[Path]) -> list[str]:
    """Run the optional dependency boundary audit."""
    return find_optional_boundary_violations(root, paths)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["cityImage"],
        help="Files/directories to scan. Defaults to production package only.",
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = [(root / path).resolve() for path in args.paths]

    findings = run_audit(root, paths)

    if findings:
        print("Optional-dependency boundary audit failed:")
        for finding in findings:
            print(f" - {finding}")
        return 1

    print("Optional-dependency boundary audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
