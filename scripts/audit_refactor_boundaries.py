"""Static refactor-boundary audit for cityImage.

This script checks that the hard refactor did not leave production/test code
importing deleted flat modules such as ``cityImage.utilities`` or
``cityImage.land_use_sparse``.

It is intentionally AST-based: string mentions in documentation or in public API
contract tests are allowed. Actual imports are not.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

DELETED_FLAT_MODULES = {
    "buildings_height",
    "buildings_landmarks",
    "buildings_load",
    "buildings_visibility",
    "colors",
    "graph_centrality",
    "graph_clean",
    "graph_consolidate",
    "graph_load",
    "graph_topology",
    "land_use_assign",
    "land_use_classify",
    "land_use_derive",
    "land_use_sparse",
    "land_use_tags",
    "land_use_utils",
    "plot",
    "utilities",
}

PACKAGE_NAME = "cityImage"

EXPECTED_NEW_PATHS = {
    "cityImage/adapters.py",
    "cityImage/angles.py",
    "cityImage/barriers.py",
    "cityImage/centrality.py",
    "cityImage/data_utils.py",
    "cityImage/geometry.py",
    "cityImage/graph.py",
    "cityImage/landmarks.py",
    "cityImage/network.py",
    "cityImage/io.py",
    "cityImage/osm.py",
    "cityImage/network_topology.py",
    "cityImage/regions.py",
    "cityImage/scoring.py",
    "cityImage/schema.py",
    "cityImage/landuse/__init__.py",
    "cityImage/landuse/assign.py",
    "cityImage/landuse/classify.py",
    "cityImage/landuse/derive.py",
    "cityImage/landuse/sparse.py",
    "cityImage/landuse/tags.py",
    "cityImage/landuse/utils.py",
    "cityImage/plotting/__init__.py",
    "cityImage/plotting/colors.py",
    "cityImage/plotting/static.py",
}


def _module_is_forbidden(module_name: str) -> bool:
    """Return True if an absolute module path points at a deleted flat module."""
    forbidden_absolute = {f"{PACKAGE_NAME}.{name}" for name in DELETED_FLAT_MODULES}

    return module_name in forbidden_absolute or any(
        module_name.startswith(f"{forbidden}.") for forbidden in forbidden_absolute
    )


def _relative_module_name(
    current_file: Path, module: str | None, level: int, root: Path
) -> str | None:
    """Resolve simple relative imports inside the cityImage package."""
    if level <= 0:
        return module

    try:
        relative = current_file.relative_to(root)
    except ValueError:
        return module

    if not relative.parts or relative.parts[0] != PACKAGE_NAME:
        return module

    package_parts = list(relative.with_suffix("").parts[:-1])
    if level > len(package_parts):
        return module

    base_parts = package_parts[: len(package_parts) - level + 1]
    if module:
        base_parts.extend(module.split("."))

    return ".".join(base_parts)


def _iter_python_files(paths: list[Path]) -> list[Path]:
    """Return Python files under the supplied files/directories."""
    python_files: list[Path] = []

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            python_files.append(path)
        elif path.is_dir():
            python_files.extend(sorted(path.rglob("*.py")))

    return sorted(set(python_files))


def find_deleted_files(root: Path) -> list[Path]:
    """Return deleted flat module files that still physically exist."""
    return sorted(
        root / PACKAGE_NAME / f"{module_name}.py"
        for module_name in DELETED_FLAT_MODULES
        if (root / PACKAGE_NAME / f"{module_name}.py").exists()
    )


def find_missing_expected_paths(root: Path) -> list[Path]:
    """Return expected refactor paths missing from the working tree."""
    return sorted(
        root / rel_path for rel_path in EXPECTED_NEW_PATHS if not (root / rel_path).exists()
    )


def find_forbidden_imports(root: Path, paths: list[Path]) -> list[str]:
    """Return human-readable forbidden import findings."""
    findings: list[str] = []

    for path in _iter_python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(f"{path}: syntax error while parsing: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _module_is_forbidden(alias.name):
                        findings.append(f"{path}:{node.lineno}: forbidden import {alias.name!r}")

            elif isinstance(node, ast.ImportFrom):
                resolved = _relative_module_name(path, node.module, node.level, root)
                if resolved and _module_is_forbidden(resolved):
                    findings.append(f"{path}:{node.lineno}: forbidden from-import {resolved!r}")

    return findings


def run_audit(root: Path, paths: list[Path]) -> list[str]:
    """Run all static boundary checks and return findings."""
    findings: list[str] = []

    for path in find_deleted_files(root):
        findings.append(f"{path}: deleted flat module file still exists")

    for path in find_missing_expected_paths(root):
        findings.append(f"{path}: expected refactor file is missing")

    findings.extend(find_forbidden_imports(root, paths))

    return findings


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=[PACKAGE_NAME, "tests", "scripts"],
        help="Files/directories to scan for Python imports.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root. Defaults to the current working directory.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = [(root / path).resolve() for path in args.paths]

    findings = run_audit(root, paths)

    if findings:
        print("Refactor-boundary audit failed:")
        for finding in findings:
            print(f" - {finding}")
        return 1

    print("Refactor-boundary audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
