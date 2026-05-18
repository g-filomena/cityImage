"""Report the cityImage module ownership map.

This is a lightweight architecture/reporting script. It does not import
cityImage modules, so it is safe to run in minimal environments.

Usage
-----
python scripts/report_module_ownership.py
python scripts/report_module_ownership.py --write docs/development/module_ownership.md
python scripts/report_module_ownership.py --fail-on-unassigned
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path("cityImage")


@dataclass(frozen=True)
class ModuleOwner:
    """Ownership metadata for one module or package."""

    path: str
    owner: str
    status: str
    optional_deps: str
    delegates_to: str
    keep_reason: str


OWNERSHIP: dict[str, ModuleOwner] = {
    "cityImage/__init__.py": ModuleOwner(
        "cityImage/__init__.py",
        "public API",
        "keep",
        "none",
        "lazy import map",
        "Top-level import contract and backwards-compatible public surface.",
    ),
    "cityImage/adapters.py": ModuleOwner(
        "cityImage/adapters.py",
        "schema/adapters",
        "keep",
        "none",
        "pandas/geopandas conventions",
        "Normalises user data into cityImage column/schema expectations.",
    ),
    "cityImage/schema.py": ModuleOwner(
        "cityImage/schema.py",
        "schema/adapters",
        "keep",
        "none",
        "pandas/geopandas conventions",
        "Defines and validates the internal nodes/edges/buildings schema.",
    ),
    "cityImage/scoring.py": ModuleOwner(
        "cityImage/scoring.py",
        "imageability scoring",
        "keep",
        "none",
        "numpy/pandas",
        "Combines Lynchian/imageability indicators into score outputs.",
    ),
    "cityImage/angles.py": ModuleOwner(
        "cityImage/angles.py",
        "graph semantics",
        "keep",
        "none",
        "math/shapely only",
        "Preserves custom endpoint-oriented street-segment angle semantics.",
    ),
    "cityImage/graph.py": ModuleOwner(
        "cityImage/graph.py",
        "graph core",
        "keep",
        "none",
        "networkx",
        "Prepared GDF -> graph and dual graph semantics.",
    ),
    "cityImage/network_topology.py": ModuleOwner(
        "cityImage/network_topology.py",
        "network topology",
        "keep",
        "none",
        "shapely/geopandas/networkx",
        "Custom geometry/topology repair operations not safely delegated wholesale.",
    ),
    "cityImage/network.py": ModuleOwner(
        "cityImage/network.py",
        "network acquisition",
        "keep thin",
        "osmnx",
        "osmnx",
        "Thin OSMnx boundary for acquisition; cityImage owns post-processing semantics.",
    ),
    "cityImage/io.py": ModuleOwner(
        "cityImage/io.py",
        "file IO bridge convenience API",
        "keep thin",
        "none",
        "GeoPandas",
        "Preserves file-based notebook/user pipelines while delegating file IO.",
    ),
    "cityImage/osm.py": ModuleOwner(
        "cityImage/osm.py",
        "OSM bridge convenience API",
        "keep thin",
        "osmnx",
        "OSMnx",
        "Preserves OSM notebook/user pipelines while delegating acquisition.",
    ),
    "cityImage/pedestrian.py": ModuleOwner(
        "cityImage/pedestrian.py",
        "pedestrian network acquisition",
        "keep thin",
        "osmnx",
        "osmnx",
        "Pedestrian-specific OSM/network workflow boundary.",
    ),
    "cityImage/centrality.py": ModuleOwner(
        "cityImage/centrality.py",
        "centrality outputs",
        "keep thin",
        "igraph",
        "igraph",
        "Preserves centrality output semantics used by scoring; delegates algorithms.",
    ),
    "cityImage/regions.py": ModuleOwner(
        "cityImage/regions.py",
        "regions/districts",
        "keep",
        "python-louvain/community",
        "networkx/community-louvain/shapely",
        "Keeps district/gateway semantics and mapping between primal/dual outputs.",
    ),
    "cityImage/barriers.py": ModuleOwner(
        "cityImage/barriers.py",
        "barriers",
        "keep",
        "none",
        "shapely/geopandas",
        "Keeps custom barrier extraction/crossing semantics.",
    ),
    "cityImage/buildings.py": ModuleOwner(
        "cityImage/buildings.py",
        "building footprints",
        "keep",
        "none",
        "geopandas/shapely",
        "Owns cityImage building-footprint preparation and building schema semantics.",
    ),
    "cityImage/landmarks.py": ModuleOwner(
        "cityImage/landmarks.py",
        "landmarks",
        "keep",
        "none",
        "shapely/geopandas",
        "Hard Lynchian landmark scoring and local landmark semantics.",
    ),
    "cityImage/height.py": ModuleOwner(
        "cityImage/height.py",
        "building height",
        "keep thin",
        "rasterio/rasterstats",
        "rasterstats/rasterio",
        "Thin raster/zonal-stat boundary plus cityImage building-height schema.",
    ),
    "cityImage/visibility3d.py": ModuleOwner(
        "cityImage/visibility3d.py",
        "3D visibility",
        "keep",
        "pyvista/dask/psutil/tqdm",
        "pyvista/dask",
        "Owns 3D sight-line workflow and output schema; delegates mesh/ray operations.",
    ),
    "cityImage/visibility2d.py": ModuleOwner(
        "cityImage/visibility2d.py",
        "2D visibility",
        "keep",
        "none",
        "shapely/geopandas",
        "Owns 2D visibility polygons/sight-line semantics used by landmarks and imageability scoring.",
    ),
    "cityImage/geometry.py": ModuleOwner(
        "cityImage/geometry.py",
        "geometry helpers",
        "keep small",
        "none",
        "shapely/geopandas/pyproj",
        "Small custom helpers used by topology/visibility/regions.",
    ),
    "cityImage/data_utils.py": ModuleOwner(
        "cityImage/data_utils.py",
        "data helpers",
        "keep small",
        "none",
        "pandas/numpy",
        "Small data-frame/scaling helpers used internally.",
    ),
    "cityImage/landuse/__init__.py": ModuleOwner(
        "cityImage/landuse/__init__.py",
        "land-use package",
        "keep",
        "none",
        "package aggregator",
        "New package namespace for land-use functions.",
    ),
    "cityImage/landuse/assign.py": ModuleOwner(
        "cityImage/landuse/assign.py",
        "land-use assignment",
        "keep",
        "none",
        "geopandas spatial joins",
        "Assigns polygon/point land uses to buildings with cityImage semantics.",
    ),
    "cityImage/landuse/classify.py": ModuleOwner(
        "cityImage/landuse/classify.py",
        "land-use classification",
        "keep",
        "none",
        "internal taxonomy",
        "Maps raw OSM/domain tags into cityImage land-use classes.",
    ),
    "cityImage/landuse/derive.py": ModuleOwner(
        "cityImage/landuse/derive.py",
        "land-use derivation",
        "keep",
        "none",
        "internal taxonomy",
        "Derives raw land-use candidates from OSM-style tag columns.",
    ),
    "cityImage/landuse/sparse.py": ModuleOwner(
        "cityImage/landuse/sparse.py",
        "sparse land-use representation",
        "keep",
        "none",
        "pandas/geopandas",
        "Preserves multi-label sparse land-use semantics.",
    ),
    "cityImage/landuse/tags.py": ModuleOwner(
        "cityImage/landuse/tags.py",
        "land-use taxonomy",
        "keep",
        "none",
        "internal constants",
        "Source of truth for OSM/domain group taxonomy.",
    ),
    "cityImage/landuse/utils.py": ModuleOwner(
        "cityImage/landuse/utils.py",
        "land-use helpers",
        "keep small",
        "none",
        "pandas/list helpers",
        "Private utility functions shared by land-use package modules.",
    ),
    "cityImage/plotting/__init__.py": ModuleOwner(
        "cityImage/plotting/__init__.py",
        "plotting package",
        "keep",
        "matplotlib/mapclassify",
        "package aggregator",
        "New package namespace for plotting helpers.",
    ),
    "cityImage/plotting/colors.py": ModuleOwner(
        "cityImage/plotting/colors.py",
        "plotting colours",
        "keep thin",
        "matplotlib",
        "matplotlib",
        "Colour-map helpers used by plotting; isolated optional dependency.",
    ),
    "cityImage/plotting/static.py": ModuleOwner(
        "cityImage/plotting/static.py",
        "static plotting",
        "keep thin",
        "matplotlib/mapclassify",
        "matplotlib/mapclassify",
        "Convenience static map plotting boundary.",
    ),
}


def iter_package_python_files(root: Path) -> list[str]:
    """Return package Python files as POSIX-style relative paths."""
    package_root = root / PACKAGE_DIR
    return sorted(
        path.relative_to(root).as_posix()
        for path in package_root.rglob("*.py")
        if path.name != "__pycache__"
    )


def build_report(root: Path) -> tuple[str, list[str], list[str]]:
    """Build markdown report, unassigned paths, and missing-owned paths."""
    actual_paths = set(iter_package_python_files(root))
    owned_paths = set(OWNERSHIP)

    unassigned = sorted(actual_paths - owned_paths)
    missing_owned = sorted(owned_paths - actual_paths)

    lines = [
        "# cityImage module ownership map",
        "",
        "This file records the current post-refactor ownership boundary.",
        "",
        "Status meanings:",
        "",
        "- `keep`: cityImage owns the semantics.",
        "- `keep thin`: cityImage owns the boundary/output semantics but delegates algorithms or acquisition.",
        "- `keep small`: small helper module retained because multiple owned modules need it.",
        "",
        "| Module | Owner | Status | Optional deps | Delegates to | Keep reason |",
        "|---|---|---|---|---|---|",
    ]

    for path in sorted(OWNERSHIP):
        item = OWNERSHIP[path]
        exists = "yes" if path in actual_paths else "missing"
        lines.append(
            "| "
            f"`{item.path}` ({exists}) | "
            f"{item.owner} | "
            f"{item.status} | "
            f"{item.optional_deps} | "
            f"{item.delegates_to} | "
            f"{item.keep_reason} |"
        )

    if unassigned:
        lines.extend(["", "## Unassigned package files", ""])
        lines.extend(f"- `{path}`" for path in unassigned)

    if missing_owned:
        lines.extend(["", "## Missing owned package files", ""])
        lines.extend(f"- `{path}`" for path in missing_owned)

    return "\n".join(lines) + "\n", unassigned, missing_owned


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--write", help="Write markdown report to this path.")
    parser.add_argument(
        "--fail-on-unassigned",
        action="store_true",
        help="Exit non-zero if unassigned or missing-owned package files are found.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report, unassigned, missing_owned = build_report(root)

    if args.write:
        output_path = Path(args.write)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(report)

    if args.fail_on_unassigned and (unassigned or missing_owned):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
