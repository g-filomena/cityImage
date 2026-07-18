"""Public API contract after the hard refactor.

These tests lock the curated top-level package boundary.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import cityImage as ci

DELETED_MODULES = {
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


CORE_PUBLIC_SYMBOLS = [
    "angle_line_geometries",
    "get_coord_angle",
    "graph_fromGDF",
    "dual_gdf",
    "dual_graph_fromGDF",
    "nodes_degree",
    "center_line",
    "gdf_multipolygon_to_polygon",
    "scaling_columnDF",
    "classify_sparse_land_uses",
    "attach_sparse_land_uses",
    "classify_land_use",
    "regions_from_dual_partition",
    "district_to_nodes_from_edges",
    "find_gateways",
    "barriers_from_osm_features",
    "network_from_file",
    "network_from_osm",
    "buildings_from_file",
    "buildings_from_osm",
    "barriers_from_osm",
    "features_from_osm",
    "validate_nodes_gdf",
    "standardize_edges_gdf",
]

HARD_REMOVED_PUBLIC_SYMBOLS = {
    "get_network_fromOSM",
    "get_network_fromFile",
    "get_buildings_fromOSM",
    "get_buildings_fromFile",
    "dict_to_df",
    "rescale_ranges",
    "resolve_lists_columns",
    "convert_numeric_columns",
    "gdf_from_geometries",
    "envelope_wgs",
    "convex_hull_wgs",
    "min_distance_geometry_gdf",
    "obtain_nodes_gdf",
    "join_nodes_edges_by_coordinates",
    "polygons_gdf_multiparts_to_singleparts",
}


OPTIONAL_SYMBOLS_BY_EXTRA = {
    "matplotlib": ["plot_gdf", "rand_cmap", "Plot"],
    "igraph": ["calculate_centrality", "straightness_centrality"],
    "rasterstats": [
        "buildings_height_from_dem_dtm",
        "buildings_base_from_dtm",
        "assign_elevations_from_rasters",
    ],
    "pyvista": ["compute_3d_sight_lines"],
}

PLOTTING_INTERNALS_NOT_TOP_LEVEL = {
    "plotOn_ax",
    "subplot",
    "plot_baseMap",
    "generate_legend_fig",
    "generate_legend_ax",
    "generate_colorbar",
    "set_axes_frame",
}


def test_lazy_public_map_does_not_reference_deleted_flat_modules():
    public_modules = set(ci._PUBLIC_SYMBOLS.values())  # noqa: SLF001

    assert DELETED_MODULES.isdisjoint(public_modules)


def test_deleted_flat_modules_are_not_part_of_the_lazy_public_contract():
    for module_name in DELETED_MODULES:
        assert module_name not in ci._PUBLIC_SYMBOLS.values()  # noqa: SLF001


def test_deleted_flat_modules_are_not_importable_once_their_files_are_removed():
    """When an old file is physically gone, its old import path must fail."""
    package_dir = Path(ci.__file__).resolve().parent

    for module_name in DELETED_MODULES:
        module_file = package_dir / f"{module_name}.py"
        if module_file.exists():
            continue

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"cityImage.{module_name}")


def test_hard_removed_symbols_are_not_top_level_api():
    for symbol in HARD_REMOVED_PUBLIC_SYMBOLS:
        assert symbol not in ci._PUBLIC_SYMBOLS  # noqa: SLF001

        with pytest.raises(AttributeError):
            getattr(ci, symbol)


def test_core_public_symbols_resolve_from_top_level():
    for symbol in CORE_PUBLIC_SYMBOLS:
        assert hasattr(ci, symbol), symbol


@pytest.mark.parametrize("dependency,symbols", OPTIONAL_SYMBOLS_BY_EXTRA.items())
def test_optional_public_symbols_resolve_when_their_dependency_is_available(dependency, symbols):
    pytest.importorskip(dependency)

    for symbol in symbols:
        assert hasattr(ci, symbol), symbol


def test_plotting_internals_are_not_top_level_public_api():
    for symbol in PLOTTING_INTERNALS_NOT_TOP_LEVEL:
        assert symbol not in ci._PUBLIC_SYMBOLS  # noqa: SLF001
        assert not hasattr(ci, symbol)
