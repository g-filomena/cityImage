"""Offline smoke tests for ``cityImage.plotting.static``.

These render with the non-interactive Agg backend and assert on the returned
Matplotlib figures. They intentionally cover many parameter branches (categorical
vs scheme vs colorbar, points/lines/polygons, base maps, size scaling) rather
than pixel output.
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import LineString, Point, Polygon

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from cityImage.plotting.static import (  # noqa: E402
    plot_gdf,
    plot_grid_gdf_columns,
    plot_grid_gdfs_column,
)

pytest.importorskip("mapclassify")

CRS = "EPSG:3857"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _polygons():
    gdf = gpd.GeoDataFrame(
        {"buildingID": [1, 2, 3], "score": [0.1, 0.5, 0.9], "kind": ["a", "b", "a"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ],
        crs=CRS,
    )
    return gdf


def _points():
    return gpd.GeoDataFrame(
        {"nodeID": [1, 2, 3], "value": [1.0, 5.0, 9.0], "kind": ["x", "y", "x"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=CRS,
    )


def _lines():
    return gpd.GeoDataFrame(
        {"edgeID": [1, 2], "flow": [3.0, 8.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 0)])],
        crs=CRS,
    )


def test_plot_gdf_plain_and_categorical_with_legend():
    assert isinstance(plot_gdf(_polygons()), Figure)  # plain map, no column
    fig = plot_gdf(_polygons(), column="kind", legend=True, black_background=False)
    assert isinstance(fig, Figure)


def test_plot_gdf_numeric_with_colorbar_and_scheme():
    cbar_fig = plot_gdf(
        _polygons(),
        column="score",
        cbar=True,
        cbar_min_max=True,
        cbar_max_symbol=True,
        axes_frame=True,
    )
    assert isinstance(cbar_fig, Figure)

    scheme_fig = plot_gdf(_lines(), column="flow", scheme="quantiles", classes=2, legend=True)
    assert isinstance(scheme_fig, Figure)


def test_plot_gdf_points_with_size_scaling_and_base_map():
    fig = plot_gdf(
        _points(),
        column="value",
        geometry_size_column="value",
        geometry_size_factor=3.0,
        base_map_gdf=_polygons(),
        base_map_color="grey",
    )
    assert isinstance(fig, Figure)


def test_plot_grid_gdfs_column_across_multiple_gdfs():
    fig = plot_grid_gdfs_column(
        gdfs=[_polygons(), _polygons()],
        column="kind",
        ncols=2,
        nrows=1,
        legend=True,
        titles=["one", "two"],
        main_title="grid",
    )
    assert isinstance(fig, Figure)


def test_plot_grid_gdfs_column_validates_layout():
    with pytest.raises(ValueError, match="appropriate combination"):
        plot_grid_gdfs_column(gdfs=[_polygons()], column="kind", ncols=3, nrows=3)


def test_plot_grid_gdf_columns_across_multiple_columns():
    fig = plot_grid_gdf_columns(
        _polygons(),
        columns=["score", "kind"],
        ncols=2,
        nrows=1,
        titles=["score", "kind"],
    )
    assert isinstance(fig, Figure)
