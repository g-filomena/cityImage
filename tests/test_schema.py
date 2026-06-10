import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from cityImage.schema import (
    SchemaError,
    ensure_building_schema_defaults,
    require_land_use_lists,
    validate_buildings_gdf,
    validate_edges_gdf,
    validate_nodes_gdf,
)


def test_validate_minimal_nodes_edges_and_buildings_schema():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:3857",
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [10], "u": [1], "v": [2]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:3857",
    )
    buildings = gpd.GeoDataFrame(
        {"buildingID": [100]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    )

    assert validate_nodes_gdf(nodes).ok
    assert validate_edges_gdf(edges).ok
    assert validate_buildings_gdf(buildings).ok


def test_validate_buildings_reports_missing_columns():
    buildings = gpd.GeoDataFrame(
        {"id": [100]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    )

    report = validate_buildings_gdf(buildings)

    assert not report.ok
    assert report.missing_columns == ("buildingID",)


def test_ensure_building_schema_defaults_adds_optional_columns():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1]},
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:3857",
    )

    out = ensure_building_schema_defaults(buildings)

    assert out.loc[0, "area"] == 4.0
    assert out.loc[0, "base"] == 0.0
    assert out.loc[0, "land_uses"] == ["unknown"]
    assert out.loc[0, "land_uses_overlap"] == [1.0]


def test_require_land_use_lists_rejects_mismatched_overlap_lengths():
    buildings = gpd.GeoDataFrame(
        {
            "buildingID": [1],
            "land_uses": [["retail", "office"]],
            "land_uses_overlap": [[1.0]],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    )

    with pytest.raises(SchemaError):
        require_land_use_lists(buildings)
