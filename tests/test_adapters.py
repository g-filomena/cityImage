import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci


def test_standardize_osmnx_like_nodes_edges():
    nodes = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[Point(0, 0), Point(1, 0)],
        index=pd.Index([100, 200], name="osmid"),
        crs="EPSG:3857",
    )
    edges = gpd.GeoDataFrame(
        {"speed": [30]},
        geometry=[LineString([(0, 0), (1, 0)])],
        index=pd.MultiIndex.from_tuples([(100, 200, 0)], names=["u", "v", "key"]),
        crs="EPSG:3857",
    )

    nodes_out = ci.standardize_nodes_gdf(nodes)
    edges_out = ci.standardize_edges_gdf(edges)

    assert list(nodes_out["nodeID"]) == [100, 200]
    assert list(nodes_out["x"]) == [0.0, 1.0]
    assert list(edges_out["u"]) == [100]
    assert list(edges_out["v"]) == [200]
    assert list(edges_out["edgeID"]) == [0]
    assert "length" in edges_out.columns


def test_standardize_buildings_with_scalar_land_use():
    buildings = gpd.GeoDataFrame(
        {"my_id": [10, 11], "lu": ["retail", None]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.standardize_buildings_gdf(
        buildings,
        building_id_column="my_id",
        land_uses_column="lu",
        default_land_use="unknown",
    )

    assert list(out["buildingID"]) == [10, 11]
    assert out.loc[0, "land_uses"] == ["retail"]
    assert out.loc[1, "land_uses"] == ["unknown"]
    assert out.loc[0, "land_uses_overlap"] == [1.0]
    assert "area" in out.columns
    assert "height" in out.columns
    assert "base" in out.columns


def test_standardize_cityimage_inputs_returns_only_provided_tables():
    buildings = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    )

    outputs = ci.standardize_cityimage_inputs(buildings_gdf=buildings)

    assert sorted(outputs) == ["buildings_gdf"]
    assert list(outputs["buildings_gdf"]["buildingID"]) == [0]
