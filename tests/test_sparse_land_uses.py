import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci


def test_classify_sparse_land_uses_from_non_osm_column():
    buildings = gpd.GeoDataFrame(
        {"lu_eng": ["church", "bank", None]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.classify_sparse_land_uses(
        buildings,
        source_column="lu_eng",
        mapping={
            "church": "religious",
            "bank": "financial",
        },
        default_land_use="unknown",
    )

    assert out["land_uses"].tolist() == [["religious"], ["financial"], ["unknown"]]


def test_attach_sparse_land_uses_from_external_polygons():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
        ],
        crs="EPSG:3857",
    )

    sparse_land_uses = gpd.GeoDataFrame(
        {"lu_eng": ["shop", "office"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]),
            Polygon([(1, 0), (2, 0), (2, 2), (1, 2)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.attach_sparse_land_uses(
        buildings,
        sparse_land_uses,
        source_column="lu_eng",
        mapping={
            "shop": "retail",
            "office": "office",
        },
        default_land_use="unknown",
    )

    assert out.loc[0, "land_uses"] == ["retail", "office"]
    assert out.loc[0, "land_uses_overlap"] == [0.5, 0.5]
    assert out.loc[1, "land_uses"] == ["unknown"]
    assert out.loc[1, "land_uses_overlap"] == [1.0]
