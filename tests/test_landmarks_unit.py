import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci


def _buildings():
    return gpd.GeoDataFrame(
        {
            "buildingID": [0, 1],
            "height": [10.0, 5.0],
            "historic": ["yes", None],
            "land_uses": [["religious"], ["residential"]],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:3857",
    )


def test_visibility_score_empty_sight_lines_returns_gdf_not_tuple():
    out = ci.visibility_score(
        _buildings(),
        sight_lines=gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"),
    )

    assert hasattr(out, "geometry")
    assert "3dvis" in out.columns
    assert out["3dvis"].eq(0.0).all()


def test_cultural_score_from_osm_uses_historic_helper():
    out = ci.cultural_score(_buildings(), from_OSM=True)

    assert out["cult"].tolist() == [1.0, 0.0]


def test_pragmatic_score_accepts_normalised_land_uses_without_overlap_column():
    out = ci.pragmatic_score(_buildings(), search_radius=10)

    assert "prag" in out.columns
    assert out["prag"].notna().all()
