import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci


def test_classify_land_use_sparse_attribute_wrapper_scalar_values():
    """The wrapper maps sparse attribute values without using OSM tags."""
    gdf = gpd.GeoDataFrame(
        {"lu_eng": ["church", "bank", "unknown"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 3,
        crs="EPSG:3857",
    )

    out = ci.classify_land_use(
        gdf,
        raw_land_use_column="lu_eng",
        new_land_use_column="land_use",
        categories=[["church"], ["bank"]],
        strings=["religious", "business_services"],
    )

    assert out["land_use"].tolist() == ["religious", "business_services", "unknown"]
