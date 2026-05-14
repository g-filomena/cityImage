from __future__ import annotations

import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd

pd.set_option("display.precision", 3)

from .utilities import downloader, gdf_multipolygon_to_polygon
from .land_use_derive import derive_land_uses_raw_fromOSM
from .land_use_classify import classify_land_uses_raws_into_OSMgroups


def _require_column(gdf: gpd.GeoDataFrame, column: str, argument_name: str) -> None:
    """Raise a clear error when a user-supplied source column is missing."""
    if column not in gdf.columns:
        raise ValueError(f"{argument_name}='{column}' was not found in the input GeoDataFrame")


def _geometry_union(geometry: gpd.GeoSeries):
    """Return a geometry union compatible with older/newer GeoPandas versions."""
    try:
        return geometry.union_all()
    except AttributeError:
        return geometry.unary_union


def _coerce_osm_height(height_series: pd.Series) -> pd.Series:
    """Coerce common OSM height strings to numeric metres when possible.

    Examples handled:
    - "12"
    - "12.5"
    - "12 m"
    - "12 meters"

    Non-standard values remain NaN.
    """
    extracted = height_series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def get_buildings_fromFile(
    input_path,
    crs,
    case_study_area=None,
    distance_from_center=1000,
    min_area=200,
    min_height=5,
    height_field=None,
    base_field=None,
    land_use_field=None,
):
    """
    Read building footprints from a file.

    Returns two GeoDataFrames:
    1. buildings within the case-study area;
    2. a larger building set containing the case-study buildings and adjacent
       obstructions.

    The non-OSM/sparse land-use route starts here: if `land_use_field` is
    provided, its values are stored as `land_use_raw`. Use
    `classify_sparse_land_uses()` or `attach_sparse_land_uses()` afterwards to
    normalise those values into `land_uses`.

    Parameters
    ----------
    input_path : str
        Path to the building footprint file.
    crs : str or pyproj.CRS
        Target projected CRS.
    case_study_area : shapely geometry or None
        Polygon defining the area of interest. If None, a buffer around the
        centroid of all valid buildings is used when `distance_from_center` is
        positive.
    distance_from_center : float or None
        Buffer radius in CRS units. If None or 0 and `case_study_area` is None,
        all valid buildings are returned as both outputs.
    min_area : float or None
        Minimum footprint area. Set to None to disable.
    min_height : float or None
        Minimum height. Applied only if `height_field` is provided and the
        source appears to contain metre-like height values.
    height_field : str or None
        Source column containing building heights. If omitted, output `height`
        is created with NaN values and no height filtering is applied.
    base_field : str or None
        Source column containing base elevation. If omitted, output `base` is
        set to 0.0.
    land_use_field : str or None
        Source column containing sparse/non-OSM land-use labels.

    Returns
    -------
    buildings_gdf : GeoDataFrame
        Buildings within the case-study area.
    obstructions_gdf : GeoDataFrame
        All valid buildings after cleaning/filtering.
    """
    obstructions_gdf = gpd.read_file(input_path).to_crs(crs)
    obstructions_gdf = obstructions_gdf[
        obstructions_gdf.geometry.notna() & ~obstructions_gdf.geometry.is_empty
    ].copy()

    obstructions_gdf["area"] = obstructions_gdf.geometry.area

    if min_area is not None:
        obstructions_gdf = obstructions_gdf[obstructions_gdf["area"] >= min_area].copy()

    if height_field is not None:
        _require_column(obstructions_gdf, height_field, "height_field")
        obstructions_gdf["height"] = pd.to_numeric(
            obstructions_gdf[height_field],
            errors="coerce",
        )

        # Preserve the original intent: filter by height only when the supplied
        # values look like actual metres rather than floor counts/classes.
        mean_height = obstructions_gdf["height"].mean(skipna=True)
        if (
            min_height is not None
            and pd.notna(mean_height)
            and mean_height > min_height
        ):
            obstructions_gdf = obstructions_gdf[
                obstructions_gdf["height"].notna()
                & (obstructions_gdf["height"] >= min_height)
            ].copy()
    else:
        obstructions_gdf["height"] = np.nan

    if base_field is None:
        obstructions_gdf["base"] = 0.0
    else:
        _require_column(obstructions_gdf, base_field, "base_field")
        obstructions_gdf["base"] = pd.to_numeric(
            obstructions_gdf[base_field],
            errors="coerce",
        ).fillna(0.0)

    if land_use_field is not None:
        _require_column(obstructions_gdf, land_use_field, "land_use_field")
        obstructions_gdf["land_use_raw"] = obstructions_gdf[land_use_field]
    else:
        obstructions_gdf["land_use_raw"] = None

    standard_columns = ["height", "base", "geometry", "area", "land_use_raw"]
    obstructions_gdf = obstructions_gdf[standard_columns].copy()
    obstructions_gdf = gdf_multipolygon_to_polygon(obstructions_gdf)
    obstructions_gdf = obstructions_gdf.reset_index(drop=True)
    obstructions_gdf["buildingID"] = obstructions_gdf.index.values.astype(int)

    if obstructions_gdf.empty:
        return obstructions_gdf.copy(), obstructions_gdf

    if (case_study_area is None) and (
        distance_from_center is None or distance_from_center == 0
    ):
        buildings_gdf = obstructions_gdf.copy()
        return buildings_gdf, obstructions_gdf

    if case_study_area is None:
        case_study_area = _geometry_union(obstructions_gdf.geometry).centroid.buffer(
            distance_from_center
        )

    buildings_gdf = obstructions_gdf[
        obstructions_gdf.geometry.within(case_study_area)
    ].copy()

    return buildings_gdf, obstructions_gdf


def get_buildings_fromOSM(
    OSMplace,
    download_method: str,
    crs=None,
    distance=1000,
    min_area=200,
    min_height=5,
):
    """
    Download, clean and classify OSM building footprints.

    This is the OSM-specific land-use route:

        OSM tags -> land_uses_raw -> land_uses

    The raw OSM `building` tag is used during classification but is not kept in
    the returned analysis table.

    `min_height` is kept for backward compatibility. If OSM returns a `height`
    column, known heights below `min_height` are filtered out while buildings
    with unknown height are retained.
    """
    tags = {"building": True}

    buildings_gdf = downloader(
        OSMplace=OSMplace,
        download_method=download_method,
        tags=tags,
        distance=distance,
    )

    if crs is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        buildings_gdf = buildings_gdf.to_crs(crs)

    buildings_gdf = buildings_gdf[
        buildings_gdf.geometry.notna() & ~buildings_gdf.geometry.is_empty
    ].copy()

    buildings_gdf = derive_land_uses_raw_fromOSM(
        buildings_gdf,
        default="residential",
    )
    buildings_gdf = classify_land_uses_raws_into_OSMgroups(
        buildings_gdf,
        land_uses_raw_column="land_uses_raw",
        new_group_column="land_uses",
    )

    if "height" in buildings_gdf.columns:
        buildings_gdf["height"] = _coerce_osm_height(buildings_gdf["height"])
        if min_height is not None:
            buildings_gdf = buildings_gdf[
                buildings_gdf["height"].isna()
                | (buildings_gdf["height"] >= min_height)
            ].copy()

    keep_columns = [
        column
        for column in ["height", "geometry", "land_uses_raw", "land_uses"]
        if column in buildings_gdf.columns
    ]
    buildings_gdf = buildings_gdf[keep_columns].copy()

    buildings_gdf = gdf_multipolygon_to_polygon(buildings_gdf)

    buildings_gdf["area"] = buildings_gdf.geometry.area
    if min_area is not None:
        buildings_gdf = buildings_gdf[buildings_gdf["area"] >= min_area].copy()

    buildings_gdf = buildings_gdf.reset_index(drop=True)
    buildings_gdf["buildingID"] = buildings_gdf.index.values.astype(int)

    return buildings_gdf


def select_buildings_by_study_area(
    larger_buildings_gdf,
    method="polygon",
    polygon=None,
    distance=1000,
):
    """
    Select buildings from a GeoDataFrame that fall within a defined study area.

    Parameters
    ----------
    larger_buildings_gdf : GeoDataFrame
        GeoDataFrame containing building polygons to filter.
    method : {"polygon", "distance"}
        - "polygon": use `polygon`.
        - "distance": buffer the centroid of all buildings by `distance`.
    polygon : shapely Polygon or MultiPolygon, optional
        Study area polygon. Required if method is "polygon".
    distance : float
        Buffer distance in CRS units if method is "distance".

    Returns
    -------
    GeoDataFrame
        Buildings within the study area.
    """
    if larger_buildings_gdf.empty:
        return gpd.GeoDataFrame(
            columns=larger_buildings_gdf.columns,
            geometry=larger_buildings_gdf.geometry.name
            if hasattr(larger_buildings_gdf, "geometry")
            else None,
            crs=getattr(larger_buildings_gdf, "crs", None),
        )

    if method == "distance":
        study_area = _geometry_union(larger_buildings_gdf.geometry).centroid.buffer(
            distance
        )
    elif method == "polygon":
        study_area = polygon
    else:
        raise ValueError("Method must be either 'polygon' or 'distance'.")

    if study_area is None:
        return gpd.GeoDataFrame(
            columns=larger_buildings_gdf.columns,
            geometry=larger_buildings_gdf.geometry.name,
            crs=larger_buildings_gdf.crs,
        )

    return larger_buildings_gdf[
        larger_buildings_gdf.geometry.within(study_area)
    ].copy()
