import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import box
from rasterstats import zonal_stats

from .utilities import gdf_multipolygon_to_polygon

def assign_building_heights_from_other_gdf(buildings_gdf, detailed_buildings_gdf, crs, base_field = 'base', height_field = 'height', min_overlap=0.4):
    """
    Assigns 'base' and 'height' attributes to each building in `buildings_gdf` by extracting
    information from a more detailed building GeoDataFrame (`detailed_buildings_gdf`).

    The assignment logic is as follows:
    - If a building fully contains a detailed building, assign the lowest 'base' and highest 'height' among contained buildings.
    - If a detailed building contains a main building, assign the lowest 'base' and highest 'height' among containers.
    - If a detailed building intersects only one main building, assign its values to that building.
    - If a detailed building intersects multiple main buildings, assign its values to the one with the highest overlap (if overlap ≥ `min_overlap`).
    - The overlap ratio is defined as (intersection_area / detailed_building_area).
    - The reverse intersection is also considered: if a main building overlaps multiple detailed buildings, attributes are borrowed from the one with the highest overlap ratio (if overlap ≥ `min_overlap`).

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        GeoDataFrame of main building polygons (to be assigned base/height).
    detailed_buildings_gdf : GeoDataFrame
        More detailed building dataset with valid 'base' and 'height' attributes.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrame. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    min_overlap : float, optional
        Minimum required overlap ratio (as a fraction, default 0.4).

    Returns
    -------
    GeoDataFrame
        Copy of `buildings_gdf` with updated 'base' and 'height' columns.

    Raises
    ------
    ValueError
        If CRS of the input GeoDataFrames does not match the provided CRS.
    """
    
    # Ensure CRS matches
    if (buildings_gdf.crs != crs) or (detailed_buildings_gdf.crs !=crs):
        raise ValueError(f"CRS mismatch: buildings_gdf ({buildings_gdf.crs}) and detailed_buildings_gdf ({detailed_buildings_gdf.crs}) must have the same CRS.")

    buildings_gdf = buildings_gdf.copy()
    detailed_buildings_gdf = gdf_multipolygon_to_polygon(detailed_buildings_gdf)
    
    detailed_buildings_gdf['base'] = detailed_buildings_gdf[base_field]
    detailed_buildings_gdf['height'] = detailed_buildings_gdf[height_field]
    
    buildings_gdf["base"] = 9999.0 
    buildings_gdf["height"] = -9999.0 
        
    # # **Step 1: Handle Full Containment (Buildings that contain detailed ones)**
    # **Step 1: Identify buildings that contain detailed buildings**
    containment = gpd.sjoin(buildings_gdf, detailed_buildings_gdf, predicate="contains", how="left")

    # Compute min(base) and max(height) for each containing building
    contained_bases = containment.groupby(containment.index)["base_right"].min()
    contained_height = containment.groupby(containment.index)["height_right"].max()

    # Apply updates for contained buildings
    buildings_gdf["base"] = buildings_gdf["base"].combine(contained_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(contained_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)
    
    # **Step 2: Identify detailed buildings that contain buildings_gdf**
    reverse_containment = gpd.sjoin(detailed_buildings_gdf, buildings_gdf, predicate="contains", how="left")
    container_bases = reverse_containment.groupby("index_right")["base_left"].min()
    container_height = reverse_containment.groupby("index_right")["height_left"].max()
    
    buildings_gdf["base"] = buildings_gdf["base"].combine(container_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(container_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # **Step 3: Handle Intersections (Buildings that partially overlap detailed ones)** 
    buildings_gdf['geo_check'] = buildings_gdf.geometry
    buildings_gdf['ix'] = buildings_gdf.index
    
    intersections = gpd.sjoin(detailed_buildings_gdf, buildings_gdf, predicate="intersects", how="left")
    intersections = intersections[intersections.geo_check != None]
    
    # Compute intersection area
    intersections["area_intersection"] = intersections.apply(lambda row: row["geometry"].intersection(row["geo_check"]).area, axis=1)

    # Compute overlap ratio (intersection_area / detailed_building_area)
    intersections["overlap_ratio"] = intersections["area_intersection"] / intersections["geometry"].area
    # Keep only valid matches where the overlap is at least `min_overlap`
    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]

    # Keep only the building in `buildings_gdf` that has the highest coverage for each detailed building
    best_matches = valid_matches.loc[valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()]
    best_matches = best_matches.set_index("ix")
    best_matches.index = best_matches.index.astype(int)
    
    # Compute min(base) and max(height) for the selected matches
    intersection_bases = best_matches.groupby(best_matches.index)["base_left"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_left"].max()

    # Apply updates efficiently
    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis = 1)
    buildings_gdf.index = buildings_gdf.index.astype(int)
    
    # **Step 4: Handle Intersections in Reverse (Detailed buildings donate attributes to smaller ones)** ----------
    detailed_buildings_gdf['geo_check'] = detailed_buildings_gdf.geometry
    buildings_gdf['ix'] = buildings_gdf.index

    # Perform spatial join: find detailed buildings that intersect with non-detailed ones
    intersections = gpd.sjoin(buildings_gdf, detailed_buildings_gdf, predicate="intersects", how="left")
    intersections = intersections[intersections.geo_check.notnull()]

    # Compute intersection area
    intersections["area_intersection"] = intersections.apply(lambda row: row["geometry"].intersection(row["geo_check"]).area, axis=1)

    # Compute overlap ratio (intersection_area / non-detailed building area)
    intersections["overlap_ratio"] = intersections["area_intersection"] / intersections["geometry"].area

    # Keep only matches where the **smaller building's** overlap is at least threshold value
    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]

    # For each **non-detailed building**, find the detailed one with the highest overlap
    best_matches = valid_matches.loc[valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()]

    # Compute min(base) and max(height) for the selected matches
    intersection_bases = best_matches.groupby(best_matches.index)["base_right"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_right"].max()

    # Apply updates efficiently: non-detailed buildings take values from detailed ones
    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # # Replace 9999.0 in "base" with NaN, and -9999.0 in "height" with NaN
    buildings_gdf["base"] = buildings_gdf["base"].replace(9999.0, np.nan)
    buildings_gdf["height"] = buildings_gdf["height"].replace(-9999.0, np.nan)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis = 1, errors = 'ignore')
    
    return buildings_gdf

def buildings_height_from_dem_dtm(buildings_gdf, dem_path, dtm_path, base_stat="mean", top_stat="max", all_touched=False, min_valid_elev=-50):
    """
    Compute per-building base elevation and height from a DEM (top) and DTM (base).

    For each building polygon:
      - base   = base_stat(DTM) using only raster values >= min_valid_elev
      - height = top_stat(DEM) using only raster values >= min_valid_elev, minus base

    Behaviour for missing coverage:
      - If a building has no valid DEM/DTM pixels (outside raster extent or all values < min_valid_elev / nodata),
        both 'base' and 'height' are set to 9999.0 (sentinel for missing height).

    Parameters
    ----------
    buildings_gdf: GeoDataFrame with polygon geometries; if it has a 'buildingID' column,
                   the returned GeoDataFrame index is set to that.
    dem_path, dtm_path: paths to single-band DEM (top) and DTM (base) rasters; these rasters
                        must share CRS, extent and resolution.
    base_stat, top_stat: statistic names understood by rasterstats (e.g. "mean", "median", "min", "max");
                         base_stat applies to the DTM, top_stat applies to the DEM.
    all_touched: forwarded to rasterstats.zonal_stats; if True, any pixel touched by a polygon
                 is included, otherwise only pixels whose center lies within the polygon.
    min_valid_elev: any raster value strictly lower than this threshold is treated as invalid
                    and ignored in both DEM and DTM.

    Returns
    -------
    GeoDataFrame in the same CRS as the original buildings, containing all original columns
    plus two new fields:
      - 'base'   (float): per-building DTM statistic, or 9999.0 if no valid DTM coverage
      - 'height' (float): per-building DEM statistic minus base, or 9999.0 if no valid DEM/DTM coverage.
    """

    # Save original CRS of the buildings
    original_crs = buildings_gdf.crs
    buildings_with_data = buildings_gdf.copy()

    # --- 1. Open rasters, check CRS, and read data while masking < min_valid_elev ---
    with rasterio.open(dem_path) as dem_src, rasterio.open(dtm_path) as dtm_src:
        dem_crs = dem_src.crs
        dtm_crs = dtm_src.crs

        if dem_crs != dtm_crs:
            raise ValueError("DEM and DTM have different CRS; reproject one of them beforehand.")

        # Reproject buildings to raster CRS (much cheaper than reprojecting rasters)
        if buildings_with_data.crs is None:
            raise ValueError("buildings_gdf has no CRS set.")
        if buildings_with_data.crs != dem_crs:
            buildings_with_data = buildings_with_data.to_crs(dem_crs)

        # Keep only buildings that intersect the raster extent (avoid useless stats / crashes)
        raster_bounds = dem_src.bounds  # (minx, miny, maxx, maxy)
        raster_geom = box(*raster_bounds)  # shapely polygon from bounds
        raster_bbox = gpd.GeoDataFrame(geometry=[raster_geom], crs=dem_crs)
        buildings_with_data = buildings_with_data[buildings_with_data.intersects(raster_bbox.loc[0, "geometry"])]

        if buildings_with_data.empty:
            raise ValueError("No buildings intersect the DEM/DTM extent.")

        # Read DEM/DTM as masked arrays (nodata from raster is automatically masked)
        dem_data = dem_src.read(1, masked=True)
        dtm_data = dtm_src.read(1, masked=True)

        # Mask all values below the minimum valid elevation (e.g. -50 m)
        dem_data = np.ma.masked_less(dem_data, min_valid_elev)
        dtm_data = np.ma.masked_less(dtm_data, min_valid_elev)

        dem_transform = dem_src.transform
        dtm_transform = dtm_src.transform

    # --- 2. Fix invalid geometries (common cause of zonal_stats failures) ---
    buildings_with_data["geometry"] = buildings_with_data.geometry.buffer(0)

    # --- 3. Zonal stats for DEM (top) using top_stat (e.g. "max") ---
    dem_stats = zonal_stats(
        buildings_with_data,
        dem_data,
        affine=dem_transform,
        stats=[top_stat],
        all_touched=all_touched,
        geojson_out=False,
        nodata=None  # nodata already handled via masking
    )
    dem_df = pd.DataFrame(dem_stats)
    dem_col = f"dem_{top_stat}"
    dem_df.columns = [dem_col]

    # --- 4. Zonal stats for DTM (base) using base_stat (e.g. "mean") ---
    dtm_stats = zonal_stats(
        buildings_with_data,
        dtm_data,
        affine=dtm_transform,
        stats=[base_stat],
        all_touched=all_touched,
        geojson_out=False,
        nodata=None
    )
    dtm_df = pd.DataFrame(dtm_stats)
    dtm_col = f"dtm_{base_stat}"
    dtm_df.columns = [dtm_col]

    # --- 5. Attach results and compute base / height ---
    buildings_with_data = buildings_with_data.reset_index(drop=True)
    buildings_with_data = pd.concat([buildings_with_data, dem_df, dtm_df], axis=1)

    buildings_with_data["base"] = buildings_with_data[dtm_col]
    buildings_with_data["height"] = buildings_with_data[dem_col] - buildings_with_data["base"]   # height = MAX(DEM) - base, using only values >= min_valid_elev

    # --- 6. Drop intermediate columns ---
    buildings_with_data = buildings_with_data.drop(columns=[dem_col, dtm_col])

    # --- 7. Reproject back to original CRS if needed ---
    if buildings_with_data.crs != original_crs:
        buildings_with_data = buildings_with_data.to_crs(original_crs)

    buildings_with_data.index = buildings_with_data.buildingID
    buildings_with_data.index.name = None
    return buildings_with_data
    
def assign_height_from_dtm(nodes_gdf: gpd.GeoDataFrame, dtm_path: str, z_col: str = "z", min_valid_elev: float = -50.0):
    """
    Sample a DTM raster to assign elevation to point nodes.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of POINT geometries (nodes).
    dtm_path : str
        Path to a single-band DTM raster readable by rasterio.
    z_col : str, default "z"
        Name of the column to store elevation values.
    min_valid_elev : float, default -1000.0
        Minimum acceptable elevation; values below this (or nodata) are set to NaN.

    Returns
    -------
    GeoDataFrame
        Copy of `nodes_gdf` with a new column `z_col` containing elevation (float).
    """
    
    # Work on a copy
    nodes_gdf_with_data = nodes_gdf.copy()
    original_crs = nodes_gdf.crs
    
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
        nodata = src.nodata

        # Reproject nodes to DTM CRS if needed
        if dtm_crs is None:
            raise ValueError("DTM has no CRS set.")
        if nodes_gdf_with_data.crs != dtm_crs:
            nodes_gdf_with_data = nodes_gdf_with_data.to_crs(dtm_crs)

        # Extract coordinates for sampling
        # geometry must be Points
        if not all(geom.geom_type == "Point" for geom in nodes_gdf_with_data.geometry):
            raise ValueError("All geometries in nodes_gdf must be Points.")

        coords = [(geom.x, geom.y) for geom in nodes_gdf_with_data.geometry]

        # Sample the raster at node locations (first band)
        sampled = list(src.sample(coords))
        # sampled is a list of arrays, one per point; we take first band [0]
        elev = np.array([vals[0] if len(vals) > 0 else np.nan for vals in sampled], dtype=float)

        # Handle nodata and invalid elevations
        if nodata is not None:
            elev[elev == nodata] = np.nan
        elev[elev < min_valid_elev] = np.nan

        nodes_gdf_with_data[z_col] = elev
        # --- 7. Reproject back to original CRS if needed ---
    
    if nodes_gdf_with_data.crs != original_crs:
        nodes_gdf_with_data = nodes_gdf_with_data.to_crs(original_crs)    

    return nodes_gdf_with_data