from __future__ import annotations

import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv

from typing import Any
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, mapping
from shapely.ops import linemerge, unary_union
pd.set_option("display.precision", 3)

import concurrent.futures
from .utilities import scaling_columnDF
from .buildings_visibility import visibility_polygon2d, compute_3d_sight_lines
    
def structural_score(buildings_gdf, obstructions_gdf, edges_gdf, advance_vis_expansion_distance = 300, neighbours_radius = 150):
    """
    The function computes the "Structural Landmark Component" sub-scores of each building.
    It considers:
    - distance from the street network:
    - advance 2d visibility polygon;
    - number of neighbouring buildings in a given radius.
         
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    edges_gdf: LineString GeoDataFrame
        Street segmetns GeoDataFrame.
    obstructions_gdf: Polygon GeoDataFrame
        Obstructions GeoDataFrame.  
    advance_vis_expansion_distance: float
        2d advance visibility - it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    neighbours_radius: float
        Neighbours - search radius for other adjacent buildings.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    buildings_gdf = buildings_gdf.copy()
    assert_all_polygons(buildings_gdf)
    
    # remove z coordinates if they are there already - issue with 2dvis
    if len(buildings_gdf.geometry.iloc[0].exterior.coords[0]) == 3: 
        buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(lambda g: type(g)([(x, y) for x, y, *_ in g.exterior.coords]))
   
    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.union_all()

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(lambda row: visibility_polygon2d(row, obstructions_gdf, sindex, max_expansion_distance=
                                    advance_vis_expansion_distance))
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(lambda row: _number_neighbours(row, obstructions_gdf, sindex, radius=neighbours_radius))

    return buildings_gdf
    
def _number_neighbours(geometry, obstructions_gdf, obstructions_sindex, radius):
    """
    The function counts the number of neighbours, in a GeoDataFrame, around a given geometry, within a
    search radius.
     
    Parameters
    ----------
    geometry: Shapely Geometry
        The geometry for which neighbors are counted.
    obstructions_gdf: GeoDataFrame
        The GeoDataFrame containing the obstructions.
    obstructions_sindex: Spatial Index
        The spatial index of the obstructions GeoDataFrame.
    radius: float
        The search radius for neighboring buildings.

    Returns
    -------
    int
        The number of neighbors.
    """
    buffer = geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)
  
def visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a': []}), method = 'longest'):
    """
    The function calculates the sub-scores of the "Visibility Landmark Component".
    - 3d visibility;
    - facade area;
    - (height).
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    sight_lines: LineString GeoDataFrame
        The Sight Lines GeoDataFrame.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0
    if ("height" not in buildings_gdf.columns) | (sight_lines.empty): 
        return buildings_gdf, sight_lines

    sight_lines['nodeID'] = sight_lines['nodeID'].astype(int)
    sight_lines['buildingID'] = sight_lines['buildingID'].astype(int)
    sight_lines['length'] = sight_lines.geometry.length
    
    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: _facade_area(row["geometry"], row["height"]), axis = 1)

    stats = sight_lines.groupby('buildingID').agg({'length': ['mean','max', 'count']}) 
    stats.columns = stats.columns.droplevel(0)
    stats.rename(columns = {"count": "nr_lines"}, inplace = True)

    stats["max"] = stats["max"].fillna(stats["max"].min())
    stats["mean"] = stats["mean"].fillna(stats["mean"].min())
    stats["nr_lines"] = stats["nr_lines"].fillna(stats["nr_lines"].min())
    stats.reset_index(inplace = True)
    columns = ["max", "mean", "nr_lines"]

    for column in columns:
        stats[column+"_sc"] = scaling_columnDF(stats[column])

    if method == 'longest':
        stats["3dvis"] = stats["max_sc"]
    elif method == 'combined':
        stats["3dvis"] = stats["max_sc"]*0.5+stats["mean_sc"]*0.25+stats["nr_lines_sc"]*0.25

    buildings_gdf = pd.merge(buildings_gdf, stats[["buildingID", "3dvis"]], on = "buildingID", how = "left") 
    buildings_gdf['3dvis'] = buildings_gdf['3dvis'].where(pd.notnull(buildings_gdf['3dvis']), 0.0)
    
    return buildings_gdf

def _facade_area(building_geometry, building_height):
    """
    Compute the approximate facade area of a building given its geometry and height.

    Parameters
    ----------
    building_geometry: Polygon
        The geometry of the building.
    building_height: float
        The height of the building.

    Returns
    -------
    float
        The computed approximate facade area of the building.
    """    
    envelope = building_geometry.envelope
    coords = mapping(envelope)["coordinates"][0]
    d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
    width = min(d)
    return width*building_height
    
def _is_historic(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() not in {"0", "no", "false", "", "none", "nan"}

def get_historic_buildings_fromOSM(
    OSMplace,
    download_method: str,
    crs: str | Any | None = None,
    distance: float = 1000,
):
    """
    Download OSM building footprints and keep only those that are tagged as
    historic and/or heritage in OSM.

    Definition of "historic building" used here
    ------------------------------------------
    A building is kept if it has at least one of:
      - historic=*
      - heritage=*
    where the tag value is not missing and not an explicit "falsey" value
    (0/no/false).

    Parameters
    ----------
    OSMplace : str | tuple | shapely.geometry.Polygon
        Passed through to `downloader` (your wrapper).
    download_method : str
        Passed through to `downloader`.
    crs : str | pyproj.CRS | None, default None
        If None, uses `ox.projection.project_gdf` (UTM-like projected CRS).
        If provided, reprojects to this CRS.
    distance : float, default 1000
        Passed through to `downloader` when using distance-based methods.

    Returns
    -------
    GeoDataFrame
        Columns:
          - geometry
          - historic : int8 (0/1)
          - area     : float (in CRS units^2)

        Note: if no historic/heritage buildings are found, returns an empty GeoDataFrame
        with the same columns.
    """
    
    tags = {"building": True, "historic": True, "heritage": True}
    historic_gdf = downloader(
        OSMplace=OSMplace,
        download_method=download_method,
        tags=tags,
        distance=distance,
    )

    if historic_gdf is None or len(historic_gdf) == 0:
        # return empty with expected schema (best effort)
        return gpd.GeoDataFrame({"geometry": [], "historic": pd.Series(dtype="int8"), "area": pd.Series(dtype="float64")})

    # Ensure expected columns exist (avoid KeyError on subset)
    if "historic" not in historic_gdf.columns:
        historic_gdf["historic"] = pd.NA
    if "heritage" not in historic_gdf.columns:
        historic_gdf["heritage"] = pd.NA

    historic_gdf = historic_gdf[["geometry", "historic", "heritage"]].copy()
    historic_gdf = historic_gdf[historic_gdf.geometry.notna()].copy()

    # Keep only historic/heritage-tagged buildings
    mask = historic_gdf["historic"].apply(_is_historic) | historic_gdf["heritage"].apply(_is_historic)
    historic_gdf = historic_gdf.loc[mask].copy()

    if len(historic_gdf) == 0:
        return gpd.GeoDataFrame({"geometry": [], "historic": pd.Series(dtype="int8"), "area": pd.Series(dtype="float64")}, crs=gdf.crs)

    # Project
    if crs is None:
        historic_gdf = ox.projection.project_gdf(historic_gdf)
    else:
        historic_gdf = historic_gdf.to_crs(crs)

    # Binary flag
    historic_gdf["historic"] = 1
    historic_gdf["historic"] = historic_gdf["historic"].astype("int8")

    historic_gdf = historic_gdf[["geometry", "historic"]].copy()
    historic_gdf["area"] = historic_gdf.geometry.area.astype("float64")

    return historic_gdf

def cultural_score(
    buildings_gdf,
    historic_elements_gdf=None,
    score_column: str | None = None,
    from_OSM: bool = False,
):
    """
    Compute a "Cultural Landmark Component" per building.

    Modes
    -----
    1) from_OSM=True
       Uses buildings_gdf["historic"] (OSM tag already on buildings) and converts to 0/1.

    2) from_OSM=False
       Uses spatial intersections between building polygons and `historic_elements_gdf`
       (points or polygons):
         - score_column is None -> count of intersecting elements per building
         - score_column provided -> sum(score_column) over intersecting elements per building

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Target building polygons. Must have a CRS.
    historic_elements_gdf : GeoDataFrame | None
        Source points/polygons of cultural/historic elements. Must have a CRS matching buildings_gdf.
        If None or empty and from_OSM=False -> returns buildings with cult = 0.
    score_column : str | None
        Name of numeric scoring column in historic_elements_gdf. If non-numeric, values are coerced
        with errors='coerce' and NaN treated as 0.
    from_OSM : bool, default False
        If True, uses buildings_gdf["historic"] only (no external GDF needed).

    Returns
    -------
    GeoDataFrame
        Copy of buildings_gdf with an added float column 'cult'.

    Notes
    -----
    - Uses sjoin(..., how="inner") and aggregates by the LEFT index, so it does NOT depend on
      GeoPandas' version-specific 'index_right' column naming.
    """

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["cult"] = 0.0

    # -------------------------
    # Mode 1: use OSM historic
    # -------------------------
    if from_OSM:
        if "historic" not in buildings_gdf.columns:
            raise ValueError("from_OSM=True requires buildings_gdf to contain a 'historic' column")
            
        buildings_gdf["cult"] = buildings_gdf["historic"].apply(_is_hist).astype("int8").astype(float)
        return buildings_gdf

    # -------------------------
    # Mode 2: spatial match
    # -------------------------
    if historic_elements_gdf is None or len(historic_elements_gdf) == 0:
        return buildings_gdf

    if buildings_gdf.crs != historic_elements_gdf.crs:
        raise ValueError("CRS mismatch: buildings_gdf and historic_elements_gdf must have the same CRS")

    left = buildings_gdf[["geometry"]].copy()
    right_cols = ["geometry"]

    if score_column is not None:
        if score_column not in historic_elements_gdf.columns:
            raise ValueError(f"score_column '{score_column}' not found in historic_elements_gdf")
        right_cols.append(score_column)

    right = historic_elements_gdf[right_cols].copy()

    # drop missing geometries defensively
    left = left[left.geometry.notna()].copy()
    right = right[right.geometry.notna()].copy()
    if left.empty or right.empty:
        return buildings_gdf

    # predicate name differs across GeoPandas versions (predicate vs op)
    try:
        joined = gpd.sjoin(left, right, how="inner", predicate="intersects")
    except TypeError:
        joined = gpd.sjoin(left, right, how="inner", op="intersects")

    if joined.empty:
        return buildings_gdf

    if score_column is None:
        counts = joined.groupby(joined.index).size().astype(float)
        buildings_gdf["cult"] = counts.reindex(buildings_gdf.index, fill_value=0.0)
        return buildings_gdf

    vals = pd.to_numeric(joined[score_column], errors="coerce").fillna(0.0)
    sums = vals.groupby(joined.index).sum().astype(float)
    buildings_gdf["cult"] = sums.reindex(buildings_gdf.index, fill_value=0.0)
    return buildings_gdf
 
def pragmatic_score(
    buildings_gdf,
    land_uses_column: str = "land_uses",
    overlaps_column: str = "land_uses_overlap",
    search_radius: float = 200,
    default_land_use: str = "residential",
):
   """
    Compute a "Pragmatic Landmark Component" (unexpectedness) for each building based on
    the composition of nearby land-use labels.

    Concept
    -------
    For a building with label L, the pragmatic component is:

        prag(L) = 1 - ( mass_of_L_in_neighborhood / total_label_mass_in_neighborhood )

    where neighborhood mass is measured within a buffer of radius `search_radius`
    around the building geometry.

    Multi-use + weighting policy
    ----------------------------
    - Buildings may have multiple land-use labels in `land_uses_column` (list[str]).
    - Neighbor contributions are WEIGHTED using `overlaps_column` (list[float]):

        * each neighbor building contributes total weight 1.0
        * its weight is split across its labels according to overlaps
        * overlaps are normalized to sum to 1 per neighbor building

      Example:
        land_uses = ["shop", "office"]
        overlaps  = [0.7, 0.3]
      contributes 0.7 mass to "shop" and 0.3 mass to "office" in the neighborhood.

    - If overlaps are missing, misaligned, or invalid for a building, equal weights
      are used for its labels.

    Building-level output for multi-use targets
    -------------------------------------------
    The score is computed for each (building, label) pair after exploding multi-use labels.
    The final per-building score is the MAX across the building's own labels:

        prag(building) = max_L prag(L)

    Missing / empty labels
    ----------------------
    - If a building has missing or empty `land_uses_column`, it is assigned
      [default_land_use] with weight 1.0.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        GeoDataFrame of building polygons. Geometry must be valid and in a projected CRS
        compatible with `search_radius` units (e.g., meters).
    land_uses_column : str, default "land_uses"
        Column containing a string label or list[str] of labels per building.
    overlaps_column : str, default "land_uses_overlap"
        Column containing list[float] overlaps aligned 1:1 with `land_uses_column`.
        Used only to weight neighbor label mass (not used to weight the final max collapse).
    search_radius : float, default 200
        Buffer radius used to define the neighborhood around each building.
    default_land_use : str, default "residential"
        Label assigned when `land_uses_column` is missing/empty.

    Returns
    -------
    GeoDataFrame
        Copy of `buildings_gdf` with an additional column:
          - "prag": float
        One value per original building row.

    Raises
    ------
    ValueError
        If `land_uses_column` or `overlaps_column` is missing from the GeoDataFrame.

    Notes
    -----
    - Neighborhood selection uses a spatial index for a bbox prefilter, then an exact
      intersects() test against the buffered geometry.
    - The building itself is excluded from its neighborhood using the original index.
    - This implementation explodes buildings to one row per label for scoring.
      For large datasets, this can be computationally heavy.

    Assumptions
    -----------
    - `overlaps_column` values are non-negative and intended to represent relative
      label importance within the building (e.g., overlap fractions). They are normalized
      internally to sum to 1 per building.
    """
    
    gdf = buildings_gdf.copy()

    if land_uses_column not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{land_uses_column}'")
    if overlaps_column not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{overlaps_column}'")

    def _as_list(v):
        if isinstance(v, list):
            return v
        if v is None:
            return []
        try:
            if pd.isna(v):
                return []
        except Exception:
            pass
        return [v]

    # normalize labels
    gdf[land_uses_column] = gdf[land_uses_column].apply(lambda v: _as_list(v) or [default_land_use])

    # normalize overlaps -> weights sum to 1 per building
    def _weights_for_row(row):
        labels = row[land_uses_column]
        w = _as_list(row[overlaps_column])

        # fallback: equal weights
        if len(labels) == 0 or len(w) != len(labels):
            return [1.0 / len(labels)] * len(labels)

        try:
            w = [float(x) for x in w]
        except Exception:
            return [1.0 / len(labels)] * len(labels)

        s = float(np.nansum(w))
        if not np.isfinite(s) or s <= 0:
            return [1.0 / len(labels)] * len(labels)

        return [x / s for x in w]

    gdf["_w_list"] = gdf.apply(_weights_for_row, axis=1)

    # zip (label, weight) into a single list column and explode once
    gdf["_lu_w"] = gdf.apply(lambda r: list(zip(r[land_uses_column], r["_w_list"])), axis=1)
    gdf_exploded = gdf.explode("_lu_w", ignore_index=False)

    # split back
    gdf_exploded[land_uses_column] = gdf_exploded["_lu_w"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    gdf_exploded["_w"] = gdf_exploded["_lu_w"].apply(lambda x: float(x[1]) if isinstance(x, tuple) else 1.0)

    # spatial index on exploded geometries
    sindex = gdf_exploded.sindex

    def _unexpectedness(building_index, building_geometry, building_label):
        buf = building_geometry.buffer(search_radius)

        candidate_idx = list(sindex.intersection(buf.bounds))
        if not candidate_idx:
            return 0.0

        possible = gdf_exploded.iloc[candidate_idx]
        matches = possible[possible.intersects(buf)]

        # exclude same building (all exploded rows share original index)
        matches = matches[matches.index != building_index]
        if matches.empty:
            return 0.0

        total_w = float(matches["_w"].sum())
        if total_w <= 0:
            return 0.0

        Nj_w = float(matches.loc[matches[land_uses_column] == building_label, "_w"].sum())
        return 1.0 - (Nj_w / total_w)

    gdf_exploded["prag_temp"] = gdf_exploded.apply(
        lambda row: _unexpectedness(row.name, row.geometry, row[land_uses_column]),
        axis=1,
    )

    # collapse per building: max across own labels
    gdf["prag"] = gdf_exploded.groupby(gdf_exploded.index)["prag_temp"].max()

    return gdf.drop(columns=["_w_list", "_lu_w"], errors="ignore")
    
def compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights):
    """
    Computes the component and global scores for a buildings GeoDataFrame, rescaling values when necessary and assigning weights to the different properties measured.

    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        The input GeoDataFrame containing buildings information.
    global_indexes_weights: dict
        Dictionary with index names (string) as keys and weights as values.
    global_components_weights: dict
        Dictionary with component names (string) as keys and weights as values.

    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame with computed scores.
    
    Examples
    --------
    # Example 1: Computing global scores for a buildings GeoDataFrame
    >>> buildings_gdf = gpd.GeoDataFrame(...)
    >>> global_indexes_weights = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.30, "2dvis": 0.30, "neigh": 0.20, "road": 0.20}
    >>> global_components_weights = {"vScore": 0.50, "sScore": 0.30, "cScore": 0.20, "pScore": 0.10}
    """  
    # scaling
    cols = {"direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"], "inverse": ["neigh", "road"]}
   
    if not (abs(sum(global_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Global components weights must sum to 1.0")
   
    # Check if vScore should be computed
    compute_vScore = (
        "vScore" in global_components_weights and
        "height" in buildings_gdf.columns and
        buildings_gdf["height"].max() > 0.0
    )
    
    # Rescale values if the column exists in buildings_gdf
    for col in cols["direct"] + cols["inverse"]:
        if col in buildings_gdf.columns:
            buildings_gdf[col + "_sc"] = scaling_columnDF(buildings_gdf[col], inverse=(col in cols["inverse"]))

    # Compute component scores
    # Visual Score
    if compute_vScore:
        buildings_gdf["vScore"] = sum(
            buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
            for col in ["fac", "height", "3dvis"] if f"{col}_sc" in buildings_gdf
        )
        buildings_gdf["vScore_sc"] = scaling_columnDF(buildings_gdf["vScore"])
        
    # Structural Score

    buildings_gdf["sScore"] = sum(
        buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
        for col in ["area", "neigh", "2dvis", "road"] if f"{col}_sc" in buildings_gdf
    )
    buildings_gdf["sScore_sc"] = scaling_columnDF(buildings_gdf["sScore"])
    
    buildings_gdf["cScore"], buildings_gdf["pScore"] = buildings_gdf["cult_sc"], buildings_gdf["prag_sc"]
    
    # Cultural and Pragmatic Scores
    if "cult_sc" in buildings_gdf.columns:
        buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    if "prag_sc" in buildings_gdf.columns:
        buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    
    # Final global score: Compute dynamically
    buildings_gdf["gScore"] = sum(
        buildings_gdf[f"{component}_sc"] * global_components_weights[component]
        for component in global_components_weights
        if f"{component}_sc" in buildings_gdf and (component != "vScore" or compute_vScore)
    )
    
    buildings_gdf["gScore_sc"] = scaling_columnDF(buildings_gdf["gScore"])
    return buildings_gdf

def compute_local_scores(buildings_gdf, local_indexes_weights, local_components_weights, rescaling_radius = 1500):
    """
    The function computes landmarkness at the local level. The components' weights may be different from the ones used to calculate the
    global score. The radius parameter indicates the extent of the area considered to rescale the landmarkness local score.
    - local_indexes_weights: keys are index names (string), items are weights.
    - local_components_weights: keys are component names (string), items are weights.
    
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        The input GeoDataFrame containing buildings information.
    local_indexes_weights: dict
        Dictionary with index names (string) as keys and weights as values.
    local_components_weights: dict
        Dictionary with component names (string) as keys and weights as values.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    
    Examples
    --------
    >>> # local landmarkness indexes weights, cScore and pScore have only 1 index each
    >>> local_indexes_weights = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.40, "2dvis": 0.00, "neigh": 0.30 , "road": 0.30}
    >>> # local landmarkness components weights
    >>> local_components_weights = {"vScore": 0.25, "sScore": 0.35, "cScore":0.10 , "pScore": 0.30} 
    """  
    
    sindex = buildings_gdf.sindex # spatial index
    
    # Validate that local_components_weights sum to 1.0
    if not (abs(sum(local_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Local components weights must sum to 1.0")

    # Initialize scores conditionally
    compute_vScore = (
        "vScore" in local_components_weights and
        "height" in buildings_gdf.columns and
        buildings_gdf["height"].max() > 0.0
    )
    if compute_vScore:
        buildings_gdf["vScore_l"] = 0.0  # Initialize only if valid height data exists
   
    buildings_gdf["sScore_l"] = 0.0
    buildings_gdf["lScore"] = 0.0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {
            executor.submit(
                _building_local_score,
                row["geometry"],
                idx,
                buildings_gdf,
                sindex,
                local_components_weights,
                local_indexes_weights,
                rescaling_radius,
            ): idx
            for idx, row in buildings_gdf.iterrows()
        }
        for future in concurrent.futures.as_completed(future_scores):
            buildingID = future_scores[future]
            try:
                buildings_gdf.loc[buildingID, "lScore"] = future.result()
            except Exception:
                buildings_gdf.loc[buildingID, "lScore"] = 0.0
    
    buildings_gdf["lScore_sc"] = scaling_columnDF(buildings_gdf["lScore"])
    return buildings_gdf
    
def _building_local_score(building_geometry, buildingID, buildings_gdf, buildings_gdf_sindex, local_components_weights, local_indexes_weights, radius):
    """
    The function computes landmarkness at the local level for a single building. 
    
    Parameters
    ----------
    building_geometry  Polygon
        The geometry of the building.
    buildingID: int
        The ID of the building.
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame containing the buildings.
    buildings_gdf_sindex: Spatial Index
        The spatial index of the buildings GeoDataFrame.
    local_components_weights: dictionary
        The weights assigned to local-level components.
    local_indexes_weights: dictionary
        The weights assigned to local-level indexes.
    radius: float
        The radius that regulates the area around the building within which the scores are recomputed.

    Returns
    -------
    score : float
        The computed local-level landmarkness score for the building.
    """
                                             
    cols = {"direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"], "inverse": ["neigh", "road"]}
    
    buffer = building_geometry.buffer(radius)
    possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    matches = buildings_gdf.iloc[list(buildings_gdf_sindex.intersection(buffer.bounds))].copy()
    matches = matches[matches.intersects(buffer)]
                
    # Rescale all values dynamically if the column exists in matches
    for column in cols["direct"] + cols["inverse"]:
        if column in matches.columns:
            matches[column + "_sc"] = scaling_columnDF(matches[column], inverse=(column in cols["inverse"]))
  
    # Compute structural score (sScore)
    if "sScore" in local_components_weights:
        matches["sScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col] for col in ["area", "2dvis", "neigh", "road"] if f"{col}_sc" in matches
        )
  
    # Recomputing visual scores only if "height" is valid
    # Determine if vScore should be computed
    compute_vScore = (
        "vScore" in local_components_weights and
        "height" in matches.columns and
        matches["height"].max() > 0.0
    )
    
    if compute_vScore:
        matches["vScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col] for col in ["fac", "height", "3dvis"] if f"{col}_sc" in matches
        )
  
    # Compute cultural and pragmatic scores if defined
    if "cScore" in local_components_weights:
        matches["cScore_l"] = matches["cult_sc"]
    if "pScore" in local_components_weights:
        matches["pScore_l"] = matches["prag_sc"]
    
    # Rescale component scores dynamically
    for component in local_components_weights.keys():
        if f"{component}_l" in matches and (component != "vScore" or compute_vScore):
            matches[f"{component}_l_sc"] = scaling_columnDF(matches[f"{component}_l"])

    # Compute the final local score
    matches["lScore"] = sum(
        matches[f"{component}_l_sc"] * local_components_weights[component]
        for component in local_components_weights
        if f"{component}_l_sc" in matches and (component != "vScore" or compute_vScore)
    )

    # Return the local score for the specified building
    return round(matches.loc[buildingID, "lScore"], 3)

def assert_all_polygons(gdf: gpd.GeoDataFrame):
    
    invalid = gdf[~gdf.geometry.apply(lambda g: isinstance(g, (Polygon)))]
    if not invalid.empty:
        raise TypeError(f"Found non-polygon geometries: {invalid.geometry.geom_type.unique().tolist()}") 