import pandas as pd
import numpy as np
import geopandas as gpd
import math
import pyproj
import osmnx as ox
import networkx as nx

from typing import List
from math import sqrt
from shapely.geometry import LineString, Point, Polygon, MultiPoint, mapping, LinearRing, MultiPolygon
from shapely.ops import unary_union, transform, nearest_points, split, linemerge
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry
import pyvista as pv
pd.set_option("display.precision", 3)

class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided""" 
    
def downloader(place, download_method, tags = None, distance = 500.0, downloading_graph = False, network_type = None):
    """
    The function downloads certain geometries from OSM, by means of OSMNX functions.
    It returns a GeoDataFrame, that could be empty when no geometries are found, with the provided tags.
    
    Parameters
    ----------
    place: str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide a Shapely Polygon in unprojected latitude-longitude degrees (EPSG:4326) CRS;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    tag: dict 
        The desired OSMN tags.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    downloading_graph: bool
        Download a graph, instead of, as by default, GeoDataFrame.
    network_type: str {"walk", "bike", "drive", "drive_service", "all", "all_private", "none"}
        It indicates type of street or other network to extract - from OSMNx paramaters.
        
    Returns
    -------
    G, geometries_gdf: NetworkX Graph, GeoDataFrame
        The resulting Graph (when downloading_graph) or a GeoDataFrame.
    """    
    download_options = {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
    if download_method not in download_options:
        raise downloadError('Provide a download method amongst {}'.format(download_options))

    download_method_dict = {
        'distance_from_address': ox.features_from_address,
        'distance_from_point': ox.features_from_point,
        'OSMplace': ox.features_from_place,
        'polygon': ox.features_from_polygon
        }
    if downloading_graph:
        download_method_dict = {
        'distance_from_address': ox.graph_from_address,
        'distance_from_point': ox.graph_from_point,
        'OSMplace': ox.graph_from_place,
        'polygon': ox.graph_from_polygon
        }
    
    download_func = download_method_dict.get(download_method)
    # using OSMNx to download data from OpenStreetMap  
    try:
        if download_func:
            if download_method in ['distance_from_address', 'distance_from_point']:
                if downloading_graph:
                    G = download_func(place, network_type = network_type, dist = distance, retain_all=True, simplify = True)
                else:
                    geometries_gdf = download_func(place, tags = tags, dist = distance)
            else:
                if downloading_graph:
                    G = download_func(place, network_type = network_type, retain_all=True, simplify = True)
                else:
                    geometries_gdf = download_func(place, tags = tags) 
                    
    except ox._errors.InsufficientResponseError:
        # Handle the InsufficientResponseError error by returning an empty GeoDataFrame
        geometries_gdf = gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry').set_crs('EPSG:4326')
        if downloading_graph:
            G=nx.empty_graph()
    if downloading_graph:
        return G
    return geometries_gdf
    
def scaling_columnDF(series, inverse=False):
    """
    Rescales the values in a dataframe's column from 0 to 1 or from 1 to 0.

    Parameters
    ----------
    series: pd.Series
        The pd.Series to rescale.
    inverse: bool, optional
        If True, rescales from 1 to 0 instead of 0 to 1. Default is False.

    Returns
    -------
    pd.Series
        The rescaled pd.Series.
    """
    if series.max() == series.min():
        # If all values are the same, return 0 for all (or 1 if inverse=True)
        scaled = pd.Series(0.0, index=series.index)
        if inverse:
            scaled = pd.Series(1.0, index=series.index)
    else:
        # Normal scaling
        scaled = (series - series.min()) / (series.max() - series.min())
        if inverse:
            scaled = 1 - scaled
    return scaled

        
    
def dict_to_df(list_dict, list_col):
    """
    It takes a list of dictionaries and creates from them a pandas DataFrame. Each dictionary becomes a Series, where keys are rows,
    values cell values. The column names are renamed trough a list of strings (list_col).
    
    Parameters
    ----------
    list_dict: list of dict
        The list of dictionaries.
    list_col: list of string
        The corresponding column names to assign to the series.
    
    Returns:
    ----------
    df: Pandas DataFrame
        The resulting DataFrame.
    """
    df = pd.DataFrame(list_dict).T
    df.columns = ["d{}".format(i) for i, col in enumerate(df, 1)]
    df.columns = list_col
    return df
     
def center_line(line_geometries):
    """
    Computes the center line for a list of LineStrings by averaging their coordinates.

    Parameters
    ----------
    line_geometries : list of LineString
        A list of LineStrings to compute the center line.

    Returns
    -------
    LineString
        The resulting center line as a LineString.
    """
    if len(line_geometries) < 2:
        raise ValueError("At least two LineStrings are required to compute a center line.")

    # Extract coordinates from all LineStrings
    all_coords = [list(line.coords) for line in line_geometries]

    # Ensure all LineStrings have the same direction (align the first and last points)
    for i in range(1, len(all_coords)):
        if all_coords[i][0] != all_coords[i - 1][-1]:
            all_coords[i] = all_coords[i][::-1]

    # Ensure all lists have the same length by trimming the longer ones
    min_length = min(len(coords) for coords in all_coords)
    all_coords = [coords[:min_length] for coords in all_coords]

    # Compute the average of coordinates for each point across all LineStrings
    center_line_coords = [
        [
            sum(coords[i][0] for coords in all_coords) / len(all_coords),
            sum(coords[i][1] for coords in all_coords) / len(all_coords),
        ]
        for i in range(min_length)
    ]

    # Create and return the resulting LineString
    return LineString(center_line_coords)

def min_distance_geometry_gdf(geometry, gdf):
    """
    Given a geometry and a GeoDataFrame, it returns the minimum distance between the geometry and the GeoDataFrame. 
    It provides also the index of the closest geometry in the GeoDataFrame.
    
    Parameters
    ----------
    geometry: Point, LineString or Polygon
    
    gdf: GeoDataFrame
    
    Returns:
    ----------
    distance, index: tuple
        The closest distance from the geometry, and the index of the closest geometry in the gdf.
    """
    sindex = gdf.sindex
    closest = sindex.nearest(geometry, return_distance = True)
    iloc = closest[0][1][0]
    distance = closest[1][0]
    index = gdf.iloc[iloc].name 
    return distance, index
 
def split_line_at_MultiPoint(line_geometry, intersections, z = 0.0):   
    """
    The function checks whether the coordinates of one or more Points in a Point Collections are part of the sequence of coordinates of a LineString.
    When this has been ascerted or fixed, the LineString line_geometry is split at each of the intersecting points in the collection and a list of lines, 
    each representing one of the sections resulting from the intersections is returned.

    Parameters
    -------
    line_geometry: LineString
        The LineString which has to be split.
    intersections: MultiPoint
        The intersecting points.
    z: float
        The z-coordinate value to assign to each vertex of the resulting lines. Default is 0.0.
        
    Returns
    -------
    lines: List of MultiLineString
        The resulting segments composing the original line_geometry.
    """
    line_geometry_tmp = line_geometry
    # iterate over each point in the intersections
    for point in intersections:
        # create a new list of coordinates from the line_geometry
        new_line_coords = list(line_geometry_tmp.coords)
        # iterate over the new line coordinates; skip the first coordinate
        for n, coord in enumerate(new_line_coords):
            if n == 0:
                continue
            # create a LineString between the previous and current coordinates
            line_section = LineString([Point(new_line_coords[n-1]), Point(coord)])
            # check if the point intersects this line section or if it is very close to it
            if ((point.intersects(line_section)) | (line_section.distance(point) < 1e-8)):
                # if so, insert the coordinates of the intersection point into the new_line_coords list
                new_line_coords.insert(n, point.coords[0])
                break
        # create a new LineString from the updated coordinates
        line_geometry_tmp = LineString([coor for coor in new_line_coords])

    # convert the intersections to a MultiPoint object
    intersections = MultiPoint(intersections)
    # split the line_geometry_tmp at each of the intersecting points
    lines = split(line_geometry_tmp, intersections)
    # create a list of individual LineString geometries from the split result
    lines_list = [line for line in lines.geoms]
    
    # create LineString objects with z-coordinate values for each vertex
    if z is not None:
        lineZ = [LineString([(coords[0], coords[1], z) for coords in line.coords]) for line in lines_list]
    # return the list of resulting line segments
    return lines_list

def merge_line_geometries(line_geometries):
    """
    Given a list of LineString geometries, this function reorders the geometries in the correct sequence based on their starting and ending points,
    and returns a merged LineString feature.
    
    Parameters:
    ----------
    line_geometries: List of LineString
        A list of LineString geometries to be merged.
        
    Returns:
    ----------
    LineString: 
        The merged LineString feature.
    """
    if not all(isinstance(line, LineString) for line in line_geometries):
        raise ValueError("Input must be a list of LineString geometries")
    if not all(isinstance(line, BaseGeometry) for line in line_geometries):
        raise ValueError("Input must be a list of valid geometries")
    if len(line_geometries) < 2:
        raise ValueError("At least 2 LineStrings are required to merge")
    
    # create a dictionary to store the "from" and "to" vertexes of each LineString as keys
    lines = {(line.coords[0], line.coords[-1]): line for line in line_geometries}
    # sort the line geometries based on their starting and ending coordinates
    line_geometries = sorted(line_geometries, key=lambda line: (line.coords[0], line.coords[-1]))
    # initialize an empty list to store the merged line geometries
    merged = []
    # rmove and store the first line geometry from the line_geometries list
    first_line = line_geometries.pop(0)
    # add the first line geometry to the merged list
    merged.append(first_line)

    # iterate over the remaining line geometries
    for line in line_geometries:
        # check if the last coordinate of the previously merged line is the same as the first coordinate of the current line
        if merged[-1].coords[-1] == line.coords[0]:
            # if so, add the current line to the merged list
            merged.append(line)
        # check if the last coordinate of the previously merged line is the same as the last coordinate of the current line
        elif merged[-1].coords[-1] == line.coords[-1]:
            # if so, add the reversed version of the current line to the merged list
            merged.append(LineString(list(reversed(line.coords))))
    
    return LineString(list(merged))
            
def envelope_wgs(gdf):
    """
    Given a GeoDataFrame it derives its envelope in the WGS coordinate system.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    
    Return
    ----------
    envelope_wgs: Polygon
        The resulting envelope.
    """
    envelope = gdf.unary_union.envelope.buffer(100)
    
    # Define a transformer for projecting from the GeoDataFrame's CRS to WGS84
    transformer = pyproj.Transformer.from_crs(gdf.crs, 'epsg:4326', always_xy=True)
    envelope_wgs = transform(transformer.transform, envelope)
    return envelope_wgs 
    
def convex_hull_wgs(gdf):
    """
    Given a GeoDataFrame it derives its convex hull in the WGS coordinate system.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        
    Return
    ----------
    convex_hull_wgs: Polygon
        The resulting hexagon.
    """
    # Compute the convex hull without buffering
    convex_hull = gdf.unary_union.convex_hull

    # Define a transformer for projecting from the GeoDataFrame's CRS to WGS84
    transformer = pyproj.Transformer.from_crs(gdf.crs, 'epsg:4326', always_xy=True)
    
    # Apply the transformation
    convex_hull_wgs = transform(transformer.transform, convex_hull)
    return convex_hull_wgs      
        
def rescale_ranges(n, range1, range2):
    """
    Given a value n and the range which it belongs to, the function rescale the value, between a different range.
        
    Parameters
    ----------
    n: int, float
        A value.
    range1: tuple
        A certain range, e.g. (0, 1) or (10.5, 100). The value n should be within this range.
    range2: tuple
        A range, e.g. (0, 1) or (10.5, 100), that should used to rescale the value.
        
    Return
    ----------
    value: float
        The rescaled value.
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    value = (delta2 * (n - range1[0]) / delta1) + range2[0]    
    return value
                         
def gdf_from_geometries(geometries, crs):
    """
    The function creates a GeoDataFrame from a list of geometries.
    
    Parameters
    ----------
    geometries: list of LineString, Polygon or Points
        The geometries to be included in the GeoDataFrame.
        
    Returns
    -------
    gdf: GeoDataFrame
        The resulting GeoDataFrame.
    """
    
    df = pd.DataFrame({'geometry': geometries})
    gdf = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    gdf['length'] = gdf['geometry'].length
    return gdf
    
def line_at_centroid(line_geometry, offset):
    """
    Given a LineString, it creates a perpendicular LineString that intersects the given geometry at its centroid.
    The offset determines the distance from the original line.
    This fictional line can be used to count precisely the number of trajectories/features intersecting a segment.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry.
    offset: float
        The offset from the geometry.
    
    Returns
    -------
    LineString
        The perpendicular line intersecting the given one.
    """
    left = line_geometry.parallel_offset(offset, 'left')
    right =  line_geometry.parallel_offset(offset, 'right')
    
    if left.geom_type == 'MultiLineString': 
        left = merge_disconnected_lines(left)
    if right.geom_type == 'MultiLineString': 
        right = merge_disconnected_lines(right)   
    
    if (left.is_empty) & (not right.is_empty): 
        left = line_geometry
    if (right.is_empty) & (not left.is_empty): 
        right = line_geometry
    
    left_centroid = left.interpolate(0.5, normalized = True)
    right_centroid = right.interpolate(0.5, normalized = True)
   
    return LineString([left_centroid, right_centroid])
    
def sum_at_centroid(line_geometry, lines_gdf, column):
    """
    Given a LineString geometry, it sums the column values of all the features in a LineString GeoDataFrame intersecting the line.
    This function should be executed per row by means of the df.apply(lambda row : ..) function.
        
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry.
    lines_gdf: LineString GeoDataFrame
        The GeoDataFrame.
    column: string
        The name of the column.
    
    Returns
    -------
    int
    """
    return lines_gdf[lines_gdf.geometry.intersects(line_geometry)][column].sum()

def polygons_gdf_multiparts_to_singleparts(polygons_gdf):
    """    
    The function extracts the individual parts of the multi-part geometries and creates a new GeoDataFrame with single part geometries.
            
    Parameters
    ----------
    polygons_gdf: GeoDataFrame
        GeoDataFrame containing building footprint geometries.
    
    Returns
    -------
    single_parts_gdf: Polygon GeoDataFrame
        The new GeoDataFrame with simplified single part building footprint geometries.
    """  
    polygons_gdf = polygons_gdf.copy()
    single_parts = gpd.geoseries.GeoSeries([geom for geom in polygons_gdf.unary_union.geoms])
    single_parts_gdf = gpd.GeoDataFrame(geometry=single_parts, crs = polygons_gdf.crs)
    
    return single_parts_gdf

def fix_multiparts_LineString_gdf(gdf):
    """
    Fixes MultiLineString geometries in a GeoDataFrame by merging them into LineString geometries.
    If there are still MultiLineString geometries present after merging, they are exploded into individual LineString geometries.

    Parameters
    ----------
    gdf: GeoDataFrame
        The input GeoDataFrame with potentially problematic MultiLineString geometries.

    Returns
    -------
    gdf: GeoDataFrame
        The fixed GeoDataFrame with MultiLineString geometries merged into LineString geometries.
    """
    gdf = gdf.copy()

    # check if the GeoDataFrame contains MultiLineString geometries
    if 'MultiLineString' in gdf.geometry.type.unique():
        # select the rows where the geometry type is MultiLineString
        condition = (gdf.geometry.type == 'MultiLineString')
        # merge MultiLineString geometries into LineString geometries
        gdf.loc[condition, 'geometry'] = gpd.GeoSeries([linemerge(geo) for geo in gdf[condition].geometry], index=gdf[condition].index)

        # check again if MultiLineString geometries are still present after merging
        if 'MultiLineString' in gdf.geometry.type.unique():
            # create a separate GeoDataFrame with only the MultiLineString geometries
            multi_gdf = gdf[gdf.geometry.type == 'MultiLineString'].copy()
            # remove the MultiLineString geometries from the original GeoDataFrame
            gdf = gdf[gdf.geometry.type == 'LineString']
            # explode the MultiLineString geometries into individual LineString geometries
            multi_gdf = multi_gdf.explode(ignore_index=True)
            # append the exploded LineString geometries back to the original GeoDataFrame
            gdf = gdf.append(multi_gdf, ignore_index=True)

    return gdf
 
def remove_lists_columns(df):
    """
    For each cell in the DataFrame, if the cell contains a list, update it to keep only the first element of the list.

    Parameters
    ----------
    df: DataFrame
        The input DataFrame.

    Returns
    -------
    df: DataFrame
        A DataFrame with the transformation applied to the relevant cells.
    """
    df = df.copy()
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        
    return df
  
def polygon_2d_to_3d(building_polygon, base, height):
    """
    Convert a 2D polygon to a 3D polygon. This function takes a 2D polygon (building_polygon) and extrudes it into 3D space, creating a pv.PolyData,
    creating a 3D polygon with a base and a height elevation.
    
    Parameters
    ----------
    building_polygon (shapely.geometry.Polygon): 
        2D polygon to be extruded.
    base (float): 
        Base height of the 3D polygon.
    height (float): 
        Height of the 3D polygon.
    
    Returns:
    ----------
    pv.PolyData: 
        A 3D polygon.
    """
    
    def reorient_coords(xy):
        """
        Reorient the coordinates of the polygon
        
        This function reorients the coordinates of a polygon if the polygon
        has counterclockwise orientation.
        
        Parameters
        ----------
        xy: list
            List of tuples, each representing a coordinate of the polygon
        
        Returns
        ----------
        list: 
            Reoriented list of tuples representing the polygon's coordinates
        """
        value = 0
        for i in range(len(xy)):
            x1, y1 = xy[i]
            x2, y2 = xy[(i+1)%len(xy)]
            value += (x2-x1)*(y2+y1)
        if value > 0:
            return xy
        else:
            return xy[::-1]
    
    poly_points = building_polygon.exterior.coords
       
    # Reorient the coordinates of the polygon
    xy = reorient_coords(poly_points)
    # Create 3D coordinates with the base height
    xyz_base = [(x,y,base) for x,y in xy]
    # Create faces of the polygon
    faces = [len(xyz_base), *range(len(xyz_base))]
    # Create the 3D polygon using pyvista
    polygon = pv.PolyData(xyz_base, faces=faces)
    
    # Extrude the 3D polygon to the specified height
    return polygon.extrude((0, 0, height - base), capping=True)
    
    
def aggregate_geometries(gdf, column_operations):
    """
    Aggregate overlapping geometries in a GeoDataFrame and perform specified operations on their attributes.

    Parameters
    ----------
    gdf: GeoDataFrame
        The input GeoDataFrame.
    column_operations: dict
        A dictionary where keys are column names and values are aggregation functions ('min', 'max', 'mean').

    Returns:
    ----------
    GeoDataFrame: 
        A new GeoDataFrame with aggregated geometries and attributes.
    """
    
    necessary = True

    while necessary:
        # Create a spatial index for faster lookup
        spatial_index = gdf.sindex
        to_remove_geometries = []

        # Loop through each geometry in the GeoDataFrame
        for idx, row in gdf.iterrows():
            # Find all geometries that contain the current geometry
            possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))
            containing_geometries = gdf.iloc[possible_matches_index]
            containing = containing_geometries[
                (containing_geometries.geometry.contains(row.geometry)) &
                (containing_geometries.index != idx) &
                (~containing_geometries.index.isin(to_remove_geometries))
            ]

            if len(containing) > 0:  # If the current geometry is contained by others
                # Get the container geometry (largest one by area)
                ix_container = containing.geometry.area.idxmax()
                row_container = gdf.loc[ix_container]

                # Update attributes based on the user-defined operations
                for column, operation in column_operations.items():
                    if operation == 'max':
                        gdf.loc[ix_container, column] = max(row[column], row_container[column])
                    elif operation == 'min':
                        gdf.loc[ix_container, column] = min(row[column], row_container[column])
                    elif operation == 'mean':
                        gdf.loc[ix_container, column] = (row[column] + row_container[column]) / 2

                # Add the current geometry to the list for removal
                to_remove_geometries.append(idx)

        # Break if no geometries were removed (no changes made)
        if len(to_remove_geometries) == 0:
            break

        # Remove aggregated geometries from the GeoDataFrame
        gdf = gdf[~gdf.index.isin(to_remove_geometries)]

    return gdf

def convert_numeric_columns(df):
    """
    Converts DataFrame columns to appropriate numeric types:
    - Object columns with integer values to `int64`
    - Object columns with float values to `float64`
    - Ensures existing numeric columns are standardized to `int64` or `float64`
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if df[col].dtype == 'float64' and df[col].dropna().mod(1).eq(0).all():
                df[col] = df[col].astype('int64')  # Convert floats that are actually ints
        except ValueError:
            pass  # Keeps non-convertible columns unchanged
    return df
    
def gdf_multipolygon_to_polygon(gdf):
    
    def convert_multipolygon_to_polygon(geometry):
        if isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
            return geometry.geoms[0]  # Extract the single Polygon
        return geometry  # Return unchanged if already Polygon or a true MultiPolygon

    gdf["geometry"] = gdf["geometry"].apply(convert_multipolygon_to_polygon)

    return gdf 

def has_interior(poly):
    """Returns True if the polygon has interior rings (holes), False otherwise."""
    if poly.geom_type == "MultiPolygon":
        return false
    return len(poly.interiors) > 0

def is_exterior_inside_another(poly, other_poly):
    """Check if the exterior of one polygon is inside another polygon"""
    if not has_interior(poly):
        return False
    coords = list(other_poly.exterior.coords)
    coords.reverse()
    return any(coords == list(ring.coords) for ring in poly.interiors)