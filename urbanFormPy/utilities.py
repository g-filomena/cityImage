import pandas as pd, numpy as np, geopandas as gpd
import math
from math import sqrt
from shapely.geometry import Point, LineString, MultiLineString

pd.set_option('precision', 10)

    
def scaling_columnDF(df, i, inverse = False):
    
    """
    it scales a column of a dataframe from 0 to 1
    
    Parameters
    df: pandas dataframe
    i: string (column name)
    ----------
    """
    df[i+'_sc'] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    if (inverse == True): df[i+'_sc'] = 1-(df[i]-df[i].min())/(df[i].max()-df[i].min())
        
	
def dict_to_df(list_dict, list_col):
    """
    It takes a list of dictionary and merge them in a df, as columns.
    
    ----------
    Parameters
    list_dict: list of dictionaries that will become df's columns
    list_col: list of the names that will be used as colums heading
    
    Returns:
    ----------
    DataFrame
    """
    
    df = pd.DataFrame(list_dict).T
    df.columns = ['d{}'.format(i) for i, col in enumerate(df, 1)]
    df.columns = list_col
    
    return(df)

	
# Angle/math functions
	
def euclidean_distance(xs, ys, xt, yt):
    """ xs stands for x source and xt for x target """
    return sqrt((xs - xt)**2 + (ys - yt)**2)

"""
math functions for angle computations
readapted for LineStrings from Abhinav Ramakrishnan answer in https://stackoverflow.com/a/28261304/7375309
"""

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]


def get_coord_angle(origin, distance, angle):

    (disp_x, disp_y) = (distance * math.sin(math.radians(angle)), distance * math.cos(math.radians(angle)))
    return (origin[0] + disp_x, origin[1] + disp_y)

def ang_geoline(geolineA, geolineB, degree = False, deflection = False, angular_change = False):
    
    """
    Given two LineStrings it computes the deflection angle between them. Returns value in degrees or radians.
    
    ----------
    Parameters
    geolineA, geolineB: LineString geometries
    degree: Boolean
    deflection: Boolean, if true computes angle of incidence, otherwise angle formed by the two vectors as diverging from the common node.
    
    Returns:
    ----------
    float
    """
    if angular_change == True: deflection = True
        
    # extracting coordinates and creates lines
    coordsA = list(geolineA.coords)
    coordsB = list(geolineB.coords)   

    x_originA = float("{0:.10f}".format(coordsA[0][0]))
    y_originA = float("{0:.10f}".format(coordsA[0][1]))
    x_secondA = float("{0:.10f}".format(coordsA[1][0]))
    y_secondA = float("{0:.10f}".format(coordsA[1][1]))
    
    x_destinationA = float("{0:.10f}".format(coordsA[-1][0]))
    y_destinationA = float("{0:.10f}".format(coordsA[-1][1]))
    x_secondlastA = float("{0:.10f}".format(coordsA[-2][0]))
    y_secondlastA = float("{0:.10f}".format(coordsA[-2][1]))
    
    x_originB = float("{0:.10f}".format(coordsB[0][0]))
    y_originB = float("{0:.10f}".format(coordsB[0][1]))
    x_secondB = float("{0:.10f}".format(coordsB[1][0]))
    y_secondB = float("{0:.10f}".format(coordsB[1][1]))
    
    x_destinationB = float("{0:.10f}".format(coordsB[-1][0]))
    y_destinationB = float("{0:.10f}".format(coordsB[-1][1]))
    x_secondlastB = float("{0:.10f}".format(coordsB[-2][0]))
    y_secondlastB = float("{0:.10f}".format(coordsB[-2][1]))
    
    if (deflection == True) & (angular_change == True):
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_secondlastA, y_secondlastA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_secondlastB, y_secondlastB))

        elif ((x_destinationA, y_destinationA) == (x_originB, y_originB)):
            lineA = ((x_secondlastA, y_secondlastA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_originB, y_originB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_destinationB, y_destinationB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_secondlastB, y_secondlastB))
            
        else:  print(issue)
            
    if (deflection == True) & (angular_change == False):
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif ((x_destinationA, y_destinationA) == (x_originB, y_originB)):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif ((x_originA, y_originA) == (x_originB, y_originB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif ((x_originA, y_originA) == (x_destinationB, y_destinationB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        else:  print(issue)
#             lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
#             lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))
    else:
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif ((x_destinationA, y_destinationA) == (y_originB, y_originB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((y_originB, y_originB), (x_destinationB, y_destinationB))

        elif ((x_originA, y_originA) == (y_originB, y_originB)):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        else: #(originA == destinationB)
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB),(x_originB, y_originB)) 
    
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    
    try:
        # Get dot prod
        dot_prod = dot(vA, vB)
        # Get magnitudes
        magA = dot(vA, vA)**0.5
        magB = dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle_rad = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        angle_deg = math.degrees(angle_rad)%360
        
    except:
        angle_deg = 0
        angle_rad = 0
        
    if degree == True: return angle_deg
    else: return angle_rad
    
def ang_geoline(geolineA, geolineB, degree = False, deflection = False):
    
    """
    Given two LineStrings it computes the deflection angle between them. Returns value in degrees or radians.
    
    ----------
    Parameters
    geolineA, geolineB: LineString geometries
    degree: Boolean
    deflection: Boolean, if true computes angle of incidence, otherwise angle formed by the two vectors as diverging from the common node.
    
    Returns:
    ----------
    float
    """
    
    # extracting coordinates and creates lines
    coordsA = list(geolineA.coords)
    coordsB = list(geolineB.coords)   

    x_originA = float("{0:.10f}".format(coordsA[0][0]))
    y_originA = float("{0:.10f}".format(coordsA[0][1]))
    x_secondA = float("{0:.10f}".format(coordsA[1][0]))
    y_secondA = float("{0:.10f}".format(coordsA[1][1]))
    
    x_destinationA = float("{0:.10f}".format(coordsA[-1][0]))
    y_destinationA = float("{0:.10f}".format(coordsA[-1][1]))
    x_secondlastA = float("{0:.10f}".format(coordsA[-2][0]))
    y_secondlastA = float("{0:.10f}".format(coordsA[-2][1]))
    
    x_originB = float("{0:.10f}".format(coordsB[0][0]))
    y_originB = float("{0:.10f}".format(coordsB[0][1]))
    x_secondB = float("{0:.10f}".format(coordsB[1][0]))
    y_secondB = float("{0:.10f}".format(coordsB[1][1]))
    
    x_destinationB = float("{0:.10f}".format(coordsB[-1][0]))
    y_destinationB = float("{0:.10f}".format(coordsB[-1][1]))
    x_secondlastB = float("{0:.10f}".format(coordsB[-2][0]))
    y_secondlastB = float("{0:.10f}".format(coordsB[-2][1]))
    
    if deflection == True:
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_secondlastA, y_secondlastA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_secondlastB, y_secondlastB))

        elif ((x_destinationA, y_destinationA) == (x_originB, y_originB)):
            lineA = ((x_secondlastA, y_secondlastA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_originB, y_originB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_destinationB, y_destinationB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_secondlastB, y_secondlastB))
            
        else: #no common vertex
            print(issue)
#             lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
#             lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))
    else:
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif ((x_destinationA, y_destinationA) == (y_originB, y_originB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((y_originB, y_originB), (x_destinationB, y_destinationB))

        elif ((x_originA, y_originA) == (y_originB, y_originB)):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        else: #(originA == destinationB)
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB),(x_originB, y_originB)) 
    
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    
    try:
        # Get dot prod
        dot_prod = dot(vA, vB)
        # Get magnitudes
        magA = dot(vA, vA)**0.5
        magB = dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle_rad = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        angle_deg = math.degrees(angle_rad)%360
        
    except:
        angle_deg = 0
        angle_rad = 0
        
    if degree == True: return angle_deg
    else: return angle_rad

def ang(lineA, lineB, degree = True):
    # Get nicer vector form
    
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    print(vA, vB)
    # Get dot prod
    dot_prod = dot(vA, vB)

    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    print(dot_prod)
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle_rad = math.acos(dot_prod/magB/magA)
    print(angle_rad)
    # Basically doing angle <- angle mod 360
    angle_deg = math.degrees(angle_rad)%360
    if degree == True: return angle_deg
    else: return angle_rad
    
def center_line(line_geo, line_geoC): 
    
    line_coordsA = list(line_geo.coords)
    line_coordsB = list(line_geoC.coords)
        
    if ((line_coordsA[0] == line_coordsB[-1]) | (line_coordsA[-1] == line_coordsB[0])): line_coordsB.reverse()  
    if line_coordsA == line_coordsB: center_line = LineString([coor for coor in line_coordsA]) 
    else:
        while len(line_coordsA) > len(line_coordsB):
            index = int(len(line_coordsA)/2)
            del line_coordsA[index]
        while len(line_coordsB) > len(line_coordsA):
            index = int(len(line_coordsB)/2)
            del line_coordsB[index]          
    
        new_line = line_coordsA
        for n, i in enumerate(line_coordsA):
            link = LineString([coor for coor in [line_coordsA[n], line_coordsB[n]]])
            np = link.centroid.coords[0]       
            new_line[n] = np
            
        new_line[0] = line_coordsA[0]
        new_line[-1] = line_coordsA[-1]
        center_line = LineString([coor for coor in new_line])

    return(center_line)


def dist_to_gdf_point(point, gpd):
    gpd = gpd.copy()
    gpd['Dist'] = gpd.apply(lambda row:  point.distance(row.geometry),axis=1)
    geoseries = gpd.loc[gpd['Dist'].argmin()]
    distance  = geoseries.Dist
    index = geoseries.name
    return distance, index

def dist_to_gdf_line(geoline, gpd):
    gpd = gpd.copy()
    gpd['Dist'] = gpd.apply(lambda row:  geoline.distance(row.geometry),axis=1)
    geoseries = gpd.loc[gpd['Dist'].argmin()]
    distance  = geoseries.Dist
    index = geoseries.name
    return distance, index

def merge_lines(geolines):
    
    first = list(geolines[0].coords)
    second = list(geolines[1].coords)
    coords = []

    reverse = False
    if first[0] == second[0]: 
        reverse = True
        first.reverse()
    if first[-1] == second[-1]: second.reverse()
    if first[0] == second[-1]:
        first.reverse()
        second.reverse()
        reverse = True
    
    coords = first + second
    last = second
    for n,i in enumerate(geolines):
        if n < 2: continue
        next_coords = list(i.coords)
        if (next_coords[-1] == last[-1]) :
            next_coords.reverse()
            last = next_coords
            
    if reverse == True: coords.reverse()
    geoline = LineString([coor for coor in coords])
    return(geoline)
            
def print_row(index_column):
    sys.stdout.write('\r')
    sys.stdout.write("at row: "+ str(index_column))
    sys.stdout.flush()
    sleep(0.0001)
    
def merge_disconnected_lines(list_lines):
    new_line = []
    for n, i in enumerate(list_lines):
        coords = list(i.coords)
        if n < len(list_lines)-1: coords.append(list_lines[n+1].coords[-1])
        new_line = new_line + coords

    geoline = LineString([coor for coor in new_line])
    return(geoline)
 
def find_actual_centroid(geoline):    
    coords = list(geoline.coords)
    if len(coords) == 2: centroid = geoline.coords[0]       
    else:
        if len(coords)%2 == 0:
            left = int((len(coords)/2)-1)
            right = int((len(coords)/2)+1)
            tmp = LineString([coords[left], coords[right]])
            centroid = tmp.centroid.coords[0]
        else:
            centroid_position = int(len(coords)/2)  
            centroid = coords[centroid_position]

    return(centroid)

    
def line_at_centroid(geoline, offset):
    left = geoline.parallel_offset(offset, 'left')
    right =  geoline.parallel_offset(offset, 'right')
    
    if left.geom_type == 'MultiLineString': left = merge_disconnected_lines(left)
    if right.geom_type == 'MultiLineString': right = merge_disconnected_lines(right)   
    
    if (left.is_empty == True) & (right.is_empty == False): left = geoline
    if (right.is_empty == True) & (left.is_empty == False): right = geoline
    
    left_centroid = find_actual_centroid(left)
    right_centroid = find_actual_centroid(right)
    
    fict = LineString([left_centroid, right_centroid])
    return(fict)

            
            
            
            
            
            


    
