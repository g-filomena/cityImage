import math
import numpy as np
from math import sqrt
from shapely.geometry import Point, LineString, MultiLineString

"""
A series of math functions for angle computations.
Readapted for LineStrings from Abhinav Ramakrishnan's post in https://stackoverflow.com/a/28261304/7375309.
"""  
def _dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
        
def get_coord_angle(origin, distance, angle):
    """
    The function returns the coordinates of the line starting from a tuple of coordinates, which forms with the y axis an angle in degree of a certain magnitude,
    given the distance from the origin.
        
    Parameters
    ----------
    origin: tuple of float
        tuple of coordinates
    distance: float
        the distance from the origin coordinates
    angle: float
        the desired angle

    Returns:
    ----------
    coords: tuple
        the resulting coordinates
    """

    (disp_x, disp_y) = (distance * math.sin(math.radians(angle)), distance * math.cos(math.radians(angle)))
    coord = (origin[0] + disp_x, origin[1] + disp_y)
    return coord

def angle_line_geometries(line_geometryA, line_geometryB, degree = False, deflection = False, angular_change = False):
    """
    Given two LineStrings it computes the deflection angle between them. Returns value in degrees or radians.
    
    Parameters
    ----------
    line_geometryA: LineString
        the first line
    line_geometryB: LineString
        the other line; it must share a vertex with line_geometryA
    degree: boolean
        if True it returns value in degree, otherwise in radians
    deflection: boolean
        if True it computes angle of incidence between the two lines, otherwise angle between vectors
    angular_change: boolean
        LineStrings are formed by two vertexes "from" and to". Within the function, four vertexes (2 per line) are considered; two of them are equal and shared between the lines.
        The common vertex is used to compute the angle, along with another vertex per line. If True it computes angle of incidence between the two lines, on the basis of the vertex in common and the second following
        (intermediate, if existing) vertexes forming each of the line. For example, if the line_geometryA has 3 vertexes composing its geometry, from, to and an intermediate one, the latter is used to compute 
        the angle along with the one which is shared with line_geometryB. When False, the angle is computed by using exclusively from and to nodes, without considering intermediate vertexes which form the line geometries.
        
    Returns:
    ----------
    angle: float
        the resulting angle in radians or degrees
    """
           
    # extracting coordinates and createing lines
    coordsA = list(line_geometryA.coords)
    x_originA, y_originA = float("{0:.10f}".format(coordsA[0][0])), float("{0:.10f}".format(coordsA[0][1]))
    x_secondA, y_secondA = float("{0:.10f}".format(coordsA[1][0])), float("{0:.10f}".format(coordsA[1][1]))
    x_destinationA, y_destinationA = float("{0:.10f}".format(coordsA[-1][0])), float("{0:.10f}".format(coordsA[-1][1]))
    x_second_lastA, y_second_lastA = float("{0:.10f}".format(coordsA[-2][0])), float("{0:.10f}".format(coordsA[-2][1]))
    
    coordsB = list(line_geometryB.coords)
    x_originB, y_originB = float("{0:.10f}".format(coordsB[0][0])), float("{0:.10f}".format(coordsB[0][1]))
    x_secondB, y_secondB = float("{0:.10f}".format(coordsB[1][0])), float("{0:.10f}".format(coordsB[1][1]))
    x_destinationB, y_destinationB  = float("{0:.10f}".format(coordsB[-1][0])), float("{0:.10f}".format(coordsB[-1][1]))
    x_second_lastB, y_second_lastB  = float("{0:.10f}".format(coordsB[-2][0])), float("{0:.10f}".format(coordsB[-2][1]))
    
    if angular_change:
        if (x_destinationA, y_destinationA) == (x_destinationB, y_destinationB):
            lineA = ((x_second_lastA, y_second_lastA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_second_lastB, y_second_lastB))

        elif (x_destinationA, y_destinationA) == (x_originB, y_originB):
            lineA = ((x_second_lastA, y_second_lastA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif (x_originA, y_originA) == (x_originB, y_originB):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif (x_originA, y_originA) == (x_destinationB, y_destinationB):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_second_lastB, y_second_lastB))
        # no common vertex      
        else: raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
    
    # deflection on the entire lines
    elif deflection:
        if (x_destinationA, y_destinationA) == (x_destinationB, y_destinationB):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif (x_destinationA, y_destinationA) == (x_originB, y_originB):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif (x_originA, y_originA) == (x_originB, y_originB):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif (x_originA, y_originA) == (x_destinationB, y_destinationB):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))
        # no common vertex   
        else: raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
    
    # angle between vectors
    else:
        if (x_destinationA, y_destinationA) == (x_destinationB, y_destinationB):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif (x_destinationA, y_destinationA) == (x_originB, y_originB):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif (x_originA, y_originA) == (x_originB, y_originB):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        elif (x_originA, y_originA) == (x_destinationB, y_destinationB):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB),(x_originB, y_originB)) 
        # no common vertex   
        else: raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
    
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    
    try:
        # Get dot prod
        dot_prod = _dot(vA, vB)
        # Get magnitudes
        magA = _dot(vA, vA)**0.5
        magB = _dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle_rad = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        angle_deg = math.degrees(angle_rad)%360
        
    except:
        angle_deg = 0.0
        angle_rad = 0.0
    
    angle = angle_rad
    if degree:
        angle = angle_deg
    return angle
    
def difference_angle_line_geometries(line_geometryA, line_geometryB):
    """
    Given two LineStrings it computes the difference of the angles that they form with the Y-axis. Returns value in degrees or radians.
    
    Parameters
    ----------
    line_geometryA: LineString
        the first line
    line_geometryB: LineString
        the other line; it must share a vertex with line_geometryA
            
    Returns:
    ----------
    difference_angle: float
        the resulting difference in degrees
    """
           
    # extracting coordinates and createing lines
    coordsA = list(line_geometryA.coords)
    x_originA, y_originA = float("{0:.10f}".format(coordsA[0][0])), float("{0:.10f}".format(coordsA[0][1]))
    x_destinationA, y_destinationA = float("{0:.10f}".format(coordsA[-1][0])), float("{0:.10f}".format(coordsA[-1][1]))
    
    coordsB = list(line_geometryB.coords)
    x_originB, y_originB = float("{0:.10f}".format(coordsB[0][0])), float("{0:.10f}".format(coordsB[0][1]))
    x_destinationB, y_destinationB  = float("{0:.10f}".format(coordsB[-1][0])), float("{0:.10f}".format(coordsB[-1][1]))
    
    if x_originA == x_destinationA: 
        angle_A = np.pi/2
    else: angle_A = np.arctan((y_destinationA-y_originA)/(x_destinationA-x_originA))  
    if x_originB == x_destinationB: 
        angle_B = np.pi/2  
    else: angle_B = np.arctan((y_destinationB-y_originB)/(x_destinationB-x_originB)) 
    angle_A = math.degrees(angle_A)%360
    angle_B = math.degrees(angle_B)%360
    
    if angle_A > 180: 
        angle_A = angle_A-180
    if angle_B > 180: 
        angle_B = angle_B-180
    difference_angle = abs(angle_A - angle_B)
        
    return difference_angle
    
def is_parallel(line_geometry_A, line_geometry_B, hard = False):
    
    difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
    if (difference_angle <= 30):
        return True
        
    line_coordsA = list(line_geometry_A.coords)
    line_coordsB = list(line_geometry_B.coords)
    if ((len(line_coordsA) == 2) | (len(line_coordsB) == 2)): 
        return False
       
    if not hard:
        # remove first coordinates (A,B)
        line_geometry_A = LineString([coor for coor in line_coordsA[1:]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): 
            return True
        
        # remove first (A) and last (B)
        line_geometry_B = LineString([coor for coor in line_coordsB[:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): 
            return True
        
        # remove last (A) and first (B)
        line_geometry_A = LineString([coor for coor in line_coordsA[:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): 
            return True
        
        # remove last coordinates (A, B)
        line_geometry_A = LineString([coor for coor in line_coordsA[:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): 
            return True
        
        if ((len(line_coordsA) == 3) | (len(line_coordsB) == 3)):
            return False
        line_geometry_A = LineString([coor for coor in line_coordsA[1:-1]])
        line_geometry_B = LineString([coor for coor in line_coordsB[1:-1]])
        difference_angle = difference_angle_line_geometries(line_geometry_A, line_geometry_B)
        if (difference_angle <= 20) & (difference_angle >= -20): 
            return True
        
    return False   
    
def is_continuation(ix_lineA, ix_lineB, edges_gdf):

    nameA = edges_gdf.loc[ix_lineA]['name']
    nameB = edges_gdf.loc[ix_lineB]['name']
    line_geometry_A = edges_gdf.loc[ix_lineA]['geometry']
    line_geometry_B = edges_gdf.loc[ix_lineB]['geometry']
    if is_parallel(line_geometry_A, line_geometry_B, hard = True): 
        return True
    return ((nameA == nameB) & (is_parallel(line_geometry_A, line_geometry_B, hard = False)))   
    
class Error(Exception):
    """Base class for other exceptions"""
    
class AngleError(Error):
    """Raised when not-intersecting lines are provided for computing angles"""
    
    
            
            
            
            
            
            


    
