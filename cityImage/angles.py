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

class AngleError(Error):
    """Raised when not-intersecting lines are provided for computing angles"""

class Settings():
    """
    A class to store and compare the coordinates of two line geometries.
    
    Attributes:
        x_originA (float): The x-coordinate of the first point of the first line geometry.
        y_originA (float): The y-coordinate of the first point of the first line geometry.
        x_secondA (float): The x-coordinate of the second point of the first line geometry.
        y_secondA (float): The y-coordinate of the second point of the first line geometry.
        x_destinationA (float): The x-coordinate of the last point of the first line geometry.
        y_destinationA (float): The y-coordinate of the last point of the first line geometry.
        x_second_lastA (float): The x-coordinate of the second-last point of the first line geometry.
        y_second_lastA (float): The y-coordinate of the second-last point of the first line geometry.
        x_originB (float): The x-coordinate of the first point of the second line geometry.
        y_originB (float): The y-coordinate of the first point of the second line geometry.
        x_secondB (float): The x-coordinate of the second point of the second line geometry.
        y_secondB (float): The y-coordinate of the second point of the second line geometry.
        x_destinationB (float): The x-coordinate of the last point of the second line geometry.
        y_destinationB (float): The y-coordinate of the last point of the second line geometry.
        x_second_lastB (float): The x-coordinate of the second-last point of the second line geometry.
        y_second_lastB (float): The y-coordinate of the second-last point of the second line geometry.
        lineA (Tuple): The coordinates of the first line used to calculate the angle
        lineB (Tuple): The coordinates of the second linne used to calculate the angle
        """
    
    def set_coordinates(self, coords, prefix):
        """
        Set the coordinates for a line.

        Parameters:
        coords (list): A list of coordinates in the form [(x1, y1), (x2, y2), ...]
        prefix (str): A string prefix to be added to the variable names for the coordinates.
                      For example, if "A" is passed as the prefix, the coordinates will be stored as
                      self.x_originA, self.y_originA, etc.
        """
        setattr(self, 'x_origin'+ prefix, float("{0:.10f}".format(coords[0][0])))
        setattr(self, 'y_origin'+ prefix, float("{0:.10f}".format(coords[0][1])))
        setattr(self, 'x_second'+ prefix, float("{0:.10f}".format(coords[1][0])))
        setattr(self, 'y_second'+ prefix, float("{0:.10f}".format(coords[1][1])))
        setattr(self, 'x_destination'+ prefix, float("{0:.10f}".format(coords[-1][0])))
        setattr(self, 'y_destination'+ prefix, float("{0:.10f}".format(coords[-1][1])))
        setattr(self, 'x_second_last'+ prefix, float("{0:.10f}".format(coords[-2][0])))
        setattr(self, 'y_second_last'+ prefix, float("{0:.10f}".format(coords[-2][1])))
    
    def set_conditions(self, calculation_type):
        """
        Given a Setting object and a calculation type, this function returns the lines that will be used to compute the angle.

        Parameters
        ----------
        Setting: object
            an object of the Setting class, which contains information about the lines
        calculation_type: string
            one of: 'vectors', 'angular_change', 'deflection'
            'vectors': computes angle between vectors
            'angular_change': computes angle of incidence between the two lines,
            'deflection': computes angle of incidence between the two lines, on the basis of the vertex in common and the second following(intermediate, if existing) vertexes forming each of the line.

        Raises
        ------
        AngleError: if the lines do not have a common vertex
        """
        if calculation_type == "angular_change":
            if (self.x_destinationA, self.y_destinationA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_second_lastA, self.y_second_lastA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_destinationB, self.y_destinationB), (self.x_second_lastB, self.y_second_lastB))
            elif (self.x_destinationA, self.y_destinationA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_second_lastA, self.y_second_lastA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_secondB, self.y_secondB))
            elif (self.x_originA, self.y_originA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_secondA, self.y_secondA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_secondB, self.y_secondB))
            elif (self.x_originA, self.y_originA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_secondA, self.y_secondA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_destinationB, self.y_destinationB), (self.x_second_lastB, self.y_second_lastB))
            # no common vertex      
            else: 
                raise AngleError("The lines do not intersect! provide lines wich have a common vertex")

        # deflection on the entire lines
        elif calculation_type == "deflection":
            if (self.x_destinationA, self.y_destinationA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_originA, self.y_originA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_destinationB, self. y_destinationB), (self.x_originB, self.y_originB))
            elif (self.x_destinationA, self.y_destinationA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_originA, self.y_originA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_destinationB, self.y_destinationB))
            elif (self.x_originA, self.y_originA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_destinationA, self.y_destinationA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_destinationB, self.y_destinationB))
            elif (self.x_originA, self.y_originA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_destinationA, self.y_destinationA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_destinationB, self.y_destinationB), (self.x_originB, self.y_originB))
            # no common vertex   
            else: 
                raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
        
        else: # calculation_type == "vectors"
            if (self.x_destinationA, self.y_destinationA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_destinationA, self.y_destinationA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_destinationB, self.y_destinationB), (self.x_originB, self.y_originB))
            elif (self.x_destinationA, self.y_destinationA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_destinationA, self.y_destinationA), (self.x_originA, self.y_originA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_destinationB, self.y_destinationB))
            elif (self.x_originA, self.y_originA) == (self.x_originB, self.y_originB):
                self.lineA = ((self.x_originA, self.y_originA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_originB, self.y_originB), (self.x_destinationB, self.y_destinationB))
            elif (self.x_originA, self.y_originA) == (self.x_destinationB, self.y_destinationB):
                self.lineA = ((self.x_originA, self.y_originA), (self.x_destinationA, self.y_destinationA))
                self.lineB = ((self.x_destinationB, self.y_destinationB),(self.x_originB, self.y_originB)) 
            # no common vertex   
            else:
                raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
   
    def __init__(self, coordsA, coordsB, calculation_type):
        """
        Initializes the class with the coordinates of two line geometries.
        
        Args:
            coordsA (list): A list of coordinates of the first line geometry.
            coordsB (list): A list of coordinates of the second line geometry.
        """
        self.set_coordinates(coordsA, 'A')
        self.set_coordinates(coordsB, 'B')
        self.set_conditions(calculation_type)

def angle_line_geometries(line_geometryA, line_geometryB, degree = False, calculation_type = 'vector'):
    """
    Given two LineStrings it computes the angle between them. Returns value in degrees or radians.
    
    Parameters
    ----------
    line_geometryA: LineString
        the first line
    line_geometryB: LineString
        the other line; it must share a vertex with line_geometryA
    degree: boolean
        if True it returns value in degree, otherwise in radians
    calculation_type: string
        one of: 'vectors', 'angular_change', 'deflection'
        'vectors': computes angle between vectors
        'angular_change': computes angle of incidence between the two lines
        'deflection': computes angle of incidence between the two lines, on the basis of the vertex in common and the second following(intermediate, if existing) vertexes forming each of the line.
    
    Returns:
    ----------
    angle: float
        the resulting angle in radians or degrees
    """
    valid_calculation_types = ['vectors', 'angular_change', 'deflection']
    if not isinstance(line_geometryA, LineString) or not isinstance(line_geometryB, LineString):
        raise TypeError("Both input must be of type shapely.geometry.LineString")
    if calculation_type not in valid_calculation_types:
        raise ValueError(f"Invalid calculation type. Choose one of: {valid_calculation_types}.")  
    # extracting coordinates and createing lines
    coordsA = list(line_geometryA.coords)
    coordsB = list(line_geometryB.coords)
    
    if len(coordsA) < 2 or len(coordsB) < 2:
        raise ValueError("Both LineString must have at least 2 coordinates")
    settings = Settings(coordsA, coordsB, calculation_type)
    
    lineA, lineB = settings.lineA, settings.lineB
        
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

         
class Error(Exception):
    """Base class for other exceptions"""
    
