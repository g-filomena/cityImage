import math
from math import sqrt
from shapely.geometry import Point, LineString, MultiLineString

    
"""
math functions for angle computations
readapted for LineStrings from Abhinav Ramakrishnan answer in https://stackoverflow.com/a/28261304/7375309
"""
	
def euclidean_distance(xs, ys, xt, yt):
    """ xs stands for x source and xt for x target """
    return sqrt((xs - xt)**2 + (ys - yt)**2)

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
		
def get_coord_angle(origin, distance, angle):

    (disp_x, disp_y) = (distance * math.sin(math.radians(angle)), distance * math.cos(math.radians(angle)))
    return (origin[0] + disp_x, origin[1] + disp_y)

def angle_line_geometries(line_geometryA, line_geometryB, degree = False, deflection = False, angular_change = False):
    
    """
    Given two LineStrings it computes the deflection angle between them. Returns value in degrees or radians.
    
    ----------
    Parameters
    line_geometryA: LineString
	line_geometryB: LineString
    degree: boolean
		If True returns value in degree, otherwise in radians
    deflection: boolean
		If True computes angle of incidence between the two lines, otherwise angle between vectors
    angular_change: boolean

    Returns:
    ----------
    float
    """
	
    if angular_change: deflection = True
        
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
        if ((x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_second_lastA, y_second_lastA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB), (x_second_lastB, y_second_lastB))

        elif ((x_destinationA, y_destinationA) == (x_originB, y_originB)):
            lineA = ((x_second_lastA, y_second_lastA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_originB, y_originB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_originB, y_originB), (x_secondB, y_secondB))

        elif ((x_originA, y_originA) == (x_destinationB, y_destinationB)):
            lineA = ((x_secondA, y_secondA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_second_lastB, y_second_lastB))
        # no common vertex      
        else:  raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
    
	# deflection on the entire lines
    elif deflection & not angular_change:
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
        if (x_destinationA, y_destinationA) == (x_destinationB, y_destinationB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((x_destinationB, y_destinationB), (x_originB, y_originB))

        elif (x_destinationA, y_destinationA) == (y_originB, y_originB)):
            lineA = ((x_destinationA, y_destinationA), (x_originA, y_originA))
            lineB = ((y_originB, y_originB), (x_destinationB, y_destinationB))

        elif (x_originA, y_originA) == (y_originB, y_originB):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_originB, y_originB), (x_destinationB, y_destinationB))

        else: (x_originA, y_originA) == (y_destinationB, y_destinationB)):
            lineA = ((x_originA, y_originA), (x_destinationA, y_destinationA))
            lineB = ((x_destinationB, y_destinationB),(x_originB, y_originB)) 
		# no common vertex   
		else: raise AngleError("The lines do not intersect! provide lines wich have a common vertex")
    
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
        angle_deg = 0.0
        angle_rad = 0.0
        
    if degree: return angle_deg
    else: return angle_rad
    
class Error(Exception):
	"""Base class for other exceptions"""
	pass
class AngleError(Error):
	"""Raised when not-intersecting lines are provided for computing angles"""
	pass
    
    
            
            
            
            
            
            


    
