from points import Pointset, Point
import numpy as np
from visvis import ssdf

def convert_2d_point_to_3d_point(center_point, vec1, vec2, pointset_2d_points):
    
    """
    Given a pointset containing 2D points (y,x), a center point (z,y,x), and two vectors which
    describe the orientation of the slice inside the 3D volume, the algorithm 
    will convert the 2D points to 3D points.
    """
    
    if pointset_2d_points.__class__ == Point:
        temp = Pointset(2)
        temp.Append(pointset_2d_points)
        pointset_2d_points = temp
    
    #Pointset to store 3D points
    pointset_3d_points = Pointset(3)

    for i in range(len(pointset_2d_points)):
        
        #Shift from point 127.5,127.5 which is the rotation point
        y_shift = pointset_2d_points[i][1] - 127.5
        x_shift = pointset_2d_points[i][0] - 127.5
        
        center = Point(center_point[2],center_point[1],center_point[0]) 
        point_3d = center + y_shift*vec1 + x_shift*vec2      
        point_3d = Point(point_3d[2],point_3d[1],point_3d[0]) 
        pointset_3d_points.Append(point_3d)
    
    return pointset_3d_points
    
