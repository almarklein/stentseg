from points import Pointset, Point
import interpolation_
from slice_from_volume import *
import stentPoints2d
from convert_2d_point_to_3d_point import *
from visvis import ssdf

def get_slice(vol, sampling, center_point, global_direction, look_for_bifurcation, alpha):
    
    """
    Given two center points and the direction of the stent, the algorithm will 
    first create a slice on the second center point. A new center point is 
    calculated on this slice, which is used to find a local direction. The 
    local direction is used to update the global direction with weigthing
    factor alpha (where 0 is following the local direction and 1 the global 
    direction). Using this new direction a slice is created on the first center
    point.
    """
    
    vec1 = Point(1,0,0)
    vec2 = Point(0,1,0)
    
    #Set second center point as rotation point
    rotation_point = Point(center_point[2], center_point[1], center_point[0]) + global_direction
    
    #Create the slice using the global direction
    slice, vec1, vec2 = slice_from_volume(vol, sampling, global_direction, rotation_point, vec1, vec2)
    
    #Search for points in this slice and filter these with the clustering method
    points_in_slice = stentPoints2d.detect_points(slice)
    
    try:
        points_in_slice_filtered = stentPoints2d.cluster_points(points_in_slice, \
        Point(128,128))
    except AssertionError:
        points_in_slice_filtered = []
    
    #if filtering did not succeed the unfiltered points are used
    if not points_in_slice_filtered:
        points_in_slice_filtered = points_in_slice
        succeeded = False
    else:
        succeeded = True
    
    #if looking for bifurcation than the radius has to be calculated
    if succeeded and look_for_bifurcation:
        
        center_point_with_radius = stentPoints2d.fit_cirlce( \
        points_in_slice_filtered)
        
        #convert 2d center point to 3d for the event that the bifurcation
        #is found
        center_point_3d_with_radius = convert_2d_point_to_3d_point( \
        Point(rotation_point[2], rotation_point[1], rotation_point[0]), vec1, vec2, center_point_with_radius)
        
        #Convert pointset into point
        center_point_3d_with_radius = Point(center_point_3d_with_radius[0])
        
        #Give the 3d center point the radius as attribute
        center_point_3d_with_radius.r = center_point_with_radius.r
        
    else:
        center_point_3d_with_radius = Point(0,0,0)
        center_point_3d_with_radius.r = []
    
    #Now the local direction of the stent has to be found. Note that it is not
    #done when the chopstick filtering did not succeed.
    if succeeded:
        better_center_point = stentPoints2d.converge_to_centre(points_in_slice_filtered, \
        Point(128,128))
    else:
        better_center_point = Point(0,0), []
    
    #If no better center point has been found, there is no need to update global
    #direction
    if better_center_point[0] == Point(0,0):
        updated_direction = global_direction
    else:
        better_center_point_3d = convert_2d_point_to_3d_point(\
        Point(rotation_point[2], rotation_point[1], rotation_point[0]), \
        vec1, vec2, better_center_point[0])
        
        
        #Change pointset into a point
        better_center_point_3d = Point(better_center_point_3d[0])
        
        #local direction (from starting center point to the better center point)
        local_direction = Point(better_center_point_3d[2]-center_point[2], \
        better_center_point_3d[1]-center_point[1], better_center_point_3d[0] \
        -center_point[0]).Normalize()
        
        #calculate updated direction
        updated_direction = ((1-alpha)*local_direction + alpha*global_direction).Normalize()

    #Now a new slice can be created using the first center point as rotation point.
    rotation_point = Point(center_point[2], center_point[1], center_point[0])
    
    slice, vec1, vec2 = slice_from_volume(vol, sampling, updated_direction, rotation_point, vec1, vec2)
    
    #Change order
    rotation_point = Point(center_point[0], center_point[1], center_point[2])
    
    #return new center point
    new_center_point = center_point + Point(updated_direction[2], \
    updated_direction[1], updated_direction[0])

    return slice, new_center_point, updated_direction, rotation_point, vec1, vec2, \
    center_point_3d_with_radius
    