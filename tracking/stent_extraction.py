from visvis import ssdf
import numpy as np
import visvis as vv
import stentPoints2d
import centerline_tracking
from points import Aarray, Point, Pointset
from slice_from_volume import *
from convert_2d_point_to_3d_point import *
from graph import Graph

def load_volume(patnr):
    
    """
    Gives back volume (pixels) and volume properties (sampling and origin)
    """
    
#     tmp = r'C:\Users\Michel\Documents\BMT\Bacheloropdracht\data\croppedAvg7_pat%02i.ssdf'
    tmp = r'C:\almar\data\dicom\cropped\croppedReg_pat%02i_gravity.bsdf'
    s = ssdf.load(tmp%patnr)
    vol = Aarray(s.vol.astype(np.float32), s.sampling, s.origin)

    #Save sampling
    sampling = s.sampling
    origin = s.origin

    return vol, sampling, origin

def from_nodes_to_graph(node_points_stent):
    
    """
    Given a set of nodes it creates a graph. The node set contains a ring number 
    (4th dimension) and direction (5th dimension)
    """
    
    graph = Graph()
    
    #Min and maximum distance when connecting nodes
    min = 10
    max = 18
    
    for i in range(len(node_points_stent)):
        
        #Take point
        point = Point(node_points_stent[i][2],node_points_stent[i][1],node_points_stent[i][0])
        
        #Add node to graph
        graph.AppendNode(point)
        
        #Pointset to store possible matches
        possible_match = Pointset(3)
        
        for j in range(len(node_points_stent)):
            if node_points_stent[j][3] == node_points_stent[i][3] and node_points_stent[j][4] != node_points_stent[i][4]:
                    if min < point.Distance(Point(node_points_stent[j][2],node_points_stent[j][1],node_points_stent[j][0])) < max:
                        possible_match.Append(Point(node_points_stent[j][2],node_points_stent[j][1],node_points_stent[j][0]))
        
        if len(possible_match) == 0:
            continue
        elif len(possible_match) == 1:
            graph.AppendNode(possible_match[0])
            graph.CreateEdge(graph[np.size(graph)-2], graph[np.size(graph)-1])
        elif len(possible_match) == 2:
            graph.AppendNode(possible_match[0]); graph.AppendNode(possible_match[1])
            graph.CreateEdge(graph[np.size(graph)-1], graph[np.size(graph)-3]); graph.CreateEdge(graph[np.size(graph)-2], graph[np.size(graph)-3])
        else:
            while True: 
                sorted = False
                for j in range(len(possible_match)-1):
                    if point.Distance(possible_match[j]) > point.Distance(possible_match[j+1]):
                        possible_match[j], possible_match[j+1] = possible_match[j+1], possible_match[j]
                        sorted = True
                if not sorted:
                    break
            graph.AppendNode(possible_match[0]); graph.AppendNode(possible_match[1])
            graph.CreateEdge(graph[np.size(graph)-1], graph[np.size(graph)-3]); graph.CreateEdge(graph[np.size(graph)-2], graph[np.size(graph)-3])

    return graph
    
def initial_center_point(vol, seed_points_i, seed_points_i_plus_1, sampling, origin, alpha):
    
    """
    Given two seed points, the algorithm will create the first center point
    and the global direction
    """
    
    #Set global direction
    global_direction = seed_points_i_plus_1 - seed_points_i
    global_direction = Point(global_direction[2], global_direction[1], \
    global_direction[0]).Normalize()
    
    #Create slice
    slice, unused, unused, unused, vec1, vec2, unused = \
    centerline_tracking.get_slice(vol, sampling, seed_points_i, global_direction, False, alpha)
    
    #Search points and filter
    point_in_slice = stentPoints2d.detect_points(slice)
    try:
        filtered = stentPoints2d.cluster_points(point_in_slice, Point(128,128))
    except AssertionError:
        filtered = []

    #Calculate new center and transform to 3d
    if filtered:
        center_point_2d = stentPoints2d.fit_cirlce(filtered)    
        better_center_3d = convert_2d_point_to_3d_point(seed_points_i, vec1, vec2, center_point_2d)
        #If not too far, use better center.
        if better_center_3d.Distance(seed_points_i) > 10:
            center_point = better_center_3d
        else:
            center_point = seed_points_i  
    else: 
        center_point = seed_points_i
   
    return center_point, global_direction
    
def find_connecting_points(pointset_slice_i, pointset_slice_i_plus_1):
    
    """
    Given a set of points in slice i, the algorithm will calculate distances 
    between these points and the points in slice i+1. If this distance is less 
    than a certain distance, the point in slice i+1 is returned as connecting
    point. If there is no connecting point found, a virtual point is created.
    """
    
    #Maximum distance for which a point is seen as connecting
    max_distance = ssdf._stent_tracking_params.max_dist
    
    #Pointsets to store points
    connecting_points = Pointset(2)
    virtual_points = Pointset(2)
   
    #for each point in slice i
    for i in range(len(pointset_slice_i)):
        
        distances = pointset_slice_i[i,:] - pointset_slice_i_plus_1[:,:]

        #for all distances from this point
        for j in range(len(distances)):
            
            #if distance is smaller or equal to the max distance
            if abs(distances[j,0]) + abs(distances[j,1]) <= max_distance:
            
                #make the point a connecting point
                connecting_points.Append(pointset_slice_i_plus_1[j])
              
        #if the point from slice i has no connecting point
        if len(np.where(abs(distances[:,0]) + abs(distances[:,1]) <= max_distance)[0]) == 0:
            
            #change the point from slice i into a virtual point
            virtual_points.Append(pointset_slice_i[i])
      
    return connecting_points, virtual_points

def check_virtual_points(virtual_points, connecting_points, center_point, vol, sampling, global_direction, alpha):
   
    """
    Given a set of virtual points, the algorithm will check in the next two 
    slices for connecting points. If there is a connecting point, the virtual 
    point will change the virtual point into a connecting point. Otherwise the 
    virtual point will be removed.
    """

    look_for_bifurcation = False
    
    #Create two temporary slices
    slice_i_plus_1, center_point_i_plus_1, global_direction_i_plus_1, unused, unused, \
    unused, unused = centerline_tracking.get_slice(vol, sampling, center_point,global_direction,look_for_bifurcation, alpha)
    slice_i_plus_2, unused, unused, unused, unused, unused, unused = \
    centerline_tracking.get_slice(vol, sampling, center_point_i_plus_1, global_direction_i_plus_1, look_for_bifurcation, alpha)
   
    #Find points in this slice
    pointset_slice_i_plus_1 = stentPoints2d.detect_points(slice_i_plus_1)
    pointset_slice_i_plus_2 = stentPoints2d.detect_points(slice_i_plus_2)
    
    #Check whether the virtual points return connecting points.
    new_connecting_points_1, still_virtual_points = find_connecting_points( \
    virtual_points, pointset_slice_i_plus_1)
    new_connecting_points_2, removed_virtual_points = find_connecting_points( \
    still_virtual_points, pointset_slice_i_plus_2)
    
    #Add the new connecting points to the existing list of connection points
    connecting_points.Extend(new_connecting_points_1)
    connecting_points.Extend(new_connecting_points_2)
    
    return connecting_points, slice_i_plus_1

def check_for_nodes(connecting_points, slice_i_plus_1):
    
    """
    Given a set of stent points, the algorithm checks whether there are any 
    duplicates. A duplicate would mean that that point was found by two stent 
    points and could therefore be a node. An additional check is done to check 
    whether there are no connecting points above the potential node. If this is 
    the case, than the point is not a node.
    """
    
    #Pointset to store 2D nodes
    point_nodes = Pointset(2)
    
    #Pointset to store point which need to be removed
    point_remove = Pointset(2)
    
    #points in slice_i_plus_1
    points_check = stentPoints2d.detect_points(slice_i_plus_1)
    
    #For every connecting point
    for i in range(len(connecting_points)):
        for j in range(len(connecting_points)):
            
            #if points are the same
            if connecting_points[i].Distance(connecting_points[j]) <= 3 and i != j:
            
                #check if this point would find a connecting point
                point = Pointset(2); point.Append(connecting_points[i])
                node_check, unused = find_connecting_points(point, points_check)
                
                #Only add node once. 
                if not node_check:
                    if not point_nodes:
                        point_nodes.Append((connecting_points[i]+connecting_points[j])/2)
                    elif point_nodes.Contains(connecting_points[i]) == False:
                        point_nodes.Append((connecting_points[i]+connecting_points[j])/2)
                    
                    #The connecting points need to be removed later
                    point_remove.append(connecting_points[i]); point_remove.append(connecting_points[i])
    
    return point_nodes, point_remove
    
def delete_nodes_from_list(connecting_points, point_remove):
    
    """
    Deletes the nodes from the connecting_points list to avoid creation of 
    virtual points.
    """
    
    for i in range(len(point_remove)):
        try: 
            connecting_points.RemoveAll(point_remove[i])
        except ValueError:
            continue

    return connecting_points

def additional_nodes(connecting_points):

    #Create pointset
    extra_nodes = Pointset(3)

    for i in range(len(connecting_points)):
        for j in range(len(connecting_points)):
            if 0 < connecting_points[i].Distance(connecting_points[j]) < 4:
                #Merge points
                extra_nodes.Append((connecting_points[i]+connecting_points[j])/2)
        
    for i in range(len(connecting_points)):
        
        #Create copy
        connecting_points_copy = connecting_points.Copy()
        
        #Remove all from copy to avoid distance = 0.
        connecting_points_copy.RemoveAll(connecting_points[i])
        
        if np.sum((connecting_points[i].Distance(connecting_points_copy))<4)==0:
            extra_nodes.Append(connecting_points[i])
    
    new_nodes = extra_nodes
    
    return new_nodes

def amount_of_nodes(vol, sampling, center_point, global_direction, look_for_bifurcation):
    
    """
    Given a center point and direction the algorithm will create a slice and 
    filter the found points. Based on the filtered points, the amount of 
    expected nodes are calculated    
    """
    
    look_for_bifurcation = False
    alpha = 1
    
    #create the slice
    slice, unused, unused, unused, unused, unused, unused = \
    centerline_tracking.get_slice(vol, sampling, center_point, global_direction, \
    look_for_bifurcation, alpha)

    #detect points
    points_in_slice = stentPoints2d.detect_points(slice)

    #filter points
    try:
        filtered_points = stentPoints2d.cluster_points(points_in_slice, Point(128,128))
    except AssertionError:
        filtered_points = points_in_slice 
    
    #if filtering was succesful
    if filtered_points:
        expected_amount_of_nodes = len(filtered_points)/2
    else:
        expected_amount_of_nodes = len(points_in_slice)/2

    return expected_amount_of_nodes, filtered_points

def check_for_possible_bifurcation(average_radius, center_point_3d_with_radius,\
    rotation_point ,found_bifurcation, global_direction):
    
    """
    This function tracks the radius of the stent. If the radius drops to less
    than 75%, the bifurcation might have been found. If so, two new starting points will be 
    calculated.
    """
    
    #Initially no new starting positions.
    new_starting_positions = []
    
    #Easier notation in rotation matrix
    gd = global_direction
    
    #To avoid wrong predictions on an early error in radius
    if len(average_radius) > 10:
        if 0.75*np.mean(average_radius) > center_point_3d_with_radius.r \
        and not found_bifurcation:
            
            if center_point_3d_with_radius.Distance(rotation_point)>0.25*np.mean(average_radius):
                
                #found bifurcation
                found_bifurcation = True
                
                #calculate new starting points.
                new_starting_positions = Pointset(3)
                
                #vector of first branch
                fb_vector = np.mat([[center_point_3d_with_radius[2] - rotation_point[2]],
                [center_point_3d_with_radius[1] - rotation_point[1]],
                [center_point_3d_with_radius[0] - rotation_point[0]]])
                
                #Create rotation matrix to rotate vector
                rotation_matrix = np.mat([[2*gd[0]**2-1, 2*gd[0]*gd[1], 2*gd[0]*gd[2]],
                [2*gd[1]*gd[2], 2*gd[1]**2-1, 2*gd[1]*gd[2]],
                [2*gd[2]*gd[0], 2*gd[1]*gd[2], 2*gd[2]**2-1]])
                
                #second branch
                sb = rotation_matrix * fb_vector
                
                sb_center_point = rotation_point + Point(sb[2], sb[1], sb[0]) 
                
                #Return new starting positions
                new_starting_positions.Append(center_point_3d_with_radius)
                new_starting_positions.Append(sb_center_point)

    #Add radius to list with radia
    if not center_point_3d_with_radius.r == []:
        average_radius.append(center_point_3d_with_radius.r)
        
    return average_radius, found_bifurcation, new_starting_positions

def connecting_stent_points(vol, sampling, center_point, global_direction, look_for_bifurcation, \
    average_radius, direction, expected_amount_of_nodes, points_in_slice, found_bifurcation, alpha, ring):
    
    """
    This algorithm uses the earlier stated functions in order to connect points
    and define nodes. 
    """

    #Nodes that are found for the current direction
    node_points_direction = Pointset(3)
    
    #Other pointsets to store points
    center_points = Pointset(3)
    stent_points = Pointset(3)
    node_points = Pointset(5)

    #Set round to 0 for the case that the amount of nodes criterion is not met.
    round = 0
    
    #Initial
    new_starting_positions = []
    
    #connecting points
    while len(node_points_direction)<expected_amount_of_nodes and round < 12:
        
        #Get slice
        slice, center_point, global_direction, rotation_point, vec1, vec2, \
        center_point_3d_with_radius = centerline_tracking.get_slice(vol, sampling, \
        center_point, global_direction, look_for_bifurcation, alpha)  
        
        #If looking for the bifurcation, check radius
        if look_for_bifurcation and direction == 1:
            if not found_bifurcation:
                average_radius, found_bifurcation, new_starting_positions \
                = check_for_possible_bifurcation(average_radius, center_point_3d_with_radius, \
                rotation_point, found_bifurcation, global_direction)
        else:
            new_starting_positions = []
        
        #Find points
        points_in_next_slice = stentPoints2d.detect_points(slice)
     
        #Find connecting points. Change not connecting points into virtual points
        connecting_points, virtual_points = find_connecting_points(points_in_slice, \
        points_in_next_slice)
        
        #Check if virtual points should be converted into connecting points
        connecting_points, temp_slice = check_virtual_points(virtual_points, \
        connecting_points, center_point, vol, sampling,  global_direction, alpha) 
      
        #Check for nodes
        point_nodes_2d, point_remove = check_for_nodes(connecting_points, temp_slice)
        
        #If nodes are found, these points should be deleted from the connecting points list
        connecting_points = delete_nodes_from_list(connecting_points, point_remove)
        
        #Convert 2d points into 3d points
        connecting_points_3d = convert_2d_point_to_3d_point(center_point, vec1,\
        vec2, connecting_points)
        point_nodes_3d = convert_2d_point_to_3d_point(center_point, vec1, vec2,\
        point_nodes_2d)
        
        #Add 3d points to their list
        center_points.Append(rotation_point)
        stent_points.Extend(connecting_points_3d)
        node_points_direction.Extend(point_nodes_3d)
        for i in range(len(point_nodes_3d)):
            node_points.Append(Point(point_nodes_3d[i][0], point_nodes_3d[i][1], point_nodes_3d[i][2], ring, direction))
        
        #Set connecting points as new base points
        points_in_slice = connecting_points       
        
        #Next round
        round=round+1
        
        #If there are three nodes found, do first one more run.
        if len(node_points_direction)>2:
            diff = 10 - round
            if diff > 0:
                round = round + diff
                
        if round == 12:
            new_nodes = additional_nodes(connecting_points_3d)
            for i in range(len(new_nodes)):
                node_points.Append(Point(new_nodes[i][0], new_nodes[i][1], new_nodes[i][2], ring, direction))
        
    return center_points, stent_points, node_points, global_direction, found_bifurcation, new_starting_positions, ring

def searching_for_ring(vol, sampling, center_point, global_direction, \
    look_for_bifurcation, average_radius, alpha, ring):
    
    """
    Given a center point and a direction, the algorithm will start by searching
    for points in a certain stent ring using the global direction. By connecting 
    points it will come to a set of nodes. After a certain amount of nodes are 
    found, the algorithm will flip the direction and will search for points in 
    the other direction until the stent ring is defined. The radius is used when
    searching for the bifurcation.        
    """
    
    #Initially, the bifurcation is not found
    found_bifurcation = False
    
    #First the initial center point and direction has to be stored in order to
    #flip the direction later
    stored_center_point = center_point
    flipped_global_direction = -1*global_direction
    
    #Create pointsets to store points
    node_points_ring = Pointset(5) #(z,y,x,ring,direction)
    stent_points_ring = Pointset(3)
    center_points_ring = Pointset(3)
    
    #Expected amount of nodes and returns initial filtered points
    expected_amount_of_nodes, initial_points_in_slice = amount_of_nodes(vol, sampling, \
    center_point, global_direction, look_for_bifurcation)
    
    #set direction: 1 for along the direction of the stent, -1 for the opposite \
    #direction
    direction = 1
    
    #Find and connect points
    center_points, stent_points, node_points, global_direction, found_bifurcation, new_starting_positions, \
    ring = connecting_stent_points(vol, sampling, center_point, global_direction, look_for_bifurcation, \
    average_radius, direction, expected_amount_of_nodes, initial_points_in_slice, found_bifurcation, alpha, ring)
    
    #Store starting positions, because the direction -1 will remove otherwise.
    if found_bifurcation:
        store_starting_positions = new_starting_positions
    
    #Add found points to pointsets
    node_points_ring.Extend(node_points)
    stent_points_ring.Extend(stent_points)
    center_points_ring.Extend(center_points)
    
    #Store last center point to use as 'jumping' site
    if len(center_points_ring):
        jump_location = center_points_ring[len(center_points_ring)-1]
    else:
        jump_location = stored_center_point
    jump_direction = global_direction

    #Flip direction
    direction = -1

    #Find and connect points for other direction
    center_points, stent_points, node_points, global_direction, found_bifurcation, new_starting_positions, \
    ring = connecting_stent_points(vol, sampling, stored_center_point, flipped_global_direction, \
    look_for_bifurcation, average_radius, direction, expected_amount_of_nodes, \
    initial_points_in_slice, found_bifurcation, alpha, ring)   
    
    if found_bifurcation:
        new_starting_positions = store_starting_positions
    
    #Add found points to pointsets
    node_points_ring.Extend(node_points)
    stent_points_ring.Extend(stent_points)
    center_points_ring.Extend(center_points)
    
    ring = ring + 1
    
    return node_points_ring, stent_points_ring, center_points_ring, \
    jump_location, jump_direction, found_bifurcation, new_starting_positions, ring
    
def stent_detection(vol, sampling, origin, seed_points):
    
    """
    This is the function that will be called in order to extract the stent
    from the CT data. 
    """

    #Point set where the final points will be stored. In node, the 4th number
    #indicates the ring number, the 5th indicates the direction where it is found
    node_points_stent = Pointset(5)
    center_points_stent = Pointset(3)
    stent_points_stent = Pointset(3)

    #Initially, the bifurcation is not found
    found_bifurcation = False
    average_radius = []
    
    #start with ring 1
    ring = 1
    
    #Weighting factor alpha
    alpha = ssdf._stent_tracking_params.weighting_factor
    distance = ssdf._stent_tracking_params.distance
    
    #From seed point to seed point except for the last seed point, where the 
    #bifurcation needs to be found.
    for i in range(len(seed_points)-2):
        
        round = 0
        
        #Don't look for the bifurcation when not connecting points in the last 
        #part of the stent
        look_for_bifurcation = False
        
        #Get initial center point and direction using the seed points
        center_point, global_direction = initial_center_point(vol, seed_points[i],
        seed_points[i+1], sampling, origin, alpha) 
        
        #While the center point has not passed the seed point
        while center_point[0] < seed_points[i+1][0] and round < 10:
            
            #Find stent, node and center points for a certain stent ring
            node_points_ring, stent_points_ring, center_points_ring, \
            jump_location, jump_direction, found_bifurcation, new_starting_positions, \
            ring = searching_for_ring(vol, sampling, center_point, global_direction,\
            look_for_bifurcation, average_radius, alpha, ring)
            
            #Set new starting center point inside the new ring
            center_point = jump_location + distance*Point(jump_direction[2],jump_direction[1],jump_direction[0])
            global_direction = jump_direction
            
            #Add points to point sets
            node_points_stent.Extend(node_points_ring)
            center_points_stent.Extend(center_points_ring)
            stent_points_stent.Extend(stent_points_ring)
            
            round = round + 1

    #For the last part of the stent where the bifurcation needs to be found
    for i in range(len(seed_points)-2,len(seed_points)-1):
        
        #Reset round to zero
        round = 0
        
        look_for_bifurcation = True
        
        center_point, global_direction = initial_center_point(vol, seed_points[i],\
        seed_points[i+1], sampling, origin, alpha)
        
        #While the bifurcation is not found
        while found_bifurcation == False and round < 10:
            
            node_points_ring, stent_points_ring, center_points_ring, \
            jump_location, jump_direction, found_bifurcation, new_starting_positions, \
            ring = searching_for_ring(vol, sampling, center_point, global_direction, \
            look_for_bifurcation, average_radius, alpha, ring)
            
            center_point = jump_location + distance * Point(jump_direction[2],jump_direction[1],jump_direction[0])
            global_direction = jump_direction
            
            
            #Add points to point sets    
            node_points_stent.Extend(node_points_ring)
            center_points_stent.Extend(center_points_ring)
            stent_points_stent.Extend(stent_points_ring)
            
            round = round + 1
    
    #Store new starting positions in another poitset, because algorithm will
    #return [] as new_starting_positions.
    starting_positions = Pointset(3)
    
    #For the case where there is no bifurcation present
    try:
        starting_positions.Append(new_starting_positions[0])# + distance * Point(jump_direction[2],jump_direction[1],jump_direction[0]))
        starting_positions.Append(new_starting_positions[1])# + distance * Point(jump_direction[2],jump_direction[1],jump_direction[0]))
    except IndexError:
        []
    
    alpha = ssdf._stent_tracking_params.weighting_factor_bif
    
    #Now points after the bifurcation will be found
    for i in range(len(starting_positions)):
        
        center_point = starting_positions[i]
        
        look_for_bifurcation = False
        
        round = 0
        
        while round <4:
            
            node_points_ring, stent_points_ring, center_points_ring, \
            jump_location, jump_direction, found_bifurcation, new_starting_positions, \
            ring = searching_for_ring(vol, sampling, center_point, global_direction, \
            look_for_bifurcation, average_radius, alpha, ring)
            
            center_point = jump_location + distance * Point(jump_direction[2],jump_direction[1],jump_direction[0])
            global_direction = jump_direction
            
            if 255<center_point[0]:
                break
            
            #Add points to point sets    
            node_points_stent.Extend(node_points_ring)
            center_points_stent.Extend(center_points_ring)
            stent_points_stent.Extend(stent_points_ring)
            
            round = round + 1

    return node_points_stent, center_points_stent, stent_points_stent

def getDefaultParams():
    
    #Default params
    params = ssdf.new()
    params.max_dist = 2
    params.weighting_factor = 0.99
    params.weighting_factor_bif = 0.95
    params.distance = 4
    return params
    
ssdf._stent_tracking_params = getDefaultParams()

def select_stent(volnr, params=None):
    
    """
    This functions calls the stent extration function. It contains seed points
    for each dataset
    """
    
    if params is None:
        params = getDefaultParams()
    ssdf._stent_tracking_params = params
    
    if volnr == 1:
        #vol 1
        vol, sampling, origin = load_volume(1)
        seed_points = Pointset(3)
        seed_points.Append(60,110,133)
        seed_points.Append(100,75,150)
        seed_points.Append(160,70,148)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 2:
        #vol 2
        vol, sampling, origin = load_volume(2)
        seed_points = Pointset(3)
        seed_points.Append(60,175,155)
        seed_points.Append(100,150,170)
        seed_points.Append(165,140,167)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 3:
        #vol 3
        vol, sampling, origin = load_volume(3)
        seed_points = Pointset(3)
        seed_points.Append(47,160,140)
        seed_points.Append(80,145,130)
        seed_points.Append(130,130,128)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 4:
        #vol 4
        vol, sampling, origin = load_volume(4)
        seed_points = Pointset(3)
        seed_points.Append(40,160,160)
        seed_points.Append(140,145,135)
        seed_points.Append(150,150,150)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 5:
        #vol 5
        vol, sampling, origin = load_volume(5)
        seed_points = Pointset(3)
        seed_points.Append(34,150,170)
        seed_points.Append(60,135,180)
        seed_points.Append(130,140,180)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 6:
        #vol 6
        vol, sampling, origin = load_volume(6)
        seed_points = Pointset(3)
        seed_points.Append(40,170,175)
        seed_points.Append(70,160,180)
        seed_points.Append(140,142,168)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 7:
        #vol 7
        vol, sampling, origin = load_volume(7)
        seed_points = Pointset(3)
        seed_points.Append(40,180,160)
        seed_points.Append(110,160,160)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    elif volnr == 8:
        #vol 8
        vol, sampling, origin = load_volume(8)
        seed_points = Pointset(3)
        seed_points.Append(60,130,190)
        seed_points.Append(80,110,180)
        seed_points.Append(120,112,140)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
        
    elif volnr == 18:
        #vol 18
        vol, sampling, origin = load_volume(18)
        seed_points = Pointset(3)
        seed_points.Append(70,170,140)
        seed_points.Append(135,150,160)
        node_points_stent, center_points_stent, stent_points_stent = stent_detection(vol, sampling, origin, seed_points)
    else:
        raise ValueError('Invalid volnr')

    plot = True
    if plot:
        points = Pointset(3)
        for i in range(len(node_points_stent)):
            points.Append(node_points_stent[i][2],node_points_stent[i][1],node_points_stent[i][0])
        
        points_2 = Pointset(3)
        for i in range(len(stent_points_stent)):
            points_2.Append(stent_points_stent[i][2],stent_points_stent[i][1],stent_points_stent[i][0]) 
        
        vv.closeAll()
        vv.volshow(vol)
        vv.plot(points_2,ls='',ms='.',mc='g',mw=4 ,alpha=0.5)
        vv.plot(points,ls='',ms='.',mc='b',mw=4 , alpha=0.9)
 
    graph = from_nodes_to_graph(node_points_stent)

    return graph

if __name__ == '__main__':
    stentnr = 2
    graph = select_stent(stentnr)