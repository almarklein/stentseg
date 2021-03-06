# Copyright 2015-2016 A. Klein, M. Koenrades

"""
Functionality to extract centerlines based on a point cloud that for example
represents the vessel wall. Works in 2D and 3D.

The public API consists of points_from_mesh() and find_centerline() and 
smooth_centerline().
"""

import numpy as np
import visvis as vv

from stentseg.utils import PointSet


def dist_over_centerline(cl, cl_point1=None, cl_point2=None, type='euclidian'):
    """ Calculate distance over centerline from centerline p1 to centerline p2,
    or entire centerline if points are not given.
    Euclidian distance or Z-distance.
    """
    import numpy as np
    
    if isinstance(cl, PointSet):
        cl = np.asarray(cl)
    if not cl_point1 is None:
        if isinstance(cl_point1, PointSet):
            cl_point1 = np.asarray(cl_point1)
            cl_point2 = np.asarray(cl_point2)
        indpoint1 = int(np.where( np.all(cl == cl_point1, axis=-1) )[0]) # -1 counts from last to the first axis
        indpoint2 = int(np.where( np.all(cl == cl_point2, axis=-1) )[0]) 
        if indpoint1 == indpoint2:
            print('Points on centerline are at same level, distance is zero')
            dist = 0
            return dist
        clpart = cl[ min(indpoint1, indpoint2):max(indpoint1, indpoint2)+1]
    else:
        clpart = cl
    vectors = np.vstack([clpart[i]-clpart[i+1] for i in range(len(clpart)-1)])
    if type == 'euclidian':
        d = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5  # 3Dvector length in mm
    elif type == 'z':
        d = abs(vectors[:,2])  # x,y,z ; 1Dvector length in mm
    dist = d.sum()
    
    return dist


def points_from_nodes_in_graph(graph):
    """
    """
    # Create set of tuples to remove duplicates
    pp = set(tuple(p) for p in graph.nodes())
    # Turn into a pointset
    return PointSet(np.array(list(pp)))


def points_from_mesh(mesh, invertZ = True):
    """ Create a point cloud (represented as a PointSet) from a visvis mesh
    object, or from a filename pointing to a .stl or .obj file.
    """
    if isinstance(mesh, str):
        mesh = vv.meshRead(mesh)
    if invertZ == True:
        for vertice in mesh._vertices:
            vertice[-1] = vertice[-1]*-1
    # Create set of tuples to remove duplicates
    pp = set(tuple(p) for p in mesh._vertices)
    # Turn into a pointset
    return PointSet(np.array(list(pp)))


def smooth_centerline(pp, n=2):
    """ Smooth a series of points by applying a size-3 averaging
    windows, n times.
    """
    for iter in range(n):
        pp2 = pp.copy()
        for i in range(1, pp.shape[0]-1):
            pp2[i] = (pp[i-1] + pp[i] + pp[i+1]) / 3
        pp = pp2
    return pp


def pp_to_graph(pp, type='oneEdge'):
    """ PointSet to graph with points connected with edges or as one edge.
    Returns graph. Can be used for centerline output.
    """
    from stentseg.stentdirect import stentgraph
    graph = stentgraph.StentGraph()
    if type=='oneEdge':
        # add single nodes
        for p in pp:
            p_as_tuple = tuple(p.flat)
            graph.add_node(p_as_tuple)
        # add one path of pp
        pstart = p_as_tuple
        pend = tuple(pp[-1].flat)
        graph.add_edge(pstart, pend, path = pp  )
    else:
        for i, p in enumerate(pp[:-1]):
            n1 = tuple(p.flat)
            n2 = tuple(pp[i+1].flat)
            # create path between nodes as PointSet
            path = PointSet(3, dtype=np.float32)
            for p in [n1, n2]:
                path.append(p)
            graph.add_edge(n1,n2, path= path)
            
    return graph


def find_centerline(pp, start, ends, step, *,
                substep=None, ndist=20, regfactor=0.2, regsteps=10,
                verbose=False):
    """ Find path from a start point to an end-point or set of end-points.
    Starting from start, will try to find a route to the closest
    of the given end-points, while maximizing the distance to the points
    in pp. Works on 2D and 3D points.
    
    Arguments:
        pp (Pointset): points to stay away from. Typically points on the
            structure that we are trying to find a centerline for.
        start (tuple, PointSet): Point or tuple indicating the start position.
        ends (tuple, list, Pointset): Point or points indicating end positions.
        step (float): the stepsize of points along the centerline.
        substep (float, optional): the stepsize for the candiate
            positions. Default is 0.2 of step.
        ndist (int): the number of closest points to take into account in
            the distance measurements. Default 20.
        regfactor (float): regularization factor between 0 and 1. Default 0.2.
        regsteps (float): Distance in number of steps from start/end at which
            there is no additional "endpoint" regularization. Default 10.
        verbose (bool): if True, print info.
    Returns a Pointset with points that are each step distance from
    each-other (except for the last step). The first point matches the
    start position and the last point matches one of the end positions.
    """
    
    substep = substep or step * 0.2
    
    # Check pp and ndim
    if not isinstance(pp, PointSet):
        raise ValueError('pp must be a pointset')
    dims = pp.shape[1]
    if dims not in (2, 3):
        raise ValueError('find_centerline() only works on 2D or 3D data')
    
    # Prepare start
    start = np.array(start)
    start.shape = -1, dims
    start = PointSet(start)
    assert start.shape[0] == 1
    assert start.shape[1] == dims
    
    # Prepare ends
    ends = np.array(ends)
    ends.shape = -1, dims
    ends = PointSet(ends)
    assert ends.shape[1] == dims
    
    # Init
    centerline = PointSet(dims)
    pos = start
    
    # Prepare for first iter
    centerline.append(pos)
    end = _closest_point(ends, pos)
    vec = (end - pos).normalize() * step

    n_estimate = (end-pos).norm() / step
    i = 0
    while pos.distance(end) > step:
        i += 1
        measure = _distance_measure(pp, pos, ndist)
        
        #if i > n_estimate * 2 or measure > 250: # * 4:
        if i > n_estimate * 2 or measure > 60: # return sooner
            print('i={} and n_estimate={}; measure={}'.format(i,n_estimate,measure))
            print('We seem to be lost. Centerline is returned')
            return centerline
            # raise RuntimeError('We seem to be lost')
        
        if verbose:
            print('iter %i, distance %1.1f of %1.1f' % (i, pos.distance(end), start.distance(end)))
        
        # Estimated position and refine using distance measure
        # Step towards end-pos, but refine by looking orthogonal to real direction
        pos1 = pos + vec
        pos2 = _find_better_pos(pp, pos1, vec, substep, ndist)
        
        # Calculate damp factor to direct regularization towards start-end direction
        reg_damper = 1.0 - float(min(pos.distance(end), pos1.distance(start))) / (regsteps*step)
        reg_damper = min(1.0, max(reg_damper, 0))
        
        # Combine original estimate with distance-based position -> regularization
        reg = min(1, regfactor + reg_damper)
        refpos = reg * pos1 + (1-reg) * pos2
        
        # Rescale so we take equally sized steps
        vec_real = (refpos - pos).normalize()
        pos = pos + vec_real * step
        
        # Store and prepare for next step
        centerline.append(pos)
        end = _closest_point(ends, pos)
        vec = ( (1-reg_damper)*vec_real + (end - pos).normalize() ).normalize() * step
    
    # Wrap up
    centerline.append(end)
    return centerline


def _distance_measure(pp, pos, n):
    """ Calculate a distance measure for the n closest points in pp to
    pos. Higher numbers relate to higher distances.
    """
    distances = pp.distance(pos)
    # Coarsely reduce number of distances (sorting is expensive)
    distances2 = distances
    while distances2.size >= n:
        distances = distances2
        distances2 = distances[distances<distances.mean()]
    # Select n distances and return weighted sum
    distances.sort()
    return (distances[:n] * np.linspace(1, 0, n)).sum()
    #return distances[:n].sum()


def _closest_point(points, pos):
    """ Select the point in points that is closest to pos.
    """
    distances = points.distance(pos)
    i = np.argmin(distances)
    return points[int(i)]  # index must be Python int to get a Pointset


def _find_better_pos(pp, *args):
    """ Given a pos, find a position orthogonal to the given vector
    that has a higher distance to the n closest points in pp.
    """
    if pp.shape[1] == 2:
        return _find_better_pos2(pp, *args)
    else:
        return _find_better_pos3(pp, *args)


def _find_better_pos2(pp, pos, vec, step, n):
    """ In 2D, we just have to search left/right along a line.
    """
    v1 = vec.normal() * step
    
    measure = _distance_measure(pp, pos, n)
    
    # for loops are like while loops with a failsafe
    
    for i in range(30):
        new_pos = pos + v1
        new_measure = _distance_measure(pp, new_pos, n)
        if new_measure <= measure:
            break
        pos, measure = new_pos, new_measure
    
    for i in range(30):
        new_pos = pos - v1
        new_measure = _distance_measure(pp, new_pos, n)
        if new_measure <= measure:
            break
        pos, measure = new_pos, new_measure
    
    return pos


def _find_better_pos3(pp, pos, vec, step, n):
    """ In 3D we search recursively left/right, forward/backward.
    Maybe this implementation could instead do a circle fit.
    """
    
    stub1, stub2 = PointSet([1, 1, 1]), PointSet([1, 0, 1])
    v1 = vec.cross(stub1)
    v1 = v1 if v1.norm() > 0 else vec.cross(stub2)
    v2 = vec.cross(v1)
    v1, v2 = v1.normalize() * step, v2.normalize() * step
    
    measure = _distance_measure(pp, pos, n)
    
    # for loops are like while loops with a failsafe
    
    for i in range(5):
        
        curpos = pos
        
        for i in range(30):
            new_pos = pos + v1
            new_measure = _distance_measure(pp, new_pos, n)
            if new_measure <= measure:
                break
            pos, measure = new_pos, new_measure
        
        for i in range(30):
            new_pos = pos - v1
            new_measure = _distance_measure(pp, new_pos, n)
            if new_measure <= measure:
                break
            pos, measure = new_pos, new_measure
        
        for i in range(30):
            new_pos = pos + v2
            new_measure = _distance_measure(pp, new_pos, n)
            if new_measure <= measure:
                break
            pos, measure = new_pos, new_measure
        
        for i in range(30):
            new_pos = pos - v2
            new_measure = _distance_measure(pp, new_pos, n)
            if new_measure <= measure:
                break
            pos, measure = new_pos, new_measure
        
        if (pos == curpos).all():
            break
    
    return pos
