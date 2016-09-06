""" Functionality for stent analysis

"""

from stentseg.utils import PointSet

def point_in_pointcloud_closest_to_p(pp, point):
    """ Find point in PointSet which is closest to a point
    Returns a PointSet with point found in PointSet and point 
    """
    vecs = pp-point
    dists_to_pp = ( (vecs[:,0]**2 + vecs[:,1]**2 + vecs[:,2]**2)**0.5 ).reshape(-1,1)
    pp_index =  list(dists_to_pp).index(dists_to_pp.min() ) # index on path
    pp_point = pp[pp_index]
    p_in_pp_and_point = PointSet(3, dtype='float32')
    [p_in_pp_and_point.append(*p) for p in (pp_point, point)]
    return p_in_pp_and_point