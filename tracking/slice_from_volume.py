from points import Point
import numpy as np
import interpolation_
from visvis import ssdf

def slice_from_volume(vol, sampling, normal, pos, prev_vec1, prev_vec2):
    
    #Get vecs
    vec1, vec2 = get_span_vectors(normal, prev_vec1, prev_vec2)
    
    vec1 = vec1 * Point(sampling[2],sampling[1],sampling[0])
    vec2 = vec2 * Point(sampling[2],sampling[1],sampling[0])
    
    #Create slice
    slice = interpolation_.slice_from_volume(vol,pos,vec1,vec2,256)
    
    return slice, vec1, vec2
    
def get_span_vectors(normal, c, d):
    """ get_span_vectors(normal, prevA, prevB) -> (a,b)
    
    Given a normal, return two orthogonal vectors which are both orthogonal
    to the normal. The vectors are calculated so they match as much as possible
    the previous vectors.
    
    """
    
    # Calculate a from previous b
    a1 = d.Cross(normal)
    
    if a1.Norm() < 0.001:
        # The normal and  d point in same or reverse direction
        # -> Calculate b from previous a
        b1 = c.Cross(normal)
        a1 = b1.Cross(normal)
    
    # Consider the opposite direction
    a2 = -1 * a1
    if c.Distance(a1) > c.Distance(a2):
        a1 = a2
    
    # Ok, calculate b
    b1 = a1.Cross(normal)
    
    # Consider the opposite
    b2 = -1 * b1
    if d.Distance(b1) > d.Distance(b2):
        b1 = b2

    # Done
    return a1.Normalize(), b1.Normalize()