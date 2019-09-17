# Author: M.A. Koenrades. Created 2018.
""" Module with functions to handle graphs or PointSets for analysis or visualization

"""
import numpy as np
import scipy.io
from stentseg.utils import PointSet

def get_graph_in_phase(graph, phasenr):
    """ Get position of model in a certain phase
    """
    from stentseg.stentdirect import stentgraph
    import numpy as np
    
    # initialize
    model_phase = stentgraph.StentGraph()
    for n1, n2 in graph.edges():
        # obtain path and deforms of nodes and edge
        path = graph.edge[n1][n2]['path']
        pathDeforms = graph.edge[n1][n2]['pathdeforms']
        # obtain path in phase
        path_phase = []
        for i, point in enumerate(path):
            pointposition = point + pathDeforms[i][phasenr]
            path_phase.append(pointposition) # points on path, one phase
        n1_phase, n2_phase = tuple(path_phase[0]), tuple(path_phase[-1]) # position of nodes
        model_phase.add_edge(n1_phase, n2_phase, path = np.asarray(path_phase), 
                            pathdeforms = np.asarray(pathDeforms))
    return model_phase

def points_from_edges_in_graph(graph, type='order'):
    """ Get all points on the paths in the graph, return as list with
    PointSets per path. If type='order' return in correct order of path points;
    there may be duplicates between edges, not within a selfloop edge.
    If type = 'noduplicates' return in random order but without duplicate points.
    """
    paths_as_pp = []
    for n1,n2 in graph.edges():
        path = graph.edge[n1][n2]['path']  # points on path include n1 n2
        if type == 'noduplicates':
            # Create set of tuples to remove duplicates; but order is lost!
            path = np.asarray(path)
            pp = set(tuple(p) for p in path)
            # turn into a pointset
            path_as_pp = PointSet(np.array(list(pp)))
        elif type == 'order':
            if path[0].all() == path[-1].all():
                path = path[1:] # remove the duplicate (in selfloop)
                path_as_pp = PointSet(path)
        paths_as_pp.append(path_as_pp)
    
    return paths_as_pp
    
    
def save_paths_as_mat(paths_as_pp, storematloc=None):
    """ Save the paths obtained as PointSets as mat file
    """
    #filename = '%s_%s_%s_%s.mat' % (ptcode, ctcode, cropname, 'edgepoints')
    #storematloc = os.path.join(targetdir, filename)
    storevar = dict()
    storevar['edgepoints'] = paths_as_pp
    scipy.io.savemat(storematloc,storevar)
    print('')
    print('points of the edges/paths were stored as .mat to {}'.format(storematloc))
    print('')


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