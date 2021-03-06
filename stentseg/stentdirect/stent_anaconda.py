# 2014-2016 Maaike Koenrades
"""
Implementation of StentDirect algorithm for the Anaconda stent graft

Functions in this module are specific to the anaconda proximal ring
Imports functions needed from stentgraph 
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import networkx as nx
import time

import visvis as vv
from visvis import ssdf

from ..utils.new_pointset import PointSet
from ..utils import gaussfun

from . import stentgraph
from .stentgraph import _sorted_neighbours, _edge_length
from .base import StentDirect


SORTBY = 'cost'


class AnacondaDirect(StentDirect):
    """ An implementation of the StentDirect algorithm targeted at 
    the Anaconda dual ring. In particular, the pruning algorithm is
    modified and seed placement is not allowed above higher th or seeds are
    removed by detection of outliers.
    Rationale: markers are placed next to wire and cause a gap in R1
    """
    
#     stentType = 'anaconda'

    def Step1(self):
        """ Step1()
        Detect seed points.
        """
        print('get mask for seedpoints anacondaRing is used')
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        
        t0 = time.time()
        
        # Detect points
        th = self._params.seed_threshold
        pp = get_stent_likely_positions(self._vol, th) # calls function below
        
        # Create nodes object from found points
        nodes = stentgraph.StentGraph()
        for p in pp:
            p_as_tuple = tuple(p.flat) # todo: perhaps seed detector should just yield list of tuples.
            nodes.add_node(p_as_tuple)
        
        t1 = time.time()
        if self._verbose:
            print()
            print('Found %i seed points, which took %1.2f s.' % (len(nodes), t1-t0))
        
        # Store the nodes
        self._nodes1 = nodes
        
        # Draw?
        if self._draw:
            self.Draw(1)
        
        return nodes

    def Step3(self, cleanNodes=True):
        """ Step3()
        Process graph to remove unwanted edges.
        """
        
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        if self._nodes2 is None:
            raise ValueError('Edges not yet calculated.')
        
        # Get nodes and params
        #nodes = stentgraph.StentGraph()
        #nodes.unpack( self._nodes2.pack() )
        nodes = self._nodes2.copy()
        params = self._params
        
        # Init times        
        t_start = time.time()
        t_clean = 0
        
        print('Step3 anacondaRing is used')
        # Iteratively prune the graph. 
        cur_edges = 0
        count = 0
        ene = params.graph_expectedNumberOfEdges
        while cur_edges != nodes.number_of_edges():
            count += 1
            cur_edges = nodes.number_of_edges()
            self._Step3_iter(nodes, cleanNodes)
        
        if cleanNodes == True:
            stentgraph.add_nodes_at_crossings(nodes) # not in loop; changes path of rings
            stentgraph.pop_nodes(nodes) # pop before corner detect or angles can not be found
            stentgraph.add_corner_nodes(nodes, th=params.graph_angleVector, angTh=params.graph_angleTh)
            stentgraph.pop_nodes(nodes)  # because removing edges/add nodes can create degree 2 nodes
            stentgraph.prune_tails(nodes, params.graph_trimLength)
            stentgraph.prune_clusters(nodes, 3) #remove residual nodes/clusters
            
            nodes = self._RefinePositions(nodes)
            stentgraph.smooth_paths(nodes, 4) # do not smooth iterative based on changing edges
            
        t0 = time.time()-t_start
        tmp = "Reduced to %i edges and %i nodes, "
        tmp += "which took %1.2f s (%i iters)"
        print(tmp % (nodes.number_of_edges(), nodes.number_of_nodes(), t0, count))
        
        # Finish
        self._nodes3 = nodes
        if self._draw:
            self.Draw(3)
        
        return nodes
    
    def _Step3_iter(self, nodes, cleanNodes=True):
        params = self._params
        ene = params.graph_expectedNumberOfEdges
        
        # prune edges prior to pop and add crossing nodes, otherwise many false nodes
        stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize) # before clean nodes
        stentgraph.prune_very_weak(nodes, params.graph_weakThreshold)
        stentgraph.prune_weak(nodes, ene, params.graph_strongThreshold)
        
        # Use our own pruning algorithm
        prune_redundant(nodes, params.graph_strongThreshold,
                               params.graph_min_strutlength,
                               params.graph_max_strutlength)
        # if cleanNodes == True:
        #     stentgraph.add_nodes_at_crossings(nodes) 
        #     # mind that adding at crossing in first iteration can lead to uncleaned edges (degree 3 nodes)
        #     prune_redundant(nodes, params.graph_strongThreshold,
        #                            params.graph_min_strutlength,
        #                            params.graph_max_strutlength)
        stentgraph.prune_tails(nodes, params.graph_trimLength)
        stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize)


# Step 1: Rationale: markers are placed next to wire, prevent seed placement here
def get_stent_likely_positions(data, th):
    """ Get a pointset of positions that are likely to be on the stent.
    If the given data has a "sampling" attribute, the positions are
    scaled accordingly. 
    
    Detection goes according to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    """
    
    # Get mask
    mask = get_mask_with_stent_likely_positions(data, th)
    
    # Convert mask to points
    indices = np.where(mask==2)  # Tuple of 1D arrays
    pp = PointSet( np.column_stack(reversed(indices)), dtype=np.float32)
    
    # Correct for anisotropy and offset
    if hasattr(data, 'sampling'):
        pp *= PointSet( list(reversed(data.sampling)) ) 
    if hasattr(data, 'origin'):
        pp += PointSet( list(reversed(data.origin)) ) 
    
    return pp


def get_mask_with_stent_likely_positions(data, th):
    """ Get a mask image where the positions that are likely to be
    on the stent, subject to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    Returns a mask, which can easily be turned into a set of points by 
    detecting the voxels with the value 2.
    
    This is a pure-Python implementation.
    """
    
    # NOTE: this pure-Python implementation is little over twice as slow
    # as the Cython implementation, which is a neglectable cost since
    # the other steps in stent segmentation take much longer. By using
    # pure-Python, installation and modification are much easier!
    # It has been tested that this algorithm produces the same results
    # as the Cython version.
    
    import numpy as np
    # Init mask
    mask = np.zeros_like(data, np.uint8)
    
    # Criterium 1A: voxel must be above th
    # Note that we omit the edges
    mask[20:-20,20:-20,20:-20] = (data[20:-20,20:-20,20:-20] > th[0]) * 3
    
    values = []
    for z, y, x in zip(*np.where(mask==3)):
        
        # Only proceed if this voxel is "free"
        if mask[z,y,x] == 3:
            
            # Set to 0 initially
            mask[z,y,x] = 0  
            
            # Get value
            val = data[z,y,x]
            
            # Get maximum of neighbours
            patch = data[z-1:z+2, y-1:y+2, x-1:x+2].copy()
            patch[1,1,1] = 0
            themax = patch.max()
            
            # Criterium 2: must be local max
            if themax > val:
                continue
            # Also ensure at least one neighbour to be *smaller*
            if (val > patch).sum() == 0:
                continue
            
            # Criterium 3: one neighbour must be above th
            if themax <= th[0]:
                continue
            
            # Criterium 1B: voxel must be below upper seed th, if given
            if len(th) ==2:
                if val > th[1]:
                    print('Seed removed by higher th: ',(z,y,x),'ctvalue=', val)
                    continue
            
            # Set, and suppress stent points at direct neightbours
            mask[z-1:z+2, y-1:y+2, x-1:x+2] = 1
            mask[z,y,x] = 2
            values.append(data[z,y,x])
    
    # remove outliers => markers
    if len(th) == 1: # no upper seed threshold given
        for z, y, x in zip(*np.where(mask==2)):
            val = data[z,y,x]
            #mean-based method
            if val-np.mean(values) > 2.5* np.std(values):
                # but do not remove if value if below 2200 HU (or False remove when normalized)
                if not val < 2200:
                    mask[z,y,x] = 1
                    values.remove(val)
                    print("seed with value {} removed as outlier".format(val) )
    
    print()
    print('Seed ctvalues: {}'.format(sorted(values)))
    
    return mask


# Step 3
def prune_weak(graph, enc, ctvalue, min_strutlength, max_strutlength):
    """ Remove weak edges, based on cost and number of expected edges (enc).
    
    All edges are tested to be eligible for removal from both nodes.
    An edge is eligible for removal if a node has more than the expected 
    number of edges, and the edge is not strong enough (does not have a CT
    value above the given value). Exceptions are those edges that are part of a
    quadrangle, or quadrilateral, with other strong edges of a specified length.
    
    Rationale: The proximal double Anaconda ring contains 4 anchor points with
    hooks. Nodes at these points connect 3 or 4 edges and should be preserverd.
    Edges in excess of the expected number of edges can only be maintained if 
    their lowest CT value is so high it proves the existance of a wire between
    the nodes or if the path is situated between two hooks at 
    the proximal fixation rings (struts).
    """
    
    # First, get a sorted list of edges
    edges = graph.edges()
    cnt1 = 0
    cnt2 = 0    
    for (n1, n2) in edges:
        c = graph[n1][n2]
        if c['ctvalue'] < ctvalue:
            # We *might* remove this edge
            
            # Check for each node what their "extra edges" are
            extra_edges1 = _sorted_neighbours(graph, n1)
            extra_edges2 = _sorted_neighbours(graph, n2)
            
            # If this edge is "extra" as seen from both ends, we *might* remove it
            if n1 in extra_edges2[enc:] and n2 in extra_edges1[enc:]:
                #print('Eligable edge for removal: ',n1,'-',n2 ,graph.edge[n1][n2])
                quadrangle = 0
                # Test whether edge n1,n2 is part of *strong* quadrangle
                for node1 in extra_edges1:
                    if quadrangle == 1:
                        break  # not allow removal when part of *strong* quadrangle
                    if node1 == n2:
                        continue  # go to next iteration, do not consider n1-n2
                    for node2 in extra_edges2:
                        if node2 == n1:
                            continue
                        if graph.has_edge(node1,node2):
                            q1 = graph.edge[node1][node2]
                            q2_path_l = _edge_length(graph, n1, node1)
                            q3_path_l = _edge_length(graph, n2, node2)
                            print('Quadrangle found with connecting edge: ',node1,'-',node2,q1)
                            if (min_strutlength < q2_path_l < max_strutlength and 
                                min_strutlength < q3_path_l < max_strutlength):
                                quadrangle = 1            # strong
                                break
                            else:                       
                                quadrangle = 2            # weak
                if quadrangle == 0:
                    graph.remove_edge(n1, n2)
                if quadrangle == 1:
                    #print('Eligable edge',n1,'-',n2,'part of *strong* quadrangle so not removed')
                    #print('******************')
                    cnt1 +=1
                if quadrangle == 2:
                    print('Eligable edge',n1,'-',n2,'part of *weak* quadrangle so removed')
                    print('q2_path_l',q2_path_l,'q3_path_l',q3_path_l)
                    print('******************')
                    graph.remove_edge(n1, n2)
                    cnt2 +=1
    print()
    print('This iteration %i eligable edges for removal were part of *strong* quadrangle'
          'so NOT removed and %i were part of *weak* quadrangle so removed' % (cnt1,cnt2))
    print()


def prune_redundant(graph, ctvalue, min_strutlength, max_strutlength):
    """
    Remove redundant edges. 
    
    A connection is redundant if a weak connection (high mcp cost) 
    connects two nodes which are already connected via two other edges
    wich are each stronger.
    
    In other words, this function tests each triangle of egdes and removes
    the weakest one. Exceptions: 
    - if its above the given ctvalue
    - if its part of a strong triangle: edge length neighbours within 
      strutlength range
    
    Rationale: 2nd ring of proximal rings is thinner and thus weaker. 
    The nodes at hooks often have 3 or more edges and can be found redundant 
    
    """
    
    # First, get a sorted list of edges (weakest first)
    edges = graph.edges()
    edges.sort(key=lambda x: graph.edge[x[0]][x[1]][SORTBY])
    if SORTBY == 'cost':
        edges.reverse()
    
    cntremove = 0
    cntspare = 0
    for (n1, n2) in edges:
        counter = _prune_redundant_edge(graph, n1, n2, ctvalue, min_strutlength, 
                    max_strutlength)
        if graph.has_edge(n1,n2):
            if counter == True:
                cntspare += 1
        else:
            cntremove += 1   
#     print()
#     print('This iteration %i redundant edges were part of a *weak*' 
#           'triangle so REMOVED' % (cntremove))
#     print('This iteration %i redundant edges were part of a *strong*'
#           'triangle so NOT REMOVED' % (cntspare))
#     print()


def _prune_redundant_edge(graph, n1, n2, min_ctvalue, min_strutlength, max_strutlength):
    """ Check if we should prune a given single redundant edge. 
    """
    
    counter = False
    
    # Do not allow connections to same node. Note: such edges should
    # not be produced by the MCP alg, but we check them to be sure
    if n1 == n2:
        print('Warning: detected node that was connected to itself; node is not removed')
        return counter
    # Get neighbours for n1 and n2
    nn1 = set(graph.edge[n1].keys())
    nn1.discard(n2)
    nn2 = set(graph.edge[n2].keys())
    nn2.discard(n1)
    
    # Get cost and ctvalue for this edge
    cost = graph.edge[n1][n2]['cost']
    ctvalue = graph.edge[n1][n2]['ctvalue']
    
    # If ct value is high enough, leave edge intact
    if ctvalue >= min_ctvalue:
        return counter
    
    # Note: the graph type that we use prohibits having more than one
    # edge between any two nodes.
    
    weak = 0

    # Check if there are two edges that are stronger than this edge
    for node1 in nn1:
        for node2 in graph.edge[node1].keys():
            if node2 == n2:
                if ((graph.edge[n1][node1]['cost'] < cost) and
                    (graph.edge[node1][node2]['cost'] < cost)):
                        weak = 1
                        path0_l = _edge_length(graph, n1, node2)
                        path1_l = _edge_length(graph, n1, node1)
                        path2_l = _edge_length(graph, node1, node2)
                        paths = [path0_l, path1_l, path2_l]
                        paths2 = paths.copy()
                        paths.sort(reverse = True) # longest first
                        # directions
                        v0 = np.subtract(n1,node2)# nodes, paths in x,y,z
                        v1 = np.subtract(n1,node1) 
                        v2 = np.subtract(node1,node2)
                        dir0 = abs(v0 / np.sqrt(v0[0]**2+v0[1]**2+v0[2]**2))
                        dir1 = abs(v1 / np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2))
                        dir2 = abs(v2 / np.sqrt(v2[0]**2+v2[1]**2+v2[2]**2))
                        dirs = [dir0, dir1, dir2]
                        ir = paths2.index(paths[2]) # ring edge
                        is1 = paths2.index(paths[0])
                        is2 = paths2.index(paths[1])
                        # Check if the triangle edges are proximal ring struts
                        if (min_strutlength < paths[0] < max_strutlength and 
                            min_strutlength < paths[1] < max_strutlength and
                            dirs[ir][2]<dirs[is1][2] and dirs[ir][2]<dirs[is2][2]):
#                                 print('Eligable edge with' , graph.edge[n1][n2],
#                                         'is part of *strong* triangle so NOT removed')
#                                 print()
#                                 print('Neighbour edge with' ,graph.edge[n1][node1],
#                                         'and pathlength', path1_l)
#                                 print('Neighbour edge with' ,graph.edge[node1][node2],
#                                         'with pathlength' , path2_l)
                                print('********nn1**********')
                                counter = True
                                # mark spared edges; prevent removal in next iteration
                                graph.add_node(n1, spared=True)
                                graph.add_node(n2, spared =True)
                                return counter                          
    if weak == 1:
#         print('Eligable edge with' ,graph.edge[n1][n2],' is part of *weak* triangle so removed')
#         print()
#         print('Neighbour edge with' , graph.edge[n1][node1], 'and pathlength', path1_l)
#         print('Neighbour edge with' ,graph.edge[node1][node2], 'and pathlength', path2_l)
#         print('******************')
        graph.remove_edge(n1, n2)
        return counter
    
    for node1 in nn2:
        for node2 in graph.edge[node1].keys():
            if node2 == n1:
                if ((graph.edge[n2][node1]['cost'] < cost) and
                    (graph.edge[node1][node2]['cost'] < cost)):
                        weak = 1
                        path0_l = _edge_length(graph, n2, node2)
                        path1_l = _edge_length(graph, n2, node1)
                        path2_l = _edge_length(graph, node1, node2)
                        paths = [path0_l, path1_l, path2_l]
                        paths2 = paths.copy()
                        paths.sort(reverse = True) # longest first
                        # directions
                        v0 = np.subtract(n2,node2)# nodes, paths in x,y,z
                        v1 = np.subtract(n2,node1) 
                        v2 = np.subtract(node1,node2)
                        dir0 = abs(v0 / np.sqrt(v0[0]**2+v0[1]**2+v0[2]**2))
                        dir1 = abs(v1 / np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2))
                        dir2 = abs(v2 / np.sqrt(v2[0]**2+v2[1]**2+v2[2]**2))
                        dirs = [dir0, dir1, dir2]
                        ir = paths2.index(paths[2]) # ring edge
                        is1 = paths2.index(paths[0])
                        is2 = paths2.index(paths[1])
                        # Check if the triangle edges are proximal ring struts
                        if (min_strutlength < paths[0] < max_strutlength and 
                            min_strutlength < paths[1] < max_strutlength and
                            dirs[ir][2]<dirs[is1][2] and dirs[ir][2]<dirs[is2][2]):
#                                         print('Eligable edge with' , graph.edge[n1][n2],
#                                               'is part of *strong* triangle so NOT removed')
#                                         print()
#                                         print('Neighbour edge with' ,graph.edge[n1][node1],
#                                               'and pathlength' , path1_l)
#                                         print('Neighbour edge with' ,graph.edge[node1][node2],
#                                               'with pathlength' , path2_l)
#                                         print('********nn2**********')
                                counter = True
                                # mark spared edges; prevent removal in next iteration
                                graph.add_node(n1, spared=True)
                                graph.add_node(n2, spared =True)
                                return counter
    if weak == 1:
        graph.remove_edge(n1, n2) 
        return counter
                    








