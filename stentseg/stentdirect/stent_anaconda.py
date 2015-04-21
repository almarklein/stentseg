"""
Implementation of StentDirect algorithm for the Anaconda stent graft

Functions in this module are specific to the anaconda proximal ring
Imports functions needed from stentgraph 
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import networkx as nx

import visvis as vv
from visvis import ssdf

from ..utils.new_pointset import PointSet
from ..utils import gaussfun

from . import stentgraph
from .stentgraph import _sorted_neighbours
from .base import StentDirect


SORTBY = 'cost'


class AnacondaDirect(StentDirect):
    """ An implementation of the StentDirect algorithm targeted at 
    the Anaconda stent graft. In particular, the pruning algorithm is
    modified.
    """
    
#     stentType = 'anaconda'
    
    def _Step3_iter(self, nodes, cleanNodes=True):
        params = self._params
        ene = params.graph_expectedNumberOfEdges
        
        # prune edges prior to pop and add crossing nodes, otherwise many false nodes
        stentgraph.prune_very_weak(nodes, params.graph_weakThreshold)
        stentgraph.prune_weak(nodes, ene, params.graph_strongThreshold)
        
        # Use our own pruning algorithm
        prune_redundant(nodes, params.graph_strongThreshold,
                               params.graph_min_strutlength,
                               params.graph_max_strutlength)
        if cleanNodes == True:
            stentgraph.pop_nodes(nodes)
            stentgraph.add_nodes_at_crossings(nodes) 
            # mind that adding at crossing in first iteration can lead to uncleaned edges (degree 3 nodes)
            stentgraph.pop_nodes(nodes)  # because adding nodes can leave other redundant
            prune_redundant(nodes, params.graph_strongThreshold,
                                   params.graph_min_strutlength,
                                   params.graph_max_strutlength)
        stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize)
        stentgraph.prune_tails(nodes, params.graph_trimLength)


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


def _edge_length(graph, n1, n2):
    """ 
    Get path length for this edge in mm
    
    """
    path = np.asarray(graph.edge[n1][n2]['path'])  # [x,y,z]
    v = []
    for point in range(len(path)-1):
        v.append(path[point]-path[point+1])
    v = np.vstack(v)
    d = (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5  # vector length in mm
    path_l = d.sum() 
    return path_l


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
    print()
    print('This iteration %i eligable edges for removal were part of a *weak*' 
          'triangle so removed' % (cntremove))
    print('This iteration %i eligable edges for removal were part of a *strong*'
          'triangle so NOT removed' % (cntspare))
    print()


def _prune_redundant_edge(graph, n1, n2, min_ctvalue, min_strutlength, max_strutlength):
    """ Check if we should prune a given single redundant edge. 
    """
    
    counter = False
    
    # Do not allow connections to same node. Note: such edges should
    # not be produced by the MCP alg, but we check them to be sure
    if n1 == n2:
        print('Warning: detected node that was connected to itself.')
        graph.remove_edge(n1, n2)
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
                        paths.sort(reverse = True) # longest first
                        # Check if the triangle edges are proximal ring struts
                        if (min_strutlength < paths[0] < max_strutlength and 
                            min_strutlength < paths[1] < max_strutlength):
                                print('Eligable edge with' , graph.edge[n1][n2],
                                        'is part of *strong* triangle so NOT removed')
                                print()
                                print('Neighbour edge with' ,graph.edge[n1][node1],
                                        'and pathlength', path1_l)
                                print('Neighbour edge with' ,graph.edge[node1][node2],
                                        'with pathlength' , path2_l)
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
                        paths.sort(reverse = True) # longest first
                        # Check if the triangle edges are proximal ring struts
                        if (min_strutlength < paths[0] < max_strutlength and 
                            min_strutlength < paths[1] < max_strutlength):
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
                    








