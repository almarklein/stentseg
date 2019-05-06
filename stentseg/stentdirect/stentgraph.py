"""
Next version for stentseg-specific graph class and functionality.
Use networkx instead of vispy.util.graph.
In progress. Step 1 and step2 of stentdirect work now.

The functions in this module that strat with "prune" are intended to
do no harm to the graph and it *should* be possible to apply them
recursively and in any order to clean up a graph that represents a 
stent.
 
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import networkx as nx

import visvis as vv
from visvis import ssdf

from stentseg.utils.new_pointset import PointSet
from stentseg.utils import gaussfun

assert nx.__version__ < "2", "This code need NetworkX v1. Use pip install networkx==1.11"

# todo: what if we sort by CT value?
SORTBY = 'cost'


class StentGraph(nx.Graph):
    """ A graph class specific to the stentseg package.
    It has functionality for drawing in visvis and exporting to ssdf.
    """
    
    def Draw(self, *args, **kwargs):
        """ Backward compatibility.
        """
        return self.draw(*args, **kwargs)
    
    def draw(self, mc='b', lc='g', mw=7, lw=0.6, alpha=0.5, axes=None, simple=False):
        """ Draw in visvis.
        """
        
        # we can only draw if we have any nodes
        if not self.number_of_nodes():
            return
        
        # Convert nodes to Poinset
        ppn = PointSet(3)
        for n in self.nodes_iter():
            ppn.append(*n)
        
        # Convert connections to Pointset
        ppe = PointSet(3)
        if simple:
            # Edges only
            for n1, n2 in self.edges_iter():
                ppe.append(*n1)
                ppe.append(*n2)
        else:
            # Paths
            for n1, n2 in self.edges_iter():
                path = self.edge[n1][n2]['path']
                ppe.append(path[0])
                for p in path[1:-1]:
                    ppe.append(p)
                    ppe.append(p)
                ppe.append(path[-1])     
        
        # Plot markers (i.e. nodes)
        vv.plot(ppn, ms='o', mc=mc, mw=mw, ls='', alpha=alpha, 
                axes=axes, axesAdjust=False)
        
        # Plot lines
        vv.plot(ppe, ls='+', lc=lc, lw=lw, ms='', alpha=alpha, 
                axes=axes, axesAdjust=False)
    
    
    def pack(self):
        """ Pack the graph to an ssdf struct.
        This method is not stent-specific and is a generic graph export to ssdf.
        """
        # Prepare
        s = ssdf.new()
        s.nodes = []
        s.edges = []
        # Serialize
        s.graph = ssdf.loads(ssdf.saves(self.graph))
        for node in self.nodes_iter():
            #node_info = {'node': node, 'attr': self.node[node]}
            node_info = node, self.node[node]
            s.nodes.append(node_info)
        for edge in self.edges_iter():
            #edge_info = {'node1': edge[0], 'node2':edge[1], 'attr': self.edge[edge[0]][edge[1]]}
            edge_info = edge[0], edge[1], self.edge[edge[0]][edge[1]]
            s.edges.append(edge_info)
        return s
    
    
    def unpack(self, s):
        """ Populate the graph with the given ssdf struct, which should
        be a grapgh export via the pack method.
        """
        self.clear()
        self.graph.update(s.graph.__dict__)
        for node_info in s.nodes:
            #node, attr = node_info['node'], node_info['attr'].__dict__
            node, attr = node_info
            node = node if not isinstance(node, list) else tuple(node)
            attr = attr if isinstance(attr, dict) else attr.__dict__
            self.add_node(node, **attr)
        for edge_info in s.edges:
            #node1, node2, attr = edge_info['node1'], edge_info['node2'], node_info['attr'].__dict__
            node1, node2, attr = edge_info
            node1 = node1 if not isinstance(node1, list) else tuple(node1)
            node2 = node2 if not isinstance(node2, list) else tuple(node2)
            attr = attr if isinstance(attr, dict) else attr.__dict__
            self.add_edge(node1, node2, **attr)



def check_path_integrity(graph):
    """ Verify that each path is represented in the right order.
    """
    for n1, n2 in graph.edges():
        path = graph.edge[n1][n2]['path']
        if n1 > n2:
            n1, n2 = n2, n1
        assert np.isclose(path[0], n1, 0, 1).all()
        assert np.isclose(path[-1], n2, 0, 1).all()
    

def prune_very_weak(graph, ctvalue):
    """ Remove very weak edges
    
    All edges are evaluated and their ctvalue is compared to the given
    threshold. If it is lower, the edge is very bad and removed.
    """
    
    for (n1, n2) in graph.edges():
        c = graph[n1][n2]
        if c['ctvalue'] < ctvalue:
            graph.remove_edge(n1, n2)



def _sorted_neighbours(graph, node):
    """ For a node in a graph, get a list of neighbours that is sorted
    by the cost of the edge.
    """
    # Create tuples (node, edge_dict) for each neighbour
    neighbours = [n2 for n2 in graph.edge[node].keys()]
    
    # Sort that list
    sorter = lambda x: graph.edge[node][x][SORTBY]
    neighbours.sort(key=sorter)
    return neighbours



def prune_weak(graph, enc, ctvalue):
    """ Remove weak edges, based on cost and number of expected edges (enc).
    
    All edges are tested to be elligible for removal from both nodes.
    An edge is elligible for removal if a node has more than the 
    expected number of edges, and the edge is not strong enough (does not
    have a CT value above the given value).
    
    Rationale: Edges with a path that contains a very low CT
    value can always be discarded. Edges in excess of the expected
    number of edges can only be maintained if their lowest CT 
    value is so high it proves the existance of a wire between
    the nodes.
    """
    
    # First, get a sorted list of edges
    edges = graph.edges()
    # Sorting not needed, we sort at the neighbours
    #edges.sort(key=lambda x: graph.edge[x[0]][x[1]]['ctvalue'])
    
    for (n1, n2) in edges:
        c = graph[n1][n2]
        if c['ctvalue'] < ctvalue:
            # We *might* remove this edge
            
            # Check for each node what their "extra edges" are
            extra_edges1 = _sorted_neighbours(graph, n1)[enc:]
            extra_edges2 = _sorted_neighbours(graph, n2)[enc:]
            
            # If this edge is "extra" as seen from both ends, we remove it
            if n1 in extra_edges2 and n2 in extra_edges1:
                if (graph.node[n1].get('spared', False) and
                   graph.node[n2].get('spared', False) ):
                    pass  # explicitly prevent removal
                else:
                    graph.remove_edge(n1, n2)



def prune_tails(graph, maxlength):
    """ Remove all tails that are smaller or equal than maxlength nodes.
    """
    visited_nodes = set()
    nodes_to_remove = set()
    
    def pop(node, trail):
        if len(trail) >= maxlength:
            return  # Do not remove
        # Get next node
        neighbours = list(graph.edge[node].keys())
        if not trail:
            nextnode = neighbours[0]
        else:
            nextnode = neighbours[0] if neighbours[1] == trail[-1] else neighbours[1]
        # Pop
        trail.append(node)
        visited_nodes.add(node)
        # Proceed to next or remove trail?
        if nx.degree(graph, nextnode) == 2:
            pop(nextnode, trail)
        else:
            nodes_to_remove.update(set(trail))
    
    # Iterate over all nodes
    for node in graph.nodes():
        if node in visited_nodes:
            pass
        elif nx.degree(graph, node) == 1:
            pop(node, [])
    
    # Remove the nodes
    for node in nodes_to_remove:
        graph.remove_node(node)



def prune_clusters(graph, minsize):
    """ Remove all small clusters of interconnected nodes (i.e.
    connected components) that are smaller than minsize nodes.
    """
    # Thanks to networkx this one is easy!
    for cluster in list(nx.connected_components(graph)):
        if len(cluster) < minsize:
            if cluster[0] in graph.nodes_with_selfloops(): #pop can create cluster of edge to self
                pass # prevent removal
            else:
                graph.remove_nodes_from(cluster)



def prune_redundant(graph, ctvalue):
    """
    Remove redundant edges. 
    
    A connection is redundant if a weak connection (high mcp cost) 
    connects two nodes which are already connected via two other edges
    wich are each stronger.
    
    In other words, this function tests each triangle of egdes and removes
    the weakest one (except if its above the given ctvalue).
    
    """
    
    # First, get a sorted list of edges (weakest first)
    edges = graph.edges()
    edges.sort(key=lambda x: graph.edge[x[0]][x[1]][SORTBY])
    if SORTBY == 'cost':
        edges.reverse()
    
    for (n1, n2) in edges:
        _prune_redundant_edge(graph, n1, n2, ctvalue)


def _prune_redundant_edge(graph, n1, n2, min_ctvalue):
    """ Check if we should prune a given single redundant edge. 
    """
    
    # Do not allow connections to same node. Note: such edges should
    # not be produced by the MCP alg, but we check them to be sure
    if n1 == n2:
        print('Warning: detected node that was connected to itself; node is not removed.')
#         graph.remove_edge(n1, n2) # do not remove as rings can consist of 1 node connected to itself
        return
    
    # Get neightbours for n1 and n2
    nn1 = set(graph.edge[n1].keys())
    nn1.discard(n2)
    nn2 = set(graph.edge[n2].keys())
    nn2.discard(n1)
    
    # Get cost and ctvalue for this edge
    cost = graph.edge[n1][n2]['cost']
    ctvalue = graph.edge[n1][n2]['ctvalue']
    
    # If ct value is high enough, leave edge intact
    if ctvalue >= min_ctvalue:
        return
    
    # Note: the graph type that we use prohibits having more than one
    # edge between any two nodes.
    
    # Check if there are two edges that are stronger than this edge
    for node1 in nn1:
        for node2 in graph.edge[node1].keys():
            if node2 == n2:
                if ((graph.edge[n1][node1]['cost'] < cost) and
                    (graph.edge[node1][node2]['cost'] < cost)):
                    graph.remove_edge(n1, n2)
                    return
    
    for node1 in nn2:
        for node2 in graph.edge[node1].keys():
            if node2 == n1:
                if ((graph.edge[n2][node1]['cost'] < cost) and
                    (graph.edge[node1][node2]['cost'] < cost)):
                    graph.remove_edge(n1, n2)
                    return


def _pop_node(graph, node):
    """ Pop a node from the graph and connect its two neightbours.
    The paths are combined and ctvalue and cost are taken into account.
    Returns the returned edge (or None if the node could not be popped).
    
    If the three nodes under consideration (the given and its two
    neighbours) form a clique, the clique is reduced to a single node
    which is connected to itself. The remaining node is the smallest
    of the neighbours of the given node.
    """
    
    #  n1 -- n2 -- n3  (n2 is popped)
    n2 = node
    
    # We can only pop nodes that have two neighbours
    assert graph.degree(n2) == 2
    
    # Get the neihbours and the corresponding edges
    n1, n3 = list(graph.edge[n2].keys())
    edge12 = graph.edge[n1][n2]
    edge23 = graph.edge[n2][n3]
    
    # Get the paths and conbine into one new path
    # The start of a path is at the "smallest" node (stentmcp.py)
    path12 = edge12['path']
    path23 = edge23['path']
    # First make sure the path is from n2-n1-n3
    if n2 < n1: 
        path12 = PointSet(np.flipud(path12))
    if n3 < n2:
        path23 = PointSet(np.flipud(path23))
    path = PointSet(path12.shape[1])
    path.extend(path12[:-1])  # Avoid that n2 occurs twice in the path
    path.extend(path23)
    # Flip if necessary
    if n3 < n1:
        path = PointSet(np.flipud(path)).copy()  # make C_CONTIGUOUS
    
    # Verify path order
    if False:
        if n1 < n3:
            assert np.isclose(path[0], n1).all()
            assert np.isclose(path[-1], n3).all()
        else:
            assert np.isclose(path[0], n3).all()
            assert np.isclose(path[-1], n1).all()
    
    # Calculate new ctvalue and cost
    cost = sum([edge12['cost'], edge23['cost']])
    ctvalue = min(edge12['ctvalue'], edge23['ctvalue'])
    
    # Check if this is a cluster of three nodes...
    neighbours1 = set(graph.edge[n1].keys())
    neighbours3 = set(graph.edge[n3].keys())
    collapse3 = False
    if n3 in neighbours1: # == n1 in neighbours3
        # So n1 and n3 are connected. We cannot just pop n2, because
        # each two nodes can have just one connection. We either pop
        # two nodes, or not pop at all.
        #
        # Determine what node we can collapse
        collapse3 = True
        collapsable = set()
        if neighbours1 == set([n2, n3]):
            collapsable.add(n1)
        if neighbours3 == set([n1, n2]):
            collapsable.add(n3)
        #
        if not collapsable: 
            #print('cannot pop %s' % repr(n2))
            return  # we cannot pop this node!
        else:
            node_to_collapse = min(collapsable)  # in case both were ok
        # Get edge
        edge13 = graph.edge[n1][n3]
        # Path from 1-3 is always reversed (because now we need it
        # in the other* order that we defined above)
        path13 = edge13['path']
        path13_via2 = path
        path = PointSet(path.shape[1])
        if node_to_collapse == min([n1, n3]):  # node to keep is NOT min
            path.extend(np.flipud(path13))
            path.extend(path13_via2)
        else:  # node to keep is min
            path.extend(path13_via2)
            path.extend(np.flipud(path13))
            
        # Cost and ctvalue is easy
        cost = sum([edge13['cost'], cost])
        ctvalue = min(edge13['ctvalue'], ctvalue)
    
    # Pop the node, make new edge
    if collapse3:
        # Keep the smallest of the two
        graph.remove_node(n2)
        graph.remove_node(node_to_collapse)
        node_to_keep = n1 if node_to_collapse == n3 else n3
        #assert np.isclose(path[0], node_to_keep).all()
        #assert np.isclose(path[-1], node_to_keep).all()
        graph.add_edge(node_to_keep, node_to_keep, cost=cost, ctvalue=ctvalue, path=path)
        return (node_to_keep, node_to_keep)
    else:
        graph.remove_node(n2)
        graph.add_edge(n1, n3, cost=cost, ctvalue=ctvalue, path=path)
        return (n1, n3)


def pop_nodes(graph):
    """ Pop all nodes with degree 2 (having exactly two edges). A series
    of connected nodes can thus be reduced by a single node with an
    edge that connects to itself. After this, the cornerpoints need to
    be detected.
    """
    
    for node in list(graph.nodes()):
        try:
            neighbours = list(graph.edge[node].keys())
        except KeyError:
            continue  # node popped as a cluster of three in _pop_node()
        if len(neighbours) == 2:
            if node not in neighbours:  # cannot pop if we only connect to self
                if graph.node[node].get('corner', False):
                    pass  # explicitly prevent popping when corner=True
                elif graph.node[node].get('nopop', False):
                    pass  # explicitly prevent popping when nopop=True
                else:
                    _pop_node(graph, node)


def create_mesh(graph, radius=1.0, fullPaths=True):
    """ Create a polygonal model from the stent and return it as
    a visvis.BaseMesh object. To draw the mesh, instantiate
    a normal mesh using vv.Mesh(vv.gca(), thisMesh).
    """
    from visvis.processing import lineToMesh, combineMeshes
    from visvis import Pointset  # lineToMesh does not like the new PointSet class
    
    # Init list of meshes
    meshes = []
    
    for n1, n2 in graph.edges():
        # Obtain path of edge and make mesh
        if fullPaths:                
            path = graph.edge[n1][n2]['path']
            path = Pointset(path)  # Make a visvis pointset
        else:
            path = Pointset(3)
            path.append(n1); path.append(n2)
        meshes.append( lineToMesh(path, radius, 8) )
    
    # Combine meshes and return
    if meshes:
        return combineMeshes(meshes)
    else:
        return None


def _detect_corners(path, th=5, smoothFactor=2.0, angTh=45):
    """ detectCorners(path, th=5, smoothFactor=3, angTh=45)
    Return the indices on the given path were corners are detected.
    """
    
    # Initialize angles
    angles = np.zeros((len(path),), dtype=np.float64)
    
    # angle to radians
    angTh = angTh * np.pi / 180
    
    for i in range(len(path)):
        
        # Are we far enough down the wire that we can calculate 
        # the vectors?
        if i >= th*2:
            
            # Get current point, point between vector, and point at end
            p1 = path[i]                
            p2 = path[i-th]  # center point              
            p3 = path[i-2*th]
            
            # Calculate vectors          
            vec1 = p1-p2
            vec2 = p2-p3
            
            # Calculate and store angle between them, (abs for 2D)
            angles[i-th] = abs(vec1.angle(vec2))
    
    # Smooth angles and take derivative
    angles2 = angles
    angles = gaussfun.gfilter(angles, smoothFactor)# * smoothFactor**0.5
    #print([int(i*180/np.pi) for i in angles])
    
    # Detect local maximuma
    tmp = angles[th:-th]
    localmax = (    (tmp > angles[th-1:-th-1])
                &   (tmp > angles[th+1:-th+1]) 
                &   (tmp > angTh)
                )
    
    # Return indices
    I, = np.where(localmax)
    return [i+th for i in I]


def _add_corner_to_edge(graph, n1, n2, **kwargs):
    
    # Get path and locations to split it
    edge = graph.edge[n1][n2]
    path = edge['path']
    I = _detect_corners(path, **kwargs)
    
    # Ensure n1 < n2
    if not np.isclose(path[0], n1, 0, 0.5).all():
        n1, n2 = n2, n1
    assert np.isclose(path[0], n1, 0, 0.5).all()
    assert np.isclose(path[-1], n2, 0, 0.5).all()
    
    
    if I:
        # Split path in multiple sections
        paths = []
        i_prev = 0
        for i in I:
            paths.append( path[i_prev:i+1] )  # note the overlap
            i_prev = i
        paths.append( path[i_prev:] )
        
        # Create new nodes (and insert)
        node2s = []
        for i in I:
            tmp = tuple(path[i].flat)
            node2s.append(tmp)
            graph.add_node(tmp, corner=True)
        node2s.append(n2)
        
        # Create new edges (and connect)
        node1 = n1
        for i in range(len(paths)):
            node2 = node2s[i]
            subpath = paths[i]
            if node1 > node2:  subpath = PointSet(np.flipud(subpath))
            graph.add_edge(node1, node2, path=subpath,
                           cost=edge['cost'], ctvalue=edge['ctvalue'])
            node1 = node2
        
        # Check whether we should still process a bit...
        popThisNode = None
        if n1 == n2 and len(I) > 1:
            popThisNode = n1
        
        # Disconnect old edge
        graph.remove_edge(n1, n2)
        
        # Pop node and process new bit if we have to.
        # We use recursion, but can only recurse once 
        if popThisNode and graph.degree(popThisNode) <= 2:                
            cnew = _pop_node(graph, popThisNode)
            if cnew is not None:
                _add_corner_to_edge(graph, *cnew, **kwargs)


def add_corner_nodes(graph, **kwargs):
    """ Detects positions on each edge where it bends and places a new
    node at these positions.
    """
    for n1, n2 in graph.edges():
        _add_corner_to_edge(graph, n1, n2, **kwargs)


def _get_pairs_of_neighbours(graph, node):
    """ For a given node, return all possible pairs of neighbours.
    Returns a list of tuple pairs. Each tuple has the smallest element 
    first. The list is also sorted.
    """
    neighbours = list(graph.edge[node].keys())
    
    pairs = []
    
    n = len(neighbours)
    for i1 in range(0, n):
        for i2 in range(i1+1, n):
            n1, n2 = neighbours[i1], neighbours[i2]
            if n1 > n2:
                n1, n2 = n2, n1
            pairs.append((n1, n2))
    
    pairs.sort()
    return pairs


def add_nodes_at_crossings(graph):
    """ Reposition crossings to discart duplicate paths; i.e. place a node
    at the position where two commin paths diverge.
    """
    # Check integrity of the graph
    check_path_integrity(graph)
    
    # Keep processing the whole graph until there are no more changes
    graph_changed = True
    while graph_changed:
        graph_changed = False
        
        # For each node ...
        for node in graph.nodes():
            
            # Process this node until there are no more changes
            graph_changed_now = True
            while graph_changed_now:
                graph_changed_now = _add_nodes_at_crossings_for_node(graph, node)
                graph_changed = graph_changed or graph_changed_now


def _add_nodes_at_crossings_for_node(graph, node):
    
    # Get all neightbour pars for the current node
    neighbour_pairs = _get_pairs_of_neighbours(graph, node)
    
    # Check the path of all pairs ...
    for node1, node2 in neighbour_pairs:
        
        # Get edge and path arrays
        edge1 = graph[node][node1]
        edge2 = graph[node][node2]
        path1, path2 = edge1['path'], edge2['path']
        
        # Flip paths if necessary so that path[0] == node for both paths
        if node > node1:  # if not np.isclose(path1[0], node).all()
            path1 = np.flipud(path1)
        if node > node2:  # if not np.isclose(path2[0], node).all()
            path2 = np.flipud(path2)
        
        # Walk the paths until they diverge
        maxwalk = min(path1.shape[0], path2.shape[0])
        pathbreak = 0
        for i in range(maxwalk):
            if not np.allclose(path1[i], path2[i]):
                pathbreak = i - 1
                break
        else:
            pathbreak = maxwalk
        
        if pathbreak == maxwalk:
            # One path is completely redundant: strip that path
            
            # Swap node1/node2 so that node1 has the shortest path
            if path1.shape[0] > path2.shape[0]:
                node1, node2 = node2, node1
                edge1, edge2 = edge2, edge1
                path1, path2 = path2, path1
            # Trim path2
            path2_ = path2[pathbreak-1:]
            if node1 > node2:  path2_= np.flipud(path2_)
            # Now remove edge node-node2 and replace with node1, node2
            graph.remove_edge(node, node2)
            graph.add_edge(node1, node2, path=PointSet(path2_), 
                        cost=edge2['cost'], ctvalue=edge2['ctvalue'])
            return True
        
        elif pathbreak > 0:
            # A part of the path was the same: insert new node
            
            # Define new node and paths
            new_node = tuple(path1[pathbreak].flat)
            commonpath = path1[:pathbreak+1]
            path1_ = path1[pathbreak:]
            path2_ = path2[pathbreak:]
            # Swap paths?
            if node > new_node:  commonpath = np.flipud(commonpath)
            if new_node > node1:  path1_= np.flipud(path1_)
            if new_node > node2:  path2_ = np.flipud(path2_)
            # Remove old edges
            graph.remove_edge(node, node1)
            graph.remove_edge(node, node2)
            # Add new edges
            graph.add_node(new_node, crossing=True)  # mark the node
            graph.add_edge(node, new_node, path=PointSet(commonpath), 
                    cost=min(edge1['cost'], edge2['cost']), 
                    ctvalue=max(edge1['ctvalue'], edge2['ctvalue']),)
            graph.add_edge(new_node, node1, path=PointSet(path1_), 
                        cost=edge1['cost'], ctvalue=edge1['ctvalue'])
            graph.add_edge( new_node, node2, path=PointSet(path2_), 
                        cost=edge2['cost'], ctvalue=edge2['ctvalue'])
            return True
    
    # No changes (note that this is beyond the loop
    return False


def smooth_paths(graph, ntimes=2, closed=False):
    for n1, n2 in graph.edges():
        path = graph.edge[n1][n2]['path']
        if not closed:
            for iter in range(ntimes):
                tmp = path[1:-1] + path[0:-2] + path[2:]
                path[1:-1] = tmp / 3.0
        elif closed:
            # if edge is closed and thus connected to self
            assert path[0].all() == path[-1].all() # compare PointSets
            assert graph.degree(n1) == 2 # not 3, not being connected to other node also
            for iter in range(ntimes):
                pp2 = path.copy()
                for i in range(0, path.shape[0]-1):
                    if i==0:
                        pp2[i] = (path[i-2] + path[i] + path[i+1]) / 3
                        pp2[-1] = pp2[i]
                    else:
                        pp2[i] = (path[i-1] + path[i] + path[i+1]) / 3
                path = pp2
            # Now change graph
            try:
                cost = graph.edge[n1][n2]['cost']
                ctvalue = graph.edge[n1][n2]['ctvalue']
                # remove old node and path
                graph.remove_node(n1)
                new_node = tuple(path[0].flat)
                graph.add_node(new_node) # mind that node attributes are ignored 
                graph.add_edge(new_node, new_node, path = path, cost = cost, ctvalue = ctvalue )
            except KeyError:
                # remove old node and path
                graph.remove_node(n1)
                new_node = tuple(path[0].flat)
                graph.add_node(new_node) # mind that node attributes are ignored 
                graph.add_edge(new_node, new_node, path = path )
            

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
    path_length = d.sum() 
    return path_length
