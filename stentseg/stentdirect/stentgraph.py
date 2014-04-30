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

import networkx as nx
from stentseg.utils.new_pointset import PointSet
import visvis as vv
from visvis import ssdf


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
    
    def draw(self, mc='g', lc='y', mw=7, lw=0.6, alpha=0.5, axes=None, simple=False):
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
            self.add_node(node, attr)
        for edge_info in s.edges:
            #node1, node2, attr = edge_info['node1'], edge_info['node2'], node_info['attr'].__dict__
            node1, node2, attr = edge_info
            node1 = node1 if not isinstance(node1, list) else tuple(node1)
            node2 = node2 if not isinstance(node2, list) else tuple(node2)
            attr = attr if isinstance(attr, dict) else attr.__dict__
            self.add_edge(node1, node2, attr)



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
    # Sorting not needed, we sort at the neightbours
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
            graph.remove_nodes_from(cluster)



def prune_redundant(graph, ctvalue):
    """
    Remove redundant edges. 
    
    A connection is redundant if a weak connection (high mcp cost) 
    connects two nodes which are already connected via two other nodes
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
        print('Warning: detected node that was connected to itself.')
        graph.remove_edge(n1, n2)
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



class TestStentGraph:
    
    
    def test_prune_redundant1(self):
        """ Test removing redundant edges on a graph with two triangles
        that are connected by a single edge.
        """
        
        # Create two triangles that are connected with a single edge
        graph = StentGraph()
        graph.add_edge(11, 12, cost=1, ctvalue=50)
        graph.add_edge(12, 13, cost=3, ctvalue=50)
        graph.add_edge(13, 11, cost=2, ctvalue=50)
        #
        graph.add_edge(21, 22, cost=2, ctvalue=60)
        graph.add_edge(22, 23, cost=3, ctvalue=60)
        graph.add_edge(23, 21, cost=1, ctvalue=60)
        #
        graph.add_edge(21, 11, cost=4, ctvalue=10)
        
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 7
        
        prune_redundant(graph, 55)
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 6
        
        prune_redundant(graph, 55)
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 6
        
        prune_redundant(graph, 65)
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 5
        
        prune_tails(graph, 2)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
    
    
    def test_prune_redundant2(self):
        """ Test removing redundant edges on a graph with two triangles
        that are connected by a two edges, twice.
        """
        
        # Create two triangles that are connected with a single edge
        graph = StentGraph()
        graph.add_edge(11, 12, cost=1, ctvalue=50)
        graph.add_edge(12, 13, cost=3, ctvalue=50)
        graph.add_edge(13, 11, cost=2, ctvalue=50)
        #
        graph.add_edge(21, 22, cost=2, ctvalue=60)
        graph.add_edge(22, 23, cost=3, ctvalue=60)
        graph.add_edge(23, 21, cost=1, ctvalue=60)
        #
        graph.add_edge(21, 1, cost=4, ctvalue=10)
        graph.add_edge(1, 11, cost=4, ctvalue=10)
        #
        graph.add_edge(22, 2, cost=4, ctvalue=10)
        graph.add_edge(2, 12, cost=4, ctvalue=10)
        
        
        assert graph.number_of_nodes() == 8
        assert graph.number_of_edges() == 10
        
        prune_redundant(graph, 55)
        assert graph.number_of_nodes() == 8
        assert graph.number_of_edges() == 10-1
        
        prune_redundant(graph, 55)
        assert graph.number_of_nodes() == 8
        assert graph.number_of_edges() == 10-1
        
        prune_redundant(graph, 65)
        assert graph.number_of_nodes() == 8
        assert graph.number_of_edges() == 10-2
        
        prune_tails(graph, 2)
        assert graph.number_of_nodes() == 8-2
        assert graph.number_of_edges() == 10-2-2
    
    
    def test_prune_tails(self):
        
        graph = StentGraph()
        graph.add_edge(1, 2, cost=2, ctvalue=50)
        graph.add_edge(2, 3, cost=2, ctvalue=50)
        graph.add_edge(3, 1, cost=2, ctvalue=50)
        
        # Tail from 1
        graph.add_edge(1, 11, cost=3, ctvalue=50)
        graph.add_edge(11, 12, cost=3, ctvalue=50)
        graph.add_edge(12, 13, cost=3, ctvalue=50)
        graph.add_edge(13, 14, cost=3, ctvalue=50)
        
        # Tail from 2
        graph.add_edge(2, 21, cost=3, ctvalue=50)
        graph.add_edge(21, 22, cost=3, ctvalue=50)
        graph.add_edge(22, 23, cost=3, ctvalue=50)
        
        assert graph.number_of_nodes() == 3+4+3
        assert graph.number_of_edges() == 3+4+3
        
        prune_tails(graph, 3)
        assert graph.number_of_nodes() == 3+4
        assert graph.number_of_edges() == 3+4
        
        prune_tails(graph, 9)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3
    
    
    def test_prune_clusters(self):
        # Create two small cliques
        graph = StentGraph()
        graph.add_edge(1, 2, cost=2, ctvalue=50)
        graph.add_edge(2, 3, cost=2, ctvalue=50)
        graph.add_edge(3, 1, cost=2, ctvalue=50)
        #
        graph.add_edge(4, 5, cost=2, ctvalue=50)
        graph.add_edge(5, 6, cost=2, ctvalue=50)
        graph.add_edge(6, 7, cost=2, ctvalue=50)
        graph.add_edge(7, 4, cost=2, ctvalue=50)
        
        # Connect them
        graph.add_edge(1, 4, cost=3, ctvalue=50)
        
        # Also add loose node
        graph.add_nodes_from([101, 102])
        
        # Remove cliques and check that nothing happened
        prune_clusters(graph, 4)
        assert graph.number_of_edges() == 8
        assert graph.number_of_nodes() == 7
        
        # Remove connection
        graph.remove_edge(1, 4)
        
        # Remove cliques and check that one clique is removed
        prune_clusters(graph, 4)
        assert graph.number_of_edges() == 4
        assert graph.number_of_nodes() == 4
        
        # Remove cliques and check that one clique is removed
        prune_clusters(graph, 5)
        assert graph.number_of_edges() == 0
        assert graph.number_of_nodes() == 0
    
    
    def test_very_weak(self):
        
        # Create simple graph
        graph = StentGraph()
        graph.add_edge(1, 4, ctvalue=50)
        graph.add_edge(1, 5, ctvalue=40)
        graph.add_edge(1, 2, ctvalue=30)
        graph.add_edge(1, 3, ctvalue=20)
        
        # Remove weak edges
        th = 35
        prune_very_weak(graph, th)
        
        # Check result
        assert graph.number_of_edges() == 2
        for (n1, n2) in graph.edges_iter():
            assert graph[n1][n2]['ctvalue'] > th
    
    
    def test_weak1(self):
        """ 2
          / | \
        5 - 1 - 3
          \ | /
            4 
        """
        
        # Test that indeed only weakest are removed
        graph = StentGraph()
        graph.add_edge(1, 2, cost=2, ctvalue=50)
        graph.add_edge(1, 3, cost=3, ctvalue=50)  # gets removed
        graph.add_edge(1, 4, cost=4, ctvalue=50)  # gets removed
        graph.add_edge(1, 5, cost=1, ctvalue=50)
        #
        graph.add_edge(2, 3, cost=1, ctvalue=50)
        graph.add_edge(3, 4, cost=1, ctvalue=50)
        graph.add_edge(4, 5, cost=1, ctvalue=50)
        graph.add_edge(5, 2, cost=1, ctvalue=50)
        
        prune_weak(graph, 2, 80)
        
        # Check result
        assert graph.number_of_edges() == 6
        for e in graph.edges_iter():
            assert e not in [(1, 3), (1, 4)]
    
    
    def test_weak2(self):
        """ 2     5
          / |     | \
        3 - 1  -  4 - 6
        """
        
        # Test that indeed only weakest are removed
        graph = StentGraph()
        graph.add_edge(1, 2, cost=2, ctvalue=50)
        graph.add_edge(2, 3, cost=2, ctvalue=50)
        graph.add_edge(3, 1, cost=2, ctvalue=50)
        #
        graph.add_edge(4, 5, cost=2, ctvalue=50)
        graph.add_edge(5, 6, cost=2, ctvalue=50)
        graph.add_edge(6, 4, cost=2, ctvalue=50)
        
        # Connect two subgraphs with weaker connection
        graph.add_edge(1, 4, cost=3, ctvalue=50)
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 6
        for e in graph.edges_iter():
            assert e not in [(1, 4)]
        
        
        # Again, now with lower cost (stronger connection)
        graph.add_edge(1, 4, cost=1, ctvalue=50)
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 7
        
        
        # Again, now with high ct value
        graph.add_edge(1, 4, cost=3, ctvalue=90)
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 7
    
    
    def test_weak3(self):
        """ 2     456
          / |     |
        3 - 1  -  0 - 789
        """
        
        # Test that indeed only weakest are removed
        graph = StentGraph()
        graph.add_edge(1, 2, cost=2, ctvalue=50)
        graph.add_edge(2, 3, cost=2, ctvalue=50)
        graph.add_edge(3, 1, cost=2, ctvalue=50)
        #
        graph.add_edge(4, 5, cost=2, ctvalue=50)
        graph.add_edge(5, 6, cost=2, ctvalue=50)
        graph.add_edge(6, 4, cost=2, ctvalue=50)
        #
        graph.add_edge(7, 8, cost=2, ctvalue=50)
        graph.add_edge(8, 9, cost=2, ctvalue=50)
        graph.add_edge(9, 7, cost=2, ctvalue=50)
        
        
        # Connect three subgraphs
        graph.add_edge(0, 1, cost=2, ctvalue=50)
        graph.add_edge(0, 4, cost=3, ctvalue=50)  # gets removed
        graph.add_edge(0, 7, cost=2, ctvalue=50)
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 9+2
        for e in graph.edges_iter():
            assert e not in [(0, 4)]
        
        
        # Connect three subgraphs
        graph.add_edge(0, 1, cost=1, ctvalue=50)
        graph.add_edge(0, 4, cost=1, ctvalue=50)
        graph.add_edge(0, 7, cost=2, ctvalue=50)  # gets removed
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 9+2
        for e in graph.edges_iter():
            assert e not in [(0, 7)]
        
        
        # Connect three subgraphs
        graph.add_edge(0, 1, cost=3, ctvalue=50)
        graph.add_edge(0, 4, cost=4, ctvalue=90)  # None gets removed
        graph.add_edge(0, 7, cost=3, ctvalue=50)
        
        # Prune
        prune_weak(graph, 2, 80)
        # Check result
        assert graph.number_of_edges() == 9+3
    
    
    
    def test_pack1(self):
        # Custom stent
        g = StentGraph(summary='dit is een stent!', lala=3)
        g.add_node((10,20), foo=3)
        g.add_node((30,40), foo=5)
        g.add_edge((1,1), (2,2), bar=10)
        g.add_edge((10,20),(1,1), bar=20)
        
        fname = '/home/almar/test.ssdf'
        ssdf.save(fname, g.pack())
        
        g2 = StentGraph()
        g2.unpack(ssdf.load(fname))
        
        #print(nx.is_isomorphic(g, g2))
        assert nx.is_isomorphic(g, g2)
    
    
    def test_pack2(self):
        # Auto generate
        import random
        n = 500
        p=dict((i,(random.gauss(0,2),random.gauss(0,2))) for i in range(n))
        g_ = nx.random_geometric_graph(n, 0.1, dim=3, pos=p)
        
        g = StentGraph(summary='dit is een stent!', lala=3)
        g.add_nodes_from(g_.nodes_iter())
        g.add_edges_from(g_.edges_iter())
        
        
        fname = '/home/almar/test.ssdf'
        ssdf.save(fname, g.pack())
        
        g2 = StentGraph()
        g2.unpack(ssdf.load(fname))
        
        #print(nx.is_isomorphic(g, g2))
        assert nx.is_isomorphic(g, g2)



if __name__ == "__main__":
    
    # Run test. Nose is acting weird. So wrote a little test runner myself:
    test = TestStentGraph()
    for m in dir(test):
        if m.startswith('test_'):
            print('Running %s ... ' % m, end='')
            try:
                getattr(test, m)()
            except AssertionError as err:
                print('Fail')
                raise
            except Exception:
                print('Error')
                raise
            else:
                print("Ok")
    
    # Create simple graph
    graph = StentGraph()
    graph.add_edge(1, 4, cost=5)
    graph.add_edge(1, 5, cost=4)
    graph.add_edge(1, 2, cost=3)
    graph.add_edge(1, 3, cost=2)