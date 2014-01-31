"""
Next version for stentseg-specific graph class and functionality.
Use networkx instead of vispy.util.graph.
In progress. Step 1 and step2 of stentdirect work now.
"""

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
        
        # Convert edges to Pointset
        ppe = PointSet(3)
        for n1, n2 in self.edges_iter():
            ppe.append(*n1)
            ppe.append(*n2)
        
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
    
    The edges for each node are evaluated. If an edge is elligible
    for removal, it is checked whether the same is true when evaluating
    that edge from the other side.
    
    Which of the edges fall in the expected set is determined based on 
    the cost. Of the remaining edges, the lowest CT value on the path
    is used to decide which edges are considered strong enough.
    
    Rationale: Edges with a path that contains a very low CT
    value can always be discarted. Edges in excess of the expected
    number of edges can only be maintained if their lowest CT 
    value is so high it proves the existance of a wire between
    the nodes.
    """
    
    # First, get a sorted list of edges
    edges = graph.edges()
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
    
    popped_nodes = set()
    
    def _pop(node, count):
        if count > maxlength:
            pass
        elif nx.degree(graph, node) == 1:
            # Pop
            graph.remove_node(node)
            popped_nodes.add(node)
            # Next
            nextnode = graph.edge[node].keys()[0]
            _pop(nextnode, count+1)
    
    for node in graph.nodes():
        if node in popped_nodes:
            pass
        else:
            _pop(node, 0)



def remove_clusters(graph, minsize):
    """ Remove all small clusters of interconnected nodes (i.e.
    connected components) that are smaller than minsize nodes.
    """
    # Thanks to networkx this one is easy!
    for cluster in list(nx.connected_components(graph)):
        if len(cluster) < n:
            graph.remove_nodes_from(cluster)



class TestStentGraph:
    
    
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
        
        prune_tails(graph, 4)
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 5
        # todo: I was working right here ...
        
        
        
        
        
    def test_remove_clusters(self):
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
        remove_clusters(graph, 4)
        assert graph.number_of_edges() == 8
        assert graph.number_of_nodes() == 7
        
        # Remove connection
        graph.remove_edge(1, 4)
        
        # Remove cliques and check that one clique is removed
        remove_clusters(graph, 4)
        assert graph.number_of_edges() == 4
        assert graph.number_of_nodes() == 4
        
        # Remove cliques and check that one clique is removed
        remove_clusters(graph, 5)
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