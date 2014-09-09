from __future__ import print_function, division, absolute_import

import numpy as np
import networkx as nx
from visvis import ssdf

from stentseg.utils.new_pointset import PointSet
from stentseg.stentdirect.stentgraph import (StentGraph, check_path_integrity,
        _get_pairs_of_neighbours, add_nodes_at_crossings, 
        _detect_corners, _add_corner_to_edge, 
        _pop_node, pop_nodes, 
        prune_very_weak, prune_weak,
        prune_clusters, prune_redundant, prune_tails,)


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
    
    
    def test_pop_node(self):
        
        # Create paths
        path1 = PointSet(2)
        path1.append(1, 11)
        path1.append(1, 12)
        path2 = PointSet(2)
        path2.append(1, 12)
        path2.append(1, 13)
        #
        path12 = PointSet(2)
        path12.append(1, 11)
        path12.append(1, 12) 
        path12.append(1, 13)
        
        # create 4 nodes (6-7-8-9), remove 8
        graph = StentGraph()
        graph.add_edge(6, 7, cost=4, ctvalue=70)
        graph.add_edge(7, 8, cost=2, ctvalue=50, path=path1)
        graph.add_edge(8, 9, cost=3, ctvalue=60, path=path2)
        
        # Pop
        _pop_node(graph, 8)
        
        # Check
        assert graph.number_of_nodes() == 3
        assert 8 not in graph.nodes()
        assert graph.edge[7][9]['ctvalue'] == 50
        assert graph.edge[7][9]['cost'] == 5
        assert np.all(graph.edge[7][9]['path'] == path12)
        
        
        # create 4 nodes (6-8-7-9), remove 7
        graph = StentGraph()
        graph.add_edge(6, 8, cost=4, ctvalue=70)
        graph.add_edge(8, 7, cost=2, ctvalue=50, path=np.flipud(path1))
        graph.add_edge(7, 9, cost=3, ctvalue=60, path=path2)
        
        # Pop
        _pop_node(graph, 7)
        
        # Check
        assert graph.number_of_nodes() == 3
        assert 7 not in graph.nodes()
        assert graph.edge[8][9]['ctvalue'] == 50
        assert graph.edge[8][9]['cost'] == 5
        assert np.all(graph.edge[8][9]['path'] == path12)
        
        
        # create 4 nodes (7-8-6-9), remove 8
        graph = StentGraph()
        graph.add_edge(7, 8, cost=4, ctvalue=70, path=np.flipud(path2))
        graph.add_edge(8, 6, cost=2, ctvalue=50, path=path1)
        graph.add_edge(6, 9, cost=3, ctvalue=60)
        
        # Pop
        _pop_node(graph, 8)
        
        # Check
        assert graph.number_of_nodes() == 3
        assert 8 not in graph.nodes()
        assert graph.edge[6][7]['ctvalue'] == 50
        assert graph.edge[6][7]['cost'] == 6
        assert np.all(graph.edge[6][7]['path'] == path12)
        
        
        # create 3 nodes in a cycle. It should remove all but one
        graph = StentGraph()
        graph.add_edge(7, 8, cost=4, ctvalue=70, path=path1)
        graph.add_edge(8, 9, cost=2, ctvalue=50, path=path2)
        graph.add_edge(9, 7, cost=3, ctvalue=60, path=path2)
        
        # Pop
        _pop_node(graph, 8)
        
        # Check
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 1
        assert 8 not in graph.nodes()
        n = graph.nodes()[0]
        assert len(graph.edge[n][n]['path']) == 6-1
        
        
        # create 3 nodes in a cycle, with one subbranch
        graph = StentGraph()
        graph.add_edge(7, 8, cost=4, ctvalue=70, path=path1)
        graph.add_edge(8, 9, cost=2, ctvalue=50, path=path2)
        graph.add_edge(9, 7, cost=3, ctvalue=60, path=path2)
        graph.add_edge(7, 4, cost=3, ctvalue=60, path=path2)
        
        # Pop
        _pop_node(graph, 8)
        
        # Check
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 2
        assert 8 not in graph.nodes()
        assert len(graph.edge[7][7]['path']) == 6-1
    
    
    def test_pop_nodes(self):
        
        # Create dummy paths
        path1 = PointSet(2)
        path1.append(1, 11)
        path1.append(1, 12)
        
        # create 4 nodes (6-7-8-9), remove 8
        graph = StentGraph()
        graph.add_edge(6, 7, cost=4, ctvalue=70, path=path1)
        graph.add_edge(7, 8, cost=2, ctvalue=50, path=path1)
        graph.add_edge(8, 9, cost=3, ctvalue=60, path=path1)
        graph0 = graph.copy()
        
        # Pop straight line
        graph = graph0.copy()
        pop_nodes(graph)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph.edge[6][9]['path'].shape[0] == 3+1
        
        # Pop cycle
        graph = graph0.copy()
        graph.add_edge(9, 6, cost=3, ctvalue=60, path=path1)
        pop_nodes(graph)
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 1
        n = graph.nodes()[0]
        assert graph.edge[n][n]['path'].shape[0] == 4+1+1 # cycle
        # arbitrary what node stayed around
        
        # Pop with one side branch popping
        graph = graph0.copy()
        graph.add_edge(7, 2, cost=3, ctvalue=60, path=path1)
        pop_nodes(graph)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 3
        assert graph.edge[7][9]['path'].shape[0] == 2+1
        
        # Pop with one prevent popping
        graph = graph0.copy()
        graph.node[7]['nopop'] = True
        pop_nodes(graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        assert graph.edge[7][9]['path'].shape[0] == 2+1

    
    def test_detect_corners(self):
        path = PointSet(3)
        path.append(10, 2, 0)
        path.append(11, 3, 0)
        path.append(12, 4, 0)
        path.append(13, 5, 0)
        path.append(14, 6, 0)  
        path.append(15, 7, 0)  # top
        path.append(16, 6, 0)
        path.append(17, 5, 0)
        path.append(18, 4, 0)
        path.append(19, 3, 0)
        path.append(20, 2, 0)  # bottom
        path.append(21, 3, 0)
        path.append(22, 4, 0)
        path.append(23, 5, 0)
        path.append(24, 6, 0)
        path.append(25, 7, 0)  # top
        path.append(26, 6, 0)
        path.append(27, 5, 0)
        path.append(28, 4, 0)
        path.append(29, 3, 0)
        path0 = path
        
        for i in range(3):
            path = path0.copy()
            path[:,2] = path[:,i]
            path[:,i] = 0
                
            # Test that _detect_corners detects the indices correctly
            I = _detect_corners(path, smoothFactor=1)
            assert I == [5, 10, 15]
            
            # Test that _add_corner_to_edge constructs the graph and splits 
            # the path in the correct way
            graph = StentGraph()
            n1, n5 = tuple(path[0].flat), tuple(path[-1].flat)
            n2, n3, n4 =  tuple(path[5].flat),  tuple(path[10].flat),  tuple(path[15].flat)
            graph.add_edge(n1, n5, path=path, cost=0, ctvalue=0)
            _add_corner_to_edge(graph, n1, n5, smoothFactor=1)
            
            assert graph.number_of_nodes() == 5
            assert graph.number_of_edges() == 4
            for n in [n1, n2, n3, n4, n5]:
                assert n in graph.nodes()
            path12, path23, path34, path45 = path[0:6], path[5:11], path[10:16], path[15:20]
            if n1 > n2: path12 = np.flipud(path12)
            if n2 > n3: path23 = np.flipud(path23)
            if n3 > n4: path34 = np.flipud(path34)
            if n4 > n5: path45 = np.flipud(path45)
            assert np.all(graph.edge[n1][n2]['path'] == path12)
            assert np.all(graph.edge[n2][n3]['path'] == path23)
            assert np.all(graph.edge[n3][n4]['path'] == path34)
            assert np.all(graph.edge[n4][n5]['path'] == path45)
    
    
    def test_pairs(self):
    
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(1, 5)
        #
        graph.add_edge(2, 6)
        graph.add_edge(2, 7)
        #
        graph.add_edge(3, 8)
        
        pairs1 = _get_pairs_of_neighbours(graph, 1)
        assert pairs1 == [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        
        pairs2 = _get_pairs_of_neighbours(graph, 2)
        assert pairs2 == [(1, 6), (1, 7), (6, 7)]
        
        pairs3 = _get_pairs_of_neighbours(graph, 3)
        assert pairs3 == [(1, 8)]
        
        pairs4 = _get_pairs_of_neighbours(graph, 4)
        assert pairs4 == []
        


    def test_add_nodes_at_crossings1(self):
        # N4---N1=====-------N2
        #             |
        #             N3
        path1 = PointSet(3)  # path from n1 to n2
        path1.append(10, 2, 0)
        path1.append(10, 3, 0)
        path1.append(10, 4, 0)
        path1.append(10, 5, 0)
        #
        path3 = path1.copy()  # path to n3
        #
        path1.append(10, 6, 0)
        path1.append(10, 7, 0)
        path1.append(10, 8, 0)
        #
        path3.append(11, 5, 0)
        path3.append(12, 5, 0)
        path3.append(13, 5, 0)
        #
        path4 = PointSet(3)  # path to n4
        path4.append(10, 0, 0)
        path4.append(10, 1, 0)
        path4.append(10, 2, 0)
        
        graph = nx.Graph()
        n1 = tuple(path1[0].flat)
        n2 = tuple(path1[-1].flat)
        n3 = tuple(path3[-1].flat)
        n4 = tuple(path4[0].flat)
        graph.add_edge(n1, n2, path=path1, cost=3, ctvalue=3)
        graph.add_edge(n1, n3, path=path3, cost=3, ctvalue=3)
        graph.add_edge(n1, n4, path=path4, cost=3, ctvalue=3)
        
        # Pre-check
        assert len(graph.nodes()) == 4
        for n in (n1, n2, n3, n4):
            assert n in graph.nodes()
        # Deal with crossongs
        add_nodes_at_crossings(graph)
        # Check result
        check_path_integrity(graph)
        assert len(graph.nodes()) == 5
        added_node = 10, 5, 0
        for n in (n1, n2, n3, n4, added_node):
            assert n in graph.nodes()
    
    
    def test_add_nodes_at_crossings2(self):
        
        # N4---N1=====-------====N2
        #             |     |
        #             N3    N5
        path1 = PointSet(3)  # path from n1 to n2
        path1.append(10, 2, 0)
        path1.append(10, 3, 0)
        path1.append(10, 4, 0)
        path1.append(10, 5, 0)
        #
        path3 = path1.copy()  # path to n3
        #
        path1.append(10, 6, 0)
        path1.append(10, 7, 0)
        path1.append(10, 8, 0)
        path1.append(10, 9, 0)
        #
        path3.append(11, 5, 0)
        path3.append(12, 5, 0)
        path3.append(13, 5, 0)
        #
        path4 = PointSet(3)  # path to n4
        path4.append(10, 0, 0)
        path4.append(10, 1, 0)
        path4.append(10, 2, 0)
        # 
        path5 = PointSet(3)  # path from n2 to n5 (note the order)
        path5.append(10, 9, 0)  # dup path1
        path5.append(10, 8, 0)  # dup path1
        path5.append(10, 7, 0)  # dup path1
        path5.append(11, 7, 0)
        path5.append(12, 7, 0)
        path5.append(13, 7, 0)
        
        graph = nx.Graph()
        n1 = tuple(path1[0].flat)
        n2 = tuple(path1[-1].flat)
        n3 = tuple(path3[-1].flat)
        n4 = tuple(path4[0].flat)
        n5 = tuple(path5[-1].flat)
        graph.add_edge(n1, n2, path=path1, cost=3, ctvalue=3)
        graph.add_edge(n1, n3, path=path3, cost=3, ctvalue=3)
        graph.add_edge(n1, n4, path=path4, cost=3, ctvalue=3)
        graph.add_edge(n5, n2, path=path5, cost=3, ctvalue=3)
        
        # Pre-check
        assert len(graph.nodes()) == 5
        for n in (n1, n2, n3, n4, n5):
            assert n in graph.nodes()
        # Deal with crossongs
        add_nodes_at_crossings(graph)
        # Check result
        check_path_integrity(graph)
        assert len(graph.nodes()) == 7
        added_node1 = 10, 5, 0
        added_node2 = 10, 7, 0
        for n in (n1, n2, n3, n4, n5, added_node1, added_node2):
            assert n in graph.nodes()
    
    
    def test_add_nodes_at_crossings3(self):
        # N4---N1>>>>>======-------N2
        #             |     |
        #             N3    N5
        path1 = PointSet(3)  # path from n1 to n2
        path1.append(10, 2, 0)
        path1.append(10, 3, 0)
        path1.append(10, 4, 0)
        path1.append(10, 5, 0)
        #
        path3 = path1.copy()  # path to n3
        path3.append(11, 5, 0)
        path3.append(12, 5, 0)
        path3.append(13, 5, 0)
        # 
        path1.append(10, 6, 0)
        path1.append(10, 7, 0)
        #
        path5 = path1.copy()
        path5.append(11, 7, 0)
        path5.append(12, 7, 0)
        path5.append(13, 7, 0)
        #
        path1.append(10, 8, 0)
        path1.append(10, 9, 0)
        #
        path4 = PointSet(3)  # path to n4
        path4.append(10, 0, 0)
        path4.append(10, 1, 0)
        path4.append(10, 2, 0)
        
        graph = nx.Graph()
        n1 = tuple(path1[0].flat)
        n2 = tuple(path1[-1].flat)
        n3 = tuple(path3[-1].flat)
        n4 = tuple(path4[0].flat)
        n5 = tuple(path5[-1].flat)
        graph.add_edge(n1, n2, path=path1, cost=3, ctvalue=3)
        graph.add_edge(n1, n3, path=path3, cost=3, ctvalue=3)
        graph.add_edge(n1, n4, path=path4, cost=3, ctvalue=3)
        graph.add_edge(n1, n5, path=path5, cost=3, ctvalue=3)
        
        # Pre-check
        assert len(graph.nodes()) == 5
        for n in (n1, n2, n3, n4, n5):
            assert n in graph.nodes()
        # Deal with crossongs
        add_nodes_at_crossings(graph)
        # Check result
        check_path_integrity(graph)
        assert len(graph.nodes()) == 7
        added_node1 = 10, 5, 0
        added_node2 = 10, 7, 0
        for n in (n1, n2, n3, n4, n5, added_node1, added_node2):
            assert n in graph.nodes()
    

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
